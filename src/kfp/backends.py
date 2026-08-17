"""Inference backends.

The pipeline must be runnable end to end *before* any GPU is available,
otherwise you cannot tell whether a bad number is a model failure or a plumbing
failure. Three backends, one interface:

  EchoBackend       deterministic template answer built from the prompt.
                    Not a model. Exists to test plumbing only.
  RuleBackend       answers by consulting the rule engine. This is the ORACLE.
  HFBackend         real transformers model, loaded lazily so that importing
                    this module never requires torch.

Every backend returns raw text, which then goes through the same parser as a
real model. That matters: it means the parser is exercised by the smoke test
rather than being first tried out on real generations.
"""
from __future__ import annotations

import re
from typing import Protocol


class Backend(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str:
        ...


class EchoBackend:
    """Plumbing test only. Picks the first product named in the prompt.

    Do NOT report this as a system. It is deliberately dumb: if it scores above
    chance on the personalisation metrics, the metrics are broken.
    """

    name = "echo"

    def __init__(self, catalog: dict):
        self.names = {
            p["product_id"]: p.get("name_ru", p["product_id"])
            for p in catalog["products"]
        }

    def generate(self, system: str, user: str) -> str:
        m = re.search(r"^(hb_[a-z_]+) — (.+?);", user, re.M)
        if not m:
            return "Сізге қолайлы өнім табылмады."
        return f"Сізге {m.group(2)} ұсынамын."


class RuleBackend:
    """Oracle: renders the rule engine's decision as natural language.

    Provides the ceiling for every metric and, just as usefully, a *reference
    answer* for supervised fine-tuning that is not the output of another LLM.
    This is what keeps the training data free of the self-distillation loop.
    """

    name = "oracle_rules"

    def __init__(self, catalog: dict):
        self.by_id = {p["product_id"]: p for p in catalog["products"]}
        self._scenario: dict | None = None

    def bind(self, scenario: dict) -> None:
        self._scenario = scenario

    def generate(self, system: str, user: str) -> str:
        s = self._scenario
        if s is None:
            raise RuntimeError("RuleBackend.bind(scenario) must be called first")
        if s["gold_action"] == "refuse":
            reason = ""
            for item in s["ineligible_products"]:
                if item["violated_rules"]:
                    reason = item["violated_rules"][0]
                    break
            return (
                "Өкінішке қарай, қазіргі жағдайыңызға сай өнім ұсына алмаймын. "
                f"Себебі: талаптарға сай келмейсіз ({reason})."
            )
        p = self.by_id[s["gold_product_id"]]
        a = p.get("attributes", {})
        parts = [f"Сізге {p.get('name_ru', p['product_id'])} ұсынамын"]
        if "rate_annual_pct" in a:
            parts.append(f"жылдық мөлшерлеме {a['rate_annual_pct']}%")
        if "min_amount_kzt" in a:
            parts.append(f"ең төменгі сома {a['min_amount_kzt']} теңге")
        text = ", ".join(parts) + "."
        if s["gold_action"] == "refer_to_advisor":
            text += " Нақты шарттарды бөлімшедегі маманмен талқылағаныңыз жөн."
        return text


class HFBackend:
    """Real model. torch/transformers imported lazily inside __init__."""

    def __init__(self, model_id: str, max_new_tokens: int = 220,
                 temperature: float = 0.0, seed: int = 0,
                 load_in_4bit: bool = True, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id.split("/")[-1]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch = torch
        torch.manual_seed(seed)

        kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                pass

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.name += f"+lora({adapter_path.rstrip('/').split('/')[-1]})"
        self.model.eval()

    def generate(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()


class SegmentRoutedHFBackend:
    """The proposed method: one adapter per behavioural segment, routed at
    inference time by the client's segment.

    Adapters are loaded once and switched via PEFT's named-adapter API rather
    than reloading the base model per request, which would make the evaluation
    unusably slow.
    """

    def __init__(self, model_id: str, adapter_paths: dict[int, str], **kw):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"{model_id.split('/')[-1]}+seglora(k={len(adapter_paths)})"
        self.max_new_tokens = kw.get("max_new_tokens", 220)
        self.temperature = kw.get("temperature", 0.0)
        self.torch = torch
        torch.manual_seed(kw.get("seed", 0))

        self.tok = AutoTokenizer.from_pretrained(model_id)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto")

        items = sorted(adapter_paths.items())
        first_seg, first_path = items[0]
        self.model = PeftModel.from_pretrained(
            base, first_path, adapter_name=f"seg{first_seg}")
        for segid, path in items[1:]:
            self.model.load_adapter(path, adapter_name=f"seg{segid}")
        self.model.eval()
        self._segment = first_seg

    def bind_segment(self, segment: int) -> None:
        self._segment = segment
        self.model.set_adapter(f"seg{segment}")

    def generate(self, system: str, user: str) -> str:
        return HFBackend.generate(self, system, user)  # type: ignore[arg-type]
