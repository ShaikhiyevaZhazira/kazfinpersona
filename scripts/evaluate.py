#!/usr/bin/env python3
"""Evaluate systems on KazFinPersona.

Two modes:

  --predictions FILE   score a real model's outputs (JSONL with
                       {scenario_id, answer_text} or {scenario_id,
                       pred_action, pred_product_id})

  --simulate           score four synthetic policies whose behaviour is known
                       by construction. This is a *metric validation* run: it
                       checks that the metric suite separates a model that
                       genuinely uses the client profile from one that has
                       memorised `intent -> most frequent product`.

Run --simulate before trusting any number this suite produces on a real model.
If a profile-blind policy scores well on your personalization metric, the
metric is broken, not the policy.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfp import eligibility  # noqa: E402
from kfp.eval import metrics as M, stats as S  # noqa: E402


# ------------------------------------------------------------------ loading

def load_scenarios(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build_records(scenarios: list[dict], predictions: dict) -> list[dict]:
    recs = []
    for s in scenarios:
        p = predictions.get(s["scenario_id"])
        if p is None:
            continue
        recs.append({
            "scenario_id": s["scenario_id"],
            "client_id": s["client_id"],
            "intent": s["intent"],
            "gold_action": s["gold_action"],
            "gold_product_id": s["gold_product_id"],
            "eligible_products": s["eligible_products"],
            "ineligible_products": s["ineligible_products"],
            "rationale_facts": s["rationale_facts"],
            "counterfactual_of": s["counterfactual_of"],
            "pred_action": p.get("pred_action"),
            "pred_product_id": p.get("pred_product_id"),
            "asserted_numbers": p.get("asserted_numbers", []),
            "answer_text": p.get("answer_text", ""),
            "unparseable": p.get("unparseable", False),
        })
    return recs


def make_pairs(records: list[dict]) -> list[tuple[dict, dict]]:
    by_id = {r["scenario_id"]: r for r in records}
    pairs = []
    for r in records:
        origin = r.get("counterfactual_of")
        if origin and origin in by_id:
            pairs.append((by_id[origin], r))
    return pairs


# --------------------------------------------------------------- simulation

class Policy:
    """Base class for synthetic policies used in metric validation."""
    name = "base"

    def __init__(self, catalog, train_scenarios, seed=0):
        self.catalog = catalog
        self.by_id = {p["product_id"]: p for p in catalog["products"]}
        self.rng = np.random.default_rng(seed)
        self.train = train_scenarios

    def predict(self, s: dict) -> dict:
        raise NotImplementedError

    def _numbers_for(self, pid):
        if not pid:
            return []
        attrs = self.by_id[pid].get("attributes", {})
        out = []
        for v in attrs.values():
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out


class OraclePolicy(Policy):
    """Perfect knowledge of the rules. Upper bound."""
    name = "oracle"

    def predict(self, s):
        return {"pred_action": s["gold_action"],
                "pred_product_id": s["gold_product_id"],
                "asserted_numbers": self._numbers_for(s["gold_product_id"])}


class ProfileBlindPolicy(Policy):
    """Answers from the question alone: intent -> modal product in train.

    This is the failure mode the previous paper could not detect. It should
    score respectably on product-match accuracy and ~0 on counterfactual
    correctness. If it does not, the metric suite is wrong.
    """
    name = "profile_blind"

    def __init__(self, catalog, train_scenarios, seed=0):
        super().__init__(catalog, train_scenarios, seed)
        counts = collections.defaultdict(collections.Counter)
        for s in train_scenarios:
            if s["gold_action"] == "recommend" and s["gold_product_id"]:
                counts[s["intent"]][s["gold_product_id"]] += 1
        self.mode = {k: c.most_common(1)[0][0] for k, c in counts.items()}

    def predict(self, s):
        pid = self.mode.get(s["intent"])
        return {"pred_action": "recommend" if pid else "refuse",
                "pred_product_id": pid,
                "asserted_numbers": self._numbers_for(pid)}


class EligibilityBlindPolicy(Policy):
    """Uses the profile for topical fit but ignores the eligibility rules.

    Designed to score well on accuracy-family metrics while producing a high
    constraint-violation rate — i.e. exactly the model a bank cannot deploy.
    """
    name = "eligibility_blind"

    def predict(self, s):
        on_goal = [p["product_id"] for p in self.catalog["products"]
                   if s["intent"] in p.get("goals", [])]
        if not on_goal:
            return {"pred_action": "refuse", "pred_product_id": None,
                    "asserted_numbers": []}
        pid = s["gold_product_id"] if s["gold_product_id"] in on_goal else on_goal[0]
        return {"pred_action": "recommend", "pred_product_id": pid,
                "asserted_numbers": self._numbers_for(pid)}


class NoisyOraclePolicy(Policy):
    """Oracle corrupted with probability p. Models a competent but imperfect
    system, so the metric suite is exercised in the middle of its range."""
    name = "noisy_oracle"

    def __init__(self, catalog, train_scenarios, seed=0, p_err=0.3):
        super().__init__(catalog, train_scenarios, seed)
        self.p_err = p_err

    def predict(self, s):
        if self.rng.random() > self.p_err:
            return {"pred_action": s["gold_action"],
                    "pred_product_id": s["gold_product_id"],
                    "asserted_numbers": self._numbers_for(s["gold_product_id"])}
        ids = list(self.by_id.keys())
        pid = ids[int(self.rng.integers(len(ids)))]
        return {"pred_action": "recommend", "pred_product_id": pid,
                "asserted_numbers": [float(self.rng.integers(5, 40))]}

class AlwaysOpenPolicy(Policy):
    """Коллапс на самый нестрогий продукт: нулевые нарушения, нулевая персонализация."""
    name = "always_open"

    def predict(self, s):
        pid = "hb_deposit_qorjin"
        return {"pred_action": "recommend", "pred_product_id": pid,
                "asserted_numbers": self._numbers_for(pid)}


# ------------------------------------------------------------------ scoring

def score(records: list[dict], catalog: dict) -> dict:
    pairs = make_pairs(records)
    base_only = [r for r in records if not r["counterfactual_of"]]
    out = {
        "action_accuracy": M.action_accuracy(base_only),
        "product_match_accuracy": M.product_match_accuracy(base_only),
        "constraint_violation_rate": M.constraint_violation_rate(base_only),
        "hard_violation_rate": M.hard_violation_rate(base_only),
        "profile_sensitivity": M.profile_sensitivity(pairs),
        "counterfactual_correctness": M.counterfactual_correctness(pairs),
        "sensitivity_gap": M.sensitivity_correctness_gap(pairs),
        "answer_diversity": M.answer_diversity(base_only),
        "unparseable_rate": M.unparseable_rate(base_only),
        "profile_sensitivity_parsed": M.profile_sensitivity_parsed(pairs),
        "n_pairs": len(pairs),
    }
    out.update(M.refusal_metrics(base_only))
    out.update(M.fact_grounding(base_only, catalog))
    return out


def per_item_correct(records: list[dict]) -> list[float]:
    """Per-item score vector for the paired significance tests."""
    return [
        1.0 if (r["pred_action"] == r["gold_action"]
                and r["pred_product_id"] == r["gold_product_id"]) else 0.0
        for r in records if not r["counterfactual_of"]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v0.1"))
    ap.add_argument("--catalog", default=str(ROOT / "config" / "products_halyk.json"))
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "results" / "eval"))
    args = ap.parse_args()

    data = Path(args.data)
    catalog = eligibility.load_catalog(args.catalog)
    scenarios = load_scenarios(data / "scenarios.jsonl")
    train = [s for s in scenarios if s["split"] == "train"]
    test = [s for s in scenarios if s["split"] == "test"]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.simulate:
        preds = {json.loads(l)["scenario_id"]: json.loads(l)
                 for l in open(args.predictions, encoding="utf-8")}
        recs = build_records(test, preds)
        res = score(recs, catalog)
        (outdir / "metrics.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # ---- metric validation run
    table, per_item = {}, {}
    for cls in (OraclePolicy, ProfileBlindPolicy, EligibilityBlindPolicy, NoisyOraclePolicy, AlwaysOpenPolicy):
        runs, vecs = [], None
        for seed in args.seeds:
            pol = cls(catalog, train, seed=seed)
            preds = {s["scenario_id"]: pol.predict(s) for s in test}
            recs = build_records(test, preds)
            runs.append(score(recs, catalog))
            if vecs is None:
                vecs = per_item_correct(recs)
        table[cls.name] = S.aggregate_seeds(runs)
        per_item[cls.name] = vecs

    cols = ["action_accuracy", "product_match_accuracy", "answer_diversity",
            "profile_sensitivity", "counterfactual_correctness",
            "constraint_violation_rate", "refusal_recall"]

    print(f"\n{'system':<20}" + "".join(f"{c[:14]:>16}" for c in cols))
    print("-" * (20 + 16 * len(cols)))
    for sysname, row in table.items():
        cells = ""
        for c in cols:
            d = row.get(c)
            cells += f"{d['mean']:>16.3f}" if d and not np.isnan(d["mean"]) else f"{'--':>16}"
        print(f"{sysname:<20}{cells}")

    tests = {}
    for name in per_item:
        if name == "oracle":
            continue
        tests[f"oracle_vs_{name}"] = S.paired_test(per_item["oracle"], per_item[name])
    corrected = S.holm_bonferroni({k: v["pvalue"] for k, v in tests.items()
                                   if not np.isnan(v["pvalue"])})

    print("\npaired tests vs oracle (Holm-corrected):")
    for k, v in corrected.items():
        print(f"  {k:<32} p_adj={v['p_adjusted']:.2e}  significant={v['significant']}")

    (outdir / "simulation.json").write_text(
        json.dumps({"table": table, "tests": tests, "corrected": corrected},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (outdir / "simulation_table.tex").write_text(
        S.results_to_latex(table, cols,
                           caption="Metric validation on synthetic policies.",
                           label="tab:metric-validation"), encoding="utf-8")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
