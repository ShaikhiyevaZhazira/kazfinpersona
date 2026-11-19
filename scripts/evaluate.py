#!/usr/bin/env python3
"""Evaluate systems on KazFinPersona.

Three modes:

  --predictions FILE   score one model checkpoint (JSONL, one row per
                       scenario with `scenario_id` and `answer_text`). The
                       answer is RE-PARSED here, so a parser change never
                       requires regenerating outputs.

  --simulate           score the synthetic reference policies on the full test
                       split. This is a *metric validation* run: the oracle
                       must score 1.000, a profile-blind policy must score
                       CC = 0. Run it before trusting any model number.

  --compare FILE       the table that goes in the paper: the model checkpoint
                       and every reference policy scored on EXACTLY THE SAME
                       scenario ids (the ones present in the checkpoint). The
                       full-test simulation numbers are NOT comparable with a
                       model scored on a 200-item sample, because the sample
                       is deliberately enriched in counterfactual origins.

Parser: strict is the primary result; lenient (one-edit product ids, unique
short names) is reported next to it as a sensitivity row.
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
from kfp.eval.parsing import AnswerParser, PARSER_VERSION  # noqa: E402


# ------------------------------------------------------------------ loading

def load_scenarios(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_predictions(path: str) -> dict[str, dict]:
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            row = json.loads(l)
            out[row["scenario_id"]] = row
    return out


def build_records(scenarios: list[dict], predictions: dict,
                  parser: AnswerParser | None = None) -> list[dict]:
    """Join gold with predictions.

    If a parser is given and the prediction carries `answer_text`, the
    prediction is re-derived from the text. Policies have no text and keep
    their stored pred_* fields.
    """
    recs = []
    for s in scenarios:
        p = predictions.get(s["scenario_id"])
        if p is None:
            continue
        text = p.get("answer_text", "")
        if parser is not None and text is not None and "answer_text" in p:
            q = parser.parse(text)
            pred_action, pred_pid = q["action"], q["product_id"]
            unparseable = q["unparseable"]
            numbers = q["asserted_numbers"]
            rec_ids = q["recommended_ids"]
        else:
            pred_action = p.get("pred_action")
            pred_pid = p.get("pred_product_id")
            unparseable = p.get("unparseable", False)
            numbers = p.get("asserted_numbers", [])
            rec_ids = p.get("recommended_ids")        # None for policies
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
            "pred_action": pred_action,
            "pred_product_id": pred_pid,
            "asserted_numbers": numbers,
            "answer_text": text,
            "unparseable": unparseable,
            "recommended_ids": rec_ids,
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
    """Knows the client's goal-relevant product but ignores eligibility rules.

    NOTE: this is an *oracle-assisted* policy, not a deployable one. It picks
    the gold product whenever the gold product serves the question's goal, so
    its product-match accuracy is 1.000 by construction; otherwise it takes
    the first goal-relevant product in the catalogue. Its role is to show what
    "perfect topical fit, zero rule awareness" looks like on the compliance
    metrics (constraint_violation_rate), not to act as a competitive baseline.
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
    """Collapse onto the least restrictive product: zero violations, zero
    personalization."""
    name = "always_open"

    def predict(self, s):
        pid = "hb_deposit_qorjin"
        return {"pred_action": "recommend", "pred_product_id": pid,
                "asserted_numbers": self._numbers_for(pid)}


POLICIES = (OraclePolicy, ProfileBlindPolicy, EligibilityBlindPolicy,
            NoisyOraclePolicy, AlwaysOpenPolicy)


# ------------------------------------------------------------------ scoring

def score(records: list[dict], catalog: dict) -> dict:
    pairs = make_pairs(records)
    base_only = [r for r in records if not r["counterfactual_of"]]
    out = {
        "action_accuracy": M.action_accuracy(base_only),
        "product_match_accuracy": M.product_match_accuracy(base_only),
        "product_match_single": M.product_match_single(base_only),
        "multi_product_rate": M.multi_product_rate(base_only),
        "constraint_violation_rate": M.constraint_violation_rate(base_only),
        "hard_violation_rate": M.hard_violation_rate(base_only),
        "profile_sensitivity": M.profile_sensitivity(pairs),
        "counterfactual_correctness": M.counterfactual_correctness(pairs),
        "sensitivity_gap": M.sensitivity_correctness_gap(pairs),
        "answer_diversity": M.answer_diversity(base_only),
        "unparseable_rate": M.unparseable_rate(base_only),
        "profile_sensitivity_parsed": M.profile_sensitivity_parsed(pairs),
    }
    out.update(M.counts(base_only, pairs))
    out.update(M.refusal_metrics(base_only))
    out.update(M.fact_grounding(base_only, catalog))
    return out


def per_item_correct(records: list[dict]) -> dict[str, float]:
    """scenario_id -> 1/0 (action and product both right), base items only."""
    return {
        r["scenario_id"]: 1.0 if (r["pred_action"] == r["gold_action"]
                                  and r["pred_product_id"] == r["gold_product_id"]) else 0.0
        for r in records if not r["counterfactual_of"]
    }


def run_policies(catalog, train, items, seeds):
    """Score every reference policy on `items` for each seed.

    Returns (table, per_item_vectors_of_seed0, pair_breakdown_of_seed0).
    """
    table, vecs, brk = {}, {}, {}
    for cls in POLICIES:
        runs = []
        for seed in seeds:
            pol = cls(catalog, train, seed=seed)
            preds = {s["scenario_id"]: pol.predict(s) for s in items}
            recs = build_records(items, preds)
            runs.append(score(recs, catalog))
            if seed == seeds[0]:
                vecs[cls.name] = per_item_correct(recs)
                brk[cls.name] = M.pair_breakdown(make_pairs(recs))
        table[cls.name] = S.aggregate_seeds(runs)
    return table, vecs, brk


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"


COLS = ["action_accuracy", "product_match_accuracy", "answer_diversity",
        "profile_sensitivity", "counterfactual_correctness",
        "constraint_violation_rate", "refusal_recall"]


def print_table(rows: dict[str, dict], cols=COLS) -> None:
    print(f"\n{'system':<24}" + "".join(f"{c[:14]:>16}" for c in cols))
    print("-" * (24 + 16 * len(cols)))
    for name, row in rows.items():
        print(f"{name:<24}" + "".join(f"{_fmt(row.get(c)):>16}" for c in cols))


def flat_mean(row: dict) -> dict:
    """aggregate_seeds output ({metric: {'mean':..}}) -> {metric: mean}."""
    return {k: (v["mean"] if isinstance(v, dict) and "mean" in v else v)
            for k, v in row.items()}


def score_model(path, test, catalog, lenient):
    parser = AnswerParser(catalog, lenient=lenient)
    preds = load_predictions(path)
    recs = build_records(test, preds, parser=parser)
    return recs, score(recs, catalog)


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v0.1"))
    ap.add_argument("--catalog", default=str(ROOT / "config" / "products_halyk.json"))
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--compare", default=None,
                    help="model checkpoint JSONL; policies are scored on the "
                         "same scenario ids")
    ap.add_argument("--parser", choices=["strict", "lenient"], default="strict",
                    help="parser mode for --predictions (primary = strict)")
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
    print(f"parser v{PARSER_VERSION}")

    # ---- one model
    if args.predictions:
        recs, res = score_model(args.predictions, test, catalog,
                                args.parser == "lenient")
        res["parser"] = f"v{PARSER_VERSION}-{args.parser}"
        (outdir / "metrics.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=2, default=float))
        print("\npair breakdown:",
              json.dumps(M.pair_breakdown(make_pairs(recs)), indent=2, default=float))
        return

    # ---- model vs reference policies on identical items
    if args.compare:
        ids = set(load_predictions(args.compare))
        items = [s for s in test if s["scenario_id"] in ids]
        missing = ids - {s["scenario_id"] for s in items}
        if missing:
            print(f"WARNING: {len(missing)} checkpoint ids are not in the test "
                  f"split (e.g. {sorted(missing)[:3]})")
        table, vecs, brk = run_policies(catalog, train, items, args.seeds)

        rows = {k: flat_mean(v) for k, v in table.items()}
        model_res = {}
        for mode in ("strict", "lenient"):
            recs, res = score_model(args.compare, test, catalog, mode == "lenient")
            model_res[mode] = res
            rows[f"MODEL ({mode})"] = res
            if mode == "strict":
                model_vec = per_item_correct(recs)
                model_brk = M.pair_breakdown(make_pairs(recs))

        print(f"\nsame-items comparison: n_items={len(items)} "
              f"n_base={int(rows['oracle']['n_base'])} "
              f"n_pairs={int(rows['oracle']['n_pairs'])}")
        print_table(rows)

        # Wilson interval on the model's headline proportions
        m = model_res["strict"]
        k_pm = round(m["product_match_accuracy"] * m["n_gold_recommend"])
        print("\nmodel PMA  = %s  [%s, %s]  (n=%d gold-recommend)" % (
            _fmt(m["product_match_accuracy"]),
            *[_fmt(x) for x in M.wilson(k_pm, m["n_gold_recommend"])],
            m["n_gold_recommend"]))
        k_ps = round(m["profile_sensitivity"] * m["n_pairs"])
        k_cc = round(m["counterfactual_correctness"] * m["n_pairs"])
        print("model PS   = %s  [%s, %s]  (n=%d pairs)" % (
            _fmt(m["profile_sensitivity"]),
            *[_fmt(x) for x in M.wilson(k_ps, m["n_pairs"])], m["n_pairs"]))
        print("model CC   = %s  [%s, %s]  (n=%d pairs)" % (
            _fmt(m["counterfactual_correctness"]),
            *[_fmt(x) for x in M.wilson(k_cc, m["n_pairs"])], m["n_pairs"]))

        # paired tests, model vs each policy, aligned by scenario id
        tests = {}
        for name, vec in vecs.items():
            common = sorted(set(vec) & set(model_vec))
            tests[f"model_vs_{name}"] = S.paired_test(
                [model_vec[i] for i in common], [vec[i] for i in common])
        ok = {k: v["pvalue"] for k, v in tests.items()
              if not np.isnan(v["pvalue"])}
        corrected = S.holm_bonferroni(ok) if ok else {}
        print("\npaired tests, model vs policy (Holm-corrected):")
        for k, v in corrected.items():
            print(f"  {k:<32} p_adj={v['p_adjusted']:.2e}  significant={v['significant']}")

        print("\npair breakdown (model, strict):")
        for k, v in model_brk.items():
            print(f"  {k:<28} n={v['n']:<3} PS={_fmt(v['profile_sensitivity'])} "
                  f"CC={_fmt(v['counterfactual_correctness'])}")

        (outdir / "compare.json").write_text(json.dumps({
            "parser_version": PARSER_VERSION, "n_items": len(items),
            "policies": table, "model": model_res,
            "model_pair_breakdown": model_brk, "policy_pair_breakdown": brk,
            "tests": tests, "corrected": corrected,
        }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\nwritten to {outdir / 'compare.json'}")
        return

    # ---- metric validation run on the full test split
    if not args.simulate:
        ap.error("choose one of --predictions, --compare, --simulate")

    table, vecs, _ = run_policies(catalog, train, test, args.seeds)
    print_table({k: flat_mean(v) for k, v in table.items()})

    tests = {}
    for name, vec in vecs.items():
        if name == "oracle":
            continue
        ids = sorted(vecs["oracle"])
        tests[f"oracle_vs_{name}"] = S.paired_test(
            [vecs["oracle"][i] for i in ids], [vec[i] for i in ids])
    corrected = S.holm_bonferroni({k: v["pvalue"] for k, v in tests.items()
                                   if not np.isnan(v["pvalue"])})
    print("\npaired tests vs oracle (Holm-corrected):")
    for k, v in corrected.items():
        print(f"  {k:<32} p_adj={v['p_adjusted']:.2e}  significant={v['significant']}")

    (outdir / "simulation.json").write_text(
        json.dumps({"table": table, "tests": tests, "corrected": corrected},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (outdir / "simulation_table.tex").write_text(
        S.results_to_latex(table, COLS,
                           caption="Metric validation on synthetic policies.",
                           label="tab:metric-validation"), encoding="utf-8")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
