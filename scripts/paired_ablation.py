#!/usr/bin/env python3
"""Paired analysis of two checkpoints scored on the SAME scenario ids.

    python scripts/paired_ablation.py --data data/v0.8 \
        --a results/preds/<run_none>/predictions.jsonl \
        --b results/preds/<run_profile>/predictions.jsonl \
        --label-a none --label-b profile

What it adds to `evaluate.py --compare`:

  * exact McNemar test on product match (gold-recommend base items) -- the
    test for statements like "24 versus 31 of 85";
  * the number of DISTINCT ORIGIN scenarios among the counterfactual pairs
    (pairs that share an origin are not independent);
  * PS and CC with a cluster bootstrap over origins (percentile 95% interval),
    and the paired difference B - A with the same bootstrap.

Strict parser, parser version taken from kfp.eval.parsing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


# ----------------------------------------------------------- pure functions
def pma_vector(recs: list[dict]) -> dict[str, float]:
    """scenario_id -> 1/0 product match, gold-recommend BASE items only."""
    return {r["scenario_id"]: float(r["pred_product_id"] == r["gold_product_id"])
            for r in recs if r["gold_action"] == "recommend" and not r["counterfactual_of"]}


def mcnemar_exact(a: dict[str, float], b: dict[str, float]) -> dict:
    ids = sorted(set(a) & set(b))
    only_a = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    only_b = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"n_items": len(ids),
            "correct_a": int(sum(a[i] for i in ids)),
            "correct_b": int(sum(b[i] for i in ids)),
            "only_a_correct": only_a, "only_b_correct": only_b, "p_exact": p}


def pair_indicators(pairs: list[tuple[dict, dict]]) -> list[tuple[str, float, float]]:
    """(origin_id, ps_i, cc_i, cf_id) per pair; same definitions as kfp.eval.metrics."""
    out = []
    for base, cf in pairs:
        changed = (base["pred_action"], base["pred_product_id"]) != (cf["pred_action"], cf["pred_product_id"])
        correct = (cf["pred_action"] == cf["gold_action"]
                   and cf["pred_product_id"] == cf["gold_product_id"])
        out.append((base["scenario_id"], float(changed), float(changed and correct), cf["scenario_id"]))
    return out


def _cluster_arrays(ind):
    clusters = defaultdict(list)
    for o, ps, cc, cf_id in ind:
        clusters[o].append((ps, cc, cf_id))
    keys = sorted(clusters)
    return keys, clusters


def cluster_bootstrap(ind, n_boot=10000, seed=0) -> dict:
    """PS and CC with a percentile interval from resampling ORIGINS."""
    keys, cl = _cluster_arrays(ind)
    if not keys:
        return {"n_pairs": 0, "n_origins": 0}
    s_ps = np.array([sum(x[0] for x in cl[k]) for k in keys])
    s_cc = np.array([sum(x[1] for x in cl[k]) for k in keys])
    n = np.array([len(cl[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    ps_b = s_ps[idx].sum(1) / n[idx].sum(1)
    cc_b = s_cc[idx].sum(1) / n[idx].sum(1)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n_pairs": int(n.sum()), "n_origins": len(keys),
            "max_pairs_per_origin": int(n.max()),
            "PS": float(s_ps.sum() / n.sum()), "PS_ci": q(ps_b),
            "CC": float(s_cc.sum() / n.sum()), "CC_ci": q(cc_b)}


def cluster_bootstrap_diff(ind_a, ind_b, n_boot=10000, seed=0) -> dict:
    """B - A on pairs present in both, resampling origins."""
    da = {x[3]: x for x in ind_a}
    db = {x[3]: x for x in ind_b}
    common = sorted(set(da) & set(db))
    clusters = defaultdict(list)
    for c in common:
        clusters[da[c][0]].append((da[c][1], db[c][1], da[c][2], db[c][2]))
    keys = sorted(clusters)
    if not keys:
        return {}
    arr = {m: np.array([[sum(x[j] for x in clusters[k]) for k in keys]] ).ravel()
           for m, j in (("psa", 0), ("psb", 1), ("cca", 2), ("ccb", 3))}
    n = np.array([len(clusters[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    nb = n[idx].sum(1)
    d_ps = (arr["psb"][idx].sum(1) - arr["psa"][idx].sum(1)) / nb
    d_cc = (arr["ccb"][idx].sum(1) - arr["cca"][idx].sum(1)) / nb
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    tot = n.sum()
    return {"n_pairs": int(tot), "n_origins": len(keys),
            "dPS": float((arr["psb"].sum() - arr["psa"].sum()) / tot), "dPS_ci": q(d_ps),
            "dCC": float((arr["ccb"].sum() - arr["cca"].sum()) / tot), "dCC_ci": q(d_cc)}


# --------------------------------------------------------------------- main
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    import evaluate as E
    from kfp import eligibility
    from kfp.eval.parsing import AnswerParser, PARSER_VERSION

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--catalog", default=str(root / "config" / "products_halyk.json"))
    ap.add_argument("--a", required=True, help="checkpoint JSONL of the reference run")
    ap.add_argument("--b", required=True, help="checkpoint JSONL of the run to compare")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=str(root / "results" / "eval" / "paired_ablation.json"))
    args = ap.parse_args()

    catalog = eligibility.load_catalog(args.catalog)
    scen = E.load_scenarios(Path(args.data) / "scenarios.jsonl")
    test = [s for s in scen if s["split"] == "test"]
    parser = AnswerParser(catalog, lenient=False)
    recs = {}
    for lab, path in (("a", args.a), ("b", args.b)):
        recs[lab] = E.build_records(test, E.load_predictions(path), parser=parser)

    ids_a = {r["scenario_id"] for r in recs["a"]}
    ids_b = {r["scenario_id"] for r in recs["b"]}
    if ids_a != ids_b:
        print(f"WARNING: checkpoints differ in items ({len(ids_a ^ ids_b)} ids); using the intersection")
    common = ids_a & ids_b
    for lab in recs:
        recs[lab] = [r for r in recs[lab] if r["scenario_id"] in common]

    mc = mcnemar_exact(pma_vector(recs["a"]), pma_vector(recs["b"]))
    ind = {lab: pair_indicators(E.make_pairs(recs[lab])) for lab in recs}
    boot_a = cluster_bootstrap(ind["a"], args.boot)
    boot_b = cluster_bootstrap(ind["b"], args.boot)
    diff = cluster_bootstrap_diff(ind["a"], ind["b"], args.boot)

    la, lb = args.label_a, args.label_b
    print(f"parser v{PARSER_VERSION} (strict); items in common: {len(common)}")
    print(f"\nPMA (gold-recommend base items): {la} {mc['correct_a']}/{mc['n_items']}, "
          f"{lb} {mc['correct_b']}/{mc['n_items']}")
    print(f"  correct only in {la}: {mc['only_a_correct']}; only in {lb}: {mc['only_b_correct']}; "
          f"exact McNemar p = {mc['p_exact']:.4f}")
    for lab, b in ((la, boot_a), (lb, boot_b)):
        print(f"\n{lab}: {b['n_pairs']} pairs from {b['n_origins']} distinct origins "
              f"(max {b['max_pairs_per_origin']} per origin)")
        print(f"  PS = {b['PS']:.3f} [{b['PS_ci'][0]:.3f}, {b['PS_ci'][1]:.3f}]   "
              f"CC = {b['CC']:.3f} [{b['CC_ci'][0]:.3f}, {b['CC_ci'][1]:.3f}]  (cluster bootstrap over origins)")
    if diff:
        print(f"\n{lb} - {la}: dPS = {diff['dPS']:+.3f} [{diff['dPS_ci'][0]:+.3f}, {diff['dPS_ci'][1]:+.3f}]   "
              f"dCC = {diff['dCC']:+.3f} [{diff['dCC_ci'][0]:+.3f}, {diff['dCC_ci'][1]:+.3f}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"parser_version": PARSER_VERSION, "mcnemar_pma": mc,
                                          "bootstrap_a": boot_a, "bootstrap_b": boot_b,
                                          "difference_b_minus_a": diff}, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
