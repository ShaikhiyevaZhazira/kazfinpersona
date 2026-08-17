#!/usr/bin/env python3
"""Build the KazFinPersona dataset end to end.

    python scripts/build_dataset.py --n-clients 3000 --seed 42 --out data/v0.1

Everything is deterministic given --seed. Run it with three seeds and report
mean +/- std, per the experimental protocol.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfp import counterfactual, eligibility, profiles as prof_mod  # noqa: E402
from kfp import scenarios as scen_mod, split as split_mod          # noqa: E402
from kfp import suitability, transactions as txn_mod, validate     # noqa: E402


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=3000)
    ap.add_argument("--cold-start-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats", default=str(ROOT / "config" / "kz_stats.yaml"))
    ap.add_argument("--catalog", default=str(ROOT / "config" / "products_halyk.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "v0.1"))
    ap.add_argument("--as-of", default="2026-06-01")
    args = ap.parse_args()

    out = Path(args.out)
    cfg = yaml.safe_load(open(args.stats, encoding="utf-8"))
    catalog = eligibility.load_catalog(args.catalog)
    as_of = dt.date.fromisoformat(args.as_of)

    import collections as _c
    prov = _c.Counter(p.get("eligibility_source", "assumed") for p in catalog["products"])
    print(f"[0/6] catalogue: {len(catalog['products'])} products, eligibility provenance {dict(prov)}")
    if prov.get("assumed", 0) > 0:
        print(f"      WARNING: {prov['assumed']} products have ASSUMED eligibility rules.")
        print("      The compliance metric measures those assumptions, not Halyk policy.")
        print("      This must be stated in the paper's Limitations until they are collected.")

    print(f"[1/6] generating {args.n_clients} client profiles ...")
    clients = prof_mod.generate_profiles(cfg, args.n_clients, args.seed)

    print("[2/6] generating transactions and life events ...")
    n_cold = int(args.n_clients * args.cold_start_frac)
    all_txns, all_events, feats_by_client, cold_flags = [], [], {}, {}
    for i, p in enumerate(clients):
        cold = i < n_cold
        cold_flags[p.client_id] = cold
        txns, events = txn_mod.generate_transactions(
            p, cfg, seed=args.seed + i, end_date=as_of, cold_start=cold
        )
        all_txns.extend(t.to_dict() for t in txns)
        all_events.extend(e.to_dict() for e in events)
        feats_by_client[p.client_id] = suitability.behavioural_features(p, txns, as_of)

    print("[3/6] building scenarios from the rule engine ...")
    all_scen = []
    for i, p in enumerate(clients):
        all_scen.extend(
            scen_mod.build_scenarios(
                p, feats_by_client[p.client_id], catalog,
                eligibility, suitability,
                seed=args.seed + 100000 + i,
                cold_start=cold_flags[p.client_id],
            )
        )

    print("[4/6] assigning leak-safe splits ...")
    all_scen = split_mod.assign_splits(all_scen, seed=args.seed)
    test_clients = {s.client_id for s in all_scen if s.split == "test"}

    print(f"[5/6] generating counterfactual pairs (test clients only: {len(test_clients)}) ...")
    cf_profiles, cf_scen = [], []
    by_client = {}
    for s in all_scen:
        by_client.setdefault(s.client_id, []).append(s)
    for i, p in enumerate(clients):
        if p.client_id not in test_clients:
            continue
        scs = by_client.get(p.client_id, [])
        if not scs:
            continue
        cp, cs = counterfactual.make_counterfactuals(
            p, feats_by_client[p.client_id], scs, catalog,
            eligibility, suitability, seed=args.seed + 200000 + i,
        )
        for s in cs:
            origin = next(x for x in cp if x.client_id == s.client_id)
            verdicts = eligibility.evaluate_all(origin, catalog)
            s.rationale_facts = scen_mod.rationale_facts(
                catalog, s.gold_product_id, s.gold_action, verdicts, origin
            )
        cf_profiles.extend(cp)
        cf_scen.extend(cs)

    scen_all = split_mod.assign_splits(all_scen + cf_scen, seed=args.seed)
    report = split_mod.split_report(scen_all)

    print("[6/6] validating distributions ...")
    val_rows = validate.run_all(clients, cfg)

    write_jsonl(out / "clients.jsonl", [c.to_dict() for c in clients])
    write_jsonl(out / "clients_counterfactual.jsonl", [c.to_dict() for c in cf_profiles])
    write_jsonl(out / "transactions.jsonl", all_txns)
    write_jsonl(out / "life_events.jsonl", all_events)
    write_jsonl(out / "scenarios.jsonl", [s.to_dict() for s in scen_all])

    (out / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "validation.json").write_text(
        json.dumps(val_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "validation_table.tex").write_text(validate.to_latex(val_rows), encoding="utf-8")
    (out / "build_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- split report ---")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\ntransactions: {len(all_txns)}   life events: {len(all_events)}")
    print(f"counterfactual clients: {len(cf_profiles)}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
