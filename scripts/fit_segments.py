#!/usr/bin/env python3
"""Fit behavioural segments on the training split and assign them everywhere.

    python scripts/fit_segments.py --data data/v0.4 --seed 42

Writes:
  <data>/segments.json        client_id -> segment
  <data>/segment_model.pkl    fitted transformer + kmeans (train-only)
  results/segments/           silhouette curve, centroid table, LaTeX
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfp import segments as seg, suitability  # noqa: E402
from kfp.schemas import ClientProfile, Transaction  # noqa: E402


def load(data: Path, as_of: dt.date):
    clients = {}
    for line in open(data / "clients.jsonl", encoding="utf-8"):
        d = json.loads(line)
        clients[d["client_id"]] = ClientProfile(**d)
    for f in ("clients_counterfactual.jsonl",):
        p = data / f
        if p.exists():
            for line in open(p, encoding="utf-8"):
                d = json.loads(line)
                clients[d["client_id"]] = ClientProfile(**d)

    tx = collections.defaultdict(list)
    for line in open(data / "transactions.jsonl", encoding="utf-8"):
        d = json.loads(line)
        tx[d["client_id"]].append(Transaction(**d))

    # A counterfactual client is the SAME person with exactly one profile
    # attribute changed, so it inherits the origin's transaction history.
    # Generating fresh (or empty) histories for them was a real defect: with no
    # transactions every behavioural feature collapsed to zero, all
    # counterfactuals landed in one cluster, and the test split came out 72%
    # one segment against 13% in train -- a distribution shift entirely
    # manufactured by the pipeline.
    #
    # Inheriting is also the right causal manipulation: holding behaviour fixed
    # means any change in the model's answer must come from the perturbed
    # attribute. A useful side effect is that both members of a pair route to
    # the same adapter, so routing cannot explain the difference either.
    feats = {}
    for cid, p in clients.items():
        own = tx.get(cid, [])
        if not own and "-CF-" in cid:
            own = tx.get(cid.split("-CF-")[0], [])
        feats[cid] = suitability.behavioural_features(p, own, as_of)
    return clients, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v0.4"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=None, help="skip selection, force k")
    ap.add_argument("--as-of", default="2026-06-01")
    ap.add_argument("--out", default=str(ROOT / "results" / "segments"))
    args = ap.parse_args()

    data = Path(args.data)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    as_of = dt.date.fromisoformat(args.as_of)

    clients, feats = load(data, as_of)

    # split membership comes from the scenarios file; a counterfactual client
    # inherits test by construction
    split_of = {}
    for line in open(data / "scenarios.jsonl", encoding="utf-8"):
        s = json.loads(line)
        split_of[s["client_id"]] = s["split"]

    train_ids = [c for c in clients if split_of.get(c) == "train"]
    all_ids = list(clients.keys())
    if not train_ids:
        raise SystemExit("no training clients found - check scenarios.jsonl")

    Xtr = seg.to_matrix(feats, train_ids)
    Xall = seg.to_matrix(feats, all_ids)

    if args.k is None:
        sel = seg.select_k(Xtr, seed=args.seed)
        k = sel["best_k"]
        print("silhouette by k (fitted on train only):")
        for kk, v in sel["curve"].items():
            mark = "  <-- selected" if kk == k else ""
            print(f"  k={kk:2d}  {v:.4f}{mark}")
        spread = max(sel["curve"].values()) - min(sel["curve"].values())
        if sel["best_score"] < 0.25:
            print(f"\n  NOTE: peak silhouette {sel['best_score']:.3f} is low. The behavioural")
            print("  space has no crisp cluster structure. Report segments as a pragmatic")
            print("  partition for adapter routing, NOT as discovered client types.")
        if spread < 0.05:
            print(f"  NOTE: curve is flat (range {spread:.3f}); k is weakly identified.")
    else:
        sel = {"curve": {}, "best_k": args.k, "best_score": None}
        k = args.k

    model = seg.SegmentModel(k, seed=args.seed).fit(Xtr)
    labels_all = model.predict(Xall)
    assign = {cid: int(l) for cid, l in zip(all_ids, labels_all)}

    (data / "segments.json").write_text(
        json.dumps(assign, ensure_ascii=False), encoding="utf-8")
    seg.save(model, str(data / "segment_model.pkl"))

    desc = seg.describe_segments(Xall, labels_all)

    # sanity: are segments balanced enough to train an adapter on each?
    print(f"\nsegment sizes (k={k}, assigned over {len(all_ids)} clients):")
    for d in desc:
        tag = "  <-- too small to train an adapter" if d["n"] < 100 else ""
        print(f"  seg {d['segment']}  n={d['n']:5d}  {d['share']:6.1%}{tag}")
        for f in d["distinctive"]:
            print(f"        {f['feature']:20s} {f['value']:>12,.3f}  (z={f['z_vs_population']:+.2f})")

    # segment composition per split, to confirm the mapping transfers
    by_split = collections.defaultdict(collections.Counter)
    for cid, l in assign.items():
        by_split[split_of.get(cid, "unknown")][l] += 1
    print("\nsegment share by split (train-fitted model applied to val/test):")
    for sp in ("train", "val", "test"):
        tot = sum(by_split[sp].values()) or 1
        row = "  ".join(f"s{c}:{by_split[sp][c]/tot:5.1%}" for c in range(k))
        print(f"  {sp:6s} n={tot:5d}   {row}")

    (outdir / "selection.json").write_text(
        json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "segments_description.json").write_text(
        json.dumps(desc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {outdir} and {data}")


if __name__ == "__main__":
    main()
