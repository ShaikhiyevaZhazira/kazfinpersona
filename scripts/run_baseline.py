#!/usr/bin/env python3
"""Run a baseline over the test split and write predictions.jsonl.

No GPU:
    python scripts/run_baseline.py --data data/v0.4 --backend oracle
    python scripts/run_baseline.py --data data/v0.4 --backend echo

With GPU:
    python scripts/run_baseline.py --data data/v0.4 --backend hf \\
        --model Qwen/Qwen2.5-7B-Instruct --context profile --language kk

Output goes to results/preds/<run_id>/predictions.jsonl and is scored by
    python scripts/evaluate.py --data data/v0.4 --predictions <that file>

The run_id encodes every axis of the experiment, so two runs can never
silently overwrite each other -- a mistake that quietly destroys ablation
tables.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfp import backends as B, eligibility, prompts, suitability  # noqa: E402
from kfp.eval.parsing import AnswerParser  # noqa: E402
from kfp.schemas import ClientProfile, Transaction  # noqa: E402


def load_all(data: Path, as_of: dt.date):
    clients = {}
    for f in ("clients.jsonl", "clients_counterfactual.jsonl"):
        p = data / f
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            clients[d["client_id"]] = ClientProfile(**d)

    tx = collections.defaultdict(list)
    for line in open(data / "transactions.jsonl", encoding="utf-8"):
        d = json.loads(line)
        tx[d["client_id"]].append(Transaction(**d))

    feats = {}
    for cid, p in clients.items():
        own = tx.get(cid, [])
        if not own and "-CF-" in cid:
            own = tx.get(cid.split("-CF-")[0], [])   # see fit_segments.py
        feats[cid] = suitability.behavioural_features(p, own, as_of)

    segs = {}
    sp = data / "segments.json"
    if sp.exists():
        segs = json.load(open(sp, encoding="utf-8"))

    scen = [json.loads(l) for l in open(data / "scenarios.jsonl", encoding="utf-8")]
    return clients, feats, segs, scen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v0.4"))
    ap.add_argument("--catalog", default=str(ROOT / "config" / "products_halyk.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", choices=["echo", "oracle", "hf", "seglora"],
                    default="oracle")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter-dir", default=None,
                    help="for seglora: dir containing seg0/, seg1/, ...")
    ap.add_argument("--context", choices=["none", "persona", "profile", "full"],
                    default="profile")
    ap.add_argument("--catalogue", choices=["closed", "listed", "filtered"],
                    default="listed")
    ap.add_argument("--language", choices=["kk", "ru", "mixed"], default="kk")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--as-of", default="2026-06-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    catalog = eligibility.load_catalog(args.catalog)
    clients, feats, segs, scen = load_all(data, dt.date.fromisoformat(args.as_of))
    items = [s for s in scen if s["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    tag = args.model.split("/")[-1] if args.model else args.backend
    run_id = (f"{tag}__ctx-{args.context}__cat-{args.catalogue}"
              f"__lang-{args.language}__seed{args.seed}")
    outdir = Path(args.out) if args.out else ROOT / "results" / "preds" / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    if args.backend == "echo":
        backend = B.EchoBackend(catalog)
    elif args.backend == "oracle":
        backend = B.RuleBackend(catalog)
    elif args.backend == "hf":
        if not args.model:
            raise SystemExit("--model required for --backend hf")
        backend = B.HFBackend(args.model, seed=args.seed, adapter_path=args.adapter)
    else:
        if not (args.model and args.adapter_dir):
            raise SystemExit("--model and --adapter-dir required for seglora")
        adir = Path(args.adapter_dir)
        paths = {int(p.name.replace("seg", "")): str(p)
                 for p in sorted(adir.glob("seg*")) if p.is_dir()}
        if not paths:
            raise SystemExit(f"no seg*/ subdirs under {adir}")
        backend = B.SegmentRoutedHFBackend(args.model, paths, seed=args.seed)

    parser = AnswerParser(catalog)
    t0 = time.time()
    rows = []
    for i, s in enumerate(items, 1):
        cid = s["client_id"]
        profile = clients[cid].to_dict()
        seg = segs.get(cid)

        system, user = prompts.build_prompt(
            s, profile, catalog,
            feats=feats.get(cid), segment=seg,
            context=args.context, catalogue=args.catalogue,
            language=args.language,
        )
        if isinstance(backend, B.RuleBackend):
            backend.bind(s)
        if isinstance(backend, B.SegmentRoutedHFBackend) and seg is not None:
            backend.bind_segment(int(seg))

        text = backend.generate(system, user)
        parsed = parser.parse(text)
        rows.append({
            "scenario_id": s["scenario_id"],
            "client_id": cid,
            "segment": seg,
            "answer_text": text,
            "pred_action": parsed["action"],
            "pred_product_id": parsed["product_id"],
            "asserted_numbers": parsed["asserted_numbers"],
            "unparseable": parsed["unparseable"],
        })
        if i % 200 == 0:
            print(f"  {i}/{len(items)}  {(time.time()-t0)/i:.2f}s/item")

    with open(outdir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg = vars(args) | {"run_id": run_id, "n_items": len(rows),
                        "backend_name": getattr(backend, "name", args.backend),
                        "elapsed_s": round(time.time() - t0, 1)}
    (outdir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    unparse = sum(r["unparseable"] for r in rows) / max(len(rows), 1)
    print(f"\n{run_id}")
    print(f"  {len(rows)} items in {time.time()-t0:.1f}s   unparseable {unparse:.1%}")
    print(f"  -> {outdir/'predictions.jsonl'}")


if __name__ == "__main__":
    main()
