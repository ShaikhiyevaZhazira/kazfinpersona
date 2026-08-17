#!/usr/bin/env bash
# End-to-end smoke test. No GPU required. Should finish in a few minutes.
# The oracle MUST score 1.000 on accuracy metrics and 0.000 on violations.
# Anything else means the pipeline is broken -- do not proceed to real models.
set -e
cd "$(dirname "$0")/.."

echo "=== 1/4 build dataset ==="
python scripts/build_dataset.py --n-clients 2000 --seed 42 --out data/v0.5

echo "=== 2/4 fit behavioural segments (train only) ==="
python scripts/fit_segments.py --data data/v0.5 --seed 42

echo "=== 3/4 run oracle baseline ==="
python scripts/run_baseline.py --data data/v0.5 --backend oracle --language kk

echo "=== 4/4 score it ==="
python scripts/evaluate.py --data data/v0.5 \
  --predictions results/preds/oracle__ctx-profile__cat-listed__lang-kk__seed0/predictions.jsonl \
  --out results/eval/oracle

echo
echo "=== metric validation on synthetic policies ==="
python scripts/evaluate.py --data data/v0.5 --simulate
