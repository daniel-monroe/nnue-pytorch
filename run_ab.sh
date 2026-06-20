#!/usr/bin/env bash
# Autonomous A/B: baseline then pawn-feature run, sequential (1 GPU).
# Identical config + identical train/val (held-out, disjoint) data; only --features differs.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python.exe
TRAIN=$(tr '\n' ' ' < out_attr/train_files.txt)
# --validation-datasets appends one value per occurrence -> repeat the flag per file
VALARGS=$(awk '{printf "--validation-datasets %s ", $0}' out_attr/val_files.txt)

COMMON="$VALARGS --validation-size 100000 --check-val-every-n-epoch 2 \
  --epoch-size 2000000 --max-epochs 4 --seed 42 \
  --num-workers 2 --no-pin-memory --data-loader-queue-size 4 \
  --network-save-period 4"

echo "[$(date +%H:%M:%S)] === BASELINE start ==="
$PY train.py $TRAIN --features "Full_Threats+HalfKAv2_hm^" \
  --default-root-dir logs_baseline $COMMON > logs_baseline_run.log 2>&1
echo "[$(date +%H:%M:%S)] === BASELINE done (exit $?) ==="

echo "[$(date +%H:%M:%S)] === PAWN1K start ==="
$PY train.py $TRAIN \
  --features "Full_Threats+HalfKAv2_hm^+PawnStructLeft:1024+PawnStructRight:1024" \
  --default-root-dir logs_pawn1k $COMMON > logs_pawn1k_run.log 2>&1
echo "[$(date +%H:%M:%S)] === PAWN1K done (exit $?) ==="

echo "[$(date +%H:%M:%S)] === ALL DONE ==="
