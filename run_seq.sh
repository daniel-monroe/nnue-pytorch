#!/usr/bin/env bash
# Sequential A/B at full GPU (two NNUE trainings don't fit one 8GB GPU together).
# Identical config + identical disjoint train/val data; only --features differs.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python.exe
TRAIN=$(tr '\n' ' ' < out_attr/train_files.txt)
VALARGS=$(awk '{printf "--validation-datasets %s ", $0}' out_attr/val_files.txt)

COMMON="$VALARGS --validation-size 200000 --check-val-every-n-epoch 1 \
  --epoch-size 8000000 --max-epochs 4 --batch-size 8192 --seed 42 \
  --num-workers 2 --no-pin-memory --data-loader-queue-size 4 \
  --network-save-period 1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] === BASELINE start ==="
$PY train.py $TRAIN --features "Full_Threats+HalfKAv2_hm^" \
  --default-root-dir logs_baseline $COMMON > logs_baseline_run.log 2>&1
echo "[$(date +%H:%M:%S)] === BASELINE done (exit $?) ==="

echo "[$(date +%H:%M:%S)] === PAWN4K start ==="
$PY train.py $TRAIN \
  --features "Full_Threats+HalfKAv2_hm^+PawnStructLeft:4096+PawnStructRight:4096" \
  --default-root-dir logs_pawn4k $COMMON > logs_pawn4k_run.log 2>&1
echo "[$(date +%H:%M:%S)] === PAWN4K done (exit $?) ==="
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
