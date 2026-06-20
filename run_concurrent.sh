#!/usr/bin/env bash
# Concurrent A/B on one GPU: baseline + pawn(4096+4096) at the same time.
# Each capped to 45% VRAM. Identical config + identical (disjoint) train/val data.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python.exe
TRAIN=$(tr '\n' ' ' < out_attr/train_files.txt)
VALARGS=$(awk '{printf "--validation-datasets %s ", $0}' out_attr/val_files.txt)

COMMON="$VALARGS --validation-size 200000 --check-val-every-n-epoch 1 \
  --epoch-size 4000000 --max-epochs 4 --batch-size 4096 --seed 42 \
  --num-workers 1 --no-pin-memory --data-loader-queue-size 3 \
  --network-save-period 1"

export NNUE_GPU_MEM_FRACTION=0.46
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] launching BOTH runs concurrently"

$PY train.py $TRAIN --features "Full_Threats+HalfKAv2_hm^" \
  --default-root-dir logs_baseline $COMMON > logs_baseline_run.log 2>&1 &
PID_BASE=$!
echo "  baseline PID $PID_BASE"

$PY train.py $TRAIN \
  --features "Full_Threats+HalfKAv2_hm^+PawnStructLeft:4096+PawnStructRight:4096" \
  --default-root-dir logs_pawn4k $COMMON > logs_pawn4k_run.log 2>&1 &
PID_PAWN=$!
echo "  pawn4k   PID $PID_PAWN"

wait $PID_BASE; echo "[$(date +%H:%M:%S)] BASELINE done (exit $?)"
wait $PID_PAWN; echo "[$(date +%H:%M:%S)] PAWN4K done (exit $?)"
echo "[$(date +%H:%M:%S)] ALL DONE"
