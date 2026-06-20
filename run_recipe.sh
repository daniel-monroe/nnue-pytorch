#!/usr/bin/env bash
# A/B matching nettest threats.yaml FINAL STEP (Stage 3) hyperparameters exactly,
# scaled to this laptop (batch/epochs/data). Only --features differs between runs.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python.exe
TRAIN=$(tr '\n' ' ' < out_attr/train_files.txt)
VALARGS=$(awk '{printf "--validation-datasets %s ", $0}' out_attr/val_files.txt)

# --- Exact Stage-3 recipe hyperparameters (common_run_options + advanced_stage_options) ---
RECIPE="--l1 1024 --l2 31 --optimizer-name rangerlite --factorized-weight-decay 0.001 \
  --random-fen-skipping 10 --early-fen-skipping 18 --soft-early-fen-skipping 32 \
  --pc-y0=-0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 \
  --ply-x1 0.0 --ply-y1 0.025 --ply-x2 22.0 --ply-y2 0.05 --ply-x3 25.5 --ply-y3 0.20 --ply-x4 29.5 --ply-y4 0.80 \
  --pow-exp 2.435 --qp-asymmetry 0.23 \
  --in-scaling 300.0 --out-scaling 350.0 --in-offset 300.0 --out-offset 300.0 \
  --one-cycle-warmup-pct 0.033 --one-cycle-final-div 2000 \
  --jitter-lambda-sample 0.0034 --jitter-lambda-batch 0.0062 --jitter-decay-lambda-batch 0.999 \
  --start-lambda 0.74 --end-lambda 0.74 --lr 0.00065"

# --- Laptop-scaled compute (identical for both runs) ---
# 8M/epoch / batch 8192 = 977 steps/epoch * 6 epochs = 5862 total steps -> one-cycle-steps
SCALE="--batch-size 8192 --epoch-size 8000000 --max-epochs 6 --one-cycle-steps 5862 \
  $VALARGS --validation-size 200000 --check-val-every-n-epoch 1 \
  --num-workers 2 --no-pin-memory --data-loader-queue-size 4 \
  --network-save-period 1 --seed 42"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] === BASELINE (recipe) start ==="
$PY train.py $TRAIN --features "Full_Threats+HalfKAv2_hm^" \
  --default-root-dir logs_baseline $RECIPE $SCALE > logs_baseline_run.log 2>&1
echo "[$(date +%H:%M:%S)] === BASELINE done (exit $?) ==="

echo "[$(date +%H:%M:%S)] === PAWN4K (recipe) start ==="
$PY train.py $TRAIN \
  --features "Full_Threats+HalfKAv2_hm^+PawnStructLeft:4096+PawnStructRight:4096" \
  --default-root-dir logs_pawn4k $RECIPE $SCALE > logs_pawn4k_run.log 2>&1
echo "[$(date +%H:%M:%S)] === PAWN4K done (exit $?) ==="
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
