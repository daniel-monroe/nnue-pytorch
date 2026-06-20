# Pawn-Structure NNUE Input Feature — Implementation & Recreation Guide

Adds two new NNUE input feature sets, **`PawnStructLeft`** and **`PawnStructRight`**,
modelled on the existing `Full_Threats` feature. Each maps a position's per-side
pawn structure (files a–d / e–h) to a single active feature: the frequency rank of
that structure in a vocabulary of the most common structures. Vocabulary size per
side is configurable. Weights are clamped to the same int8-quantization range as
the threat weights.

This file documents everything needed to recreate the work and the A/B training
experiment that compares a baseline net against one with these features added,
configured to match the **final step of nettest PR #346**
(<https://github.com/vondele/nettest/pull/346>).

---

## 1. What was added / changed

**New files**
| File | Purpose |
|------|---------|
| `model/modules/features/pawn_struct.py` | `PawnStruct` `InputFeature` (left/right), configurable `N`, 1 active feature, weights clamped to threat range. |
| `data_loader/cpp/pawn_vocab.h` | Auto-generated frequency-ordered vocabulary (20000 left + 20000 right `{ownP, enemyp}` bitboard pairs). Array index = feature index. |
| `gen_pawn_vocab.py` | Generates `pawn_vocab.h` from the counter's `_vocab.txt` files. |
| `pawn_structs_mt.cpp` | Multi-threaded pawn-structure counter over binpack data; emits ranked lists (boards + compact `_vocab.txt`) and cumulative coverage. |
| `_extract_subset.cpp` | Streams the first N entries of a binpack into a smaller binpack (same data, RAM-friendly subset). |
| `report_ab.py` | One-line live status: both runs' held-out `val_loss` + pawn weight magnitudes (mmap-read from checkpoint). |
| `read_tb.py` | Dumps TensorBoard scalar series (e.g. `val_loss`). |
| `run_recipe.sh` | The canonical A/B launcher: baseline then pawn4096, matching the PR #346 Stage-3 recipe (scaled compute). |
| `run_ab.sh`, `run_seq.sh`, `run_concurrent.sh` | Earlier/alternate launchers (sequential / concurrent attempts). `run_concurrent.sh` is kept to document that two trainings do **not** fit one 8 GB GPU. |

**Modified files**
| File | Change |
|------|--------|
| `model/modules/features/__init__.py` | Parse parametric components `PawnStructLeft:<N>` / `PawnStructRight:<N>` (default N=1024). |
| `data_loader/cpp/training_data_loader.cpp` | `PawnStructExtractor` (+ `pawn_canon`, key hash), name parsing, `#include "pawn_vocab.h"`. |
| `train.py` | (a) Skip `torch.compile` when Triton is unavailable (Windows). (b) Optional `NNUE_GPU_MEM_FRACTION` env to cap per-process VRAM. |

### Feature design notes
- **Canonicalization**: per side, `(white_pawns&mask, black_pawns&mask)` is folded to a
  canonical form = lexicographic min of `(wp,bp)` and `(bswap64(bp), bswap64(wp))`
  (color-swap = vertical flip). This makes the feature **perspective-symmetric** — the
  same index fires for both the white-POV and black-POV accumulators. (Verified: a
  position and its color-mirror produce identical indices.)
- **Out-of-vocabulary** structures (rank ≥ N) emit no feature.
- **Index space**: `PawnStructLeft:N` contributes `N` inputs, `PawnStructRight:N`
  another `N`, appended after `Full_Threats` (60720) and `HalfKAv2_hm` (24576).
- **NEVER prune HalfKA** — this work only *adds* features; the threat-pruning rule is unrelated.

---

## 2. Prerequisites

- `clang++` with C++20 (`-march=native`), used for all `.cpp` here and the loader DLL.
- Python venv at `.venv/` (Lightning, torch, tensorboard). Use `.venv/Scripts/python.exe`.
- Training data (binpacks). This experiment uses **linrock test60** data already present
  under `data/linrock/test60/`. The nettest recipe's full `stage_two_plus_binpacks`
  (35 files, 2021–2024, ~100s of GB) are **not** used — only the two locally-present
  recipe files `test60-2021-11` and `test60-2021-12`.

---

## 3. Recreation steps

### Step 1 — Count pawn structures → vocabulary
```bash
clang++ -O2 -std=c++20 -DNDEBUG -march=native -pthread -I data_loader/cpp \
  -o pawn_structs_mt.exe pawn_structs_mt.cpp

# <dir> <capPerThread> <topN> <outfile> <threads> <reportSecs>
./pawn_structs_mt.exe data/linrock/test60/test60-2020-2tb7p \
  3000000 50 out_attr/pawn_structures.txt 16 30
```
Produces (in `out_attr/`): `pawn_structures.txt` (coverage + top-50 boards),
`pawn_structures_{full,left,right}_all.txt` (every structure as a labelled board),
`pawn_structures_{full,left,right}_vocab.txt` (compact `count<TAB>pct<TAB>fen`).

### Step 2 — Generate the C++ vocabulary header
```bash
.venv/Scripts/python.exe gen_pawn_vocab.py \
  --left  out_attr/pawn_structures_left_vocab.txt \
  --right out_attr/pawn_structures_right_vocab.txt \
  --max 20000 --out data_loader/cpp/pawn_vocab.h
```
> The committed `pawn_vocab.h` was generated from a ~1.6 M-position sample; the top-1024/4096
> are stable. Regenerate from a larger run for a definitive vocabulary **before** training a
> net you intend to keep (changing the vocab changes feature identity).

### Step 3 — Build the data loader with pawn support
Quick (no PGO):
```bash
clang++ -O2 -std=c++20 -DNDEBUG -march=native -shared -pthread -I data_loader/cpp \
  -o build/training_data_loader.dll \
  data_loader/cpp/training_data_loader.cpp data_loader/cpp/training_data_loader_abi.cpp
```
Or the project's PGO build (now includes the pawn code): `bash compile_data_loader.sh`.

### Step 4 — (Optional) verify feature indices
```bash
.venv/Scripts/python.exe - <<'PY'
import ctypes
from data_loader._native import c_lib
fs=b'Full_Threats+HalfKAv2_hm+PawnStructLeft:1024+PawnStructRight:512'
fen=b'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
arr=(ctypes.c_char_p*1)(fen); z=(ctypes.c_int*1)(0); p=(ctypes.c_int*1)(30)
b=c_lib.dll.get_sparse_batch_from_fens(fs,1,arr,z,p,z).contents
print('num_inputs', b.num_inputs)  # 60720+24576+1024+512 = 86832
PY
```
Note: the loader is passed the **input** feature name (`HalfKAv2_hm`, no `^`).
Start position → left/right index 1 (all-pawns-intact); empty board → index 0.

### Step 5 — Extract RAM-friendly recipe-data subsets
The full recipe binpacks (3 GB each) thrash a 15 GB / 8 GB-GPU laptop. Extract subsets
(same 2021 recipe positions, game-delta encoded so tiny):
```bash
clang++ -O2 -std=c++20 -DNDEBUG -march=native -I data_loader/cpp \
  -o _extract_subset.exe _extract_subset.cpp

./_extract_subset.exe data/linrock/test60/test60-2021-11-nov-12tb7p.min-v2.binpack \
  out_attr/recipe_train.binpack 20000000          # 20M train  (~45 MB)
./_extract_subset.exe data/linrock/test60/test60-2021-12-dec-12tb7p.min-v2.binpack \
  out_attr/recipe_val.binpack 2000000             # 2M held-out val (~4 MB)

printf 'out_attr/recipe_train.binpack\n' > out_attr/train_files.txt
printf 'out_attr/recipe_val.binpack\n'   > out_attr/val_files.txt
```

### Step 6 — Run the A/B (matches nettest PR #346 Stage-3 recipe)
```bash
bash run_recipe.sh > logs_recipe_driver.log 2>&1
```
Runs **baseline** (`Full_Threats+HalfKAv2_hm^`) then **pawn4096**
(`+PawnStructLeft:4096+PawnStructRight:4096`) sequentially — identical config, only
`--features` differs. Outputs under `logs_baseline/` and `logs_pawn4k/`.

### Step 7 — Watch results
```bash
.venv/Scripts/python.exe report_ab.py                       # one-line live status
.venv/Scripts/python.exe -m tensorboard.main --logdir .     # curves for both runs
.venv/Scripts/python.exe read_tb.py logs_baseline/lightning_logs/version_<N> val_loss
```

---

## 4. Exact configuration (PR #346 final step = Stage 3)

Source: `vondele/nettest` `threats.yaml`
(<https://github.com/vondele/nettest/blob/master/threats.yaml#L197-L206> for Stage 3,
`#L46-L67` for the net/common options, `#L88-L112` for `advanced_stage_options`).

**Matched exactly** (in `run_recipe.sh`, `$RECIPE`):
```
features Full_Threats+HalfKAv2_hm^   l1 1024   l2 31   l3 32(default)
optimizer-name rangerlite            factorized-weight-decay 0.001
random-fen-skipping 10   early-fen-skipping 18   soft-early-fen-skipping 32
pc-y0 -0.20  pc-y1 0.45  pc-y2 1.0  pc-y3 0.95  pc-y4 0.75
ply: x1 0.0/y1 0.025  x2 22.0/y2 0.05  x3 25.5/y3 0.20  x4 29.5/y4 0.80
pow-exp 2.435   qp-asymmetry 0.23
in-scaling 300  out-scaling 350  in-offset 300  out-offset 300
one-cycle-warmup-pct 0.033   one-cycle-final-div 2000
jitter-lambda-sample 0.0034  jitter-lambda-batch 0.0062  jitter-decay-lambda-batch 0.999
start-lambda 0.74   end-lambda 0.74   lr 0.65e-3
```

**Deviations (scaled to laptop, applied identically to both A/B runs):**
| Param | PR #346 | Here | Why |
|-------|---------|------|-----|
| batch-size | 65536 | 8192 | 8 GB GPU |
| max-epochs | 3000 ×3 reps | 6 | time |
| one-cycle-steps | 4578000 | 5862 (= total steps) | schedule must span the run |
| validation-size | 250000000 | 200000 | time |
| resume | previous_model (Stage 2) | from scratch | no Stage-1/2 models |
| data | 35 binpacks 2021–2024 (~100s GB) | 20M/2M subset of test60 **2021-11 / 2021-12** | only these recipe files local |
| trainer code | `TonyCongqianWang@b4fd490` (fork) | `official-stockfish@89d5725` (= current nettest master trainer; this repo's HEAD) | fork would shelve the pawn feature |

**Net identity:** architecture / feature set / quantization are **identical** to PR #346, so
the produced `.nnue` has the same structure. Trained *weights* will differ (different trainer
fork, data, and compute). Bit-identical reproduction would require that fork + the full data +
multi-day multi-GPU compute.

---

## 5. Findings so far

- **Pipeline verified end-to-end**: features parse, the C++ loader emits correct indices
  (e.g. 87,344 inputs for `...+PawnStructLeft:4096+PawnStructRight:4096`), training runs,
  weights clip to ±127/256, validation is on truly held-out games.
- **First A/B (small data, 4 epochs)**: baseline vs `+pawn1024` held-out `val_loss` were
  ~identical; pawn weights stayed ~1.01× their random init (`mean|w|` 0.0156 → 0.0157) —
  i.e. **undertrained**, which is why there was no effect. Clipping never engaged
  (max|w| ≈ 0.08–0.11 ≪ 0.496 bound).
- Conclusion: a real verdict needs far more training than a laptop allows; these sparse
  features need many epochs to earn nonzero weight.

### Data-side asymmetry (from the counter)
Right side (e–h, kingside) has ~32 % more distinct structures and is less concentrated than
the left (a–d, queenside) — kingside castling + king-safety pawn moves churn more. So for
equal coverage prefer `right ≥ left` vocab size, not the reverse.

---

## 6. Not yet wired (clearly separable follow-ups)
- **`serialize.py`** `.nnue` export for the pawn feature blocks (training works; export does not).
- **Stockfish inference** for the pawn features (so a net with them can actually play).
Both are needed before a pawn-feature net can be tested in games; neither is required to
train and measure `val_loss`.
