# Tiled feature-transformer forward + backward (default on CUDA)

`model/modules/feature_transformer/tiled_ft_forward.py` and `tiled_ft_backward.py` replace the per-position
gather and the atomic scatter of the fused double feature transformer (`FusedDoubleFtFunction`,
`fused_ft_functions.py`) when CUDA + CuPy are available. `tiled_ft_fused.py` connects them: the tiled
forward produces the two raw accumulators (white, black), and a per-position epilogue applies the
perspective mix, clamp and pairwise product, writing the same `l0_` and `clamped_out` as
`fused_ft_kernel.py`. The backward computes the accumulator gradients from `grad_l0` and `clamped_out`
(same formulas as the fused backward) and hands them to the tiled scatter. Same math, fp32 sums in a
different order.

- `NNUE_TILED_FORWARD=0` selects upstream's fused forward, `NNUE_TILED_BACKWARD=0` upstream's backward
  (direct scatter, or the tile-local aggregation where `_use_aggregated_backward` enables it). Both are
  read once at import; either combination works.
- Requirements: forward `L1 % 512 == 0`; backward `L1 % 64 == 0`. Other shapes use upstream's kernels.

## Idea

Positions in a batch share most of their active feature rows. The 2B perspective lists are ordered by
[king bucket][4 min-hash values] so that consecutive lists are similar, then grouped into tiles.

- **Backward** (tiles of 256 lists): each tile's (row, list) entries are sorted by row, and one block
  per (tile, 64-column slice) holds the tile's output gradients in shared memory. Each row segment is
  summed in registers and written with ONE atomicAdd per (row, column) per tile, instead of one per
  (position, row, column). The bias gradient comes from the tile column sums. The PSQT columns use
  plain atomics, skipping zero gradients.
- **Forward** (default: the pull gather, `tiled_ft_pull.py`, tiles of 64 lists): one block per
  (tile, 128-column slice) stages the tile's distinct rows (ascending) chunk by chunk (32 rows,
  `cp.async`, double-buffered) into shared memory; each of its 16 warps keeps the accumulators of 4
  lists in registers (lane = 4 columns) and adds, per chunk, the staged rows each list uses. Every
  output is bias + rows in ascending row order, exactly as in the 16-list gather below, so the two are
  bit-identical. `NNUE_TILED_PULL=0` selects the 16-list gather (tiles of 16 lists, accumulators in
  shared memory; it was bound by the shared-memory read + write per (entry, column)). A warp-per-list
  kernel handles the PSQT columns and stops at the end of each (left-packed) list.

## Index preparation

The preparation reads only the indices, so the loader's prefetch worker computes it on its own CUDA
stream (`data_loader.dataset._register_tiled_prep`). It stores the result in a side table keyed by
the `white_indices` tensor (weak reference + data_ptr + shape check); the batch tuple is not changed.
Without a registered preparation, the forward uses the plain kernel (`tiled_ft_forward.STATS["fallback"]`)
and the backward prepares itself (`tiled_ft_backward.STATS["fallback"]`).
`NNUE_TILED_PREFETCH=0` disables the prefetch-side preparation.

The 256-list tiles (~23k entries) are too large for one shared-memory sort; their 64-list sub-tiles
are sorted in shared memory and the four sorted runs of a tile are merged by rank (the keys are unique
across runs), which gives the same keys as a global sort. The pull forward uses the sorted sub-tiles
directly. The pull preparation (`pull_lists`: per (32-row chunk, list) counts and the chunk-major
entry stream) adds no host sync. The prefetch stream ran at high priority so the preparation keeps
ahead of the training step.

Tuning variables: `NNUE_TILED_T` / `NNUE_TILED_SLICE` / `NNUE_TILED_WARPS` / `NNUE_TILED_ORDER` /
`NNUE_TILED_SORT` / `NNUE_TILED_GRID` (backward); `NNUE_TILED_FWD_T` / `NNUE_TILED_FWD_SLICE` /
`NNUE_TILED_FWD_VEC` / `NNUE_TILED_FWD_P` (16-list forward); `NNUE_PULL_TL` / `NNUE_PULL_R` /
`NNUE_PULL_LPW` / `NNUE_PULL_VEC` / `NNUE_PULL_NS` (pull forward: lists per tile, rows per chunk,
lists per warp, columns per lane, pipeline stages); `NNUE_TILED_HK_OFF` is the first king-bucketed row
of the composed feature set (default 64368 = Full_Threats + PP_3Wide, i.e. the start of K32H3), used
only by the ordering heuristic.

## Accuracy

Real batch (B = 65536, Full_Threats+PP_3Wide+K32H3^, L1 = 1024) against an fp64 reference that uses
each method's own fp32 clamp decisions, as max |error| / max |reference|:

| | upstream fused (direct scatter) | tiled |
|---|---|---|
| `l0_` | 5.3e-7 | 4.6-5.2e-7 |
| weight gradient | 4.2-4.5e-6 | 4.3-5.0e-7 |
| bias gradient | 3.9-4.9e-6 | 5.0-7.1e-7 |

About 3 of the 2.7e8 clamp decisions per batch differ between the two forwards (fp32 sum order), which
is why their gradients differ by more than either differs from its reference. `tests/test_tiled_ft.py`
checks a 1e-6 relative bound against fp64 and compares with upstream's kernels.

## Speed

RTX 3090 (sm_86, SM clock pinned at 1500 MHz), default model, plain training loop (real loader with 16
workers, model + loss + backward + fused Adam, no Lightning/torch.compile), arms interleaved.
`upstream (aggregated)` forces upstream's tile-local aggregation backward, which is off by default on sm_86.

| batch | upstream | upstream (aggregated) | tiled | tiled vs upstream |
|---|---|---|---|---|
| 65536 | 10.6 it/s | 13.3 it/s | 18.3 it/s | 1.73x |
| 131072 | 1.80 it/s* | 2.35 it/s* | 3.24-3.33 it/s* | 1.83x |

*Measured while another job shared the GPU, so only the ratios are meaningful. Under the same sharing,
65536 gave 3.24 / 4.05 / 5.5 it/s (1.70x).

## Fused weight and activation fake quantization

`fused_weight_quant.py` replaces two elementwise chains around the FT (`NNUE_FUSED_WEIGHT_QUANT=0`
restores the reference chains). Both are bit-identical in forward and backward to the reference.

- The merged FT weight (cat of the feature weights, virtual weight repeated over the buckets, then the
  STE fake quantization) is written in one pass; the backward hands each weight its slice of the merged
  gradient and sums the virtual slice over the buckets. For the default model this is a ~570 MB tensor
  that the reference chain passed over about eight times per step.
- `fake_quantize_ft_act(l0_) * l0_correction_factor` is one kernel (fp32 roundings as PyTorch's scalar
  ops, no FMA contraction); its gradient is `grad * c`. When the tiled forward runs, the same
  roundings are applied in the FT epilogue (`act_quant` argument of `double_feature_transform`), so
  the unquantized `l0_` is never written.

RTX 3090 at 1500 MHz, default model, B = 65536, same plain loop as above: 17.5 -> 23.2 it/s (1.33x).
Per step (42.8 ms of GPU time): tiled FT forward 10.6 ms, tiled FT backward 9.7 ms, fused Adam 5.0 ms,
index preparation 4.8 ms (on the loader's stream, overlapped), layer stacks ~7 ms, weight merge 1.7 ms.
The loader alone delivers ~34 batches/s with 16 workers, so training stays GPU bound.

## tiled-ft-4

RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers, 3 epochs x 400
steps, last epoch's it/s, arms interleaved: tiled-ft-3 24.4-24.9 it/s; pull forward + high-priority
prefetch stream 25.4; + sub-tile sort and lane-major scatter loads 26.9; + fused activation
quantization and the faster tile sort 27.2-27.9. Per step (nsys, after the pull forward + sort changes): main stream 98.9% busy; tiled scatter
7.6 ms, pull gather 7.3 ms (the 16-list gather took 9.3 ms in isolation on a real batch, the pull
gather 7.0 ms), fused optimizer 6.1 ms.

Exactness: the forward (FT output, PSQT columns, fused activation quantization, and the whole model
output on real batches) is bit-identical to tiled-ft-3. The backward's per-tile sums are unchanged;
weight gradients differ from tiled-ft-3 by no more than tiled-ft-3 differs from itself between runs
(atomic order, ~9e-7 relative).

## tiled-ft-5

Same setup as tiled-ft-4 (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader
workers, 3 epochs x 400 steps, last epoch's it/s, arms interleaved): tiled-ft-4 27.85-27.96 it/s (8 runs);
tiled-ft-5 29.83-29.95 it/s (7 runs, plus one outlier at 29.02), 1.07x. With `NNUE_L1_DENSE_FORWARD=0` (the old L1 forward) tiled-ft-5
gives 29.07.

- **Tiled scatter**: the entry walk issues the shared-memory loads of 8 entries before their adds and
  fetches the next 32 keys one batch ahead. Same entry order, per-row sums and flushes; 8.3 -> 7.7 ms
  in isolation on a real batch (random output gradients). Single-tile batches (where the result is
  deterministic) give bit-identical L1-column gradients to tiled-ft-4.
- **Grouped L1 forward** (`grouped_linear._bucketed_forward`): a per-bucket fp32 (IEEE) Triton matmul
  over the routed rows replaces the NNZ-compacted kernel, whose summation order depended on
  shared-memory atomics (so it was not deterministic). Each output is the sum of 16-wide dot chunks
  in order (an identity `mov` keeps Triton from fusing them into one 1024-long chain), then the bias.
  1.79 -> 0.72 ms in isolation (K = 1024, N = 32, 32 buckets, 60% zero activations). Against fp64:
  old 2.8e-7, new 3.4e-7 max relative error; new vs old 3.8e-7 = old vs old (run to run) 3.8e-7.
  `NNUE_L1_DENSE_FORWARD=0` restores the old kernel.
- **Grouped L1 input gradient**: 128-row tiles instead of 64 (0.69 -> 0.55 ms); bit-identical (the
  reduction is the same 32-long dot per output).
- **FT weight merge** (`merge_quant4`): float4 accesses, and each thread walks the 32 buckets of one
  virtual-weight element so the virtual weight is read once. 1.73 -> 1.39 ms (~830 GB/s, DRAM
  bound); bit-identical to the scalar kernel and to the torch chain (`test_merge_quant_vectorized_bit_identical`).

Per step after these changes (nsys, ms): pull gather 7.4, tiled scatter 6.7, fused optimizer 6.3
(~840 GB/s, DRAM bound), prologue 1.65, weight merge 1.47, L1 input gradient ~1.0, epilogue 0.97,
L1 forward 0.84, gradient norm 0.76, layer-stack GEMMs (TF32) ~1.6.

Tried without gain: scatter blocks walking several column slices with the next slice prefetched into
registers (spills at 32 warps, slower at 16), warp-batched scatter tile loads (7.71 -> 7.69 ms), 192-list
tiles with 128-column slices, slice-fastest grid; a 4-way unrolled pull loop (slower), pull R = 16/64,
NS = 3, LPW = 2/8; a float4 optimizer kernel (+0.7%, and not bit-identical). Writing the merged weight
from the optimizer was not done: the FT weights can also be changed through `.data` (input weight
clipping, the optimizer's lookahead path, checkpoint loading), which does not bump the version counter,
so a merge cached by the optimizer could go stale silently.

## tiled-ft-6

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved): tiled-ft-5 29.92-29.96 it/s (7 runs);
+ fused lookahead 30.31-30.71; + FT gradient row flags 31.15-31.17; + flagged clip norm 31.44-31.48 (7 runs, 1.05x).

- **Lookahead blend in the update kernel** (`ranger_lite.py`): every 5th step `lookahead_process_step`
  made three passes over every parameter and its lookahead copy (mul_, add_, copy_; ~0.9 ms per step on
  average). On a merge step the fused RangerLite kernels now apply `p * alpha`, then the FMA PyTorch's
  `add_(la, alpha=1 - alpha)` compiles to, to the new parameter and write the lookahead copy.
  Bit-identical (torch.equal on parameters and state against the unfused path, alpha 0.5/0.3/0.8).
  `NNUE_FUSE_LOOKAHEAD=0` disables.
- **FT gradient row flags** (`model/optimizers/grad_row_flags.py`): about half of the FT rows get no
  gradient in a batch, and ~25% (103.4k of 138.1k rows seen in 2000 steps) never do. The tiled scatter
  marks the rows it adds to; the merged-weight backward registers each weight's row slice with its flags;
  the fused update uses +0 for an unflagged row instead of reading the gradient (the gradient there is
  exactly +0), and an element with g == 0 and all-zero optimizer state skips the parameter read and
  all writes (its update rewrites the same bits; the host checks that the scalars keep that true).
  A lookup matches data pointer, shape and version counter (under torch.compile the gradient reaches
  the parameter as a regenerated alias, so object identity does not work) and holds the storage, so an
  in-place change or a copy makes the optimizer read the gradient again. Bit-identical
  (tests/test_fused_optimizer.py). `NNUE_GRAD_ROW_FLAGS=0` disables.
- **Clip norm over the flagged rows** (`grad_row_flags.total_norm`): the gradient-clipping norm read
  every gradient again (0.76 ms); a kernel now sums the squares of the flagged FT rows only (0.36 ms),
  the other gradients use `torch._foreach_norm`. Same value up to fp32 summation order (< 1e-6
  relative in the test). The norms in these runs are 0.006-0.012 against the clip value 2.0, so the
  clip scale is exactly 1.0 either way. `NNUE_FLAGGED_NORM=0` restores `get_total_norm`.

The training loop is no longer purely GPU bound: under nsys the main stream was 89.8% busy, and ~7 of
35 steps waited ~16 ms each for their batch before the first forward kernel.

Tried without gain: the merged FT weight written by the optimizer (a persistent merge buffer whose
validity key is (invalidation epoch, data_ptr and version of every source weight); the fused update of
each weight wrote `rint((p + v) * s) * inv` after the virtual weights were updated; `.data` writers
call `invalidate()`). It was bit-identical (118 of 118 reused merges equal to a fresh merge in real
training) and removed the 1.44 ms merge kernel, but measured 31.34-31.39 it/s against 31.44-31.45
without it (the pull gather took ~1 ms longer in the profile, and the loop now waits on batches), so it
was not kept.

## tiled-ft-7

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved): tiled-ft-6 31.19-31.49 it/s; + pull items
32.75-32.79; + live rows 33.11-33.13; + batched scatter loads 33.87-33.90; + broadcast scatter keys
34.58-34.60 (7 of 8 runs; one at 33.37). Interleaved with tiled-ft-6 at 31.46-31.49: 1.10x.

- **Pull gather items** (`tiled_ft_pull.py`): `pull_lists` also writes, per (chunk, warp), the distinct
  staged rows the warp's lists use (ascending), each with the mask of the lists that use it. The gather
  loads such a row from shared memory once and adds it to every list in the mask (4 items' loads in
  flight), with no per-chunk prefix scan or count shuffles. 8 lists per warp, 8 warps, register cap for
  3 blocks per SM. Each list still adds its rows in ascending order: bit-identical (torch.equal on 3 real
  batches). Isolated: 6.97 -> 5.52 ms; the pull preparation (loader stream) 1.16 -> 1.36 ms.
- **Live rows** (`ranger_lite.py`): a byte per FT row, computed from the optimizer state when the state
  tensors are new or changed in place and then kept by the update kernel, marks rows whose state may be
  nonzero. An unmarked row that is also unflagged would take the dormant skip in every element, so the
  kernel skips it without reading its state (~25% of the FT rows are never touched). Bit-identical
  (`test_live_rows_bit_identical`: in-place state edits including -0.0, a non-finite gradient scale,
  lookahead steps). `NNUE_LIVE_ROWS=0` disables.
- **Scatter tile load**: every thread issues all its list-id loads, then all its gradient loads, before
  its shared-memory stores (it was one dependent load chain per element). Isolated 6.75 -> 6.08 ms.
- **Scatter keys**: each warp stores its 32 prefetched keys in a 128-byte shared buffer and reads them
  back as broadcasts instead of one shuffle per entry. Isolated 6.08 -> 5.48 ms.
  Both scatter changes keep the shared-memory contents, the walk, the flushes and the per-tile sums;
  only the order of the atomics across tiles varies, as before. Against an fp64 reference (real batch,
  random output gradients): weight gradient 5.5-6.5e-7, bias 5.0-5.7e-7 of the maximum, against
  5.6-5.8e-7 and 4.8-5.6e-7 for tiled-ft-6.

Per step after these changes (nsys, ms): pull gather 6.2, tiled scatter 6.05, fused optimizer 5.0,
cutlass TF32 GEMMs 1.7, prologue 1.6, weight merge 1.46, epilogue 1.1, torch elementwise 1.1, L1 input
gradient 0.93, L1 forward 0.75, L1 weight gradient 0.62, l2 bias sum 0.5, flagged norm 0.36, inductor
new_zeros 0.34; loader stream: tile sort 1.0, pull lists 0.92, merge runs 0.6, order key 0.5. Main
stream 98.4% busy. The gather and the scatter run 0.6-0.7 ms slower in training than in isolation
(the loader stream's preparation kernels share the SMs).

Tried without gain:
- Gather: masks built in the kernel with `__reduce_or_sync` (7.3 ms, no prep change); 16 lists per
  warp (7.0 ms); 2 or 4 lists per warp; NS = 3; float2 columns; items read as shared-memory broadcasts
  (6.1 ms: the extra 512 B per block drops the gather to 2 blocks per SM).
- Scatter: persistent blocks that prefetch the next (tile, slice) into registers (7.3 ms, register
  cap at 1024 threads); 8-entry batches changed to 4 or 16; tiles of 128 lists with 16 warps
  (+0.8% it/s, but the weight and bias gradient errors against fp64 grow 1.6-2x, 0.84-1.2e-6 and
  9.2e-7, from the extra atomics: rejected).
- Zero-filling the FT weight gradient on a side stream during the forward: 32.92-32.96 it/s against
  33.11-33.12 (it competes with the forward kernels).
- The bucket selection's backward as one `where` instead of new_zeros + index_put (bit-identical
  under torch.compile): 33.12-33.13 against 33.11-33.12.
- The prefetch stream at normal instead of high priority: 34.91, 34.94, then 32.28 and 28.42 it/s (the
  preparation then sometimes falls behind the step), so it stays at high priority.

## tiled-ft-8

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved). tiled-ft-7: 34.39-34.59 it/s.

| arm | it/s |
|---|---|
| tiled-ft-7 | 33.82-34.59 (3 of 4 runs at 34.58-34.59) |
| loader-stream preparation only, old loader split | 35.15-35.71 (loader bound) |
| loader builder ratio 0.5 (+ warp order key; other preparation changes off) | 34.94-34.95 |
| + preparation | 35.99-36.78 |
| + bucket sum | 37.03 |
| + fused epilogue | 38.02-38.03 |
| + fused prologue (tiled-ft-8) | 39.45-39.52 (5 runs), 1.14x |

- **The data loader was the limit.** train.py's loader alone (with the prefetch preparation, no model)
  delivered 35.3-36 batches/s with 16 workers, of which a quarter built batches. Builders are the
  bottleneck with this feature set (K32H3 alone: 118 batches/s, +PP_3Wide 54, +Full_Threats 35).
  Half of the workers as builders: 47-49 batches/s (3/8: 46, 5/8: 48). Same batches; only the thread
  split changes (`NNUE_LOADER_BUILDER_RATIO` overrides). The library has to be rebuilt.
- **Preparation** (loader stream; exact: the gather and scatter inputs are torch.equal on 4 real
  batches): order key with one warp per list (0.55 -> 0.10 ms isolated); sub-tile sort as a CUB
  BlockRadixSort over the 18+6 varying key bits (bitonic ~0.8 -> 0.43 ms); merge of the 4 sorted runs
  in shared memory (0.36 -> 0.30 ms); the pull preparation's torch glue and list kernel replaced by
  three kernels (distinct rows per tile, a one-block prefix, a block per tile that ranks its entries in
  shared memory and builds wstart / wn / items): pull side 1.3 -> 0.27 ms. Per batch (isolated, incl.
  the host sync) 3.55 -> 1.52 ms; in training the loader stream went from 5.1 to 2.3 ms per step.
  `NNUE_TILED_RSORT_BS=0`, `NNUE_TILED_MERGE_SH=0`, `NNUE_PULL_BUILD=0`, `NNUE_PULL_ITEMS=0` restore
  the old kernels. The gather / scatter still run ~0.9 / ~1.2 ms slower while preparation kernels
  overlap them (14 and 7 of 30 instances).
- **Fused epilogue** (`NNUE_PULL_EPI`): with `keep_acc` the gather block takes 64 columns of each half
  of L1, so a pairwise pair sits in lanes l and l ^ 16; for us / them = (1, 0) / (0, 1) the list writes
  its half of `l0_` itself. Lists with a -0 or non-finite accumulator mark their position, and a warp
  per position recomputes marked and non-binary positions with the reference epilogue. Bit-identical
  `l0_` (int32 view) and accumulators. Epilogue 1.07 ms gone, gather +0.27 ms.
- **Fused prologue** (`NNUE_TILED_PRO`): scatter blocks take 32 columns of each half and their tile
  load computes the accumulator gradients from the list's own accumulators and its half of `grad_l0`
  (the 1.6 ms prologue pass wrote 537 MB that the scatter read back). Positions with non-binary
  us / them use reference rows written by a warp per position. Values equal the prologue's except for
  the sign of zeros, which the sums (starting at +0) do not see; with non-finite inputs different
  elements may become non-finite. Single tile: torch.equal. Real batches against fp64: weight
  6.5-7.3e-7, bias 6.1-9.3e-7 of the maximum (separate prologue 6.7-7.7e-7, 6.0-10.5e-7).
- **Bucket sum** (`NNUE_BUCKET_SUM_KERNEL`): the virtual weight's gradient sums only flagged rows, in
  PyTorch's reduction order (bit-identical); 0.36 -> 0.20 ms.

Per step after these changes (nsys, 25.5 ms of GPU time, main stream 98.8% busy, ms): tiled scatter
6.24, pull gather 6.05, fused optimizer 4.88, cutlass TF32 GEMMs 1.69, weight merge 1.45, L1 input
gradient 0.79, torch elementwise 0.74 (0.64 of it the FT gradient zero fill), L1 forward 0.71, L1
weight gradient 0.60, l2 bias sum 0.46, flagged norm 0.36, inductor new_zeros 0.31, bucket sum 0.21;
loader stream 2.26: sub-tile sort 0.46, merge 0.34, unpack 0.26, list-order radix sort 0.26, pull
build 0.26, tile keys 0.13, order key 0.12.

Tried without gain: gating the loader preparation on an event after the scatter, so it overlaps the
DRAM-bound optimizer and merge instead of the gather / scatter (34.8 against 35.6 it/s while loader
bound); merge runs ranked by per-thread linear advance instead of binary search (0.30 -> 0.50 ms).

## tiled-ft-9

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved, load average 6-11 from other jobs).

| arm | it/s |
|---|---|
| tiled-ft-8 | 39.49-39.53 (3 runs; 39.45-39.52 in the tiled-ft-8 runs) |
| compact loader batches | 39.44-39.51 (GPU bound; loader alone 48 -> 78 batches/s) |
| + preparation gated behind the pull gather | 39.57-39.64 |
| + FT gradient buffer reuse (tiled-ft-9) | 39.97 (3 runs, interleaved with tiled-ft-8; 39.97-40.03 in the other runs), 1.012x |

- **Compact loader batches.** Each builder used to fill the padded `white` / `black` blocks
  (B x max_active x 2 ints = 151 MB per batch) with -1 before writing the features, and
  `pack_indices` then gathered the rows. Streamed batches now store each row's IDs right after the
  previous row's (packing is two `memcpy`s) and skip the padding; `densify_sparse_batch` rewrites a
  compact batch in place into the padded layout, and the Python dense fallback calls it.
  `get_sparse_batch_from_fens` batches stay padded. Builder microbenchmark (8 threads, same entries):
  72 -> 140 batches/s; train.py's loader alone (16 workers, prefetch preparation, no model): 48 -> 78
  batches/s. Exact: packed IDs, floats and the densified blocks hash identically on 8 batches, and with
  2 workers (deterministic order) all tensors of 20 batches from train.py's loader are byte-identical
  to tiled-ft-8's. Training was already GPU bound, so it/s did not move; the loader now has ~2x
  headroom (it was within ~25% of the step rate, and sensitive to other jobs' CPU load).
- **Preparation gate** (`NNUE_PREP_GATE`, `_AT`, `_MINQ`): the prefetch worker queues the preparation
  kernels after its host sync behind an event recorded after the next pull gather, so they overlap the
  layer stacks rather than the gather / scatter. Only while >= 2 prepared batches are queued, so a step
  never waits for its own batch; the host wait times out after 50 ms (no backward running). Gating the
  whole preparation (including its host sync) stalled the main thread: 34.3 it/s; gating after the
  scatter (overlapping the optimizer): 39.26-39.47.
- **FT gradient buffer reuse** (`NNUE_FT_GRAD_REUSE`): the 566 MB `torch.zeros` of the FT weight
  gradient (0.64 ms) is replaced by reusing the previous step's buffer, which is all +0 except the rows
  that backward touched (~69.5k of 138k); a kernel zeroes those rows (0.33 ms). Reused only if nothing
  else holds the storage (`torch._C._storage_Use_Count`) and its version is unchanged since the scatter,
  otherwise `torch.zeros`. `trainer/engine.py` now drops its clip-norm gradient list right after use
  (it kept the previous step's gradient views alive). In training the reused buffer had
  `count_nonzero == 0` and no sign bits before the scatter. 39.55-39.60 -> 39.97-40.01 it/s.

Rebuilding the loader without cmake: `./scripts/build_data_loader_gxx.sh` (g++ with the CMake Release
flags: `-march=native -O3 -DNDEBUG -std=gnu++20 -fPIC -mbmi2 -DHAS_BMI2`, no PGO) writes
`data_loader/cpp/build/libtraining_data_loader.so`. The Python side needs the new
`densify_sparse_batch` symbol, so an old library does not load with this branch's `_native.py`.

Per step now (nsys, ms): tiled scatter 6.10, pull gather 5.85, fused optimizer 4.57, weight merge 1.39,
L1 forward 0.80 / input gradient 0.68 / weight gradient 0.60, l2 stack ~2.5 in total (dense all-bucket
TF32 GEMM forward 0.52, `new_zeros` of the 268 MB all-bucket gradient 0.52 + `index_put` 0.12, its
bias sum 0.44, weight / input gradient GEMMs 0.50 / 0.38), flagged norm 0.35, zero touched rows 0.33,
bucket sum 0.20.

Tried without gain: the fused-epilogue fixup as a ballot over 32 positions per warp instead of a warp
per position (its 0.54 ms in the profile was time spent sharing the SMs with the gated preparation;
40.02-40.03 vs 40.00-40.02 it/s).

Not done (needs a precision / ordering decision): the l2 stack computes all 32 buckets with cuBLAS TF32
and selects one; a bucketed TF32 kernel would remove most of its ~2.5 ms, but the GEMM accumulation
order (and the inductor reduction order of the bias sum) would differ from cuBLAS / inductor, so the
results would not be torch.equal. Reusing a persistent all-bucket gradient buffer (zeroing only the
selected entries) is unsafe inside the inductor graph, whose memory planner may reuse that buffer.

## tiled-ft-10

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved, load average 6-9 from other jobs).

| arm | it/s |
|---|---|
| tiled-ft-9 | 39.97-40.06 (6 runs) |
| + merged weight written by the optimizer, segments without a virtual weight | 40.54-40.56 (3 runs), 1.014x; 40.48-40.51 (3 runs, next table) |
| + clip-norm squares summed by the bucket sum | 40.76-40.79 (2 runs; one at 40.34) |
| + prefetch stream at normal priority | 40.94-40.99 (3 runs; 40.94-40.99 in 4 earlier runs), 1.024x over tiled-ft-9 |

- **Merged weight from the optimizer** (`NNUE_MERGE_IN_OPTIMIZER`, `fused_weight_quant.optimizer_plan`):
  the merged FT weight lives in a persistent buffer with one key per feature segment (invalidation
  epoch, data_ptr and version of the segment's parameters). At the fused RangerLite step, the update
  kernel of a segment without a virtual weight (Full_Threats, PP_3Wide) whose key still matches also
  updates its merged rows, writing only elements whose merged bits change (`rint(p_new * s) * inv` vs
  `rint(p_old * s) * inv`; skipped elements keep p, so their merge is unchanged); the next forward skips
  those segments. The K32H3 segment is still merged by the forward: every merged row changes with the
  virtual weight, and writing it from the optimizer (reading p for the dormant rows) made the update
  kernel 0.9 ms slower than the 0.73 ms merge. `.data` writers of the FT weights invalidate:
  `ComposedFeatureTransformer.clip_weights / coalesce / init_weights / zero_virtual_weights /
  load_export_weights`, the optimizer's non-fused paths, unfused lookahead blends and `train()` / `eval()`.
  Bit-identical: `NNUE_MERGE_CHECK=1` compares every reused segment with a fresh merge (1401 of 1401
  equal over 800 training steps, including lookahead blends and the epoch-end clip);
  `test_merge_in_optimizer_bit_identical`. Per step (nsys): merge 1.39 -> 0.73 ms, update +0.04 ms.
  (tiled-ft-6 wrote the whole merge, including the virtual weight, from the optimizer and measured no
  gain; skipping the merge entirely, with stale weights, gives 42.3 it/s, the upper bound.)
- **Clip norm from the bucket sum** (`NNUE_BUCKET_SUM_SSQ`): the virtual-weight bucket sum reads every
  flagged row of the K32H3 weight gradient; it now also sums their squares (per-thread fp32, a warp sum,
  one fp64 atomic per warp) and registers the result (`grad_row_flags.register_ssq`, looked up like the
  row flags: same data pointer, shape and version). `total_norm` uses it instead of running
  `flagged_ssq` over that gradient again. Same value up to fp32 summation order, like the flagged norm
  (`test_bucket_sum_ssq_norm`: 1e-6 relative against an fp64 norm); the clip scale stays exactly 1.0 in
  these runs (norms ~0.01 against the clip value 2.0). Bucket-sum gradients unchanged (torch.equal).
- **Prefetch stream at normal priority** (`NNUE_PREFETCH_PRIORITY`, default 0; -1 restores high
  priority): at high priority the gated preparation kernels took the SMs ahead of the main stream right
  after the pull gather (the 15 us fixup kernel showed as ~0.5 ms in the profile). With the gate
  (tiled-ft-9) the preparation keeps up at normal priority: 10 runs, every epoch's rate in line, no
  slow runs (tiled-ft-7 measured normal priority without the gate: 2 of 4 runs fell to 28-32 it/s).
  Same batches and math.

Final check, interleaved with tiled-ft-9 (load average 12-23): tiled-ft-9 39.99-40.03, tiled-ft-10
40.65-41.00 it/s.

Per step now (nsys, ms; main stream 23.0 ms busy of 24.2, 95%): tiled scatter 6.05, pull gather 5.80,
fused optimizer 4.61, K32H3 merge 0.73, L1 forward 0.70 / input gradient 0.68 / weight gradient 0.60,
l2 stack (TF32 GEMMs 0.51 / 0.50 / 0.36, all-bucket gradient `new_zeros` 0.31, bias sum 0.45),
zero touched rows 0.33, bucket sum 0.20, flagged norm 0.19 (two gradients); loader stream: merge runs
0.51, tile sort 0.44, unpack 0.25, pull build 0.23.

Tried without gain (isolated, real batch):
- Pull gather: items dispatched to their list through a switch or a tree of uniform branches instead of
  LPW predicated adds (most items belong to one list: mean 1.6 lists per item), 5.8 -> 6.0-9.9 ms
  (branches, spills under the 80-register cap); LPW = 4 with 16 warps (6.0 ms); slice-fastest grid
  (7.8 ms). With the staging loads replaced by zero fills the gather takes 5.55 ms, with the item loop
  removed 5.19 ms, both removed 2.12 ms: staging and item processing each nearly fill the time.
- Tiled scatter: 32-column slices with 2-4 blocks per SM (overlapping one block's tile load with
  another's walk), 6.1 -> 9.2-12 ms. Its tile load alone takes 0.95 ms of the 6.1 ms; plain stores
  instead of the atomics would save ~1.0 ms (not exact).
- The whole merge (including K32H3 and its virtual weight) written by the optimizer: the K32H3 update
  kernel became 0.9 ms slower (it has to read p of the dormant rows whenever the virtual weight changes),
  39.4-39.9 it/s.

## tiled-ft-11

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved with tiled-ft-10, load average 2-12 from other jobs).

| arm | it/s |
|---|---|
| tiled-ft-10 | 40.92-41.01 (8 runs) |
| + per-list row masks in the pull gather | 41.07-41.18 (4 runs) |
| + K32H3 merged only at the batch's rows (tiled-ft-11) | 41.62-41.76 (5 runs; 41.64-41.78 in 2 more with the L1 tiles below), 1.017x |

- **Pull gather row masks** (`NNUE_PULL_LIST_MASKS`, default 1): per (chunk, warp) the items are turned
  into one warp-uniform 32-bit mask per list (`__reduce_or_sync` of the items' row bits), and each list
  adds its staged rows in ascending order, two loads in flight, one float4 add per row. Before, every
  item (mean 1.6 lists) issued LPW = 8 predicated adds. Each list still sums bias + rows in ascending
  row order: bit-identical (`l0_`, both accumulators and the 16-list gather reference, torch.equal on the
  int32 view: 3 real batches, two weight sets including -0 weights, with and without the fused
  activation quantization, non-binary us / them; `test_pull_forward_bit_identical`). Isolated (real
  batch, no epilogue): 5.58 -> 5.32 ms; in training 5.80 -> 5.70 ms.
- **K32H3 merged at the batch's rows only** (`NNUE_MERGE_BATCH_ROWS`, default 1): the merged FT weight
  is read only by the FT forward of the same batch, and that forward reads only the rows of its indices.
  `ComposedFeatureTransformer.forward` passes its white indices to the merge; when the batch's pull
  preparation is registered (`tiled_ft_forward.peek_prep`, the entry is not popped), `mark_rows` flags its
  distinct rows (`tile_rows`) and `merge_quant4` skips the unflagged rows of the virtual-weight segment
  (~44% of the 73,728 K32H3 rows are used per batch). The partly merged segment gets no key, so the
  persistent buffer never reuses it; without a preparation the merge is complete as before. Bit-identical:
  the used rows equal the full merge, and with all unused rows set to NaN the FT forward (l0_ and
  accumulators) is bit-identical on 3 real batches (`test_merge_batch_rows_bit_identical`). Per step:
  merge 0.73 -> 0.35 ms.

Per step now (nsys, ms): tiled scatter 6.05, pull gather 5.70, fused optimizer 4.60, L1 forward 0.70 /
input gradient 0.68 / weight gradient 0.60, l2 TF32 GEMMs 0.51 / 0.49 / 0.37, l2 bias sum 0.45, K32H3
merge 0.35, zero touched rows 0.33, l2 `new_zeros` 0.31, bucket sum 0.20, flagged norm 0.19; loader
stream: merge runs 0.53, tile sort 0.43, unpack 0.25, pull build 0.23. Main stream 94.7% busy; the
CPU launches 3-20 ms ahead of the GPU, so the main-stream gaps (~1.3 ms per step, the largest right
after the gather, where the gated preparation starts) are not host overhead; most likely the loader
stream's kernels hold the SMs there (not checked further).

Tried without gain:
- Pull gather: list per lane group (lanes 4j..4j+3 hold list j, 32 columns per lane, so a lane adds only
  its own list's rows): 6.36 ms, bound by shared-memory wavefronts (each entry's row is read per list,
  and idle lane groups still cost a wavefront); with 2 of its 8 loads it took 4.93 ms. First rows of
  2 / 4 / 8 lists loaded together before the per-list loops: 5.85 / 5.59 / 14.9 ms (spills). The mask
  loop at MINB 2 / 4, NS = 3, LPW = 4 (16 warps), or 128-list tiles (16 warps, 64 registers): 5.76-6.75
  ms. `cp.async.ca` staging, with or without a block order that puts adjacent tiles on one SM: 5.69-5.70
  ms. Measured split of the mask gather: staging alone 4.78 ms (8.9 GB from L2; rows forced into L2,
  `row & 1023`, do not make it faster, so it is L2-bandwidth bound), the list adds alone 4.00 ms,
  neither 0.82 ms. 128-list tiles stage 21% fewer bytes (3.78 ms) but their adds take 4.77 ms.
- Tiled scatter: only 1.6% of the (tile, row) segments belong to rows that one tile touches, so plain
  stores for them were not tried. Skipping atomics of +0 sums (exact: the buffer is never -0):
  39.8 against 41.1 it/s. Segment ends flagged in the keys when they are fetched (no per-entry row
  compare): 7.43 -> 7.47 ms isolated. Four keys per broadcast load: unchanged (7.42 ms). Isolated
  (random gradients, no prologue): 7.42 ms, 5.23 with plain stores, 4.96 with no atomics, 7.36 with
  the gradient shared-memory loads removed.
- L1 Triton tiles: input gradient bm = 128 / bk = 64 and weight gradient bk = 64 are bit-identical
  and faster in isolation (real bucket routing: 1.04 -> 0.62 and 0.70 -> 0.49 ms), but measured
  41.72-41.78 against 41.73-41.76 it/s in training (3 interleaved pairs), so they were not kept.

## tiled-ft-12

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved, load average 3-12 from other jobs).

Same session, master at its best (aggregated backward forced, 12-position tiles) against tiled-ft-11:
16.49-16.57 against 41.72-41.78 it/s (3 runs each), 2.52x.

| arm | it/s |
|---|---|
| tiled-ft-11 | 41.65-41.79 (8 runs) |
| + training loop on a high-priority stream (tiled-ft-12) | 42.66-42.85 (5 runs), 1.025x |
| tiled-ft-12, `NNUE_PREP_GATE=0` | 42.87, 42.87 |
| tiled-ft-12, `NNUE_PREP_GATE_MINQ=4` | 42.87, 42.85 |

- **High-priority training stream** (`trainer/engine.py`, `NNUE_MAIN_STREAM_PRIORITY`, default -1; `none`
  keeps the default stream): nsys showed the ~1.2 ms of main-stream gaps per step were the loader stream's
  preparation kernels (tile sort, pull build, unpack, merge runs, order key) taking the SMs between the
  step's kernels, mostly right after the pull gather. `fit` now runs the loop on a priority -1 stream (the
  prefetch stream stays at 0, the lowest), and makes cupy's current stream the same stream, since the fused
  optimizer's cupy kernels launch without an explicit stream (without that, they ran on the legacy stream,
  unordered with the step: NaN losses). Same kernels, batches and math; only the scheduling changes (atomic
  order as before). Losses per epoch within the tiled-ft-11 run-to-run spread. In the profile the per-step
  gaps fall to ~0.15 ms on most steps, but on ~1 step in 7 the preparation, now only filling the SMs the
  step leaves free, is not ready and the step waits ~2 ms for its batch (order key stretched to 3.5 ms).

Per step (nsys, ms, unchanged kernels): tiled scatter 6.05, pull gather 5.70, fused optimizer 4.66, TF32
GEMMs 1.52, L1 forward / input gradient / weight gradient 0.70 / 0.70 / 0.63, l2 bias sum 0.46, K32H3 merge
0.35, zero touched rows 0.34, l2 `new_zeros` 0.31.

The gate variants are within ~0.1 it/s of the default (2 runs each, not enough to change the default).

Left to try: avoid the occasional preparation starvation (gate off / MINQ 4 may help slightly, needs more runs; order key / unpack
are ungated and only run in the step's gaps), persistent scatter / gather blocks walking adjacent tiles,
larger scatter tiles (needs per-row segment statistics on real batches).

## tiled-ft-13 (prefetch queue study; no speed change kept)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU, load average 6-21 from other jobs).

Finding: on tiled-ft-12 the prefetch queue is empty on ~97% of steps (queue size after `get`: 0 on
382-395 of 400 steps, 1 on the rest). The prefetch worker is sequential: per batch ~4.7 ms of host time
in `next(iterator)` plus ~19 ms in the preparation (mostly its host sync on the low-priority stream),
together about one step (23.4 ms). So the batch is produced just in time, and since the queue never
holds 2 batches when a preparation starts, the preparation gate (`NNUE_PREP_GATE_MINQ` = 2) almost never
applies. That is why `NNUE_PREP_GATE=0` and `MINQ` 3 / 4 measure the same as the default:

| arm (tiled-ft-12 code) | it/s |
|---|---|
| default | 42.07, 42.48, 42.72, 42.84, 42.85, 42.85 |
| `NNUE_PREP_GATE=0` | 42.83, 42.84, 42.89 |
| `NNUE_PREP_GATE_MINQ=4` | 42.61, 42.86, 42.87 |
| `NNUE_PREP_GATE_MINQ=3` | 42.72, 42.82, 42.83 |
| `NNUE_PREP_GATE_MINQ=1` | 42.76 |
| `NNUE_PREP_GATE_MINQ=0` (gate always, the step waits for its own batch's gate) | 16.30 |

Added, both opt-in (default off, so the default path matches tiled-ft-12; scheduling only, same batches
and kernels):
- `NNUE_PREFETCH_FETCH_DEPTH=N` (`data_loader/dataset.py`, `_threaded_iter`): `next(iterator)` runs on its
  own thread, up to N host batches ahead, so the host work overlaps the previous batch's preparation.
- `NNUE_PREP_SPLIT=1`: the preparation's kernels after its host sync (and the forward preparation) run on a
  second low-priority stream (`tiled_ft_backward._gate_point` switches the current stream; the tensors
  made before the switch get `record_stream`), so the next batch's copy and order key do not queue
  behind them. Alone (with the inline fetch) it changed nothing (42.68 it/s, prep host time unchanged).

With both (fetch depth 2 + split), the queue holds 1 batch after every `get` and the gate is active.
nsys, second of 2 epochs x 200 steps, main-stream gaps between 0.5 and 50 ms: 138 gaps / 447 ms on
tiled-ft-12 against 20 / 202 ms (gaps <= 0.5 ms: 118 -> 70 ms). But training is slower:

| arm | it/s |
|---|---|
| tiled-ft-12 | 42.85, 42.84 |
| fetch depth 2 + split | 42.68, 42.68 |
| + `NNUE_PREP_GATE_MINQ=4` | 38.53, 41.71 |
| + `NNUE_PREP_GATE=0` | 42.57, 42.47 |

Not measured why; a guess is that with the preparation ready ahead its kernels now share the SMs with
the step's kernels (low priority does not preempt resident blocks) instead of running in the stalls.

Left to try: make the preparation itself cheaper (order key, unpack, tile sort: less work rather than
better placement); persistent gather / scatter blocks walking adjacent tiles (staging is L2-bandwidth
bound, 8.9 GB per step); small exact fusions (zero touched rows, l2 bias sum, `new_zeros`).

## tiled-ft-14 (two small studies; no speed change kept)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU, load average ~5).

- **One-hot select in `StackedLinear.select_output`** (aimed at the l2 `new_zeros` 0.31 ms + bias sum
  0.46 ms): `torch.where(ls_indices == arange(count), stacked.view(B, count, N), 0).sum(1)` instead of the
  gather, so the backward is a pointwise `where` that inductor can fuse with the bias-gradient sum.
  Bit-identical to the gather: LayerStacks output and every parameter gradient compared with
  `torch.equal` on B = 65536 random inputs, eager and compiled. But slower, so reverted:

  | arm | it/s |
  |---|---|
  | tiled-ft-12 | 42.75, 42.84, 42.83 |
  | one-hot select | 41.87, 41.90, 41.92 |

  The cause was not profiled. A guess: the forward now reads all `count` x N outputs per row instead of
  gathering N, and inductor's fused kernels cost more than the separate ops they replace.

- **Zero touched rows** (0.34 ms per step in nsys): in isolation (74496 x 1032 fp32, 40% of the rows
  touched, randomly placed) the current kernel (one 128-thread block per row) takes 0.144 ms. A 256-thread
  block per row takes 0.167 ms, one warp per row 0.146 ms, and a grid-stride loop over all float4 0.164 ms. The
  kernel itself is near its write bandwidth, so the 0.34 ms in the profile is overlap with the other
  streams (or a higher touched fraction on real batches), not kernel code. The kernel is unchanged.

Build note: these worktrees need the data loader built from the branch
(`PATH=<venv with cmake>/bin:$PATH scripts/compile_data_loader.sh`); the old `.so` in `build/` misses
`pack_sparse_batch` / `densify_sparse_batch`.

Left to try (unchanged): persistent gather / scatter blocks; a cheaper preparation (order key, unpack,
tile sort). For the l2 backward, the next step is a hand-written backward that writes the dense
gradient and the per-bucket bias sum in one pass. Its sum order would differ (reduction order).

## tiled-ft-15 (L2 cache policy for the tiled scatter)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU, load average 3-8).

Finding: the scatter's atomics mostly pay for L2 misses on the weight-gradient rows, not for the atomics
as such. In isolation (real batch, random output gradients, no prologue), the scatter takes 7.43 ms. With
every red sent to `row & 1023` (all lines L2-resident) it takes 5.50 ms, and with `row & 16383`
(4 MB per column slice) 5.83 ms. Segment statistics on the same batch: 1.28M (tile, row) segments
for 6.70M entries at T = 256 (0.96M at T = 512, 0.71M at T = 1024). 54% of the segments have
length 1.

- **L2 cache policy in `tiled_scatter`** (`NNUE_TILED_L2HINT`, default 2; 0 = tiled-ft-12 code): the
  tile's gradient loads (`Gw` / `Gb`, or the prologue's accumulators and `grad_l0`) and the entry keys
  are streaming loads (`__ldcs`), and the weight-gradient reds carry an `L2::evict_last` policy
  (`createpolicy.fractional`, fraction 0.5), so the streamed gradients no longer push the row lines out
  of L2 before the next tiles that touch the same rows update them. The operations are the same; only
  the cache policy changes (atomic order, as before). Against fp64 (B = 65536 real batch, 2 runs per
  arm): weight gradient error / max 3.6e-6 and 5.2e-6 (hint 0) against 3.8e-6 and 4.6e-6 (hint 2). Bias
  gradient error: 7.0e-7 and 7.4e-7 against 6.1e-7 and 7.9e-7. Training losses per epoch match
  (0.02050 / 0.01282 / 0.01069 against 0.02049-0.02051 / 0.01277-0.01283 / 0.01062-0.01063).

  | scatter, isolated (ms) | batch 5 | batch 3 |
  |---|---|---|
  | hint 0 (tiled-ft-12) | 7.43 | 7.51 |
  | 1: streaming loads only | 7.07 | |
  | 2: + evict_last reds, fraction 1.0 / 0.75 / 0.5 | 6.20 / 6.17 / 6.14 | 6.24 (1.0) |

  | arm | it/s |
  |---|---|
  | GPU 0: hint 0 | 42.09, 41.28, 41.47, 41.23 |
  | GPU 0: hint 2 | 42.43, 42.71, 42.24, 42.25 |
  | GPU 1: hint 0 | 42.82 |
  | GPU 1: hint 2 | 43.95, 44.03 (1.027x) |

- **Gather** (`NNUE_PULL_L2HINT`, opt-in, default 0): 1 = streaming output stores (`__stcs`), 2 = also
  `evict_last` on the `cp.async` staging. Bit-identical outputs (`torch.equal`), but no gain in isolation
  (5.31-5.33 / 5.30-5.31 / 5.35 ms). This fits the tiled-ft-11 finding that the gather's staging is
  bound by L2 bandwidth, not by L2 misses.

Also measured, not kept: scatter slices of 32 columns (16 warps: 8.87 ms, 8 warps: 11.45 ms; the key walk
doubles); slice-fastest grid 13.8 ms (each row's lines go through L2 16 times).

Left to try: other evict_last fractions (0.25) and `evict_last` / `evict_first` on the PSQT scatter
and the zero-touched-rows kernel. An order of tiles that keeps the row working set of the ~82
concurrent tiles small (the rows are the miss cost: e.g. group the tiles' shared non-king rows). An
nsys profile of the new scatter in training (not taken: the isolated numbers are on random gradients
without the prologue).

## tiled-ft-16 (scatter L2 study at the training geometry; nsys cost table)

No code change kept; this section records measurements. Isolated numbers: real batch 5
(`/dev/shm/nnlogs/batch5.pt`), random output gradients, no prologue, median of 30 calls, RTX 3090 at 1500 MHz.

Finding: the tiled-ft-15 scatter numbers (7.43 -> 6.14 ms, 5.50 ms with all rows L2-resident) were taken with
OUT = 1032 (the PSQT columns in the row, so the row pitch is 4128 B and a 64-column slice of a row usually
spans three 128 B lines). Training calls the scatter through `tiled_ft_fused.backward`, which uses
OUT = l1_size = 1024 (4096 B pitch). At OUT = 1024 the scatter is not limited by L2 misses any more:

| scatter, isolated (ms) | OUT = 1032 | OUT = 1024 |
|---|---|---|
| hint 0 (tiled-ft-12) | 7.39 | 5.46-5.48 |
| hint 2, fraction 0.5 (tiled-ft-15) | 6.14 | 5.11-5.18 |
| hint 2, rows `& 1023` (all L2-resident; debug only) | 5.42 | 5.15 |
| hint 2, rows `& 4095` / `& 16383` | | 5.08 / 5.08 |

So forcing every row into L2 gains at most ~0.05-0.1 ms at the training geometry (same GPU, same session). What
limits the scatter there is not known; my guess (not measured) is red throughput at L2 (1.28M segments x 1024 columns of
`red.f32` per step; sm_86 has no vector f32 reds).

- **Tile order / scheduling (target 1)**: reuse statistics of the (tile, row) segments, batch 5, T = 256
  (distance = tiles since the same row's previous segment): kb_minhash 1.28M segments, 69477 distinct
  rows, 96.4% / 98.6% / 99.7% of the reuses within 41 / 82 / 164 tiles (median 1). minhash: 1.79M
  segments, 94.0 / 97.4 / 99.1%; identity 2.41M, 95.3 / 98.0 / 99.4%. With ~82 concurrent tiles almost all
  reuses already fall inside the resident window, and the L2-resident bound above leaves no headroom, so no
  new order was built. Scatter at OUT = 1032: minhash 10.98 ms, identity 14.25 ms (kb_minhash 6.13): the
  segment count matters, not the reuse distance.
- **evict_last fraction (target 2)**, OUT = 1024, two interleaved rounds: 0.25: 5.113 / 5.113, 0.5: 5.113 /
  5.121, 0.75: 5.131 / 5.121, 1.0: 5.153 / 5.156 ms. 0.25-0.5 are equal; the 0.5 default stays.
- **PSQT scatter / zero-touched-rows (target 3)**: the PSQT scatter does not run in training (the fused
  path has OUT == l1_size). Streaming stores (`__stcs`) in `zero_touched_rows`: 0.367-0.368 against
  0.368-0.371 ms ("alloc" stage), no change. Not kept.

nsys, real `train.py` + torch.compile, B = 65536, 16 workers, 2 epochs x 300 steps (includes compile
warm-up; 42.79 it/s under nsys). GPU kernel ms per step (600 steps); the preparation kernels (marked *) run
on the prefetch stream and overlap the step, so the total (33.8 ms) exceeds the step time (23.4 ms):

| kernel | ms/step | | kernel | ms/step |
|---|---|---|---|---|
| pull_gather | 5.70 | | order_key * | 0.99 |
| tiled_scatter (with prologue) | 5.46 | | tile_key * | 0.75 |
| ranger_lite_update_pnm (13 launches) | 4.63 | | _bucketed_forward | 0.70 |
| tile_rsort * (median 1.35 per launch) | 2.64 | | _bucketed_input_gradient | 0.69 |
| merge_runs_sh * (median 1.93) | 2.30 | | _bucketed_weight_gradient | 0.61 |
| _unpack_kernel * | 2.06 | | 2 cutlass GEMMs | 0.53 + 0.52 |
| pull_build * | 1.63 | | merge_quant4 | 0.38 |

Next steps: the scatter's remaining cost is not L2 misses at OUT = 1024, so cutting the red count is the
remaining lever (fewer (tile, row) segments: larger tiles at the same shared memory, e.g. a bf16/fp16
staging copy would not be exact, so it would need an exactness label). `tile_rsort` has mean 2.6 ms
against a 1.35 ms median per launch: it slows down when it overlaps the main stream (overlap guessed from
the mean/median gap, not shown on a timeline).

## tiled-ft-17 (bucketed kernels for the l2 layer stack; optimizer and tile_rsort studies)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU).

- **l2 on the bucketed kernels** (`NNUE_GROUPED_STACKS`, default 1; 0 = dense path as before). The
  nsys step showed the dense l2 (64 -> 32 x 32 buckets) costing ~2.3 ms/step: a cuBLAS GEMM for all
  32 buckets (0.50 ms), and in backward a `new_zeros` of the full [B, 1024] output gradient (0.30 ms),
  `index_put` (0.06), two GEMMs (0.55 + 0.37) and the bias reduction (0.50). `StackedLinear` now uses
  `grouped_l1` (the L1 bucketed kernels, which mask K) for K % 16 == 0 and N > 1, so only each row's
  bucket is computed. Exactness: **fp32, another summation order** than cuBLAS (not bit-identical),
  same label as the L1 grouped path. Against fp64 (isolated, B = 65536, random inputs), max error / max
  of output, dx, dW, db: dense 2.6e-7 / 2.4e-7 / 1.9e-7 / 1.5e-7, bucketed 1.3e-7 / 2.4e-7 / 1.9e-7 /
  1.0e-7. Isolated fwd+bwd (eager): 3.08 -> 0.43 ms. Training losses per epoch match the baseline's run
  to run spread (base 0.02049-0.02055 / 0.01277-0.01281 / 0.01062-0.01068; new 0.02053-0.02061 /
  0.01277-0.01285 / 0.01063-0.01075).
  The output layer (128 -> 1 x 32) stays dense: bucketed it is 0.66 -> 0.53 ms in isolation, but slower
  in training (l2 only: 47.15, 47.19 it/s; l2 + output: 46.58, 46.92; interleaved).

  | arm (interleaved per GPU) | it/s |
  |---|---|
  | GPU 3: tiled-ft-16 | 43.91, 43.86 |
  | GPU 3: tiled-ft-17 (l2 bucketed) | 47.15, 47.19 (1.074x) |
  | GPU 1: tiled-ft-16 | 44.00, 43.85 |
  | GPU 1: tiled-ft-17 (final code) | 47.09, 47.21 (1.075x) |

- **Fused optimizer (target 1): at the DRAM roofline, no change.** The 13 launches are 2 large ones
  (FT weights, 75.5M and 61.2M elements: 2.27 and 2.12 ms) and 11 small ones (0.25 ms together; the 8
  tiny ones ~35 us). A float4 version (4 elements per thread, one row check per 4) was no faster in
  isolation (73728 x 1024, all rows live, flagged rows 50%: 2.70 vs 2.70 ms; 90%: 2.93 / 2.94; with the
  lookahead 3.68 / 3.69); that case moves 16 B read + 12 B written per element plus the flagged
  gradient (~2.27 GB) in 2.62 ms (no merge output), ~865 GB/s on a 936 GB/s part. Fewer bytes per element
  would need a lower-precision state (not exact); one launch for the small parameters would save at most
  ~35 us/step. Not kept; `ranger_lite.py` is unchanged.
- **tile_rsort under overlap (target 2): real, by design.** Per launch (nsys, 394 launches): alone ~0.43
  ms; overlapping the cuBLAS GEMMs (now mostly gone with the l2 change) median 1.34 ms; overlapping
  `tiled_scatter` median 6.5 ms. The prefetch stream has priority 0 under the main stream's -1, so its
  blocks only get the SMs the step leaves; the main stream stayed 98.6% busy (22.7 ms/step under nsys)
  with no waits on the preparation, so this costs no step time. No change.
- ncu performance counters are not available to this user (ERR_NVGPUCTRPERM), so no DRAM byte counts.

Next steps: profile the new step with nsys (the l2 GEMMs overlapped tile_rsort, so the prefetch
overlap pattern changes); share the route (`_route_rows`) between l1 and l2 (one launch, same
indices); the output layer's dense path (`new_zeros` [B, 32] + GEMMs) is ~0.3 ms.

## tiled-ft-18 (prefetch split stream on by default; shared l1/l2 routing; output layer as a per-row dot)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU).

**nsys of tiled-ft-17** (2 epochs x 300 steps, steps 50-250 before the end): step median 20.49 ms (mean
21.41), main stream busy 20.22 ms/step. Main stream (ms/step): pull_gather 5.70, tiled_scatter 5.43,
optimizer 4.65 (13 launches), bucketed l1 + l2 kernels 2.19 (fwd 0.77, dx 0.74, dW 0.68), merge_quant4
0.36, zero_touched_rows 0.33, bucket_sum 0.23, flagged_ssq 0.21, output layer GEMMs + `new_zeros` /
`index_put` ~0.22. Prefetch stream (overlapped, 11.4 ms/step): tile_rsort 4.4, tile_key 4.1 (median 5.6
per launch: it now overlaps the scatter), unpack 1.1, merge_runs 0.6, order_key 0.5, pull_build 0.3.
The largest main-stream gap was new: **0.52 ms at every step boundary** (optimizer end -> the next step's
first kernel). The main thread was not blocked (it kept launching; pull_gather was launched 8.5 ms ahead
of its start); the next step's first kernels wait for the next batch's preparation, which still had its
tail (merge_runs_sh, sub_distinct, pull_build, sub_offsets) queued on the single low-priority prefetch
stream behind the batch's host sync (a 12-byte D2H that waits ~12 ms for that stream). With the l2 GEMMs
gone (tiled-ft-17) the step got shorter and the preparation became just-late.

Changes:
- **`NNUE_PREP_SPLIT` default 1** (`data_loader/dataset.py`, the tiled-ft-13 option; scheduling only,
  same kernels and batches, bit-identical). The kernels after the preparation's host sync run on a second
  low-priority stream, so the next batch's copy/order key no longer queue behind them. Under nsys the
  boundary gap drops 0.52 -> 0.18 ms (median); the main stream's kernels get ~0.2 ms/step slower from the
  extra overlap. In tiled-ft-13 it measured equal (the step was then longer than the preparation).
- **One routing launch for l1 and l2** (`grouped_linear._route_rows`, `NNUE_SHARE_ROUTE`, default 1):
  the last (indices, version, count, stream) -> (rows, counts) is kept; l2 gets the same indices tensor
  in training, so its route and `counts` zero-fill are skipped (route 9 -> 5 us/step plus 2 launches).
  A different tensor or an in-place change of the indices recomputes (checked). Bit-identical.
- **Output layer (128 -> 1 x 32) as a per-row dot** (`grouped_linear.bucket_dot`, used by
  `StackedLinear` for N = 1, `NNUE_BUCKET_DOT`, default 1; 0 = GEMM as before). An autograd Function that
  torch.compile traces: forward `(x * W[idx]).sum(1) + b[idx]` (inductor fuses it into the kernel that
  makes the l3 input: 86 us -> 91 us, and the 57 us forward GEMM is gone), input gradient
  `g * W[idx]` (bit-identical to the GEMM, replaces the 74 us sgemm), weight gradient one GEMM on the
  bucket-masked [B, 32] gradient as before. Exactness: forward is **fp32 instead of TF32** (the dense GEMM
  ran as a TF32 tensor-op GEMM): max error / max against fp64 3.2e-4 -> 1.0e-7; dx bit-identical; dW and db
  unchanged path (2.9e-4 / 9.8e-8). Isolated compiled fwd+bwd (B = 65536): 0.444 -> 0.430 ms.

nsys, both code changes and the split (steps as above):

| arm | step median | step mean | main busy | boundary gap |
|---|---|---|---|---|
| tiled-ft-17 | 20.49 | 21.41 | 20.22 | 0.52 |
| + route + bucket dot | 20.45 | 21.47 | 20.12 | 0.52 |
| + split only | 20.32 | 20.89 | 20.43 | 0.08 |
| tiled-ft-18 (all three) | 20.21 | 20.74 | 20.30 | 0.18 |

Training (interleaved per GPU; other users' jobs were running on the other GPUs and the CPUs, load 5-17):

| arm | it/s |
|---|---|
| GPU 3: tiled-ft-17 | 47.16, 47.15, 47.08 |
| GPU 3: tiled-ft-18 | 47.56, 47.82, 47.81 (1.013x) |
| GPU 0: tiled-ft-17 | 46.54, 45.76 |
| GPU 0: tiled-ft-18 | 46.24, 46.57, 46.00 (1.003x) |
| GPU 0: tiled-ft-18, `NNUE_PREP_SPLIT=0` | 45.98 (one run) |

GPU 0 was noisier (spread ~0.8 it/s), so the gain there is inside the noise; GPU 3 shows +1.3%. Train
loss per epoch, all runs: tiled-ft-17 0.02055-0.02062 / 0.01275-0.01282 / 0.01059-0.01074, tiled-ft-18
0.02045-0.02060 / 0.01272-0.01284 / 0.01056-0.01068 (inside the run-to-run spread).
`tests/test_tiled_ft.py`, `tests/test_grouped_linear.py`, `test_serialization_roundtrip.py`,
`test_compiled_loss.py`, `test_tf32_parity.py` pass.

Next steps: the preparation is now on the critical path at the step boundary (0.18 ms left with the
split); removing its host sync (E, max tile, max sub-tile read back to size buffers and pick kernel
variants; upper bounds would remove it) would let the worker queue the next batch earlier. tile_key and
tile_rsort slow down 5-10x when they overlap the scatter (low priority does not preempt resident blocks).

## tiled-ft-19 (high-priority first preparation stream; host-sync and boundary studies)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU).

**nsys of tiled-ft-18** (2 epochs x 300 steps, 249 steps): step median 19.93 ms, main stream busy 20.0
ms/step; pull_gather 5.63, tiled_scatter 5.23, optimizer 4.63 (13 launches), bucketed kernels 2.27,
34 small `vectorized_elementwise` launches 0.16 ms; main-stream gaps 0.73 ms/step in all. The step
boundary gap (optimizer tail -> next step's first kernel) is median 165 us; its kernel was launched
11.7 ms earlier (98% of steps), and the prefetch kernel ending just before it is `pull_build` (175 of 250
steps): the step waits for the next batch's preparation, which finishes just-in-time.

Change:
- **First preparation stream at high priority** (`data_loader/dataset.py`: with `NNUE_PREP_SPLIT=1`,
  `NNUE_PREFETCH_PRIORITY` now defaults to -1; new `NNUE_PREP_SPLIT_PRIORITY`, default 0, for the split
  stream). The copy, order key and sort before the host sync are small and now run ahead of the step's
  kernels, so the worker's host sync returns sooner and the tail (tile_key, tile_rsort, merge, pull_build,
  still low priority on the split stream) is queued earlier. Scheduling only: same kernels and batches,
  **bit-identical**. `NNUE_PREFETCH_PRIORITY=0` restores tiled-ft-18.

  | arm (GPU 2, interleaved pairs) | it/s |
  |---|---|
  | tiled-ft-18 | 47.33, 47.34, 46.77, 47.27, 47.18 (mean 47.18) |
  | tiled-ft-19 | 47.51, 47.62, 47.63, 48.08, 47.60 (mean 47.69, 1.011x) |

  (pairs 1-3 with the env vars, 4-5 with the new defaults.)

Studies, not kept:
- **Upper bounds instead of the host sync (target 1): not possible with the current kernel variants.**
  The index blocks are padded to maxa = 288, so n_lists x maxa bounds give max tile 73728 (pull falls
  back to the 16-list gather: 39.5 it/s). Real batches: E ~5.85M, max tile 18.4-20.0k, max sub-tile
  5.1-5.2k, max list 111-113. Even data-dependent bounds (sum of the T / 64 largest list counts: 26.7k /
  6.9k) exceed the shared-memory merge (25344) and the register rsort (6144) limits, and they need the
  counts, which only the GPU has. Would need the loader (C++) to report per-batch counts plus
  device-side overflow fallbacks, or a two-phase worker (async read-back, tail finished by a second thread).
- `NNUE_PREFETCH_FETCH_DEPTH=2`: 42.49 (vs 47.87); `NNUE_PREP_GATE_MINQ=3`: 47.83, 45.71 (vs 47.87, 47.84).

`tests/test_tiled_ft.py`, `tests/test_grouped_linear.py` pass. Owner files touched: `data_loader/dataset.py`
(two stream-priority defaults, one comment).

Next steps: nsys of tiled-ft-19 (boundary gap and tile_key/rsort overlap); the two-phase worker above;
the 34 tiny elementwise launches (~0.16 ms + gaps) after the optimizer and around the loss.

## tiled-ft-20 (fused per-step tail: weight clipping, clip scale, finiteness flag)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU).

**Where the small launches come from** (nsys of tiled-ft-19, one step on the main stream, 37
`vectorized_elementwise`/reduce launches):
- after the optimizer, 13 launches, ~110 us with gaps: `clip_weights(include_input=False)` from the
  `WeightClipper` callback (l1: `repeat` of the factorized weight, 2 x `new_full`, 2 x `sub`, tensor
  `clamp_`, 70 us of kernels; l2 and output: scalar `clamp_`), then `TerminateOnNaN`'s
  `flag &= torch.isfinite(loss).all()` (abs, 2 x compare, and, and-assign);
- before the optimizer, 4 launches: `torch.clamp(clip / (total_norm + 1e-6), max=1.0)` (add,
  reciprocal, mul, clamp), after ~8 launches of `grad_row_flags.total_norm` (accumulator fills, double
  add, to-float, sqrt, `_foreach_norm`, stack, `vector_norm`; not fused, see below);
- ~11 in the loss backward (fills, muls, adds between the compiled backward kernels; not attributed).

Change: `model/optimizers/fused_tail.py` (`NNUE_FUSED_TAIL`, default 1; 0 = eager ops as before), three
CuPy kernels on the current stream, each the same fp32 operations in the same order (explicit `_rn`
intrinsics, no contraction; clamp with at::native's NaN and tie behavior): **bit-identical**
(`tests/test_fused_tail.py`: clipped weights with NaN/inf/+-0 entries, 3 clip values x ~450 norms incl.
0/inf/NaN, finiteness flags).
- `clip_hidden_weights`: the three clipped tensors in one launch (virtual bounds indexed in place
  instead of materialized): 8 launches -> 1.
- `clip_scale`: 4 launches -> 1. `and_isfinite`: 5 launches -> 1.

Post-optimizer tail under nsys: 13 launches (~110 us incl. gaps) -> 2 (`clip_hidden` 42 us,
`and_isfinite` 3 us).

Training (GPU 2, 4 interleaved pairs, load ~10):

| arm | it/s |
|---|---|
| tiled-ft-19 | 47.88, 48.03, 48.08, 48.04 (mean 48.01) |
| tiled-ft-20 | 48.16, 48.06, 48.14, 48.15 (mean 48.13, 1.0025x) |

Small but every pair favors tiled-ft-20 (+0.02 to +0.28). The gain is small because the step boundary
waits for the next batch's preparation (tiled-ft-19), so most of the tail time saved becomes boundary
idle instead. Train loss last epoch 0.0128-0.0134 both arms (run-to-run spread).
`tests/test_fused_tail.py`, `tests/test_tiled_ft.py`, `tests/test_grouped_linear.py` pass.

Owner files touched (minimal hooks, eager code kept as the fallback): `model/model.py` (import + 2-line
fast path in `clip_weights` for `include_input=False`), `trainer/engine.py` (import + clip-scale line),
`trainer/callbacks.py` (import + fused call in `TerminateOnNaN.on_train_batch_end`).

Not done / next steps:
- `grad_row_flags.total_norm` tail (~8 launches: fills, double add, to-float, sqrt, `_foreach_norm`,
  stack, `vector_norm`): fusing `vector_norm` changes the fp32 summation order (not bit-identical), so it
  was left; a fused version would be "same up to fp32 summation order" like the flagged norm itself.
- The ~11 eager launches inside the loss backward were not attributed.
- Two-phase prefetch worker (async pinned read-back of E / max tile / max sub-tile with an event, tail
  queued when the event fires while the worker starts the next batch's copy/order key): not started.
  It is the remaining lever for the boundary gap, which is now the main cost of the tail savings above.

## tiled-ft-21 (two-phase prefetch worker; opt-in, no speed change kept)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU, load ~5-8).

Added, opt-in (`NNUE_PREP_TWO_PHASE=1`, default 0; needs the split stream, i.e. `NNUE_PREP_SPLIT=1`, and
fetch depth 0): the prefetch worker no longer blocks on the preparation's 12-byte size read-back.
- `tiled_ft_backward._prepare_gen` (`prepare` is now a thin driver around it, unchanged behavior): with
  `deferred=True` the (E, max tile, max sub-tile) stack is copied to pinned memory with `non_blocking`, an
  event is recorded after the copy and the generator yields it; resumed, it reads the pinned values and
  queues the same tail kernels with the same sizes. `_gate_point(after=event)` makes the split stream wait
  on that event instead of the whole first stream (which by then holds the next batch's phase 1).
  `register_prep_begin` / `register_prep_finish` wrap the two phases and the side-table registration.
- `data_loader/dataset.py` worker loop: fetch batch N+1 (`next(iterator)`, H2D), queue its phase 1 on the
  high-priority stream, then finish batch N (event wait, tail on the split stream, tiled forward
  preparation, ready event, put). Batches are put in the iterator's order; every tensor used on the split
  stream gets `record_stream` (the item, and the phase-1 tensors via `_gate_point`); the generator keeps
  the phase-1 tensors and the pinned buffer alive until the tail is queued. At most two batches'
  preparations are in flight (one more batch of index blocks and phase-1 buffers, ~10 MB).
- Exactness: **bit-identical** batches and preparations (`tests/test_prefetch_two_phase.py`: 12 synthetic
  batches through `FixedNumBatchesDataset` with the worker in both modes; every batch tensor and every
  backward / pull preparation tensor equal, except the never-written tails of the pull preparation's
  upper-bound buffers `tile_rows`, `wstart`, `wn`, which are compared over their written part).

nsys (2 epochs x 200 steps, second half, steps > 40 ms dropped; queue size after each `get`):

| arm | step median | gaps > 20 us per step (mean) | largest gap per step (median) | queue 0 / 1 |
|---|---|---|---|---|
| tiled-ft-20 | 20.32 ms | 431 us | 226 us | 131 / 69 |
| two-phase | 20.50 ms | 431 us | 195 us | 18 / 182 |

The worker now keeps a batch ready (queue 1 on 91% of steps instead of 35%), but the main stream's idle
time per step is unchanged: on tiled-ft-20 the step boundary no longer waits for the preparation (the gap
after `and_isfinite` is 1-2 us in both arms). The largest gap of each step (~200 us) is before the
optimizer (the kernel after it is `ranger_lite_update_pnm` or an elementwise launch). Not traced; a likely
cause is that `RangerLite` reads `variance_normalized.item()` every step (normloss and stable weight decay
are on by default), a host sync that lets the GPU drain before the optimizer launches.

Training (GPU 1, interleaved):

| arm | it/s |
|---|---|
| tiled-ft-20 | 47.40, 47.28, 47.53, 47.48 (mean 47.42) |
| two-phase | 47.23, 47.35, 47.32, 47.37 (mean 47.32, 0.998x) |
| two-phase + `NNUE_PREP_GATE_MINQ=1` | 47.35, 47.36 |

So the default stays off (slightly slower, like fetch depth 2 in tiled-ft-13; not measured why, the same guess
applies: the earlier preparation's kernels overlap the step's instead of filling its stalls). Train loss last epoch 0.0106-0.0107 in all runs.
`tests/test_tiled_ft.py`, `tests/test_grouped_linear.py`, `tests/test_fused_tail.py`,
`tests/test_prefetch_two_phase.py` pass.

Owner files touched: `data_loader/dataset.py` (`_begin_tiled_prep`, `_finish_tiled_prep`, the opt-in
worker loop; the default loop unchanged).

Next steps: the per-step host sync in `RangerLite` (`variance_normalized.item()`): pass the device scalar
to the fused update kernel instead (same fp32 value, so exact) to remove the ~200 us drain before the
optimizer; then re-check the boundary and whether the two-phase worker helps once the main thread runs ahead.

## tiled-ft-22 (the ~200 us gap is not a host sync; RangerLite phase-2 path without `.item()`)

Same setup (RTX 3090 at 1500 MHz, real `train.py` + torch.compile, B = 65536, 16 loader workers,
3 epochs x 400 steps, last epoch's it/s, arms interleaved on one GPU).

**The tiled-ft-21 guess was wrong.** With the default config (`weight_decay` 0 in every group, normloss
off because `normloss_active=legacy_mode`), `RangerLite.step` takes the full fused path, which has no
host sync; the `variance_normalized.item()` path (phase 2) is only used with stable weight decay and a
nonzero `weight_decay`. nsys of tiled-ft-21 (2 epochs x 200 steps, second half, 199 steps):
- the main thread is far ahead of the GPU: the kernel after each step's largest main-stream gap was
  launched a median 73-97 ms (3.5-5 steps) before it started;
- main-thread syncs: `cudaStreamSynchronize` only at the logging points (`float(train_loss)` in
  `trainer/engine.py` and `TerminateOnNaN`'s `_finite_flag.item()`, both every `log_every_n_steps`
  = epoch / 5 steps) and at epoch ends; no per-step sync, no blocking memcpy;
- the largest gap per step (median 193-228 us, between `ranger_lite_update_pnm` launches, after
  `tiled_scatter`, or before `_bucketed_bias_gradient`) is covered about twice over by kernels on the
  other streams: per gap mostly `_unpack_kernel` (~190 us; the loader's padded-row unpack on the
  high-priority first preparation stream), then `tile_rsort`, `merge_runs_sh`, `pull_build`. So the GPU
  is busy during these gaps; the main stream waits for SMs held by the preparation, not for the host.

Change (non-default configs only; default training runs the same code as tiled-ft-21):
- `model/optimizers/ranger_lite.py`, phase-2 fused path: the decay factor `decay * lr /
  variance_normalized` is computed on the device (`full_like(..., float64).div_(vn.double()).float()`:
  fp64 division rounded to fp32, as the host did with the `.item()` value) and the two phase-2 kernels
  read it through a new pointer argument (`wd_ptr`, null = use the `wd_factor` scalar as before).
  **Bit-identical** (`tests/test_ranger_phase2_sync.py`: 12 steps, 2 groups with decay, PNM on and off,
  parameters and moments `torch.equal` to tiled-ft-21; steps after the first run under
  `torch.cuda.set_sync_debug_mode("error")`).
- Bug fix in the same path: the PNM phase-2 launch passed `scale_ptr`, which is undefined there (and not
  a parameter of `ranger_lite_update_pnm_phase2`), so stable weight decay with `weight_decay > 0` and PNM
  raised `NameError`. The gradient scale is already applied by `_foreach_mul_` before this path.

Changed lines in `ranger_lite.py` (owner file): both phase-2 kernels `float wd_factor` ->
`float wd_factor, const float* __restrict__ wd_ptr` plus `if (wd_ptr) wd_factor = *wd_ptr;`;
`_fused_update_group_phase2`: `wd_ptr = None`, the `wd_factor = decay * lr / variance_norm_value` line
replaced by the device computation, one `np.uint64(wd_ptr...)` argument per launch, the `scale_ptr,`
argument removed; `step`: `variance_norm_value = variance_normalized` (was `.item()`).

Studies, not kept:
- `_unpack_kernel` with R rows per program (2-D block) and num_warps 2/4/8: 217 us vs 231 us per batch
  in isolation (bit-identical); ~0.07% of a step, not worth it. It writes the padded 2 x 288 slots per
  position (151 MB), about 700 GB/s already; the lever would be not materializing the padding.
- Training (GPU 2, 4 interleaved rounds, load ~6-8):

  | arm | it/s |
  |---|---|
  | tiled-ft-21 | 47.63, 47.33, 47.69, 47.65 (mean 47.58) |
  | tiled-ft-22 | 47.68, 47.66, 47.71, 47.73 (mean 47.70) |
  | tiled-ft-22 + `NNUE_PREP_TWO_PHASE=1` | 47.30, 47.60, 47.66, 47.64 (mean 47.55) |
  | two-phase + `NNUE_PREFETCH_PRIORITY=0` | 47.73, 47.71, 47.74, 47.73 (mean 47.73) |

  tiled-ft-21 and tiled-ft-22 run the same code by default, so their 0.12 difference is noise (one low
  tiled-ft-21 run). Two-phase alone is again no faster. Two-phase with the first preparation stream at
  normal priority is the most consistent arm, but only +0.03 over tiled-ft-22: not significant, defaults
  unchanged. Train loss last epoch 0.0106-0.0107 in all runs.

`tests/test_tiled_ft.py`, `tests/test_grouped_linear.py`, `tests/test_fused_tail.py`,
`tests/test_prefetch_two_phase.py`, `tests/test_ranger_phase2_sync.py` pass.

Next steps: the main stream is not host-starved, and the per-step gaps are filled by preparation
kernels, so further speed has to come from less GPU work: the padded unpack (151 MB written per batch,
read again by the preparation) could be skipped by preparing from the packed CSR layout; or more pairs
of two-phase + `NNUE_PREFETCH_PRIORITY=0` to see whether its +0.03 is real.
