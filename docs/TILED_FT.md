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
entry stream) adds no host sync. The prefetch stream runs at high priority so the preparation keeps
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
