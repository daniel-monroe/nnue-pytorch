"""One-pass merge + fake quantization of the feature-transformer weight.

``ComposedFeatureTransformer.merged_weight_and_bias(fake_quantize_weights=True)`` builds the effective
weight as ``cat([w, w_k + virtual.repeat(NB, 1), ...])`` and fake-quantizes it with the STE
``hard + (x - x.detach())``. For the default model that is a ~570 MB tensor, and the chain of
repeat / add / cat / mul / round / div / sub / add makes about eight full passes over it every step,
plus their backward. Here one kernel writes ``round((w + v) * s) * (1 / s)`` per feature segment
straight into the merged tensor, and the backward hands each weight its slice of the merged gradient
(the STE gradient is the identity) and reduces the virtual weight's slice over the buckets.

The result is bit-identical to the reference chain: PyTorch's CUDA ``mul``/``div`` by a Python scalar
compute in fp32 with ``float(s)`` and ``float(1) / float(s)``, ``round`` is round-half-to-even
(``rint``), ``x - x.detach()`` is exactly zero, and the virtual gradient is the same
``view(NB, nv, L).sum(0)`` that ``repeat``'s backward performs.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from model.optimizers import grad_row_flags as _grf

try:
    import cupy as cp

    from . import tiled_ft_backward as _tb

    _HAS_CUPY = True
except Exception:  # noqa: BLE001 - optional
    _HAS_CUPY = False

_SRC = r"""
extern "C" __global__ void merge_quant(float* __restrict__ out, const float* __restrict__ w,
                                       const float* __restrict__ v, long long n, long long vn,
                                       float s, float inv) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float x = w[i];
        if (v != nullptr) x = x + v[i % vn];
        out[i] = rintf(x * s) * inv;
    }
}

// Same per-element math, 16 B per access. Thread j owns float4 j of the virtual weight (or of the
// whole segment when there is none) and walks the nb buckets, so each virtual element is read once.
extern "C" __global__ void merge_quant4(float4* __restrict__ out, const float4* __restrict__ w,
                                        const float4* __restrict__ v, int vn4, int nb,
                                        float s, float inv) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < vn4; j += blockDim.x * gridDim.x) {
        const float4 a = v != nullptr ? v[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int b = 0; b < nb; ++b) {
            const long long i = (long long)b * vn4 + j;
            float4 x = w[i];
            if (v != nullptr) { x.x = x.x + a.x; x.y = x.y + a.y; x.z = x.z + a.z; x.w = x.w + a.w; }
            out[i] = make_float4(rintf(x.x * s) * inv, rintf(x.y * s) * inv, rintf(x.z * s) * inv, rintf(x.w * s) * inv);
        }
    }
}
"""
_kernel = None
_kernel4 = None
# NNUE_FUSED_WEIGHT_QUANT=0 selects the reference cat + STE chain (read once at import).
ENABLED = os.environ.get("NNUE_FUSED_WEIGHT_QUANT", "1") != "0"


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_SRC, "merge_quant")
        _kernel.compile()
    return _kernel


def _get_kernel4():
    global _kernel4
    if _kernel4 is None:
        _kernel4 = cp.RawKernel(_SRC, "merge_quant4")
        _kernel4.compile()
    return _kernel4


def _aligned16(*ts):
    return all(t is None or t.data_ptr() % 16 == 0 for t in ts)


def segments_of(features):
    """[(weight, virtual_weight or None, num_buckets)] for the feature set, or None when some feature
    merges its weight in another way (then the caller uses the reference path)."""
    segs = []
    for f in features:
        w = f.weight
        v = getattr(f, "virtual_weight", None)
        if v is not None:
            nb = getattr(f, "NUM_BUCKETS", None)
            if nb is None or w.shape[0] != v.shape[0] * nb or w.shape[1:] != v.shape[1:]:
                return None
            segs.append((w, v, nb))
        else:
            if f.merged_weight() is not w:
                return None
            segs.append((w, None, 1))
    return segs


def usable(segs) -> bool:
    return (
        _HAS_CUPY
        and segs is not None
        and all(
            w.is_cuda and w.dtype == torch.float32 and w.is_contiguous()
            and (v is None or (v.is_cuda and v.dtype == torch.float32 and v.is_contiguous()))
            for w, v, _ in segs
        )
    )


# ---- merged weight written by the optimizer -------------------------------------------------------
# The merge is written into a persistent buffer, with one key per feature segment: the state of the
# segment's source parameters (invalidation epoch, data_ptr and version of each) whose merge the buffer
# holds. At the fused RangerLite step, the update kernel of a segment without a virtual weight whose key
# still matches also writes the merged values that change into the buffer (raw writes do not move the
# version counters, so the key stays valid), and the next forward skips that segment. Segments with a
# virtual weight (every merged row changes with it) are re-merged by the forward as before.
# Writes through ``.data`` do not move version counters either: the FT weight writers (clip_weights,
# coalesce, init, loading export weights; the optimizer's non-fused paths, lookahead blends, train() /
# eval()) call ``invalidate()``; ``p.data = x`` changes data_ptr; in-place ops on parameters bump versions.
# NNUE_MERGE_IN_OPTIMIZER=0 disables the reuse.
MERGE_IN_OPTIMIZER = os.environ.get("NNUE_MERGE_IN_OPTIMIZER", "1") != "0"
STATS = {"reused": 0, "computed": 0}
_CHECK = os.environ.get("NNUE_MERGE_CHECK", "0") != "0"   # debug: compare every reused segment with a fresh merge
_epoch = 0
_buffer = None        # (out tensor, layout, (s, inv), source parameters)
_seg_keys = None      # per segment: key of the parameters whose merge the buffer holds, or None


def invalidate() -> None:
    """Forget the merged buffer's validity (call after changing FT weights through ``.data``)."""
    global _epoch, _seg_keys
    _epoch += 1
    _seg_keys = None


def _seg_params(params, i, has_v):
    return params[i:i + 2] if has_v else params[i:i + 1]


def _key(ps):
    return (_epoch, tuple((p.data_ptr(), p._version) for p in ps))


def optimizer_plan(params_with_grad):
    """For the fused optimizer: ({id(weight): merged rows}, (s, inv)) for the segments without a virtual
    weight whose merge the buffer holds and whose weight is in ``params_with_grad`` (ids); None if there
    are none. The other segments are marked stale (the optimizer changes them without writing the merge)."""
    if _buffer is None or _seg_keys is None:
        return None
    out, layout, scale, params = _buffer
    plan = {}
    r0 = 0
    for k, (i, (has_v, _nb)) in enumerate(_iter_layout(layout)):
        w = params[i]
        if (MERGE_IN_OPTIMIZER and not has_v and _seg_keys[k] is not None and _seg_keys[k] == _key((w,))
                and id(w) in params_with_grad):
            plan[id(w)] = out[r0:r0 + w.shape[0]]
        else:
            _seg_keys[k] = None
        r0 += w.shape[0]
    return (plan, scale) if plan else None


def _merge_segment(out, w, v, nb, s, inv):
    n = w.numel()
    if not n:
        return
    vn = n // nb if v is not None else n
    with _tb._cupy_stream(out.device):
        if n % 4 == 0 and vn % 4 == 0 and vn // 4 < 2**31 and _aligned16(out, w, v):
            vn4 = vn // 4
            blocks = min((vn4 + 255) // 256, 82 * 16)
            _get_kernel4()((blocks,), (256,), (out.data_ptr(), w.data_ptr(), 0 if v is None else v.data_ptr(),
                                              np.int32(vn4), np.int32(nb if v is not None else 1), s, inv))
        else:
            blocks = min((n + 255) // 256, 65535 * 8)
            _get_kernel()((blocks,), (256,), (out.data_ptr(), w.data_ptr(), 0 if v is None else v.data_ptr(),
                                              np.int64(n), np.int64(1 if v is None else v.numel()), s, inv))


class _MergeFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale, layout, *params):
        global _buffer, _seg_keys
        s = np.float32(scale)
        inv = np.float32(1.0) / s
        rows = sum(params[i].shape[0] for i, (has_v, _) in _iter_layout(layout))
        cols = params[0].shape[1]
        ctx.layout = layout
        ctx.shapes = [p.shape for p in params]
        nseg = len(layout)
        out, keys = None, [None] * nseg
        if MERGE_IN_OPTIMIZER and _buffer is not None:
            b_out, b_layout, b_scale, b_params = _buffer
            if (b_layout == layout and b_scale == (s, inv) and len(b_params) == len(params)
                    and all(a is b for a, b in zip(b_params, params)) and b_out.shape == (rows, cols)
                    and b_out.device == params[0].device):
                out = b_out
                if _seg_keys is not None:
                    keys = _seg_keys
        if out is None:
            out = torch.empty(rows, cols, device=params[0].device, dtype=torch.float32)
        r0 = 0
        new_keys = []
        for k, (i, (has_v, nb)) in enumerate(_iter_layout(layout)):
            w = params[i]
            v = params[i + 1] if has_v else None
            o = out[r0:r0 + w.shape[0]]
            key = _key(_seg_params(params, i, has_v))
            if keys[k] is not None and keys[k] == key:
                STATS["reused"] += 1
                if _CHECK:
                    fresh = torch.empty_like(o)
                    _merge_segment(fresh, w, v, nb, s, inv)
                    same = torch.equal(fresh.view(torch.int32), o.view(torch.int32))
                    name = "check_equal" if same else "check_differ"
                    STATS[name] = STATS.get(name, 0) + 1
                    if not same or STATS["reused"] % 200 == 1:
                        print(f"[merge check] {STATS}", flush=True)
            else:
                STATS["computed"] += 1
                _merge_segment(o, w, v, nb, s, inv)
            new_keys.append(key)
            r0 += w.shape[0]
        if not MERGE_IN_OPTIMIZER:
            return out
        _buffer = (out, layout, (s, inv), tuple(params))
        _seg_keys = new_keys
        return out.view_as(out)

    @staticmethod
    def backward(ctx, grad):
        flags = _grf.lookup(grad)   # rows of the tiled FT gradient that are not exactly +0
        grad = grad.contiguous()
        grads = [None] * len(ctx.shapes)
        r0 = 0
        for i, (has_v, nb) in _iter_layout(ctx.layout):
            nr = ctx.shapes[i][0]
            g = grad[r0:r0 + nr]
            grads[i] = g
            if flags is not None:
                _grf.register(g, flags[r0:r0 + nr])
            if has_v:
                vs = ctx.shapes[i + 1]
                fl = None if flags is None else flags[r0:r0 + nr]
                grads[i + 1] = _bucket_sum(g, fl, nb, vs) if fl is not None else g.view(nb, vs[0], *vs[1:]).sum(0)
            r0 += nr
        return (None, None, *grads)


# Virtual-weight gradient: the sum over the buckets, in the order of PyTorch's reduction (4 accumulators,
# bucket b into accumulator b % 4, then ((a0 + a1) + a2) + a3), reading only the flagged rows: an unflagged
# row is exactly +0, and adding +0 changes no sum (no partial sum is -0: the gradient has no -0 entries).
_BSUM_SRC = r"""
extern "C" __global__ void bucket_sum(const float4* __restrict__ g, const unsigned char* __restrict__ flags,
                                      float4* __restrict__ out, const int nb, const int rows, const int cols4)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)rows * cols4) return;
    const int r = (int)(t / cols4), c = (int)(t - (long long)r * cols4);
    float4 a[4];
    bool has[4] = {false, false, false, false};
    for (int b = 0; b < nb; ++b) {
        if (!flags[(long long)b * rows + r]) continue;
        const float4 v = g[((long long)b * rows + r) * cols4 + c];
        const int k = b & 3;
        if (has[k]) { a[k].x += v.x; a[k].y += v.y; a[k].z += v.z; a[k].w += v.w; } else { a[k] = v; has[k] = true; }
    }
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool any = false;
    for (int k = 0; k < 4; ++k) {
        if (!has[k]) continue;
        if (any) { s.x += a[k].x; s.y += a[k].y; s.z += a[k].z; s.w += a[k].w; } else { s = a[k]; any = true; }
    }
    out[t] = s;
}
"""
_bsum_kernel = None
BUCKET_SUM = os.environ.get("NNUE_BUCKET_SUM_KERNEL", "1") != "0"


def _bucket_sum(g, flags, nb, vs):
    """g.view(nb, *vs).sum(0) for a contiguous fp32 g whose unflagged rows are +0 (bit-identical)."""
    global _bsum_kernel
    cols = int(np.prod(vs[1:])) if len(vs) > 1 else 1
    rows = vs[0]
    if not (BUCKET_SUM and _HAS_CUPY and g.is_cuda and g.dtype == torch.float32 and g.is_contiguous()
            and cols % 4 == 0 and g.data_ptr() % 16 == 0 and flags.dtype == torch.uint8 and flags.numel() == nb * rows):
        return g.view(nb, *vs).sum(0)
    if _bsum_kernel is None:
        _bsum_kernel = cp.RawKernel(_BSUM_SRC, "bucket_sum")
    out = torch.empty(*vs, dtype=torch.float32, device=g.device)
    n = rows * (cols // 4)
    with _tb._cupy_stream(g.device):
        _bsum_kernel(((n + 255) // 256,), (256,), (g.data_ptr(), flags.data_ptr(), out.data_ptr(), np.int32(nb),
                                                   np.int32(rows), np.int32(cols // 4)))
    return out


def _iter_layout(layout):
    """layout: tuple of (has_virtual, num_buckets) per feature; params are w[, v] per feature in order."""
    i = 0
    for has_v, nb in layout:
        yield i, (has_v, nb)
        i += 2 if has_v else 1


def merge_and_fake_quantize(segs, scale):
    layout = tuple((v is not None, nb) for _, v, nb in segs)
    params = [p for w, v, _ in segs for p in ((w, v) if v is not None else (w,))]
    return _MergeFakeQuant.apply(scale, layout, *params)


# ---- FT activation: fake_quantize_ft_act(l0_) * l0_correction_factor in one pass --------------------
# Reference: hard = x.mul(s).add(eps).floor().div(s); out = (hard + (x - x.detach())) * c. PyTorch's CUDA
# scalar mul/add/div run in fp32 with float(s), float(eps) and float(1) / float(s) (a scalar divisor is
# applied as a reciprocal multiply), so the kernel uses the same roundings with no FMA contraction.
# The STE gradient is grad * c (identity when c == 1).
_ACT_SRC = r"""
extern "C" __global__ void act_quant(float* __restrict__ out, const float* __restrict__ x, long long n,
                                     float s, float eps, float inv, float c) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float h = __fmul_rn(floorf(__fadd_rn(__fmul_rn(x[i], s), eps)), inv);
        out[i] = __fmul_rn(h, c);
    }
}
"""
_act_kernel = None


class _ActFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, eps, corr):
        global _act_kernel
        if _act_kernel is None:
            _act_kernel = cp.RawKernel(_ACT_SRC, "act_quant")
            _act_kernel.compile()
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        s = np.float32(scale)
        if n:
            with _tb._cupy_stream(x.device):
                _act_kernel((min((n + 255) // 256, 65535 * 8),), (256,),
                            (out.data_ptr(), x.data_ptr(), np.int64(n), s, np.float32(eps),
                             np.float32(1.0) / s, np.float32(corr)))
        ctx.corr = corr
        return out

    @staticmethod
    def backward(ctx, grad):
        return (grad if ctx.corr == 1.0 else grad * ctx.corr), None, None, None


def act_usable(x: torch.Tensor) -> bool:
    return ENABLED and _HAS_CUPY and x.is_cuda and x.dtype == torch.float32


def fake_quantize_act_and_scale(x, scale, eps, corr):
    return _ActFakeQuant.apply(x, scale, eps, corr)
