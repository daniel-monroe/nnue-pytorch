"""One-launch replacements for small per-step eager op chains after the backward (tiled-ft-20).

* ``clip_hidden_weights``: ``NNUEModel.clip_weights(include_input=False)`` (8 launches: repeat,
  2 x new_full, 2 x sub, 3 x clamp_) as one kernel over all clipped tensors.
* ``clip_scale``: ``torch.clamp(clip / (total_norm + 1e-6), max=1.0)`` (4 launches: add,
  reciprocal, mul, clamp) as one single-thread kernel.
* ``and_isfinite``: ``flag &= torch.isfinite(loss).all()`` for a 0-dim loss (5 launches) as one.

All three do the same fp32 operations in the same order as the eager ops (no contraction), so the
results are bit-identical. ``NNUE_FUSED_TAIL=0`` keeps the eager ops.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("NNUE_FUSED_TAIL", "1") != "0"

_SRC = r"""
// clamp as at::native clamp (tensor or scalar bounds): NaN stays NaN, then max(x, lo), then min(., hi)
__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    if (x != x) return x;
    x = x < lo ? lo : x;
    return x > hi ? hi : x;
}
// Segment s: n[s] elements of p[s]; v[s] != 0 -> virtual bounds lo/hi - v[(i / cols % vr) * vc + i % cols % vc]
extern "C" __global__ void clip_hidden(const long long* ptrs, const long long* n, const long long* cols,
                                       const long long* vptr, const int* vr, const int* vc,
                                       const float* lo, const float* hi, int nseg) {
    for (int s = 0; s < nseg; ++s) {
        float* p = (float*)ptrs[s];
        const float* v = (const float*)vptr[s];
        const int cnt = (int)n[s], c = (int)cols[s], r = vr[s], q = vc[s];   // numel < 2^31 (checked)
        const float l = lo[s], h = hi[s];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cnt; i += gridDim.x * blockDim.x) {
            float a = l, b = h;
            if (v) {
                const int row = i / c, col = i - row * c;
                const float w = v[(row % r) * q + col % q];
                a = __fsub_rn(l, w);
                b = __fsub_rn(h, w);
            }
            p[i] = clampf(p[i], a, b);
        }
    }
}
extern "C" __global__ void clip_scale(const float* norm, float clip, float* out) {
    const float r = __fdiv_rn(1.0f, __fadd_rn(norm[0], 1e-6f));
    const float s = __fmul_rn(r, clip);
    out[0] = (s != s) ? s : (s > 1.0f ? 1.0f : s);
}
extern "C" __global__ void and_isfinite(const float* x, bool* flag) {
    const float a = x[0];
    flag[0] = flag[0] && (a == a) && (fabsf(a) != __int_as_float(0x7f800000));
}
"""

_mod = None
_clip_cache: dict = {}


def _module():
    global _mod
    if _mod is None:
        import cupy as cp

        _mod = cp.RawModule(code=_SRC)
    return _mod


def _stream(dev):
    from model.modules.feature_transformer import tiled_ft_backward as _tb

    return _tb._cupy_stream(dev)


def clip_hidden_weights(model) -> bool:
    """Fused ``clip_weights(include_input=False)``. Returns False (nothing done) if not applicable."""
    if not ENABLED:
        return False
    key = id(model)
    ent = _clip_cache.get(key)
    if ent is None:
        segs = []
        for group in model.weight_clipping:
            if "min_weight" not in group and "max_weight" not in group:
                continue
            lo, hi = group.get("min_weight"), group.get("max_weight")
            if lo is None or hi is None:
                return False
            vp = group.get("virtual_params")
            for p in group["params"]:
                if not (p.is_cuda and p.dtype == torch.float32 and p.is_contiguous() and p.dim() == 2
                        and p.numel() < 2**31 - 2**24):
                    return False
                if vp is not None:
                    if not (vp.is_cuda and vp.dtype == torch.float32 and vp.is_contiguous() and vp.dim() == 2):
                        return False
                    if p.shape[0] % vp.shape[0] or p.shape[1] % vp.shape[1]:
                        return False
                segs.append((p, vp, float(lo), float(hi)))
        if not segs:
            return False
        dev = segs[0][0].device
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)  # noqa: E731
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)  # noqa: E731
        f32 = lambda xs: torch.tensor(xs, dtype=torch.float32, device=dev)  # noqa: E731
        ent = dict(
            segs=segs,
            ptrs=[s[0].data_ptr() for s in segs] + [s[1].data_ptr() if s[1] is not None else 0 for s in segs],
            n=i64([s[0].numel() for s in segs]),
            cols=i64([s[0].shape[1] for s in segs]),
            vr=i32([s[1].shape[0] if s[1] is not None else 1 for s in segs]),
            vc=i32([s[1].shape[1] if s[1] is not None else 1 for s in segs]),
            lo=f32([s[2] for s in segs]),
            hi=f32([s[3] for s in segs]),
            dev=dev,
        )
        k = len(segs)
        ent["p"] = i64(ent["ptrs"][:k])
        ent["v"] = i64(ent["ptrs"][k:])
        _clip_cache[key] = ent
    # parameters replaced (e.g. checkpoint load into new tensors) -> rebuild
    k = len(ent["segs"])
    if [s[0].data_ptr() for s in ent["segs"]] + [s[1].data_ptr() if s[1] is not None else 0 for s in ent["segs"]] != ent["ptrs"]:
        del _clip_cache[key]
        return clip_hidden_weights(model)
    import numpy as np

    total = max(s[0].numel() for s in ent["segs"])
    grid = min((total + 255) // 256, 82 * 4)
    with _stream(ent["dev"]):
        _module().get_function("clip_hidden")(
            (grid,), (256,),
            (ent["p"].data_ptr(), ent["n"].data_ptr(), ent["cols"].data_ptr(), ent["v"].data_ptr(),
             ent["vr"].data_ptr(), ent["vc"].data_ptr(), ent["lo"].data_ptr(), ent["hi"].data_ptr(), np.int32(k)),
        )
    return True


def clip_scale(total_norm: torch.Tensor, clip: float) -> torch.Tensor:
    """``torch.clamp(clip / (total_norm + 1e-6), max=1.0)`` for a 0-dim fp32 CUDA tensor."""
    if not (ENABLED and total_norm.is_cuda and total_norm.dtype == torch.float32 and total_norm.numel() == 1):
        return torch.clamp(float(clip) / (total_norm + 1e-6), max=1.0)
    import numpy as np

    out = torch.empty((), dtype=torch.float32, device=total_norm.device)
    with _stream(total_norm.device):
        _module().get_function("clip_scale")((1,), (1,), (total_norm.data_ptr(), np.float32(clip), out.data_ptr()))
    return out


def and_isfinite(flag: torch.Tensor, loss: torch.Tensor) -> bool:
    """``flag &= torch.isfinite(loss).all()`` in place for a 0-dim fp32 CUDA loss. False if not applicable."""
    if not (ENABLED and loss.is_cuda and loss.dtype == torch.float32 and loss.numel() == 1
            and flag.dtype == torch.bool and flag.is_contiguous()):
        return False
    loss = loss.detach().contiguous()
    with _stream(loss.device):
        _module().get_function("and_isfinite")((1,), (1,), (loss.data_ptr(), flag.data_ptr()))
    return True
