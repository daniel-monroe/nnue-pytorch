# RangerLite Optimizer (2026)
# A refactored, unbloated, and bug-fixed derivative of Ranger21.
#
# Original Ranger21 implementation by @lessw2020:
# URL: https://github.com/lessw2020/Ranger21
#
# Modifications and Refactoring by @TonyCongqianWang

import collections
import math
import os
import warnings

import torch

from . import grad_row_flags as _grf

try:
    from model.modules.feature_transformer import fused_weight_quant as _fwq
except Exception:  # noqa: BLE001 - optional
    _fwq = None


def _invalidate_merge():
    if _fwq is not None:
        _fwq.invalidate()

try:
    import cupy as cp
    import numpy as np

    _HAS_CUPY = True
except Exception:  # noqa: BLE001
    cp = None
    np = None
    _HAS_CUPY = False


# Fold the lookahead blend into the fused update kernels (NNUE_FUSE_LOOKAHEAD=0 disables).
_FUSE_LOOKAHEAD = os.environ.get("NNUE_FUSE_LOOKAHEAD", "1") != "0"
# Use the FT gradient's row flags (grad_row_flags) in the fused update (NNUE_GRAD_ROW_FLAGS=0 disables).
_ROW_FLAGS = os.environ.get("NNUE_GRAD_ROW_FLAGS", "1") != "0"
# Skip whole rows whose state is all +0 and that get no gradient, without reading them (NNUE_LIVE_ROWS=0 disables).
_LIVE_ROWS = os.environ.get("NNUE_LIVE_ROWS", "1") != "0"


def _identity_scalars(lr_div_bc1, eps, bias_correction2, a, b, c):
    """Whether, with these fp32 scalars, an element with g == 0 and all-zero state keeps its exact
    parameter and state bits (+0 products and sums, p - (+0) == p): all finite, lr >= 0, eps > 0,
    bias correction > 0, the momentum factors a, b >= 0 and the divisor c > 0."""
    v = [float(np.float32(x)) for x in (lr_div_bc1, eps, bias_correction2, a, b, c)]
    return all(math.isfinite(x) for x in v) and v[0] >= 0 and v[1] > 0 and v[2] > 0 and v[3] >= 0 and v[4] >= 0 and v[5] > 0


class _RangerLiteFusedKernels:
    """Cached CuPy fused update kernels.

    Three variants are provided per optimizer flavour:
    - "full":    variance update + moment update + parameter update in one launch.
                 Used when no global variance reduction is required.
    - "variance": updates variance_ma for the stable-weight-decay normalization.
    - "phase2":  moment update + optional stable weight decay + parameter update,
                 reading pre-computed variance_ma.
    """

    _compiled = False

    @classmethod
    def _compile(cls):
        if cls._compiled or not _HAS_CUPY:
            return
        source = r"""
typedef long long int64_t;

// merged FT weight element as fused_weight_quant.merge_quant without a virtual weight: rint(w * s) * inv
__device__ __forceinline__ float merge_q(float x, float ms, float minv) {
    return __fmul_rn(rintf(__fmul_rn(x, ms)), minv);
}

extern "C" __global__ void ranger_lite_update_pnm(
    float* __restrict__ p,
    const float* __restrict__ grad,
    const float* __restrict__ grad_scale,
    float* __restrict__ grad_ma,
    float* __restrict__ neg_grad_ma,
    float* __restrict__ variance_ma,
    int64_t n,
    float beta2,
    float one_minus_beta2,
    float bias_correction2,
    float beta1_sq,
    float one_minus_beta1_sq,
    float one_plus_pnm,
    float pnm_factor,
    float noise_norm,
    float lr_div_bc1,
    float eps,
    int64_t step,
    float wd_factor,
    float* __restrict__ la_params,
    float la_alpha,
    float la_beta,
    const unsigned char* __restrict__ row_flags,
    int row_shift,
    int dormant_skip,
    unsigned char* __restrict__ live_rows,
    float* __restrict__ mout,
    float ms,
    float minv
) {
    // mout != nullptr: the merged FT weight (fused_weight_quant) of this parameter, holding the merge of
    // the old values; the elements whose merged bits change are rewritten (skipped elements keep p)
    // live_rows[r] == 0: every state element of row r is +0 (bits). A row that is also unflagged has
    // g = +0 * scale == 0 for a finite scale, so each of its elements would take the dormant skip below;
    // the whole row is skipped without reading it. A row that may get a nonzero state is marked live.
    const bool scale_ok = grad_scale == nullptr || isfinite(*grad_scale);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (live_rows != nullptr) {
            const int64_t r = i >> row_shift;
            const bool fl = row_flags[r] != 0;
            if (!live_rows[r]) {
                if ((fl || !scale_ok) && (i & ((1LL << row_shift) - 1)) == 0) live_rows[r] = 1;
                else if (dormant_skip && !fl && scale_ok) continue;
            }
        }
        // rows without a flag hold exactly +0 (grad_row_flags), so their gradient is not read
        float g = (row_flags == nullptr || row_flags[i >> row_shift]) ? grad[i] : 0.0f;
        if (grad_scale != nullptr) g = __fmul_rn(g, *grad_scale);
        const float vm_old = variance_ma[i];
        const float gm = grad_ma[i];
        const float ngm = neg_grad_ma[i];
        // g == 0 and all-zero state: the update below rewrites the same values (host checks the scalars)
        if (dormant_skip && g == 0.0f && (__float_as_uint(vm_old) | __float_as_uint(gm) | __float_as_uint(ngm)) == 0u) continue;
        float vm = vm_old * beta2 + g * g * one_minus_beta2;
        variance_ma[i] = vm;
        float denom = sqrtf(vm / bias_correction2) + eps;

        float pgm = (step & 1LL) ? gm : ngm;
        float pngm = (step & 1LL) ? ngm : gm;

        float new_pgm = pgm * beta1_sq + g * one_minus_beta1_sq;
        float pnm_val = (new_pgm * one_plus_pnm - pngm * pnm_factor) / noise_norm;
        float new_p = p[i];
        const float p_old = new_p;
        if (wd_factor != 0.0f) {
            new_p = new_p * (1.0f - wd_factor);
        }
        new_p = new_p - lr_div_bc1 * (pnm_val / denom);
        if (la_params != nullptr) {
            // this step's lookahead blend: p.mul_(alpha).add_(la, alpha=1 - alpha); la.copy_(p)
            new_p = __fmaf_rn(la_beta, la_params[i], __fmul_rn(new_p, la_alpha));   // add_ contracts to an FMA
            la_params[i] = new_p;
        }
        p[i] = new_p;
        if (mout != nullptr) {
            const float q = merge_q(new_p, ms, minv);
            if (__float_as_uint(q) != __float_as_uint(merge_q(p_old, ms, minv))) mout[i] = q;
        }

        if (step & 1LL) {
            grad_ma[i] = new_pgm;
        } else {
            neg_grad_ma[i] = new_pgm;
        }
    }
}

extern "C" __global__ void ranger_lite_update_adam(
    float* __restrict__ p,
    const float* __restrict__ grad,
    const float* __restrict__ grad_scale,
    float* __restrict__ grad_ma,
    float* __restrict__ variance_ma,
    int64_t n,
    float beta2,
    float one_minus_beta2,
    float bias_correction2,
    float beta1,
    float one_minus_beta1,
    float lr_div_bc1,
    float eps,
    float wd_factor,
    float* __restrict__ la_params,
    float la_alpha,
    float la_beta,
    const unsigned char* __restrict__ row_flags,
    int row_shift,
    int dormant_skip,
    unsigned char* __restrict__ live_rows,
    float* __restrict__ mout,
    float ms,
    float minv
) {
    // mout != nullptr: the merged FT weight (fused_weight_quant) of this parameter, holding the merge of
    // the old values; the elements whose merged bits change are rewritten (skipped elements keep p)
    // live_rows[r] == 0: every state element of row r is +0 (bits). A row that is also unflagged has
    // g = +0 * scale == 0 for a finite scale, so each of its elements would take the dormant skip below;
    // the whole row is skipped without reading it. A row that may get a nonzero state is marked live.
    const bool scale_ok = grad_scale == nullptr || isfinite(*grad_scale);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (live_rows != nullptr) {
            const int64_t r = i >> row_shift;
            const bool fl = row_flags[r] != 0;
            if (!live_rows[r]) {
                if ((fl || !scale_ok) && (i & ((1LL << row_shift) - 1)) == 0) live_rows[r] = 1;
                else if (dormant_skip && !fl && scale_ok) continue;
            }
        }
        // rows without a flag hold exactly +0 (grad_row_flags), so their gradient is not read
        float g = (row_flags == nullptr || row_flags[i >> row_shift]) ? grad[i] : 0.0f;
        if (grad_scale != nullptr) g = __fmul_rn(g, *grad_scale);
        const float vm_old = variance_ma[i];
        const float gm_old = grad_ma[i];
        // g == 0 and all-zero state: the update below rewrites the same values (host checks the scalars)
        if (dormant_skip && g == 0.0f && (__float_as_uint(vm_old) | __float_as_uint(gm_old)) == 0u) continue;
        float vm = vm_old * beta2 + g * g * one_minus_beta2;
        variance_ma[i] = vm;
        float denom = sqrtf(vm / bias_correction2) + eps;

        float gm = gm_old * beta1 + g * one_minus_beta1;
        grad_ma[i] = gm;

        float new_p = p[i];
        const float p_old = new_p;
        if (wd_factor != 0.0f) {
            new_p = new_p * (1.0f - wd_factor);
        }
        new_p = new_p - lr_div_bc1 * (gm / denom);
        if (la_params != nullptr) {
            // this step's lookahead blend: p.mul_(alpha).add_(la, alpha=1 - alpha); la.copy_(p)
            new_p = __fmaf_rn(la_beta, la_params[i], __fmul_rn(new_p, la_alpha));   // add_ contracts to an FMA
            la_params[i] = new_p;
        }
        p[i] = new_p;
        if (mout != nullptr) {
            const float q = merge_q(new_p, ms, minv);
            if (__float_as_uint(q) != __float_as_uint(merge_q(p_old, ms, minv))) mout[i] = q;
        }
    }
}

extern "C" __global__ void ranger_lite_update_variance(
    const float* __restrict__ grad,
    float* __restrict__ variance_ma,
    int64_t n,
    float beta2,
    float one_minus_beta2
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        variance_ma[i] = variance_ma[i] * beta2 + g * g * one_minus_beta2;
    }
}

extern "C" __global__ void ranger_lite_update_pnm_phase2(
    float* __restrict__ p,
    const float* __restrict__ grad,
    float* __restrict__ grad_ma,
    float* __restrict__ neg_grad_ma,
    const float* __restrict__ variance_ma,
    int64_t n,
    float bias_correction2,
    float beta1_sq,
    float one_minus_beta1_sq,
    float one_plus_pnm,
    float pnm_factor,
    float noise_norm,
    float lr_div_bc1,
    float eps,
    int64_t step,
    float wd_factor
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float vm = variance_ma[i];
        float denom = sqrtf(vm / bias_correction2) + eps;

        float g = grad[i];
        float gm = grad_ma[i];
        float ngm = neg_grad_ma[i];
        float pgm = (step & 1LL) ? gm : ngm;
        float pngm = (step & 1LL) ? ngm : gm;

        float new_pgm = pgm * beta1_sq + g * one_minus_beta1_sq;
        float pnm_val = (new_pgm * one_plus_pnm - pngm * pnm_factor) / noise_norm;
        float new_p = p[i];
        if (wd_factor != 0.0f) {
            new_p = new_p * (1.0f - wd_factor);
        }
        new_p = new_p - lr_div_bc1 * (pnm_val / denom);
        p[i] = new_p;

        if (step & 1LL) {
            grad_ma[i] = new_pgm;
        } else {
            neg_grad_ma[i] = new_pgm;
        }
    }
}

extern "C" __global__ void ranger_lite_update_adam_phase2(
    float* __restrict__ p,
    const float* __restrict__ grad,
    float* __restrict__ grad_ma,
    const float* __restrict__ variance_ma,
    int64_t n,
    float bias_correction2,
    float beta1,
    float one_minus_beta1,
    float lr_div_bc1,
    float eps,
    float wd_factor
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float vm = variance_ma[i];
        float denom = sqrtf(vm / bias_correction2) + eps;

        float g = grad[i];
        float gm = grad_ma[i] * beta1 + g * one_minus_beta1;
        grad_ma[i] = gm;

        float new_p = p[i];
        if (wd_factor != 0.0f) {
            new_p = new_p * (1.0f - wd_factor);
        }
        new_p = new_p - lr_div_bc1 * (gm / denom);
        p[i] = new_p;
    }
}
"""
        cls._pnm_kernel = cp.RawKernel(source, "ranger_lite_update_pnm")
        cls._adam_kernel = cp.RawKernel(source, "ranger_lite_update_adam")
        cls._variance_kernel = cp.RawKernel(source, "ranger_lite_update_variance")
        cls._pnm_phase2_kernel = cp.RawKernel(source, "ranger_lite_update_pnm_phase2")
        cls._adam_phase2_kernel = cp.RawKernel(source, "ranger_lite_update_adam_phase2")
        cls._pnm_kernel.compile()
        cls._adam_kernel.compile()
        cls._variance_kernel.compile()
        cls._pnm_phase2_kernel.compile()
        cls._adam_phase2_kernel.compile()
        cls._compiled = True


class RangerLite(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-7,
        pnm_activate=True,
        pnm_momentum=1.0,
        lookahead_active=True,
        lookahead_mergetime=5,
        lookahead_blending_alpha=0.5,
        normloss_active=True,
        normloss_factor=1e-4,
        use_legacy_scoping_bug=False,
        use_stable_weight_decay=True,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "pnm_momentum": pnm_momentum,
            "normloss_factor": normloss_factor,
        }
        super().__init__(params, defaults)
        self._live_cache = {}   # param -> (state tensors, their versions, live row flags); see _live_rows

        self.lookahead_active = lookahead_active
        self.lookahead_mergetime = lookahead_mergetime
        self.lookahead_alpha = lookahead_blending_alpha
        self.lookahead_step = 0

        self.pnm_active = pnm_activate
        self.normloss_active = normloss_active
        self.eps = eps
        self.param_size = 0

        self.use_legacy_scoping_bug = use_legacy_scoping_bug
        self.use_stable_weight_decay = use_stable_weight_decay

        if not _HAS_CUPY and torch.cuda.is_available():
            warnings.warn(
                "RangerLite: CuPy is not available; using the Python update path. "
                "Install CuPy to enable the fused optimizer kernel on CUDA.",
                stacklevel=2,
            )
        elif _HAS_CUPY and (self.normloss_active or self.use_legacy_scoping_bug):
            warnings.warn(
                "RangerLite: normloss or legacy mode disables the fused CUDA "
                "update kernel; using the Python update path.",
                stacklevel=2,
            )

    def _init_state(self, p):
        """Lazy state initialization for a parameter."""
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["variance_ma"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            if self.lookahead_active:
                state["lookahead_params"] = torch.clone(p.data)
            if self.pnm_active:
                state["neg_grad_ma"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

    def _live_rows(self, p, rows, factors):
        """Per-row flags of ``p`` for the fused update: 0 means every state element of the row is +0
        (bits). Computed from the state when the state tensors are new or were changed in place (version
        counter); afterwards the update kernel marks each row that may get a nonzero state (flagged, or a
        non-finite gradient scale). None when a state factor could turn +0 into another value."""
        if not all(math.isfinite(float(np.float32(f))) and float(np.float32(f)) >= 0.0 for f in factors):
            self._live_cache.pop(p, None)
            return None
        state = self.state[p]
        ts = tuple(state[k] for k in ("grad_ma", "neg_grad_ma", "variance_ma") if k in state)
        ent = self._live_cache.get(p)
        if (ent is None or len(ent[0]) != len(ts) or any(a is not b for a, b in zip(ent[0], ts))
                or ent[1] != tuple(t._version for t in ts) or ent[2].numel() != rows):
            live = torch.zeros(rows, dtype=torch.bool, device=p.device)
            for t in ts:
                live |= (t.detach().reshape(rows, -1).view(torch.int32) != 0).any(1)
            ent = (ts, tuple(t._version for t in ts), live.to(torch.uint8))
            self._live_cache[p] = ent
        return ent[2]

    def _can_fuse_group_full(self, group):
        """Full fused path requires no stable_weight_decay and no normloss."""
        if not _HAS_CUPY or self.use_legacy_scoping_bug or self.normloss_active:
            return False
        if self.use_stable_weight_decay and group.get("weight_decay", 0.0) != 0.0:
            return False
        return self._can_fuse_group_phase2(group)

    def _can_fuse_group_phase2(self, group):
        """Phase-2 fused path allows weight decay but not normloss/legacy."""
        if not _HAS_CUPY or self.use_legacy_scoping_bug or self.normloss_active:
            return False
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                return False
            if p.dtype != torch.float32 or p.device.type != "cuda":
                return False
            if not p.is_contiguous() or not grad.is_contiguous():
                return False
            state = self.state[p]
            if len(state) == 0:
                continue
            if (
                state["grad_ma"].dtype != torch.float32
                or state["grad_ma"].device != p.device
                or not state["grad_ma"].is_contiguous()
            ):
                return False
            if (
                state["variance_ma"].dtype != torch.float32
                or state["variance_ma"].device != p.device
                or not state["variance_ma"].is_contiguous()
            ):
                return False
            if self.pnm_active and (
                state["neg_grad_ma"].dtype != torch.float32
                or state["neg_grad_ma"].device != p.device
                or not state["neg_grad_ma"].is_contiguous()
            ):
                return False
        return True

    def _fused_update_group(self, group, grad_scale=None, lookahead_merge=False, merge_plan=None):
        """Apply the fused full update kernel (variance + moment + param). With ``lookahead_merge``
        the kernel also applies this step's lookahead blend (``lookahead_process_step``) to the new
        parameter value, so the parameters and lookahead copies are not passed over again."""
        _RangerLiteFusedKernels._compile()
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        decay = group.get("weight_decay", 0.0)
        pnm_factor = group["pnm_momentum"]
        eps = group["eps"]
        one_minus_beta2 = 1.0 - beta2
        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        one_plus_pnm = 1.0 + pnm_factor
        noise_norm = math.sqrt((1.0 + pnm_factor) ** 2 + pnm_factor**2)

        wd_factor = decay * lr if not self.use_stable_weight_decay else 0.0
        scale_ptr = np.uint64(grad_scale.data_ptr() if grad_scale is not None else 0)
        la_alpha = np.float32(self.lookahead_alpha)
        la_beta = np.float32(1.0 - self.lookahead_alpha)

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            self._init_state(p)

            state["step"] += 1
            step = state["step"]
            bias_correction2 = 1.0 - beta2**step

            p_arr = cp.from_dlpack(torch.to_dlpack(p.detach()))
            g_arr = cp.from_dlpack(torch.to_dlpack(grad))
            gm_arr = cp.from_dlpack(torch.to_dlpack(state["grad_ma"]))
            vm_arr = cp.from_dlpack(torch.to_dlpack(state["variance_ma"]))

            la_ptr = np.uint64(state["lookahead_params"].data_ptr() if lookahead_merge else 0)
            rf = _grf.lookup(grad) if _ROW_FLAGS else None
            if rf is not None:
                cols = grad.shape[1]
                if cols & (cols - 1) or not grad.is_contiguous() or p.numel() != grad.numel():
                    rf = None  # the row length must be a power of two
            rf_ptr = np.uint64(rf.data_ptr() if rf is not None else 0)
            rf_shift = np.int32(grad.shape[1].bit_length() - 1 if rf is not None else 0)

            def live_ptr(*factors, p=p, grad=grad, rf=rf):
                if rf is None or not _LIVE_ROWS:
                    return np.uint64(0)
                live = self._live_rows(p, grad.shape[0], factors)
                return np.uint64(live.data_ptr() if live is not None else 0)
            m = merge_plan[0].get(id(p)) if merge_plan is not None else None
            if m is not None and (not p.is_contiguous() or not m.is_contiguous() or m.shape != p.shape):
                raise RuntimeError("merge plan does not match the parameter")
            m_args = (
                np.uint64(m.data_ptr() if m is not None else 0),
                np.float32(merge_plan[1][0] if m is not None else 1.0),
                np.float32(merge_plan[1][1] if m is not None else 1.0),
            )
            block = 512
            grid = (p.numel() + block - 1) // block

            if self.pnm_active:
                effective_step = ((step + 1) // 2) * 2
                bias_correction1 = 1.0 - beta1**effective_step
                lr_div_bc1 = lr / bias_correction1
                ngm_arr = cp.from_dlpack(torch.to_dlpack(state["neg_grad_ma"]))
                dormant = rf is not None and not lookahead_merge and wd_factor == 0.0 and _identity_scalars(
                    lr_div_bc1, eps, bias_correction2, one_plus_pnm, pnm_factor, noise_norm)
                args = (
                    p_arr,
                    g_arr,
                    scale_ptr,
                    gm_arr,
                    ngm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(beta2),
                    np.float32(one_minus_beta2),
                    np.float32(bias_correction2),
                    np.float32(beta1_sq),
                    np.float32(one_minus_beta1_sq),
                    np.float32(one_plus_pnm),
                    np.float32(pnm_factor),
                    np.float32(noise_norm),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                    np.int64(step),
                    np.float32(wd_factor),
                    la_ptr,
                    la_alpha,
                    la_beta,
                    rf_ptr,
                    rf_shift,
                    np.int32(1 if dormant else 0),
                    live_ptr(beta2, one_minus_beta2, beta1_sq, one_minus_beta1_sq),
                    *m_args,
                )
                _RangerLiteFusedKernels._pnm_kernel(
                    grid=(grid,), block=(block,), args=args
                )
            else:
                bias_correction1 = 1.0 - beta1**step
                lr_div_bc1 = lr / bias_correction1
                dormant = rf is not None and not lookahead_merge and wd_factor == 0.0 and _identity_scalars(
                    lr_div_bc1, eps, bias_correction2, beta1, 1.0, 1.0)
                args = (
                    p_arr,
                    g_arr,
                    scale_ptr,
                    gm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(beta2),
                    np.float32(one_minus_beta2),
                    np.float32(bias_correction2),
                    np.float32(beta1),
                    np.float32(1.0 - beta1),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                    np.float32(wd_factor),
                    la_ptr,
                    la_alpha,
                    la_beta,
                    rf_ptr,
                    rf_shift,
                    np.int32(1 if dormant else 0),
                    live_ptr(beta2, one_minus_beta2, beta1, 1.0 - beta1),
                    *m_args,
                )
                _RangerLiteFusedKernels._adam_kernel(
                    grid=(grid,), block=(block,), args=args
                )

    def _fused_update_group_phase2(self, group, variance_norm_value=None):
        """Apply the fused phase-2 update kernel (moment + optional decay + param).

        Assumes variance_ma has already been updated by the caller.
        """
        _RangerLiteFusedKernels._compile()
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        decay = group.get("weight_decay", 0.0)
        pnm_factor = group["pnm_momentum"]
        eps = group["eps"]
        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        one_plus_pnm = 1.0 + pnm_factor
        noise_norm = math.sqrt((1.0 + pnm_factor) ** 2 + pnm_factor**2)

        wd_factor = 0.0
        if decay:
            if self.use_stable_weight_decay:
                assert variance_norm_value is not None, (
                    "variance_norm_value must be computed when use_stable_weight_decay is True and decay > 0"
                )
                wd_factor = decay * lr / variance_norm_value
            else:
                wd_factor = decay * lr

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            step = state["step"]
            bias_correction2 = 1.0 - beta2**step

            p_arr = cp.from_dlpack(torch.to_dlpack(p.detach()))
            g_arr = cp.from_dlpack(torch.to_dlpack(grad))
            gm_arr = cp.from_dlpack(torch.to_dlpack(state["grad_ma"]))
            vm_arr = cp.from_dlpack(torch.to_dlpack(state["variance_ma"]))

            block = 512
            grid = (p.numel() + block - 1) // block

            if self.pnm_active:
                effective_step = ((step + 1) // 2) * 2
                bias_correction1 = 1.0 - beta1**effective_step
                lr_div_bc1 = lr / bias_correction1
                ngm_arr = cp.from_dlpack(torch.to_dlpack(state["neg_grad_ma"]))
                args = (
                    p_arr,
                    g_arr,
                    scale_ptr,
                    gm_arr,
                    ngm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(bias_correction2),
                    np.float32(beta1_sq),
                    np.float32(one_minus_beta1_sq),
                    np.float32(one_plus_pnm),
                    np.float32(pnm_factor),
                    np.float32(noise_norm),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                    np.int64(step),
                    np.float32(wd_factor),
                )
                _RangerLiteFusedKernels._pnm_phase2_kernel(
                    grid=(grid,), block=(block,), args=args
                )
            else:
                bias_correction1 = 1.0 - beta1**step
                lr_div_bc1 = lr / bias_correction1
                args = (
                    p_arr,
                    g_arr,
                    gm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(bias_correction2),
                    np.float32(beta1),
                    np.float32(1.0 - beta1),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                    np.float32(wd_factor),
                )
                _RangerLiteFusedKernels._adam_phase2_kernel(
                    grid=(grid,), block=(block,), args=args
                )

    def _apply_weight_decay(self, p, decay, lr, variance_normalized):
        if not decay:
            return
        if self.use_stable_weight_decay:
            assert variance_normalized is not None, (
                "variance_normalized must be computed when use_stable_weight_decay is True and decay > 0"
            )
            p.data.mul_(1 - decay * lr / variance_normalized)
        else:
            p.data.mul_(1 - decay * lr)

    def unit_norm(self, x):
        """
        Calculates the L2 norm of each sub-unit (row/filter) in a parameter tensor.
        Returns a tensor of norms with the same number of dimensions as x (via keepdim).
        Examples:
            - Linear: (out, in) -> Norm per neuron
            - Conv2d: (out, in, h, w) -> Norm per filter
            - Embedding: (vocab, dim) -> Norm per vector
        """
        xlen = x.ndim

        if xlen <= 1:
            return x.norm(p=2.0, keepdim=False)
        dim = tuple(range(1, xlen))
        return x.norm(dim=dim, keepdim=True, p=2.0)

    @torch.no_grad()
    def step(self, closure=None, grad_scale=None):
        """``grad_scale``: optional 0-dim float32 CUDA tensor every gradient is multiplied by first
        (gradient clipping). The fused full-update kernels apply it while reading the gradient,
        which saves a pass over the gradients; other paths scale the gradients in place first."""
        loss = None
        if closure is not None and isinstance(closure, collections.abc.Callable):
            with torch.enable_grad():
                loss = closure()

        first_param = next(
            (
                p
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ),
            None,
        )

        if first_param is None:
            return loss  # No grads to process

        needs_variance_sum = self.normloss_active or (
            self.use_stable_weight_decay
            and any(g.get("weight_decay", 0.0) != 0.0 for g in self.param_groups)
        )

        # Fast path A: no variance sum needed and all groups can use the single
        # fused kernel that updates variance, moments and parameters at once.
        if not needs_variance_sum:
            all_fusable = all(self._can_fuse_group_full(g) for g in self.param_groups)
            if all_fusable:
                # On a lookahead merge step the blend runs inside the update kernels (same fp32
                # roundings as lookahead_process_step).
                la_merge = (
                    self.lookahead_active
                    and _FUSE_LOOKAHEAD
                    and self.lookahead_step + 1 >= self.lookahead_mergetime
                )
                # The update kernels of the FT weights without a virtual weight also write their part of
                # the merged FT weight of the next forward (fused_weight_quant).
                with_grad = {id(p) for g in self.param_groups for p in g["params"] if p.grad is not None}
                plan = _fwq.optimizer_plan(with_grad) if _fwq is not None else None
                if plan is None:
                    _invalidate_merge()
                    for group in self.param_groups:
                        self._fused_update_group(group, grad_scale, la_merge)
                else:
                    for group in self.param_groups:
                        self._fused_update_group(group, grad_scale, la_merge, merge_plan=plan)
                _grf.clear()
                if la_merge:
                    self.lookahead_step = 0
                elif self.lookahead_active:
                    self.lookahead_process_step()
                return loss
        _invalidate_merge()   # the paths below change the parameters without writing the merged weight

        if grad_scale is not None:
            grads = [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
            torch._foreach_mul_(grads, grad_scale)

        # Fast path B: every group that has work can use the phase-2 fused kernel.
        # We run a fused variance kernel per parameter, reduce the debiased sums,
        # then run the phase-2 update kernel per parameter.
        all_phase2_fusable = all(
            self._can_fuse_group_phase2(g) for g in self.param_groups
        )
        if all_phase2_fusable:
            _RangerLiteFusedKernels._compile()

            active_params = []
            param_size = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._init_state(p)
                    state = self.state[p]
                    state["step"] += 1
                    active_params.append((p, group))
                    param_size += p.numel()

            if not active_params:
                return loss

            for p, group in active_params:
                state = self.state[p]
                beta1, beta2 = group["betas"]
                grad = p.grad
                vm_arr = cp.from_dlpack(torch.to_dlpack(state["variance_ma"]))
                g_arr = cp.from_dlpack(torch.to_dlpack(grad))

                block = 512
                grid = (p.numel() + block - 1) // block
                args = (
                    g_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(beta2),
                    np.float32(1.0 - beta2),
                )
                _RangerLiteFusedKernels._variance_kernel(
                    grid=(grid,), block=(block,), args=args
                )

            variance_norm_value = None
            if needs_variance_sum:
                variance_ma_sum = None
                for p, group in active_params:
                    state = self.state[p]
                    beta1, beta2 = group["betas"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    partial = state["variance_ma"].sum() / bias_correction2
                    if variance_ma_sum is None:
                        variance_ma_sum = partial
                    else:
                        variance_ma_sum += partial
                if not self.param_size:
                    self.param_size = param_size
                variance_normalized = torch.sqrt(
                    variance_ma_sum / self.param_size
                ).clamp_min(self.eps)
                variance_norm_value = variance_normalized.item()

            for group in self.param_groups:
                self._fused_update_group_phase2(group, variance_norm_value)

            if self.lookahead_active:
                self.lookahead_process_step()
            return loss

        variance_ma_sum = torch.zeros(1, device=first_param.device)
        param_size = 0
        leaked_p = None

        # Phase 1: Accumulate variance_ma_sum for stable weight decay
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                leaked_p = p
                param_size += p.numel()
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Sparse matrix not supported")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["grad_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["variance_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    if self.lookahead_active:
                        state["lookahead_params"] = torch.clone(p.data)

                    if self.pnm_active:
                        # PNM components
                        state["neg_grad_ma"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                state["step"] += 1
                beta1, beta2 = group["betas"]

                # Variance computation
                bias_correction2 = 1 - beta2 ** state["step"]
                variance_ma = state["variance_ma"]

                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if needs_variance_sum:
                    variance_ma_debiased = variance_ma / bias_correction2
                    variance_ma_sum += variance_ma_debiased.sum()

        variance_normalized = None
        if self.use_stable_weight_decay and needs_variance_sum:
            if not self.param_size:
                if not param_size:
                    # No trainable params
                    return loss
                self.param_size = param_size

            variance_normalized = torch.sqrt(
                variance_ma_sum / self.param_size
            ).clamp_min(self.eps)

        # Phase 2: Apply weight decay and update weights
        for group in self.param_groups:
            decay = group["weight_decay"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            pnm_factor = group["pnm_momentum"]
            normloss_factor = group["normloss_factor"]

            # --- LEGACY BUG BEHAVIOR ---
            # This correctly targets the last parameter of the PREVIOUS group
            # (or the absolute last parameter of the network for Group 0)
            if self.use_legacy_scoping_bug and leaked_p is not None:
                self._apply_weight_decay(leaked_p, decay, lr, variance_normalized)
                if self.normloss_active:
                    unorm = self.unit_norm(leaked_p.data)
                    corr = 2 * normloss_factor * (1 - torch.div(1, unorm + self.eps))
                    leaked_p.data.mul_(1 - lr * corr)
            # ---------------------------

            for p in group["params"]:
                # --- MIMIC PYTHON VARIABLE LEAK ---
                # Python reassigns 'p' at the start of every loop iteration.
                # We capture that reassignment so the next group targets this group's final tensor.
                if self.use_legacy_scoping_bug:
                    leaked_p = p

                if p.grad is None:
                    continue

                state = self.state[p]

                # --- CORRECT BEHAVIOR ---
                if not self.use_legacy_scoping_bug:
                    self._apply_weight_decay(p, decay, lr, variance_normalized)
                    # Norm Loss
                    if self.normloss_active:
                        unorm = self.unit_norm(p.data)
                        corr = (
                            2 * normloss_factor * (1 - torch.div(1, unorm + self.eps))
                        )
                        p.data.mul_(1 - lr * corr)
                # ------------------------

                grad = p.grad
                step = state["step"]

                grad_ma = state["grad_ma"]
                variance_ma = state["variance_ma"]

                bias_correction2 = 1 - beta2**step

                # Despite the comment. Ranger21 doesnt actually use variance_ma_max for denominator
                denom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )

                if self.pnm_active:
                    # PNM Adam Core Setup
                    if step % 2 == 1:
                        pnm_grad_ma, pnm_neg_grad_ma = (
                            state["grad_ma"],
                            state["neg_grad_ma"],
                        )
                    else:
                        pnm_grad_ma, pnm_neg_grad_ma = (
                            state["neg_grad_ma"],
                            state["grad_ma"],
                        )

                    # Update neg_grad_ma
                    pnm_grad_ma.mul_(beta1**2).add_(grad, alpha=(1 - beta1**2))

                    if self.use_legacy_scoping_bug:
                        # Legacy calculation
                        bias_correction1 = 1 - beta1**step
                        noise_norm = math.sqrt((1 + beta2) ** 2 + beta2**2)

                    else:
                        # Corrected: Bias updated with exact number of effective steps
                        effective_step = ((step + 1) // 2) * 2
                        bias_correction1 = 1 - beta1**effective_step
                        # Corrected: Normalization calculated from pnm_factor like in original paper. However lr tuning is likely still required.
                        noise_norm = math.sqrt((1 + pnm_factor) ** 2 + pnm_factor**2)

                    pnm_val = (
                        pnm_grad_ma.mul(1 + pnm_factor)
                        .add(pnm_neg_grad_ma, alpha=-pnm_factor)
                        .mul(1 / noise_norm)
                    )

                    step_size = lr / bias_correction1
                    p.addcdiv_(pnm_val, denom, value=-step_size)

                else:
                    # Standard Adam update
                    bias_correction1 = 1 - beta1**step

                    # Standard EMA for the first moment
                    grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)

                    step_size = lr / bias_correction1
                    p.addcdiv_(grad_ma, denom, value=-step_size)

        if self.lookahead_active:
            self.lookahead_process_step()

        return loss

    #   Lookahead merge process
    def lookahead_process_step(self):
        """handles blending of params for lookahead step"""

        if not self.lookahead_active:
            return
        self.lookahead_step += 1

        if self.lookahead_step >= self.lookahead_mergetime:
            self.lookahead_step = 0
            _invalidate_merge()   # the blend below writes the parameters through .data
            # merge lookahead cached params and save current ones
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.data.mul_(self.lookahead_alpha).add_(
                        param_state["lookahead_params"],
                        alpha=1.0 - self.lookahead_alpha,
                    )
                    # save for next merge
                    param_state["lookahead_params"].copy_(p.data)

    def train(self):
        """Switches optimizer to training mode. For optimizers with train/eval behavior, this should enable training behavior."""
        _invalidate_merge()
        if self.lookahead_active:
            self.restore_for_training()

    def eval(self):
        """Switches optimizer to eval mode. For optimizers with train/eval behavior, this should enable evaluation behavior (e.g. swapping in slow weights)."""
        _invalidate_merge()
        if self.lookahead_active:
            self.swap_for_inference()

    def swap_for_inference(self):
        """Safely loads slow weights for eval/saving. Idempotent."""
        if not self.lookahead_active:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "lookahead_params" in state and "is_swapped" not in state:
                    # Only swap if we haven't already swapped.
                    # Perform pointer swap to avoid unnecessary tensor copies.
                    tmp_ref = p.data
                    p.data = state["lookahead_params"]
                    state["lookahead_params"] = tmp_ref
                    state["is_swapped"] = True

    def restore_for_training(self):
        """Restores fast weights for training. Idempotent."""
        if not self.lookahead_active:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "is_swapped" in state:
                    # Intentionally overwrite lookahead_params with the current
                    # to preserve any changes made during eval mode (e.g. swa)
                    tmp_ref = state["lookahead_params"]
                    state["lookahead_params"] = p.data
                    p.data = tmp_ref
                    del state["is_swapped"]
