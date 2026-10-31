"""tiled-ft-23: preparation and PSQT kernels read the packed (CSR) rows bit-identically to the padded rows."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import _csr_rows, _transport
from model.modules.feature_transformer import tiled_ft_backward as TB
from model.modules.feature_transformer import tiled_ft_forward as TF

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or _transport.unpack_sparse_indices is None or not TB._HAS_CUPY,
    reason="needs CUDA, Triton and CuPy")


def _flat(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj.detach().cpu().clone())
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flat(v, out)
    elif hasattr(obj, "__dict__"):
        for k in sorted(vars(obj)):
            if k not in ("tile_rows", "wstart", "wn"):    # pull prep: partly uninitialized (torch.empty upper bounds)
                _flat(vars(obj)[k], out)
    elif isinstance(obj, (int, float, str)) or obj is None:
        out.append(obj)
    return out


def _batch(B, maxa, seed):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(2):
        x = torch.randint(0, 22528, (B, maxa), generator=g, dtype=torch.int32)
        n = torch.randint(0, maxa + 1, (B, 1), generator=g)
        x[torch.arange(maxa)[None, :] >= n] = -1
        rows.append(x)
    offsets = [torch.cat((torch.zeros(1, dtype=torch.int32), (d >= 0).sum(1).cumsum(0).int())) for d in rows]
    ids = [d[d >= 0] for d in rows]
    meta = torch.randint(0, 100, (2 * B,), generator=g, dtype=torch.int32)
    packed = torch.cat((*offsets, *ids, meta)).cuda()
    return rows, meta, packed, ids[0].numel(), ids[1].numel()


def _unpack(packed, B, maxa, nw, nb, csr):
    old = _csr_rows.ENABLED
    _csr_rows.ENABLED = csr
    try:
        out = _transport.unpack_sparse_indices(packed.clone(), B, maxa, nw, nb)
    finally:
        _csr_rows.ENABLED = old
    return out, out[: B * maxa].view(B, maxa), out[B * maxa: 2 * B * maxa].view(B, maxa)


@pytest.mark.parametrize("B,maxa,seed", [(4096, 32, 1), (3000, 64, 2)])
def test_csr_prep_forward_backward_identical(B, maxa, seed):
    rows, meta, packed, nw, nb = _batch(B, maxa, seed)
    out_d, wd, bd = _unpack(packed, B, maxa, nw, nb, False)
    out_c, wc, bc = _unpack(packed, B, maxa, nw, nb, True)
    assert _csr_rows.lookup(wd, bd) is None and _csr_rows.lookup(wc, bc) is not None
    assert torch.equal(out_c[2 * B * maxa:], meta.cuda())               # metadata copied eagerly
    assert torch.equal(TB.list_counts(wc, bc), TB.list_counts(wd, bd))
    for a, b in zip(TB.list_order(wd, bd, with_counts=True), TB.list_order(wc, bc, with_counts=True)):
        assert torch.equal(a, b)
    bp_d, bp_c = TB.prepare(wd, bd), TB.prepare(wc, bc)
    fa, fb = _flat(bp_d, []), _flat(bp_c, [])
    assert len(fa) == len(fb) and all(torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y for x, y in zip(fa, fb))
    l1, out = 512, 520
    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(22528, out, generator=g, device="cuda")
    bias = torch.randn(out, generator=g, device="cuda")
    res = []
    for w, b in ((wd, bd), (wc, bc)):
        fp = TF.prepare(w, b)
        fw, fbk = TF.tiled_forward(w, b, weight, bias, l1, fp)
        gw = torch.randint(-3, 4, (B, out), generator=g, device="cuda").float()   # integers: atomics order-exact
        gb = torch.randint(-3, 4, (B, out), generator=g, device="cuda").float()
        gwt, gbt = TB.tiled_backward(w, b, gw, gb, 22528, l1, prep=TB.prepare(w, b))
        res.append([fw.clone(), fbk.clone(), gwt.clone(), gbt.clone()])
        g = torch.Generator(device="cuda").manual_seed(seed)
        torch.randn(22528, out, generator=g, device="cuda"); torch.randn(out, generator=g, device="cuda")
    for x, y in zip(*res):
        assert torch.equal(x, y)
    # materialize: the padded rows appear, later lookups use them
    _csr_rows.materialize(wc)
    assert torch.equal(wc.cpu(), rows[0]) and torch.equal(bc.cpu(), rows[1]) and torch.equal(out_c, out_d)
    assert _csr_rows.lookup(wc, bc) is None
