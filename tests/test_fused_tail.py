"""tiled-ft-20 fused tail kernels are bit-identical to the eager ops they replace."""
import types

import torch

from model.model import NNUEModel
from model.optimizers import fused_tail as ft


def _model(dev):
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda *s: (torch.randn(*s, generator=g) * 2).to(dev)  # noqa: E731
    p1, v1, p2, p3 = mk(256, 1024), mk(16, 512), mk(512, 32), mk(8, 32)
    p1[0, :5] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0])
    m = types.SimpleNamespace(weight_clipping=[
        {"params": [p1], "min_weight": -1.98, "max_weight": 1.98, "virtual_params": v1},
        {"params": [p2], "min_weight": -1.5, "max_weight": 1.5},
        {"params": [p3], "min_weight": -0.7, "max_weight": 0.7},
    ], input=None)
    return m


def test_clip_hidden_weights():
    dev = torch.device("cuda")
    b = _model(dev)
    ft.ENABLED = False
    try:
        NNUEModel.clip_weights(b, include_input=False)
    finally:
        ft.ENABLED = True
    a2 = _model(dev)
    assert ft.clip_hidden_weights(a2)
    for ga, gb in zip(a2.weight_clipping, b.weight_clipping):
        x, y = ga["params"][0], gb["params"][0]
        assert torch.equal(x.view(torch.int32), y.view(torch.int32))


def test_clip_scale_and_isfinite():
    dev = torch.device("cuda")
    vals = torch.cat([torch.rand(20000) * 10, torch.rand(20000) * 1e-3, torch.rand(2000) * 1e6,
                      torch.tensor([0.0, float("inf"), float("nan"), 0.999999, 1.0, 1.000001])]).to(dev)
    for clip in (1.0, 0.3, 7.25):
        for v in vals[::97].tolist() + vals[-6:].tolist():
            t = torch.tensor(v, device=dev)
            ref = torch.clamp(clip / (t + 1e-6), max=1.0)
            got = ft.clip_scale(t, clip)
            assert torch.equal(ref.view(torch.int32), got.view(torch.int32)) or (ref.isnan() and got.isnan()), (v, clip)
    for v in (1.0, float("nan"), float("inf"), -float("inf"), 0.0):
        for f0 in (True, False):
            f = torch.tensor(f0, device=dev)
            r = torch.tensor(f0, device=dev)
            l = torch.tensor(v, device=dev)
            assert ft.and_isfinite(f, l)
            r &= torch.isfinite(l).all()
            assert bool(f) == bool(r)
