"""RangerLite phase-2 fused path (stable weight decay with weight_decay > 0) without the host sync.

The decay factor decay * lr / variance_normalized is computed on the device (fp64 division, rounded to
fp32, as the host did) and read by the update kernels, so the parameters and moments must equal those of
tiled-ft-21 (which read variance_normalized.item()) bit for bit, and steps after the first must not sync.
"""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASE = "86636c6"  # tiled-ft-21


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "model.optimizers"
    spec.loader.exec_module(m)
    return m


def _old_module(tmp_path):
    src = subprocess.run(["git", "show", f"{BASE}:model/optimizers/ranger_lite.py"], cwd=ROOT,
                         capture_output=True, text=True)
    if src.returncode:
        pytest.skip("reference commit not available")
    s = src.stdout
    i = s.index("def _fused_update_group_phase2")
    j = s.index("                    scale_ptr,\n", i)  # the base passed an undefined name here
    s = s[:j] + s[j + len("                    scale_ptr,\n"):]
    p = tmp_path / "rl_old.py"
    p.write_text(s)
    return _load(str(p), "model.optimizers.rl_old")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("pnm", [True, False])
def test_phase2_device_decay_bit_identical(tmp_path, pnm):
    new = _load(os.path.join(ROOT, "model/optimizers/ranger_lite.py"), "model.optimizers.rl_new")
    old = _old_module(tmp_path)
    torch.manual_seed(0)
    shapes = [(1000, 64), (64,), (32, 16), (7,)]
    init = [torch.randn(s, device="cuda") for s in shapes]
    grads = [[torch.randn(s, device="cuda") * 10 ** (k % 5 - 2) for s in shapes] for k in range(12)]
    res = []
    for mod in (old, new):
        ps = [torch.nn.Parameter(t.clone()) for t in init]
        groups = [{"params": ps[:2], "lr": 8.75e-4, "weight_decay": 0.03},
                  {"params": ps[2:], "lr": 3e-3, "weight_decay": 1e-5}]
        opt = mod.RangerLite(groups, lr=1.0, weight_decay=0.0, normloss_active=False,
                             use_stable_weight_decay=True, pnm_activate=pnm)
        out = []
        for k, g in enumerate(grads):
            for p, gg in zip(ps, g):
                p.grad = gg.clone()
            if mod is new and k > 0:
                torch.cuda.set_sync_debug_mode("error")
            try:
                opt.step()
            finally:
                torch.cuda.set_sync_debug_mode(0)
            out.append([p.detach().clone() for p in ps]
                       + [opt.state[p][n].clone() for p in ps for n in ("grad_ma", "variance_ma")])
        res.append(out)
    assert not torch.equal(res[1][-1][0], init[0])
    for sa, sb in zip(*res):
        for a, b in zip(sa, sb):
            assert torch.equal(a, b)
