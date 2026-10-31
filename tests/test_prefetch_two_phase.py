"""The two-phase prefetch worker (NNUE_PREP_TWO_PHASE, tiled-ft-21) delivers the same batches in the same
order with bit-identical tiled preparations as the one-phase worker."""
import os

import pytest
import torch

from data_loader.dataset import FixedNumBatchesDataset
from model.modules.feature_transformer import tiled_ft_backward as TB
from model.modules.feature_transformer import tiled_ft_forward as TF
from model.modules.feature_transformer import tiled_ft_pull as TP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Batches(torch.utils.data.IterableDataset):
    def __init__(self, n, B, maxa, seed):
        self.n, self.B, self.maxa, self.seed = n, B, maxa, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.n):
            def idx():
                x = torch.randint(0, 22528, (self.B, self.maxa), generator=g, dtype=torch.int32)
                n = torch.randint(self.maxa // 4, self.maxa + 1, (self.B, 1), generator=g)
                x[torch.arange(self.maxa)[None, :] >= n] = -1
                return x
            us = torch.randint(0, 2, (self.B, 1), generator=g).float()
            yield (us, 1 - us, idx(), idx(), torch.rand(self.B, 1, generator=g))


def _flat(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj.detach().cpu().clone())
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flat(v, out)
    elif hasattr(obj, "__dict__"):
        for k in sorted(vars(obj)):
            _flat(vars(obj)[k], out)
    elif isinstance(obj, (int, float, str)) or obj is None:
        out.append(obj)
    return out


def _collect(two_phase, n=12, B=4096, maxa=32, seed=3):
    os.environ["NNUE_PREP_TWO_PHASE"] = "1" if two_phase else "0"
    ds = FixedNumBatchesDataset(_Batches(n, B, maxa, seed), n, pin_memory=True, device=torch.device("cuda"))
    res = []
    for i in range(n):
        item = ds[i]
        bp, fp = TB.lookup_prep(item[2]), TF.lookup_prep(item[2])
        torch.cuda.synchronize()
        if fp is not None and hasattr(fp, "tile_rows"):
            # pull preparation: tile_rows is allocated with E entries (torch.empty) but only its first
            # drow_start[-1] (distinct rows per tile) are written and read, and wstart / wn (upper bound of
            # chunks) only their first chunk_start[-1] * NW; compare the written parts
            nw = int(fp.chunk_start[-1]) * TP.NW
            trimmed = {"tile_rows": fp.tile_rows[: int(fp.drow_start[-1])], "wstart": fp.wstart[:nw], "wn": fp.wn[:nw]}
            fp_flat = _flat([trimmed.get(k, v) for k, v in sorted(vars(fp).items())], [])
        else:
            fp_flat = _flat(fp, [])
        res.append((_flat(item, []), _flat(bp, []), fp_flat, bp is not None, fp is not None))
    os.environ.pop("NNUE_PREP_TWO_PHASE")
    return res


def _eq(a, b, where=""):
    assert len(a) == len(b), where
    for j, (x, y) in enumerate(zip(a, b)):
        where_j = f"{where} field {j}"
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.view(torch.uint8) if x.dtype.is_floating_point else x, y.view(torch.uint8) if y.dtype.is_floating_point else y), where_j
        else:
            assert x == y, where_j


def test_two_phase_same_batches_and_preps():
    one, two = _collect(False), _collect(True)
    assert len(one) == len(two) == 12
    for i, (a, b) in enumerate(zip(one, two)):
        assert a[3] and a[4] and b[3] and b[4]   # both preparations registered and found
        for k, (x, y) in enumerate(zip(a[:3], b[:3])):
            _eq(x, y, f"batch {i} part {k}")
