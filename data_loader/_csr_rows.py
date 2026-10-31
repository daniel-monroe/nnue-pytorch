"""Compact (CSR) feature rows of a packed batch, readable without the padded unpack (tiled-ft-23).

With compact transport the loader copies one packed int32 block per batch (white / black row offsets,
white / black IDs, piece counts, queen buckets). ``unpack_sparse_indices`` used to expand it into the
padded (B, max_active) rows (151 MB written per batch at B = 65536). Now it only copies the per-position
metadata and registers the packed block here, keyed by the padded output; the preparation and PSQT
kernels read the rows from the packed block (``lookup``). Any other reader must call ``materialize``
first, which runs the padded unpack on the current stream (once per batch). NNUE_CSR_PREP=0 restores
the eager unpack.
"""

import os
import threading
import weakref

import torch

ENABLED = os.environ.get("NNUE_CSR_PREP", "1") != "0"
STATS = {"registered": 0, "lookups": 0, "materialized": 0}

_lock = threading.Lock()
# data_ptr(padded output) -> [weakref(output), packed, B, max_active, num_white, num_black, materialized]
_table: dict = {}


def register(output: torch.Tensor, packed: torch.Tensor, batch: int, max_active: int, nw: int, nb: int) -> None:
    with _lock:
        for k in [k for k, e in _table.items() if e[0]() is None]:
            del _table[k]
        _table[output.data_ptr()] = [weakref.ref(output), packed, batch, max_active, nw, nb, False]
        STATS["registered"] += 1


def _entry(white: torch.Tensor, black: torch.Tensor | None):
    if not _table or not white.is_cuda:
        return None
    with _lock:
        e = _table.get(white.data_ptr())
    if e is None or e[6]:
        return None
    out = e[0]()
    B, maxa = e[2], e[3]
    if out is None or out.data_ptr() != white.data_ptr() or tuple(white.shape) != (B, maxa) or not white.is_contiguous():
        return None
    if black is not None and (black.data_ptr() != white.data_ptr() + B * maxa * 4 or tuple(black.shape) != (B, maxa)):
        return None
    return e


def lookup(white: torch.Tensor, black: torch.Tensor | None = None) -> torch.Tensor | None:
    """The packed block behind the padded rows ``white`` (and ``black``) while they are not
    materialized, else None. Records the block on the current stream (the caller reads it there)."""
    e = _entry(white, black)
    if e is None:
        return None
    STATS["lookups"] += 1
    packed = e[1]
    packed.record_stream(torch.cuda.current_stream(packed.device))
    return packed


def materialize(white: torch.Tensor) -> None:
    """Write the padded rows of ``white``'s batch (no-op when they exist already)."""
    e = _entry(white, None)
    if e is None:
        return
    from ._cuda_transport import unpack_into

    packed = e[1]
    packed.record_stream(torch.cuda.current_stream(packed.device))
    unpack_into(packed, e[0](), e[2], e[3], e[4], e[5])
    e[6] = True
    STATS["materialized"] += 1
