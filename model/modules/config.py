from dataclasses import dataclass
from typing import Annotated

import tyro


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 31
    """Size of second hidden layer."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Size of third hidden layer."""
    num_value_bins: Annotated[int, tyro.conf.arg(name="num-value-bins")] = 128
    """Number of bins for the auxiliary categorical value head, which predicts a
    distribution over the teacher value from the same L3 (32-neuron) activation
    that feeds the scalar output. 0 disables the head. The head is training-only
    and is never exported to the .nnue / used at inference."""
    aux_piece_head: Annotated[bool, tyro.conf.arg(name="aux-piece-head")] = False
    """Enable the auxiliary piece-type head: a Linear(-> 6) predicting which of the
    six piece types was moved, from the 1024-dim feature-transformer output.
    Training-only; never exported to the .nnue."""
    aux_piece_head_l2: Annotated[bool, tyro.conf.arg(name="aux-piece-head-l2")] = False
    """Variant of aux_piece_head that taps the 32-dim second-layer activation (l2x_)
    instead of the 1024-dim FT output. Requires aux_piece_head=True."""
