import torch
from torch import nn

from .input_feature import InputFeature


class PawnStruct(InputFeature):
    """Pawn-structure input feature, modelled on Full_Threats.

    One feature set covers a half-board (4 files): ``left`` = files a-d,
    ``right`` = files e-h. Each position activates at most one feature: the
    rank (0-based) of its (color-folded) pawn structure in a frequency-ordered
    vocabulary of size ``num_inputs``. Structures outside the top-``num_inputs``
    most common produce no active feature (out-of-vocabulary -> dropped).

    Like threats these carry no piece values, so the PSQT columns are zeroed,
    and the weights are clamped to the same int8-quantization-safe range as the
    threat weights.
    """

    MAX_ACTIVE_FEATURES = 1
    EXPORT_WEIGHT_DTYPE = torch.int8

    # Distinct base hashes per side; the configured vocabulary size is folded
    # in so a net is only loadable against a feature set of the same size.
    _BASE_HASH = {"left": 0x5A1F0001, "right": 0x5A1F0002}

    def __init__(self, num_outputs: int, *, side: str, num_inputs: int):
        super().__init__()
        assert side in ("left", "right"), side
        assert num_inputs > 0, num_inputs

        self.side = side
        self.num_outputs = num_outputs

        self.NUM_INPUTS = num_inputs
        self.NUM_REAL_FEATURES = num_inputs
        name = ("PawnStructLeft" if side == "left" else "PawnStructRight") + f":{num_inputs}"
        self.FEATURE_NAME = name
        self.INPUT_FEATURE_NAME = name
        self.HASH = (self._BASE_HASH[side] ^ (num_inputs & 0xFFFFFFFF)) & 0xFFFFFFFF

        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        """Pawn-structure features have no piece values, so PSQT columns are zero."""
        L1 = self.num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    def clip_weights(self, quantization) -> None:
        """Clamp to the same quantization-safe range as the threat weights."""
        self.weight.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )


def make_pawn_struct(side: str, num_inputs: int):
    """Return a callable ``num_outputs -> PawnStruct`` for use as a feature component."""
    return lambda num_outputs: PawnStruct(
        num_outputs, side=side, num_inputs=num_inputs
    )
