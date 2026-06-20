import argparse
from collections.abc import Callable
from dataclasses import dataclass

from .composed import ComposedFeatureTransformer
from .full_threats import FullThreats
from .halfka_v2_hm import HalfKav2Hm
from .input_feature import InputFeature
from .pawn_struct import make_pawn_struct

import tyro
from typing import Annotated


_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "Full_Threats": FullThreats,
}

# Parametric components configured as "<Name>:<num_inputs>" (e.g. PawnStructLeft:1024).
_DEFAULT_PAWN_INPUTS = 1024
_PARAMETRIC_COMPONENTS = {
    "PawnStructLeft": lambda n: make_pawn_struct("left", n),
    "PawnStructRight": lambda n: make_pawn_struct("right", n),
}


def _get_component(part: str) -> Callable[[int], InputFeature]:
    if part in _FEATURE_COMPONENTS:
        return _FEATURE_COMPONENTS[part]
    base, sep, count = part.partition(":")
    if base in _PARAMETRIC_COMPONENTS:
        n = int(count) if sep else _DEFAULT_PAWN_INPUTS
        return _PARAMETRIC_COMPONENTS[base](n)
    raise KeyError(part)


def get_feature_cls(name: str) -> list[Callable[[int], InputFeature]]:
    return [_get_component(p) for p in name.split("+")]


def get_available_features() -> list[str]:
    return list(_FEATURE_COMPONENTS.keys()) + [
        f"{k}:<N>" for k in _PARAMETRIC_COMPONENTS
    ]


@dataclass(kw_only=True)
class FeatureConfig:
    features: Annotated[
        str,
        tyro.conf.arg(
            help="The feature set to use. Available: "
            + ", ".join(get_available_features())
            + ". Combine with +, e.g. Full_Threats+HalfKAv2_hm^"
        ),
    ] = "Full_Threats+HalfKAv2_hm^"


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default="Full_Threats+HalfKAv2_hm^",
        help="The feature set to use. Available: "
        + ", ".join(get_available_features())
        + ". Combine with +, e.g. Full_Threats+HalfKAv2_hm^",
    )


__all__ = [
    "ComposedFeatureTransformer",
    "HalfKav2Hm",
    "FullThreats",
    "InputFeature",
    "get_feature_cls",
    "get_available_features",
    "add_feature_args",
    "FeatureConfig",
]
