"""Synthetic markets with known ground truth (positive/negative/trap controls)."""

from __future__ import annotations

from exchange.synthetic.generators import (
    CONTROLS,
    GroundTruth,
    SyntheticSeries,
    bars_from_close,
    futures_ladder,
    make_control,
    make_index,
    mean_reverting,
    momentum,
    random_walk,
    regime_switch,
    seasonal,
    signal_plus_noise,
    structural_break,
    with_lookahead_trap,
)

__all__ = [
    "CONTROLS",
    "GroundTruth",
    "SyntheticSeries",
    "bars_from_close",
    "futures_ladder",
    "make_control",
    "make_index",
    "mean_reverting",
    "momentum",
    "random_walk",
    "regime_switch",
    "seasonal",
    "signal_plus_noise",
    "structural_break",
    "with_lookahead_trap",
]
