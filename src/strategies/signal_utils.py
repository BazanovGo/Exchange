"""Small helpers for building boolean signal series."""

from __future__ import annotations

import pandas as pd


def rising_edge(condition: pd.Series) -> pd.Series:
    """True on the bar where a boolean condition switches False -> True."""
    cond = condition.fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


def falling_edge(condition: pd.Series) -> pd.Series:
    """True on the bar where a boolean condition switches True -> False."""
    cond = condition.fillna(False).astype(bool)
    return ~cond & cond.shift(1, fill_value=False)
