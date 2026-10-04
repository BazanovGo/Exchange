"""Structural scanner: measure the market before searching it."""

from __future__ import annotations

from exchange.scanner.diagnostics import (
    ALL_DIAGNOSTICS,
    MIN_OBSERVATIONS,
    Diagnostic,
    adf_test,
    autocorrelation,
    hurst_exponent,
    mean_reversion_half_life,
    partial_autocorrelation,
    seasonality,
    structural_stability,
    tail_behaviour,
    variance_ratio,
    variance_ratio_profile,
    volatility_clustering,
)
from exchange.scanner.report import ScanResult, scan, scan_table

__all__ = [
    "ALL_DIAGNOSTICS",
    "MIN_OBSERVATIONS",
    "Diagnostic",
    "ScanResult",
    "adf_test",
    "autocorrelation",
    "hurst_exponent",
    "mean_reversion_half_life",
    "partial_autocorrelation",
    "scan",
    "scan_table",
    "seasonality",
    "structural_stability",
    "tail_behaviour",
    "variance_ratio",
    "variance_ratio_profile",
    "volatility_clustering",
]
