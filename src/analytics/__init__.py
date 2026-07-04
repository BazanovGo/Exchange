"""Result analysis: standardized performance reports."""

from src.analytics.features import (
    REGIME_ORDER,
    efficiency_ratio,
    higher_tf_trend,
    label_regime,
    market_features,
    redundant_pairs,
    trade_feature_dataset,
)
from src.analytics.report import PerformanceReport, key_metrics, portfolio_metrics

__all__ = [
    "PerformanceReport",
    "key_metrics",
    "portfolio_metrics",
    "market_features",
    "label_regime",
    "REGIME_ORDER",
    "efficiency_ratio",
    "higher_tf_trend",
    "trade_feature_dataset",
    "redundant_pairs",
]
