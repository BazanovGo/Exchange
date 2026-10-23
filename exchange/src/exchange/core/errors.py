"""Exception hierarchy.

A single root (:class:`ExchangeError`) lets callers catch everything this
package raises without also swallowing unrelated ``ValueError``s.
"""

from __future__ import annotations


class ExchangeError(Exception):
    """Root of every error raised by this package."""


class DataError(ExchangeError):
    """Raised when input data violates a schema or quality invariant."""


class DataQualityError(DataError):
    """Raised when a dataset fails a blocking data-quality check."""


class ConfigError(ExchangeError):
    """Raised for malformed or contradictory configuration."""


class ExecutionError(ExchangeError):
    """Raised when the execution model cannot produce a legal fill."""


class LookaheadError(ExchangeError):
    """Raised when a temporal-integrity invariant is violated.

    Deliberately its own type: a lookahead violation is never a recoverable
    condition, it means a result is invalid.
    """


class ReproducibilityError(ExchangeError):
    """Raised when an experiment cannot be reproduced from its record."""


class NotEnoughDataError(DataError):
    """Raised when a statistic is requested on too short a sample."""
