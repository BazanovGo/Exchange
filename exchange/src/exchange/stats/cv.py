"""Purged cross-validation for time series.

Why standard K-fold is invalid here
------------------------------------
Ordinary K-fold assumes observations are exchangeable. Financial bars are
not, in two specific ways that both leak:

1. **Serial correlation.** A test observation immediately after a training
   observation shares most of its information. The model is effectively
   scored on data it has already seen.
2. **Overlapping outcomes.** A signal formed at bar ``t`` and held for 20
   bars has an outcome that spans ``[t, t+20]``. If bar ``t`` is in the
   training set and bar ``t+5`` in the test set, the two labels overlap and
   the "out-of-sample" score is contaminated.

The fix (López de Prado, *Advances in Financial Machine Learning*, ch. 7):
**purging** drops training observations whose outcome window overlaps the
test set, and an **embargo** additionally drops training observations for a
short window *after* the test set, because serial correlation flows forward
too. The audited project's walk-forward used adjacent windows with no gap of
either kind.

What this module provides
-------------------------
``PurgedKFold``
    Sequential folds with purging and embargo.
``CombinatorialPurgedCV``
    López de Prado's CPCV: split the sample into ``n_groups`` blocks, use
    every combination of ``k`` of them as test, and reassemble the results
    into many distinct backtest **paths** rather than a single one. The
    dispersion across paths is the quantity of interest — a strategy whose
    Sharpe ranges from -0.5 to 2.5 depending on which slices were held out
    does not have a Sharpe of 1.0, it has no reliable Sharpe at all.
``WalkForward``
    The anchored/rolling walk-forward split, also purged and embargoed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError, LookaheadError, NotEnoughDataError


@dataclass(frozen=True)
class Split:
    """One train/test partition, expressed as positional indices."""

    train: np.ndarray
    test: np.ndarray
    label: str = ""
    purged: int = 0        # training observations removed by purging
    embargoed: int = 0     # training observations removed by the embargo

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)

    def describe(self) -> dict:
        return {
            "label": self.label,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "purged": self.purged,
            "embargoed": self.embargoed,
        }


def _purge_and_embargo(
    train_mask: np.ndarray,
    test_positions: np.ndarray,
    *,
    horizon: int,
    embargo: int,
) -> tuple[np.ndarray, int, int]:
    """Remove training observations that leak into or out of the test set.

    ``horizon`` is the number of bars a signal's outcome spans. An
    observation at position ``i`` in training is purged when its outcome
    window ``[i, i + horizon]`` intersects any test position.
    """
    n = train_mask.size
    if test_positions.size == 0:
        return train_mask, 0, 0

    purged = np.zeros(n, dtype=bool)
    embargoed = np.zeros(n, dtype=bool)
    test_start, test_end = int(test_positions.min()), int(test_positions.max())

    # Purge: a training bar whose outcome window reaches into the test block,
    # and any training bar inside the test block's own span.
    if horizon > 0:
        lo = max(0, test_start - horizon)
        purged[lo:test_start] = True
        purged[test_start:test_end + 1] = True

    # Embargo: bars immediately after the test block, whose information is
    # still correlated with it.
    if embargo > 0:
        hi = min(n, test_end + 1 + embargo)
        embargoed[test_end + 1: hi] = True

    removed_purge = int((train_mask & purged).sum())
    removed_embargo = int((train_mask & embargoed & ~purged).sum())
    return train_mask & ~purged & ~embargoed, removed_purge, removed_embargo


@dataclass(frozen=True)
class PurgedKFold:
    """K sequential folds with purging and an embargo.

    Parameters
    ----------
    n_splits:
        Number of test blocks. Blocks are contiguous in time, never shuffled
        — shuffling would train on the future.
    horizon:
        Bars a signal's outcome spans (the holding period). Purging removes
        training bars whose outcome overlaps the test block.
    embargo:
        Bars after each test block excluded from training. A common choice is
        ~1% of the sample, or the autocorrelation horizon of returns.
    """

    n_splits: int = 5
    horizon: int = 1
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ConfigError("n_splits must be >= 2")
        if self.horizon < 0 or self.embargo < 0:
            raise ConfigError("horizon and embargo must be non-negative")

    def split(self, n: int) -> Iterator[Split]:
        if n < self.n_splits * 4:
            raise NotEnoughDataError(f"{n} observations is too few for {self.n_splits} purged folds")
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1]
            test = np.arange(lo, hi)
            mask = np.ones(n, dtype=bool)
            mask[lo:hi] = False
            mask, purged, embargoed = _purge_and_embargo(
                mask, test, horizon=self.horizon, embargo=self.embargo
            )
            yield Split(
                train=np.flatnonzero(mask),
                test=test,
                label=f"fold{k}",
                purged=purged,
                embargoed=embargoed,
            )

    def n_paths(self) -> int:
        return 1


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV).

    López de Prado, *Advances in Financial Machine Learning*, ch. 12.

    The sample is cut into ``n_groups`` contiguous blocks; every combination
    of ``n_test_groups`` blocks in turn becomes the test set. This yields
    ``C(n_groups, n_test_groups)`` splits and, because each block appears in
    the test set of many splits, the results can be recombined into

        ``paths = C(n_groups, k) * k / n_groups``

    distinct full-sample out-of-sample paths. A single walk-forward gives
    exactly one path and therefore one number with no dispersion; CPCV gives
    a *distribution*, which is what a robustness claim needs.

    With the default 6 groups and 2 test groups: 15 splits, 5 paths.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    horizon: int = 1
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.n_groups < 3:
            raise ConfigError("n_groups must be >= 3")
        if not 1 <= self.n_test_groups < self.n_groups:
            raise ConfigError("n_test_groups must be in [1, n_groups)")
        if self.horizon < 0 or self.embargo < 0:
            raise ConfigError("horizon and embargo must be non-negative")

    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def n_paths(self) -> int:
        """Number of reconstructable out-of-sample paths."""
        return self.n_splits() * self.n_test_groups // self.n_groups

    def group_bounds(self, n: int) -> list[tuple[int, int]]:
        edges = np.linspace(0, n, self.n_groups + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(self.n_groups)]

    def split(self, n: int) -> Iterator[Split]:
        if n < self.n_groups * 8:
            raise NotEnoughDataError(f"{n} observations is too few for {self.n_groups} CPCV groups")
        bounds = self.group_bounds(n)
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            mask = np.ones(n, dtype=bool)
            test_parts = []
            purged = embargoed = 0
            for g in combo:
                lo, hi = bounds[g]
                mask[lo:hi] = False
                test_parts.append(np.arange(lo, hi))
            # Purge and embargo around each test block independently: the
            # blocks are disjoint in time and each has its own boundary.
            for part in test_parts:
                mask, p, e = _purge_and_embargo(mask, part, horizon=self.horizon, embargo=self.embargo)
                purged += p
                embargoed += e
            yield Split(
                train=np.flatnonzero(mask),
                test=np.concatenate(test_parts),
                label="groups=" + ",".join(map(str, combo)),
                purged=purged,
                embargoed=embargoed,
            )

    def paths(self, n: int) -> list[list[tuple[int, Split]]]:
        """Group the splits into full-sample paths.

        Each path visits every group exactly once as test data. Returns a
        list of paths; a path is a list of ``(group_index, split)`` pairs
        covering the whole sample.
        """
        splits = list(self.split(n))
        bounds = self.group_bounds(n)
        by_group: dict[int, list[Split]] = {g: [] for g in range(self.n_groups)}
        for split in splits:
            tested = {
                g for g, (lo, hi) in enumerate(bounds)
                if np.any((split.test >= lo) & (split.test < hi))
            }
            for g in tested:
                by_group[g].append(split)
        n_paths = self.n_paths()
        return [
            [(g, by_group[g][p]) for g in range(self.n_groups)]
            for p in range(n_paths)
        ]


@dataclass(frozen=True)
class WalkForward:
    """Anchored or rolling walk-forward, with purging and an embargo.

    Parameters
    ----------
    train_size / test_size / step:
        In bars. ``step`` defaults to ``test_size`` (non-overlapping tests).
    anchored:
        ``True`` grows the training window from the start of the sample;
        ``False`` rolls a fixed-length window.
    """

    train_size: int = 750
    test_size: int = 250
    step: int | None = None
    anchored: bool = False
    horizon: int = 1
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.train_size < 10 or self.test_size < 5:
            raise ConfigError("train_size >= 10 and test_size >= 5 are required")

    @property
    def stride(self) -> int:
        return self.step or self.test_size

    def split(self, n: int) -> Iterator[Split]:
        if n < self.train_size + self.test_size:
            raise NotEnoughDataError(
                f"{n} bars is too short for train={self.train_size} + test={self.test_size}"
            )
        w = 0
        start = 0
        while start + self.train_size + self.test_size <= n:
            train_lo = 0 if self.anchored else start
            train_hi = start + self.train_size
            test = np.arange(train_hi, train_hi + self.test_size)

            mask = np.zeros(n, dtype=bool)
            mask[train_lo:train_hi] = True
            mask, purged, embargoed = _purge_and_embargo(
                mask, test, horizon=self.horizon, embargo=self.embargo
            )
            yield Split(
                train=np.flatnonzero(mask),
                test=test,
                label=f"window{w}",
                purged=purged,
                embargoed=embargoed,
            )
            start += self.stride
            w += 1

    def n_paths(self) -> int:
        return 1


def split_summary(splits: Sequence[Split]) -> pd.DataFrame:
    """Table of every split's shape — the audit trail for a validation run."""
    return pd.DataFrame([s.describe() for s in splits])


def assert_no_overlap(splits: Sequence[Split]) -> None:
    """Raise if any split's train and test sets intersect.

    Cheap, and it has caught real bugs: an off-by-one in a purge boundary is
    invisible in a metric and fatal to a conclusion.
    """
    for split in splits:
        overlap = np.intersect1d(split.train, split.test, assume_unique=False)
        if overlap.size:
            raise LookaheadError(
                f"split {split.label!r}: {overlap.size} observation(s) appear in both train and test"
            )
