"""Deterministic execution.

Reproducibility is a first-class requirement, so randomness is never
implicit. Every component that needs randomness takes a
:class:`numpy.random.Generator` built from an explicit seed, and
:func:`deterministic_mode` additionally pins the interpreter-level sources
of nondeterminism (hash randomisation cannot be changed after start-up, so
it is reported rather than silently assumed).
"""

from __future__ import annotations

import hashlib
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

DEFAULT_SEED = 20260902


def rng(seed: int | None = None) -> np.random.Generator:
    """Return a PCG64 generator seeded explicitly.

    ``None`` maps to :data:`DEFAULT_SEED` rather than to entropy: an
    unseeded research run is a bug, not a convenience.
    """
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def spawn(parent: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Split ``parent`` into ``n`` independent child generators.

    Uses ``SeedSequence`` spawning, so parallel or reordered consumption of
    the children gives the same streams as sequential consumption.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    seeds = parent.bit_generator.seed_seq.spawn(n)  # type: ignore[union-attr]
    return [np.random.Generator(np.random.PCG64(s)) for s in seeds]


def derive_seed(*parts: object, base: int = DEFAULT_SEED) -> int:
    """Derive a stable 63-bit seed from ``base`` and a tuple of labels.

    Stable across processes (unlike :func:`hash`) because it goes through
    blake2b of the parts' string forms.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode())
    for part in parts:
        h.update(b"\x00")
        h.update(str(part).encode())
    return int.from_bytes(h.digest(), "big") >> 1


@dataclass(frozen=True)
class DeterminismReport:
    """What the process could and could not pin down."""

    seed: int
    pythonhashseed: str | None
    hash_randomisation_disabled: bool

    @property
    def fully_deterministic(self) -> bool:
        return self.hash_randomisation_disabled

    def summary(self) -> str:
        """One line for ``exchange doctor``."""
        phs = self.pythonhashseed if self.pythonhashseed is not None else "unset"
        state = "fully pinned" if self.fully_deterministic else "PYTHONHASHSEED not pinned"
        return f"seed={self.seed} PYTHONHASHSEED={phs} ({state})"

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "pythonhashseed": self.pythonhashseed,
            "hash_randomisation_disabled": self.hash_randomisation_disabled,
        }


def deterministic_mode(seed: int = DEFAULT_SEED) -> DeterminismReport:
    """Pin every seed this process controls; report what it cannot.

    ``PYTHONHASHSEED`` must be set before the interpreter starts, so this
    function reports its state instead of pretending to change it. The
    package never relies on set/dict iteration order for numeric output, so
    an unpinned hash seed does not by itself break reproducibility — it is
    reported for completeness.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    phs = os.environ.get("PYTHONHASHSEED")
    return DeterminismReport(
        seed=seed,
        pythonhashseed=phs,
        hash_randomisation_disabled=phs == "0",
    )


@contextmanager
def temporary_seed(seed: int):
    """Run a block with global RNG state saved and restored.

    Only for third-party code that insists on global state; package code
    should take a :class:`~numpy.random.Generator` instead.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
