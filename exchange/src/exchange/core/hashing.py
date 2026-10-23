"""Content hashing for datasets, configs and code.

Reproducibility claims are only meaningful if "the same data" and "the same
config" are verifiable. The audited project recorded a ``studied_at``
timestamp and nothing else, so none of its nineteen studies can be
reproduced with confidence. Every hash here is:

* **stable** across processes and platforms (no :func:`hash`, no pickle);
* **canonical** — the same logical content hashes the same regardless of
  column order, dict key order or float formatting noise;
* **cheap** — blake2b over a canonical byte stream.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_DIGEST_SIZE = 16  # 128-bit: ample for provenance, short enough to read


def _new():
    return hashlib.blake2b(digest_size=_DIGEST_SIZE)


def hash_bytes(data: bytes) -> str:
    h = _new()
    h.update(data)
    return h.hexdigest()


def hash_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Hash a file's raw bytes."""
    h = _new()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: object) -> str:
    """Serialise ``obj`` deterministically: sorted keys, normalised scalars."""
    return json.dumps(_canonicalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonicalise(obj: object) -> object:
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        value = float(obj)
        if np.isnan(value):
            return "__nan__"
        if np.isinf(value):
            return "__inf__" if value > 0 else "__-inf__"
        # 17 significant digits round-trips an IEEE754 double exactly and is
        # free of platform formatting differences.
        return float(f"{value:.17g}")
    if isinstance(obj, np.ndarray):
        return [_canonicalise(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return f"{obj.value}ns"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _canonicalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, set | frozenset):
        return sorted(_canonicalise(v) for v in obj)  # type: ignore[type-var]
    if isinstance(obj, Sequence):
        return [_canonicalise(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def hash_config(obj: object) -> str:
    """Hash any JSON-serialisable configuration object canonically."""
    return hash_bytes(canonical_json(obj).encode())


def hash_dataframe(df: pd.DataFrame, *, include_index: bool = True) -> str:
    """Hash a DataFrame's content, independent of column order.

    Columns are hashed individually and their digests combined in sorted
    name order, so reordering columns does not change the result while
    changing a single value does.
    """
    h = _new()
    if include_index:
        h.update(b"index:")
        h.update(_array_bytes(df.index.to_numpy()))
    for name in sorted(map(str, df.columns)):
        col = df[name]
        if isinstance(col, pd.DataFrame):  # duplicated column labels
            for i in range(col.shape[1]):
                h.update(f"|{name}#{i}:".encode())
                h.update(_array_bytes(col.iloc[:, i].to_numpy()))
            continue
        h.update(f"|{name}:".encode())
        h.update(_array_bytes(col.to_numpy()))
    return h.hexdigest()


def _array_bytes(arr: np.ndarray) -> bytes:
    """Canonical byte representation of a 1-D array."""
    if arr.dtype.kind in "mM":
        return np.ascontiguousarray(arr.astype("datetime64[ns]").astype("int64")).tobytes()
    if arr.dtype.kind == "f":
        return np.ascontiguousarray(arr.astype("float64")).tobytes()
    if arr.dtype.kind in "iub":
        return np.ascontiguousarray(arr).tobytes()
    return " ".join(map(repr, arr.tolist())).encode()


def short(digest: str, n: int = 12) -> str:
    """Human-facing abbreviation of a digest."""
    return digest[:n]
