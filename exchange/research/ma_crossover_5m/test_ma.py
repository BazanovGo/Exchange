"""Проверка: рекуррентные реализации против прямых формул по определению."""
import numpy as np
import pytest

import ma as M

rng = np.random.default_rng(7)
N = 6000
C = 80 + np.cumsum(rng.normal(0, 0.05, N))
V = rng.integers(1, 5000, N).astype(float)


def ref_sma(x, n):
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = x[t - n + 1:t + 1].mean()
    return out


def ref_wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = (x[t - n + 1:t + 1] * w).sum() / w.sum()
    return out


def ref_vwma(x, v, n):
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = (x[t - n + 1:t + 1] * v[t - n + 1:t + 1]).sum() / v[t - n + 1:t + 1].sum()
    return out


def ref_recursive(x, n, alpha):
    out = np.full(len(x), np.nan)
    out[n - 1] = x[:n].mean()
    for t in range(n, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def ref_tma(x, n):
    """TMA по определению через треугольные веса, без вложенных SMA."""
    if n % 2:
        m = (n + 1) // 2
        a, b = m, m
    else:
        a, b = n // 2, n // 2 + 1
    w = np.convolve(np.ones(a), np.ones(b))       # треугольник длины a+b-1 = n
    w = w / w.sum()
    out = np.full(len(x), np.nan)
    L = len(w)
    for t in range(L - 1, len(x)):
        out[t] = (x[t - L + 1:t + 1] * w[::-1]).sum()
    return out


@pytest.mark.parametrize("n", [2, 3, 7, 50, 99, 250])
def test_all_types(n):
    np.testing.assert_allclose(M.sma(C, n), ref_sma(C, n), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(M.wma(C, n), ref_wma(C, n), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(M.vwma(C, V, n), ref_vwma(C, V, n), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(M.ema(C, n), ref_recursive(C, n, 2 / (n + 1)), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(M.ssma(C, n), ref_recursive(C, n, 1 / n), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(M.tma(C, n), ref_tma(C, n), rtol=0, atol=1e-9, equal_nan=True)


def test_tma_window_is_n():
    for n in (2, 5, 10, 11, 250):
        out = M.tma(C, n)
        assert np.isnan(out[n - 2]) and not np.isnan(out[n - 1])


def test_resync_no_drift():
    x = 1e4 + np.cumsum(rng.normal(0, 1, 200_000))
    np.testing.assert_allclose(M.sma(x, 17)[-1], x[-17:].mean(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(M.wma(x, 31)[-1], ref_wma(x[-31:], 31)[-1], rtol=0, atol=1e-7)
