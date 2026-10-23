"""Калибровка тестов устойчивости на данных с известной истиной."""
import numpy as np

import robust as R


def test_rc_spa_null_and_signal():
    rng = np.random.default_rng(1)
    T, N = 120, 3000
    null = rng.normal(0, 1, (N, T))
    r = R.reality_check_spa(null, months=slice(0, T), B=400, seed=2, chunk=1000)
    assert r["rc_p"] > 0.05 and r["spa_p"] > 0.05
    sig = null.copy()
    sig[17] += 0.6                      # одна модель с настоящим преимуществом
    r2 = R.reality_check_spa(sig, months=slice(0, T), B=400, seed=2, chunk=1000)
    assert r2["spa_p"] < 0.05 and r2["spa_best"] == 17


def test_spa_rejection_rate_on_noise_is_nominal():
    """Доля отказов SPA на чистом шуме — около номинальных 5%, а не 60–100%,
    как у первой версии со студентизацией бутстрэп-дисперсией."""
    rng = np.random.default_rng(9)
    for T in (28, 60):
        p = [R.reality_check_spa(rng.normal(0, 10, (1500, T)), months=slice(0, T), B=300,
                                 seed=r, chunk=1500)["spa_p"] for r in range(40)]
        assert np.mean(np.array(p) < 0.05) <= 0.125


def test_pbo_null_near_half_and_signal_low():
    rng = np.random.default_rng(3)
    null = rng.normal(0, 1, (2000, 96))
    assert 0.3 < R.pbo_cscv(null, n_blocks=8, n_splits=70, seed=1)["pbo"] < 0.7
    sig = null.copy()
    sig[:50] += 0.5
    assert R.pbo_cscv(sig, n_blocks=8, n_splits=70, seed=1)["pbo"] < 0.1


def test_plateau_and_walk_forward_shapes():
    shape = (1, 1, 20, 1, 1, 10)
    v = np.arange(200, dtype=float)
    p = R.plateau(v, shape, size=(3, 3))
    assert p.shape == (200,) and np.isfinite(p).all()
    rng = np.random.default_rng(4)
    m = rng.normal(0, 1, (200, 48))
    steps = R.walk_forward(m, shape, train=24, test=6, criterion="plateau", min_active=1)
    assert len(steps) == 4 and steps[0]["test_start"] == 24


def test_plateau_does_not_invent_neighbours_at_the_edge():
    """Одна выбросная точка на краю: при честном усреднении её плато ниже,
    чем у точки в середине ровной области того же уровня."""
    shape = (1, 1, 30, 1, 1, 30)
    v = np.full(shape, 0.5)
    v[0, 0, -1, 0, 0, 15] = 5.0                 # выброс на краю по медленной
    p = R.plateau(v.ravel(), shape, size=(7, 7)).reshape(shape)
    # честно: 4 реальных строки × 7 = 28 соседей, 0.5 + 4.5/28 = 0.66;
    # с mode="nearest" край повторялся бы ещё 3 раза: 0.5 + 4.5·4/49 = 0.87
    assert p[0, 0, -1, 0, 0, 15] < 0.75


def test_dsr_penalises_trials():
    rng = np.random.default_rng(5)
    x = rng.normal(0.3, 1, 120)
    a = R.deflated_sharpe(x, 1, 0.01)["dsr"]
    b = R.deflated_sharpe(x, 1e6, 0.01)["dsr"]
    assert b < a
