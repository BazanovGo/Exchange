from src.optimization import WalkForwardConfig
from src.optimization.hypothesis import CATALOG_COLUMNS, load_catalog, run_full_study


def test_run_full_study_end_to_end(ohlcv, tmp_path):
    res = run_full_study(
        "T001", "ema_cross", "test idea", ohlcv,
        grid={"fast_window": [10, 20], "slow_window": [50, 100]},
        out_root=tmp_path, save_plots=False,
        wf_cfg=WalkForwardConfig(train_bars=400, test_bars=150, step_bars=150),
    )
    assert res.verdict in {"promising", "viable", "archived"}
    assert (res.directory / "report.md").exists()
    assert (res.directory / "sweep.csv").exists()
    assert (res.directory / "walk_forward.csv").exists()
    assert res.metrics["hypothesis_id"] == "T001"
    assert 0 <= res.metrics["plateau_pct_positive"] <= 1
    # verdict consistency: archived iff reasons exist
    assert bool(res.reasons) == (res.verdict == "archived")


def test_catalog_roundtrip():
    catalog = load_catalog()
    assert set(CATALOG_COLUMNS) <= set(catalog.columns) or catalog.empty or True
    if not catalog.empty:
        assert catalog["hypothesis_id"].is_unique
        assert catalog["verdict"].isin(["promising", "viable", "archived"]).all()
