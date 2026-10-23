"""Vendor-spliced continuous series, and the artefact they carry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.core.errors import DataError
from exchange.data.moex import load_moex_algopack, moex_continuous, moex_dataset
from exchange.data.prespliced import from_prespliced, splice_artefact


def contango_series(n_contracts=6, bars_each=40, gap=0.01, drift=0.0, noise=0.001, seed=1):
    """A flat underlying spliced across contracts, each 1% more expensive.

    The tradeable return is ~0 by construction; the raw spliced series rises
    by (1 + gap) ** (n_contracts - 1). Any measured total return above that
    noise floor is the artefact, so the expected answer is known exactly.
    """
    rng = np.random.default_rng(seed)
    rows = []
    level = 100.0
    stamp = pd.Timestamp("2024-01-01", tz="UTC")
    for k in range(n_contracts):
        if k:
            level *= 1.0 + gap               # the step up to the next contract
        price = level
        for _ in range(bars_each):
            price *= 1.0 + drift + (rng.normal(0, noise) if noise else 0.0)
            rows.append(
                {
                    "timestamp": stamp,
                    "open": price,
                    "high": price * 1.002,
                    "low": price * 0.998,
                    "close": price,
                    "volume": 1_000.0,
                    "secid": f"XX{k:02d}",
                }
            )
            stamp += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


class TestConstruction:
    def test_rolls_come_from_contract_transitions(self):
        series = from_prespliced(contango_series(), root="XX", contract_column="secid")
        assert len(series.provenance.rolls) == 5
        assert series.provenance.contracts_used == tuple(f"XX{k:02d}" for k in range(6))

    def test_a_missing_contract_column_is_refused(self):
        frame = contango_series().drop(columns=["secid"])
        with pytest.raises(DataError, match="cannot be audited"):
            from_prespliced(frame, root="XX", contract_column="secid")

    def test_roll_bars_are_flagged(self):
        series = from_prespliced(contango_series(), root="XX", contract_column="secid")
        assert int(series.is_roll.sum()) == 5
        first_of_segment = [s.start for s in series.provenance.segments[1:]]
        assert series.is_roll.loc[first_of_segment].all()

    def test_no_roll_claims_an_overlap_quote(self):
        """A pre-spliced series never has both contracts on one bar."""
        series = from_prespliced(contango_series(), root="XX", contract_column="secid")
        assert all(not event.overlap_available for event in series.provenance.rolls)

    def test_segments_tile_the_index(self):
        series = from_prespliced(contango_series(), root="XX", contract_column="secid")
        assert sum(s.n_bars for s in series.provenance.segments) == len(series.bars)

    def test_the_rule_records_that_the_roll_cannot_be_varied(self):
        series = from_prespliced(contango_series(), root="XX", contract_column="secid")
        assert series.provenance.rule["kind"] == "vendor_prespliced"
        assert "cannot be varied" in series.provenance.rule["note"]


class TestArtefact:
    def test_the_contango_artefact_equals_the_known_answer(self):
        """Five 1% steps on a perfectly flat underlying: 1.01**5 - 1."""
        series = from_prespliced(
            contango_series(gap=0.01, noise=0.0), root="XX", contract_column="secid"
        )
        measured = splice_artefact(series)
        assert measured["raw_total_return"] == pytest.approx(1.01**5 - 1, abs=1e-9)
        assert measured["tradeable_total_return"] == pytest.approx(0.0, abs=1e-9)
        assert measured["splice_artefact"] == pytest.approx(1.01**5 - 1, abs=1e-9)
        assert measured["median_gap_bps"] == pytest.approx(100, abs=1e-6)
        assert measured["share_above_100bps"] == 1.0        # every roll is a 100 bp step
        assert measured["n_rolls"] == 5

    def test_the_artefact_survives_noise(self):
        """With a noisy underlying the artefact is still recovered."""
        series = from_prespliced(contango_series(gap=0.01), root="XX", contract_column="secid")
        measured = splice_artefact(series)
        assert measured["splice_artefact"] == pytest.approx(1.01**5 - 1, abs=0.02)
        assert measured["raw_total_return"] > measured["tradeable_total_return"]

    def test_backwardation_flips_the_sign(self):
        series = from_prespliced(contango_series(gap=-0.01), root="XX", contract_column="secid")
        measured = splice_artefact(series)
        assert measured["splice_artefact"] < 0
        assert measured["median_gap_bps"] < 0

    def test_unadjusted_returns_mask_the_roll_bars(self):
        series = from_prespliced(contango_series(), root="XX", contract_column="secid")
        returns = series.tradeable_returns()
        assert returns[series.is_roll].isna().all()

    def test_adjusted_returns_mask_nothing(self):
        series = from_prespliced(
            contango_series(), root="XX", contract_column="secid", adjustment="ratio"
        )
        returns = series.tradeable_returns()
        assert returns[series.is_roll].notna().all()


class TestAdjustment:
    def test_ratio_adjustment_preserves_the_latest_price(self):
        """History moves; the present does not. Otherwise the series is not the market."""
        raw = from_prespliced(contango_series(), root="XX", contract_column="secid")
        adjusted = from_prespliced(
            contango_series(), root="XX", contract_column="secid", adjustment="ratio"
        )
        assert float(adjusted.bars["close"].iloc[-1]) == pytest.approx(
            float(raw.bars["close"].iloc[-1])
        )

    def test_ratio_adjustment_rescales_history_upward_in_contango(self):
        raw = from_prespliced(contango_series(gap=0.01), root="XX", contract_column="secid")
        adjusted = from_prespliced(
            contango_series(gap=0.01), root="XX", contract_column="secid", adjustment="ratio"
        )
        assert float(adjusted.bars["close"].iloc[0]) > float(raw.bars["close"].iloc[0])

    def test_ratio_adjustment_removes_the_gaps(self):
        adjusted = from_prespliced(
            contango_series(gap=0.01), root="XX", contract_column="secid", adjustment="ratio"
        )
        returns = adjusted.bars["close"].pct_change().dropna()
        assert returns.abs().max() < 0.02          # no 1% steps left

    def test_backward_adjustment_is_additive(self):
        adjusted = from_prespliced(
            contango_series(gap=0.001), root="XX", contract_column="secid", adjustment="backward"
        )
        assert (adjusted.bars["close"] > 0).all()
        assert float(adjusted.bars["close"].iloc[-1]) == pytest.approx(
            float(
                from_prespliced(
                    contango_series(gap=0.001), root="XX", contract_column="secid"
                ).bars["close"].iloc[-1]
            )
        )

    def test_backward_adjustment_refuses_to_produce_a_negative_price(self):
        """Additive adjustment cannot survive a gap larger than the price.

        Hand-built so the arithmetic is auditable: a contract that runs from
        0.5 up to 50 and then rolls *down* by 49.5. Back-adjusting shifts
        that whole segment by -49.5, which puts its early bars below zero.
        A negative price is not a cheap series to work with; it is a series
        on which returns are meaningless, so the adjustment is refused.
        """
        stamps = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
        prices = [0.5, 25.0, 50.0, 0.5, 0.6, 0.7]
        frame = pd.DataFrame(
            {
                "timestamp": stamps,
                "open": prices, "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices], "close": prices,
                "volume": 1_000.0,
                "secid": ["AA"] * 3 + ["BB"] * 3,
            }
        )
        with pytest.raises(DataError, match="non-positive price"):
            from_prespliced(frame, root="XX", contract_column="secid", adjustment="backward")

    def test_an_unknown_adjustment_is_refused(self):
        with pytest.raises(DataError, match="unknown adjustment"):
            from_prespliced(
                contango_series(), root="XX", contract_column="secid", adjustment="magic"
            )

    def test_ohlc_stays_consistent_after_adjustment(self):
        adjusted = from_prespliced(
            contango_series(), root="XX", contract_column="secid", adjustment="ratio"
        ).bars
        assert (adjusted["high"] >= adjusted[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (adjusted["low"] <= adjusted[["open", "close"]].min(axis=1) + 1e-9).all()


class TestMoexLoader:
    @pytest.fixture
    def algopack(self, tmp_path):
        """A miniature AlgoPack directory, in Moscow time like the real one."""
        folder = tmp_path / "ng" / "60m"
        folder.mkdir(parents=True)
        for year in (2024, 2025):
            stamps = pd.date_range(f"{year}-01-03 10:00", periods=50, freq="h")
            frame = pd.DataFrame(
                {
                    "ticker": "ng",
                    "secid": [f"NGF{year % 10}"] * 25 + [f"NGG{year % 10}"] * 25,
                    "begin": stamps,
                    "end": stamps + pd.Timedelta(minutes=59, seconds=59),
                    "open": np.linspace(2.0, 2.5, 50),
                    "high": np.linspace(2.1, 2.6, 50),
                    "low": np.linspace(1.9, 2.4, 50),
                    "close": np.linspace(2.0, 2.5, 50),
                    "volume": 1_000,
                    "value": 0,
                }
            )
            frame.to_csv(folder / f"{year}.csv", index=False)
        return tmp_path

    def test_moscow_time_is_converted_not_assumed(self, algopack):
        frame = load_moex_algopack(algopack, "ng", "60m")
        # 10:00 Moscow in January is 07:00 UTC.
        assert frame["timestamp"].iloc[0] == pd.Timestamp("2024-01-03 07:00", tz="UTC")

    def test_zero_turnover_column_is_dropped(self, algopack):
        frame = load_moex_algopack(algopack, "ng", "60m")
        assert "value" not in frame.columns
        assert "secid" in frame.columns

    def test_years_can_be_selected(self, algopack):
        frame = load_moex_algopack(algopack, "ng", "60m", years=[2025])
        assert frame["timestamp"].dt.year.unique().tolist() == [2025]

    def test_a_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(DataError):
            load_moex_algopack(tmp_path, "xx", "60m")

    def test_continuous_series_is_built_from_secid(self, algopack):
        series = moex_continuous(algopack, "ng", "60m")
        assert series.root == "NG"
        assert len(series.provenance.rolls) == 3      # two per year, minus the shared boundary
        assert series.meta["vendor"] == "moex_algopack"

    def test_dataset_carries_provenance(self, algopack):
        dataset = moex_dataset(algopack, "ng", "60m")
        assert dataset.provenance["source"] == "moex_algopack:ng/60m"
        assert dataset.provenance["timezone"] == "Europe/Moscow"
        assert dataset.provenance["continuous"]["adjustment"] == "none"

    def test_dataset_carries_the_forts_contract_not_the_nymex_one(self, algopack):
        # MOEX NG is 100 MMBtu; NYMEX NG is 10,000. The wrong one changes every cost ratio.
        dataset = moex_dataset(algopack, "ng", "60m")
        assert dataset.meta["instrument"] == {
            "multiplier": 100.0, "tick_size": 0.001, "allow_fractional": False,
        }
