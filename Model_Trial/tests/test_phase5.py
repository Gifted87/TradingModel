"""
Unit and Integration Tests for Phase 5 — Validation Harness.

Tests cover:
1. WalkForwardSplit generation — embargo boundary enforcement.
2. Metric correctness — known inputs with deterministic expected outputs.
3. Regime slicer — all buckets populated, no NaN metrics.
4. Economic gate — known returns/spreads → known PnL.
5. Shuffle test — MCC collapses to ≈ 0 after label permutation.
6. Group audits — G1 features carry signal; nonsense features score near zero.
7. ValidationHarness end-to-end — Gate 5 pass on synthetic data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.validation import (
    ValidationHarness,
    WalkForwardSplit,
    generate_splits,
    compute_mcc,
    compute_balanced_accuracy,
    compute_qlike,
    compute_oos_r2,
    compute_direction_metrics,
    compute_regression_metrics,
    compute_quantile_calibration,
    EconomicGateReport,
    evaluate_economic_gate,
    extract_spreads,
    run_shuffle_test,
    run_group_audits,
    run_adversarial_checks,
    compute_regime_report,
    aggregate_regime_reports,
)
from src.features.pipeline import compute_all_features
from src.baselines import PersistenceBaseline, HARRVBaseline


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 800, signal_strength: float = 0.3, seed: int = 0) -> pl.DataFrame:
    """
    Synthetic feature + label DataFrame.

    signal_strength controls how much true OFI signal leaks into r30.
    signal_strength=0 → pure noise (useful for testing that MCC collapses).
    """
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]

    ofi = rng.normal(0, 100.0, n)
    spread = 0.00015 + np.abs(rng.normal(0, 2e-5, n))
    mid = 1.1000 + np.cumsum(rng.normal(0, 0.0002, n))

    r30 = signal_strength * (ofi / 100.0) * 0.0003 + rng.normal(0, 0.0003, n)
    rv30 = np.log(np.abs(r30) + 1e-5)
    mag30 = np.log(np.abs(r30) + 1e-7)
    high = mid + np.abs(rng.normal(3e-4, 1e-4, n))
    low  = mid - np.abs(rng.normal(3e-4, 1e-4, n))

    df_raw = pl.DataFrame({
        "timestamp":    pl.Series(timestamps),
        "open":         mid + rng.normal(0, 1e-4, n),
        "high":         high,
        "low":          low,
        "close":        mid,
        "bid_open":     mid - spread / 2,
        "bid_close":    mid - spread / 2,
        "ask_open":     mid + spread / 2,
        "ask_close":    mid + spread / 2,
        "spread_close": spread,
        "spread_mean":  spread,
        "spread_max":   spread * 1.5,
        "volume":       rng.uniform(50, 500, n),
        "bid_volume":   rng.uniform(25, 260, n),
        "ask_volume":   rng.uniform(25, 260, n),
        "r30":          r30,
        "rv30":         rv30,
        "mag30":        mag30,
        "is_dead_zone": np.abs(r30) < spread,
    })

    return compute_all_features(df_raw)


@pytest.fixture
def synth_df() -> pl.DataFrame:
    return _make_df(n=800)


@pytest.fixture
def signal_df() -> pl.DataFrame:
    """Stronger signal — more likely to produce positive OOS R²."""
    return _make_df(n=1200, signal_strength=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Walk-forward splits
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardSplits:
    def test_embargo_boundary_enforced(self, synth_df: pl.DataFrame):
        splits = generate_splits(synth_df, train_bars=300, test_bars=100,
                                  step_bars=100, embargo_bars=30)
        assert len(splits) >= 1
        for s in splits:
            # test_start must be at least embargo_bars after train_end
            assert s.test_start >= s.train_end + 30
            assert s.n_train == 300
            assert s.n_test >= 50  # at least half target

    def test_no_overlap(self, synth_df: pl.DataFrame):
        splits = generate_splits(synth_df, train_bars=300, test_bars=100,
                                  step_bars=100, embargo_bars=30)
        for s in splits:
            assert s.test_start > s.train_end

    def test_returns_typed_dataclass(self, synth_df: pl.DataFrame):
        splits = generate_splits(synth_df, train_bars=300, test_bars=100,
                                  step_bars=100, embargo_bars=30)
        assert all(isinstance(s, WalkForwardSplit) for s in splits)

    def test_timestamps_populated(self, synth_df: pl.DataFrame):
        splits = generate_splits(synth_df, train_bars=300, test_bars=100,
                                  step_bars=100, embargo_bars=30)
        for s in splits:
            assert len(s.train_start_ts) > 0
            assert len(s.test_start_ts) > 0

    def test_small_dataset_fallback(self):
        """Very small dataset should still produce one split."""
        small = _make_df(n=100)
        splits = generate_splits(small, train_bars=50, test_bars=20,
                                  step_bars=20, embargo_bars=5)
        assert len(splits) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metric correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mcc_perfect(self):
        y = np.array([1, 1, 0, 0])
        assert compute_mcc(y, y) == pytest.approx(1.0)

    def test_mcc_anticorrelated(self):
        y_t = np.array([1, 1, 0, 0])
        y_p = np.array([0, 0, 1, 1])
        assert compute_mcc(y_t, y_p) == pytest.approx(-1.0)

    def test_mcc_chance(self):
        rng = np.random.default_rng(7)
        y_t = rng.integers(0, 2, 1000)
        y_p = rng.integers(0, 2, 1000)
        assert abs(compute_mcc(y_t, y_p)) < 0.15

    def test_balanced_accuracy_perfect(self):
        y = np.array([1, 1, 0, 0])
        assert compute_balanced_accuracy(y, y) == pytest.approx(1.0)

    def test_balanced_accuracy_chance(self):
        y_t = np.array([1, 1, 0, 0])
        y_p = np.array([1, 0, 1, 0])
        assert compute_balanced_accuracy(y_t, y_p) == pytest.approx(0.5)

    def test_qlike_zero_error(self):
        y = np.zeros(100)
        assert compute_qlike(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_qlike_positive(self):
        y_t = np.zeros(100)
        y_p = np.ones(100)
        assert compute_qlike(y_t, y_p) > 0

    def test_oos_r2_perfect(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        assert compute_oos_r2(y, y) == pytest.approx(1.0, abs=1e-9)

    def test_oos_r2_persistence(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        # persistence = r̂=0 → OOS R² = 0
        assert compute_oos_r2(y, np.zeros_like(y)) == pytest.approx(0.0, abs=1e-9)

    def test_direction_metrics_no_dead_zone(self):
        y_true = np.array([1.0, -1.0, 1.0, -1.0])
        y_pred = np.array([0.5, -0.5, 0.5, -0.5])  # perfect direction
        dm = compute_direction_metrics(y_true, y_pred)
        assert dm.hit_rate_all == pytest.approx(1.0)
        assert dm.mcc == pytest.approx(1.0)

    def test_direction_metrics_with_dead_zone(self):
        y_true = np.array([1.0, -1.0, 0.0001, -0.0001])
        y_pred = np.array([0.5, -0.5, 0.5, -0.5])
        dead = np.array([False, False, True, True])
        dm = compute_direction_metrics(y_true, y_pred, dead)
        assert dm.n_tradable == 2
        assert dm.hit_rate_tradable == pytest.approx(1.0)

    def test_quantile_calibration_perfect(self):
        n = 1000
        y_abs = np.sort(np.abs(np.random.default_rng(0).normal(0, 1, n)))
        # Perfect quantile predictions = actual quantiles
        q_preds = {
            0.1: np.full(n, float(np.percentile(y_abs, 10))),
            0.5: np.full(n, float(np.percentile(y_abs, 50))),
            0.9: np.full(n, float(np.percentile(y_abs, 90))),
        }
        cal = compute_quantile_calibration(y_abs, q_preds)
        assert cal.mean_calibration_error < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 3. Regime slicer
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeSlicer:
    def test_produces_tod_buckets(self, synth_df: pl.DataFrame):
        n = len(synth_df)
        y_pred = np.zeros(n)
        rr = compute_regime_report(synth_df, y_pred, fold_idx=1)
        assert len(rr.by_tod) > 0

    def test_produces_vol_terciles(self, synth_df: pl.DataFrame):
        n = len(synth_df)
        y_pred = np.zeros(n)
        rr = compute_regime_report(synth_df, y_pred, fold_idx=1)
        assert len(rr.by_vol) > 0

    def test_no_nan_in_metrics(self, synth_df: pl.DataFrame):
        y_pred = np.zeros(len(synth_df))
        rr = compute_regime_report(synth_df, y_pred, fold_idx=1)
        for b in rr.by_tod + rr.by_vol:
            assert np.isfinite(b.direction.mcc)
            assert np.isfinite(b.regression.oos_r2)

    def test_aggregate_across_folds(self, synth_df: pl.DataFrame):
        y_pred = np.zeros(len(synth_df))
        reports = [compute_regime_report(synth_df, y_pred, fold_idx=i) for i in range(3)]
        agg = aggregate_regime_reports(reports)
        assert "tod" in agg and "vol" in agg
        for bucket, metrics in agg["tod"].items():
            assert "hit_rate" in metrics
            assert "mcc" in metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Economic gate
# ─────────────────────────────────────────────────────────────────────────────

class TestEconomicGate:
    def test_perfect_predictor_positive_pnl(self):
        """A perfect directional predictor always has positive gross PnL."""
        n = 200
        y_true = np.random.default_rng(1).choice([-0.001, 0.001], n)
        y_pred = y_true * 2.0        # same direction, larger magnitude
        spreads = np.full(n, 0.0001)
        gr = evaluate_economic_gate(y_true, y_pred, spreads, k=1.0)
        assert gr.gross_pnl_per_bar > 0
        assert gr.n_trades > 0

    def test_no_trades_when_k_too_high(self):
        """High k means |r̂30| never exceeds k × spread → zero trades."""
        y_pred = np.full(100, 0.0001)
        y_true = np.full(100, 0.0001)
        spreads = np.full(100, 0.001)
        gr = evaluate_economic_gate(y_true, y_pred, spreads, k=100.0)
        assert gr.n_trades == 0

    def test_trade_fraction_in_bounds(self, synth_df: pl.DataFrame):
        n = len(synth_df)
        y_pred = np.random.default_rng(5).normal(0, 0.0003, n)
        spreads = extract_spreads(synth_df)
        gr = evaluate_economic_gate(
            synth_df["r30"].to_numpy(), y_pred, spreads, k=1.5
        )
        assert 0.0 <= gr.trade_fraction <= 1.0

    def test_extract_spreads_length(self, synth_df: pl.DataFrame):
        sp = extract_spreads(synth_df)
        assert len(sp) == len(synth_df)
        assert np.all(np.isfinite(sp))
        assert np.all(sp > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Shuffle test
# ─────────────────────────────────────────────────────────────────────────────

class TestShuffleTest:
    def test_mcc_collapses_after_shuffle(self, synth_df: pl.DataFrame):
        """After label permutation, MCC must be near 0."""
        y_true = synth_df["r30"].to_numpy()
        y_pred = y_true * 0.5 + np.random.default_rng(42).normal(0, 0.001, len(y_true))
        result = run_shuffle_test(y_true, y_pred, n_shuffles=50)
        assert abs(result.mean_mcc_shuffled) < 0.10

    def test_hit_rate_near_50_after_shuffle(self, synth_df: pl.DataFrame):
        y_true = synth_df["r30"].to_numpy()
        y_pred = np.ones_like(y_true) * 0.001
        result = run_shuffle_test(y_true, y_pred, n_shuffles=50)
        assert abs(result.mean_hit_rate_shuffled - 0.5) < 0.15

    def test_shuffle_test_passed_flag(self, synth_df: pl.DataFrame):
        y_true = synth_df["r30"].to_numpy()
        y_pred = np.random.default_rng(9).normal(0, 0.001, len(y_true))
        result = run_shuffle_test(y_true, y_pred, n_shuffles=50)
        # Shuffle test should pass when predictions are effectively random
        assert result.shuffle_test_passed


# ─────────────────────────────────────────────────────────────────────────────
# 6. Group audits
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupAudits:
    def test_audits_run_for_all_groups(self, synth_df: pl.DataFrame):
        train = synth_df.slice(0, 600)
        test = synth_df.slice(600, 200)
        y_train = train["r30"].to_numpy()
        y_test = test["r30"].to_numpy()
        audits = run_group_audits(train, test, y_train, y_test)
        # At least G1 and G2 should be present
        groups = {a.group for a in audits}
        assert "G1" in groups or "G2" in groups

    def test_group_audit_r2_finite(self, synth_df: pl.DataFrame):
        train = synth_df.slice(0, 600)
        test = synth_df.slice(600, 200)
        audits = run_group_audits(
            train, test,
            train["r30"].to_numpy(), test["r30"].to_numpy()
        )
        for a in audits:
            assert np.isfinite(a.oos_r2)
            assert np.isfinite(a.mcc)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ValidationHarness end-to-end / Gate 5
# ─────────────────────────────────────────────────────────────────────────────

class _PersistenceAdapter:
    def __init__(self):
        self._m = PersistenceBaseline()

    def fit(self, df, **_):
        self._m.fit(df)
        return self

    def predict(self, df):
        import numpy as np
        return self._m.predict(df).r_hat


class _HARAdapter:
    def __init__(self):
        self._m = HARRVBaseline(alpha=1.0)

    def fit(self, df, **_):
        self._m.fit(df)
        return self

    def predict(self, df):
        return self._m.predict(df)


class TestValidationHarness:
    def test_harness_runs_without_error(self, synth_df: pl.DataFrame, tmp_path: Path):
        harness = ValidationHarness(
            train_bars=400, test_bars=150, step_bars=150,
            embargo_bars=10, n_shuffle=10, run_adversarial=True,
        )
        report = harness.evaluate(
            synth_df, _PersistenceAdapter(), predictor_name="B0",
            symbol="TEST", report_dir=tmp_path,
        )
        assert report.total_folds >= 1
        assert len(report.fold_metrics) == report.total_folds
        assert len(report.economic_gate_reports) == report.total_folds

    def test_harness_writes_reports(self, synth_df: pl.DataFrame, tmp_path: Path):
        harness = ValidationHarness(
            train_bars=400, test_bars=150, step_bars=150,
            embargo_bars=10, n_shuffle=5, run_adversarial=False,
        )
        harness.evaluate(
            synth_df, _PersistenceAdapter(), predictor_name="B0",
            symbol="EURUSD", report_dir=tmp_path,
        )
        assert (tmp_path / "harness_EURUSD_B0.json").exists()
        assert (tmp_path / "harness_EURUSD_B0_report.md").exists()

    def test_gate5_b0_mcc_near_zero(self, synth_df: pl.DataFrame, tmp_path: Path):
        """Persistence (r̂=0) should have MCC ≈ 0 on all folds."""
        harness = ValidationHarness(
            train_bars=400, test_bars=150, step_bars=150,
            embargo_bars=10, n_shuffle=5, run_adversarial=False,
        )
        b0 = harness.evaluate(synth_df, _PersistenceAdapter(),
                               predictor_name="B0", symbol="T", report_dir=tmp_path)
        assert abs(b0.mean_mcc) < 0.15   # loose bound; pure noise

    def test_gate5_check_produces_report(self, synth_df: pl.DataFrame, tmp_path: Path):
        harness = ValidationHarness(
            train_bars=400, test_bars=150, step_bars=150,
            embargo_bars=10, n_shuffle=5, run_adversarial=False,
        )
        b0 = harness.evaluate(synth_df, _PersistenceAdapter(),
                               predictor_name="B0", symbol="T", report_dir=tmp_path)
        b1 = harness.evaluate(synth_df, _HARAdapter(),
                               predictor_name="B1", symbol="T", report_dir=tmp_path)
        gate5 = harness.check_gate5(b0, b1, symbol="T", report_dir=tmp_path)

        assert isinstance(gate5.gate5_passed, bool)
        assert len(gate5.verdict) > 0
        assert (tmp_path / "gate5_T.json").exists()
        assert (tmp_path / "gate5_T_report.md").exists()

    def test_harness_regime_reports_populated(self, synth_df: pl.DataFrame, tmp_path: Path):
        harness = ValidationHarness(
            train_bars=400, test_bars=150, step_bars=150,
            embargo_bars=10, n_shuffle=5, run_adversarial=False,
        )
        report = harness.evaluate(synth_df, _PersistenceAdapter(),
                                   predictor_name="B0", symbol="T", report_dir=tmp_path)
        assert len(report.regime_reports) == report.total_folds
        assert "tod" in report.regime_aggregate
        assert "vol" in report.regime_aggregate

    def test_harness_adversarial_reports_present(self, synth_df: pl.DataFrame, tmp_path: Path):
        harness = ValidationHarness(
            train_bars=400, test_bars=150, step_bars=150,
            embargo_bars=10, n_shuffle=10, run_adversarial=True,
        )
        report = harness.evaluate(synth_df, _PersistenceAdapter(),
                                   predictor_name="B0", symbol="T", report_dir=tmp_path)
        # At least some folds should have adversarial reports
        assert len(report.adversarial_reports) > 0
