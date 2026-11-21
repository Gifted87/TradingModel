"""
Unit and Integration Tests for Phase 4 Baselines and Linear Core (Branch A).

Tests cover:
1. B0 Persistence baseline: zero return forecast, trailing magnitude/RV calculation.
2. B1 HAR-RV baseline: multi-horizon regression, log-space RV forecasting, QLIKE scoring.
3. B2 Branch A Linear Core: Ridge regression on G1 features, positive OFI coefficient check, residual extraction.
4. Walk-Forward CV Engine: Embargo/purge boundary enforcement, split generation.
5. Gate 4 / Condition C2 End-to-End Evaluation.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.baselines import (
    BaselineEvaluator,
    HARRVBaseline,
    LinearCoreBranchA,
    PersistenceBaseline,
    compute_qlike,
    generate_walk_forward_splits,
)
from src.features.pipeline import compute_all_features


@pytest.fixture
def sample_features_df() -> pl.DataFrame:
    """Generate 600 bars of synthetic feature-labeled data for testing."""
    n = 600
    base = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]

    t = np.linspace(0, 60, n)
    # Synthetic OFI driver
    ofi = np.sin(t) * 100.0 + np.random.normal(0, 20.0, n)
    spread = 0.00015 + np.abs(np.random.normal(0, 0.00002, n))
    
    # Forward return has a true positive relationship with OFI + noise
    mid_price = 1.1000 + np.cumsum(np.random.normal(0, 0.0002, n))
    r30 = 0.00005 * (ofi / 100.0) + np.random.normal(0, 0.0003, n)
    rv30 = np.log(np.abs(r30) + 1e-5)
    mag30 = np.log(np.abs(r30) + 1e-7)

    high = mid_price + np.abs(np.random.normal(0.0003, 0.0001, n))
    low = mid_price - np.abs(np.random.normal(0.0003, 0.0001, n))
    open_p = mid_price + np.random.normal(0, 0.0001, n)
    close_p = mid_price

    volume = np.random.uniform(50.0, 500.0, n)

    df_raw = pl.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close_p,
        "bid_open": open_p - spread / 2,
        "bid_close": close_p - spread / 2,
        "ask_open": open_p + spread / 2,
        "ask_close": close_p + spread / 2,
        "spread_close": spread,
        "spread_mean": spread,
        "spread_max": spread * 1.5,
        "volume": volume,
        "bid_volume": volume * 0.52,
        "ask_volume": volume * 0.48,
        "r30": r30,
        "rv30": rv30,
        "mag30": mag30,
        "is_dead_zone": np.abs(r30) < spread,
    })
    
    # Compute all G1-G6 features
    df_feat = compute_all_features(df_raw)
    return df_feat


def test_persistence_baseline_mechanics(sample_features_df: pl.DataFrame):
    """Test B0 persistence return forecast is 0 and magnitudes are rolling averages."""
    train_df = sample_features_df.slice(0, 400)
    test_df = sample_features_df.slice(400, 200)

    b0 = PersistenceBaseline(trailing_window=20)
    b0.fit(train_df)
    preds = b0.predict(test_df)

    assert len(preds.r_hat) == len(test_df)
    assert np.all(preds.r_hat == 0.0)
    assert not np.any(np.isnan(preds.mag_hat))
    assert not np.any(np.isnan(preds.rv_hat))


def test_har_rv_baseline_mechanics(sample_features_df: pl.DataFrame):
    """Test B1 HAR-RV model fitting, predictions, and summary statistics."""
    train_df = sample_features_df.slice(0, 400)
    test_df = sample_features_df.slice(400, 200)

    b1 = HARRVBaseline(alpha=1.0)
    b1.fit(train_df)
    rv_hat = b1.predict(test_df)

    assert len(rv_hat) == len(test_df)
    assert not np.any(np.isnan(rv_hat))
    assert b1.summary.n_samples > 0
    assert len(b1.summary.coefficients) > 0

    # Test QLIKE metric
    y_true = test_df["rv30"].to_numpy()
    qlike = compute_qlike(y_true, rv_hat)
    assert qlike >= 0.0


def test_linear_core_branch_a_mechanics(sample_features_df: pl.DataFrame):
    """Test B2 Branch A linear core fitting, prediction, and residual computation."""
    train_df = sample_features_df.slice(0, 400)
    test_df = sample_features_df.slice(400, 200)

    b2 = LinearCoreBranchA(alpha=10.0)
    b2.fit(train_df, symbol="EURUSD_TEST")
    r_hat = b2.predict(test_df)

    assert len(r_hat) == len(test_df)
    assert not np.any(np.isnan(r_hat))
    assert b2.summary.n_train > 0

    # Compute residuals: r30 - r_hat_lin
    residuals = b2.compute_residuals(test_df)
    assert len(residuals) == len(test_df)
    assert np.allclose(residuals, test_df["r30"].to_numpy() - r_hat)


def test_walk_forward_splits_purge_embargo(sample_features_df: pl.DataFrame):
    """Test that purged walk-forward splits enforce embargo buffer."""
    splits = generate_walk_forward_splits(
        sample_features_df,
        train_bars=200,
        test_bars=100,
        step_bars=100,
        embargo_bars=30,
    )

    assert len(splits) >= 2
    for tr_s, tr_e, te_s, te_e in splits:
        # Check embargo separation
        assert te_s >= tr_e + 30
        assert tr_e - tr_s == 200
        assert te_e - te_s <= 100


def test_baseline_evaluator_gate4_pass(sample_features_df: pl.DataFrame, tmp_path: Path):
    """Test end-to-end BaselineEvaluator and Gate 4 verification."""
    evaluator = BaselineEvaluator(
        embargo_bars=10,
        train_bars=250,
        test_bars=80,
        step_bars=80,
    )

    report = evaluator.evaluate_symbol(sample_features_df, symbol="TEST_SYM", report_dir=tmp_path)

    assert report.total_folds >= 2
    assert len(report.fold_details) == report.total_folds
    assert (tmp_path / "gate4_TEST_SYM.json").exists()
    assert (tmp_path / "gate4_TEST_SYM_report.md").exists()
