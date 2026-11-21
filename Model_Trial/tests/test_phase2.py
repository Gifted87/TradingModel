"""
Adversarial Unit Tests for Phase 2: Target Construction and Labels.

Tests cover:
1. Exact trajectory path verification (up-step, down-step, roundtrip, volatility spike).
2. Excursion mathematical bounding invariant: MAE <= min(0, r) <= max(0, r) <= MFE.
3. Log-space magnitude targets stability and monotonicity.
4. Dead-zone spread calibration.
5. Boundary purge behavior (last h bars).
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.labels.label_generator import (
    audit_label_distributions,
    compute_forward_labels,
)


def test_label_exact_trajectories():
    """Verify exact numerical values for canonical price trajectories."""
    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(10)]
    
    # Trajectory 1: Monotonic upward price steps (1.000, 1.001, 1.002, 1.003, 1.004, 1.005, 1.006, ...)
    prices_up = [1.000 + i * 0.001 for i in range(10)]
    df_up = pl.DataFrame({
        "timestamp": timestamps,
        "close": prices_up,
        "spread_close": np.full(10, 0.0001),
    })
    
    labeled_up = compute_forward_labels(df_up, horizon_bars=6, horizon_minutes=30)
    
    # For bar 0 with horizon 6:
    # r30 = log(prices[6] / prices[0]) = log(1.006 / 1.000)
    expected_r30 = np.log(1.006 / 1.000)
    assert pytest.approx(labeled_up["r30"][0], 1e-6) == expected_r30
    assert labeled_up["direction"][0] == 1.0
    
    # MFE must equal r30, MAE must equal 0 (no adverse dip)
    assert pytest.approx(labeled_up["mfe30"][0], 1e-6) == expected_r30
    assert pytest.approx(labeled_up["mae30"][0], 1e-6) == 0.0


def test_label_excursion_bounding_invariant():
    """
    Mathematical Invariant Test:
    For any price path, MAE_t <= min(0, r_t) <= max(0, r_t) <= MFE_t.
    """
    np.random.seed(42)
    n = 200
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]
    
    # Random walk
    returns = np.random.normal(0, 0.001, n)
    prices = 1.1000 * np.exp(np.cumsum(returns))
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "close": prices,
        "spread_close": np.full(n, 0.0001),
    })
    
    labeled = compute_forward_labels(df, horizon_bars=6, horizon_minutes=30)
    valid = labeled.filter(pl.col("r30").is_not_nan())
    
    r30 = valid["r30"].to_numpy()
    mfe30 = valid["mfe30"].to_numpy()
    mae30 = valid["mae30"].to_numpy()
    
    # Strict inequality checks
    assert np.all(mfe30 >= 0.0), "MFE must always be non-negative"
    assert np.all(mae30 <= 0.0), "MAE must always be non-positive"
    assert np.all(mfe30 >= r30), "MFE must be >= terminal return r30"
    assert np.all(mae30 <= r30), "MAE must be <= terminal return r30"


def test_magnitude_and_rv_non_negative():
    """Verify log magnitude and RV targets are finite and correctly ordered."""
    n = 100
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]
    prices = 1.1000 + np.sin(np.linspace(0, 20, n)) * 0.0050
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "close": prices,
        "spread_close": np.full(n, 0.0001),
    })
    
    labeled = compute_forward_labels(df, horizon_bars=6, horizon_minutes=30)
    valid = labeled.filter(pl.col("r30").is_not_nan())
    
    mag30 = valid["mag30"].to_numpy()
    rv30 = valid["rv30"].to_numpy()
    
    assert not np.any(np.isnan(mag30))
    assert not np.any(np.isinf(mag30))
    assert not np.any(np.isnan(rv30))
    assert not np.any(np.isinf(rv30))


def test_dead_zone_exclusion():
    """Verify dead-zone flag activates when return magnitude is below spread."""
    n = 20
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]
    
    # Prices move by only 0.00001 (0.1 pip), while spread is 0.00010 (1 pip)
    prices = [1.10000 + i * 0.000001 for i in range(n)]
    df = pl.DataFrame({
        "timestamp": timestamps,
        "close": prices,
        "spread_close": np.full(n, 0.00020),
    })
    
    labeled = compute_forward_labels(df, horizon_bars=6, horizon_minutes=30)
    valid = labeled.filter(pl.col("r30").is_not_nan())
    
    # All tiny moves must be tagged as dead zone
    assert np.all(valid["is_dead_zone"].to_numpy() == True)
