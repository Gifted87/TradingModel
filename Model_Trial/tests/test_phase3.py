"""
Unit and Integration Tests for Phase 3 Feature Store (G1–G6).

Tests cover:
1. Microstructure features (G1): OFI rolling sums, spread level/delta, OFI/depth ratio.
2. Multi-scale momentum features (G2): Log returns across horizons, momentum acceleration.
3. Continuous indicators (G3): CCI, Stoch %K/%D, Bollinger %B/bandwidth, RSI, TRIX, MACD.
4. Time & Regime features (G5): Cyclical ToD encoding, session flags, cross-term modulations.
5. Interaction & Tail features (G4): Cross-products, absolute value transforms, distances from center.
6. Volatility features (G6): HAR log-RV, Parkinson, Garman-Klass, relative ATR.
7. Feature Registry completeness: Tracking all groups, citations, formulas.
8. FeaturePipeline: Train-fold fit, fold-safe transform, serialization (pickle), clipping.
9. Gate 3 Audit: Validation of NaN-leakage and zero-variance detection.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.features import (
    REGISTRY,
    FeaturePipeline,
    compute_all_features,
    compute_g1_microstructure,
    compute_g2_momentum,
    compute_g3_indicators,
    compute_g4_interactions,
    compute_g5_time_regime,
    compute_g6_volatility,
    run_gate3_audit,
)


@pytest.fixture
def sample_bars_df() -> pl.DataFrame:
    """Generate 500 bars of realistic 5m OHLCV synthetic data for testing."""
    n = 500
    base = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]

    # Generate synthetic price series with trend, oscillations, and noise
    t = np.linspace(0, 50, n)
    mid_price = 1.1000 + 0.0050 * np.sin(t) + 0.0020 * np.cos(t * 3) + np.cumsum(np.random.normal(0, 0.0002, n))
    spread = 0.00015 + np.abs(np.random.normal(0, 0.00003, n))

    high = mid_price + np.abs(np.random.normal(0.0003, 0.0001, n))
    low = mid_price - np.abs(np.random.normal(0.0003, 0.0001, n))
    open_p = mid_price + np.random.normal(0, 0.0001, n)
    close_p = mid_price

    volume = np.random.uniform(50.0, 500.0, n)
    bid_volume = volume * 0.52
    ask_volume = volume * 0.48

    df = pl.DataFrame({
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
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
    })
    return df


def test_individual_feature_groups(sample_bars_df: pl.DataFrame):
    """Test that all individual feature groups compute without error and add expected columns."""
    df = sample_bars_df

    # G1
    df_g1 = compute_g1_microstructure(df)
    assert "g1_spread_level" in df_g1.columns
    assert "g1_ofi_1bar" in df_g1.columns
    assert "g1_ofi_2bar_sum" in df_g1.columns
    assert "g1_ofi_depth_ratio_3bar" in df_g1.columns

    # G6
    df_g6 = compute_g6_volatility(df)
    assert "g6_log_rv_5m" in df_g6.columns
    assert "g6_log_rv_1d" in df_g6.columns
    assert "g6_log_parkinson_1bar" in df_g6.columns
    assert "g6_log_garman_klass_1bar" in df_g6.columns
    assert "g6_log_atr14" in df_g6.columns
    assert "g6_rel_atr14" in df_g6.columns

    # G2
    df_g2 = compute_g2_momentum(df)
    assert "g2_return_5m" in df_g2.columns
    assert "g2_return_24h" in df_g2.columns
    assert "g2_momentum_accel_5m_30m" in df_g2.columns

    # G3
    df_g3 = compute_g3_indicators(df)
    assert "g3_cci_14" in df_g3.columns
    assert "g3_stoch_k_14" in df_g3.columns
    assert "g3_stoch_d_14" in df_g3.columns
    assert "g3_bollinger_pct_b_20" in df_g3.columns
    assert "g3_bollinger_bandwidth_20" in df_g3.columns
    assert "g3_rsi_14" in df_g3.columns
    assert "g3_trix_14" in df_g3.columns
    assert "g3_macd_12_26_line" in df_g3.columns
    assert "g3_macd_12_26_hist" in df_g3.columns


def test_compute_all_features_dependency_chain(sample_bars_df: pl.DataFrame):
    """Test full dependency pipeline: G1 -> G6 -> G2 -> G3 -> G5 -> G4."""
    df_full = compute_all_features(sample_bars_df)

    # Check G5 features
    assert "g5_tod_sin" in df_full.columns
    assert "g5_tod_cos" in df_full.columns
    assert "g5_session_london" in df_full.columns
    assert "g5_ofi_x_tod_sin" in df_full.columns
    assert "g5_atr_x_london" in df_full.columns

    # Check G4 features (cross-products and tail transforms)
    assert "g4_rsi14_x_bb_pct_b" in df_full.columns
    assert "g4_ofi_x_spread" in df_full.columns
    assert "g4_cci14_x_atr14" in df_full.columns
    assert "g4_abs_ofi_1bar" in df_full.columns
    assert "g4_rsi14_dist_from_50" in df_full.columns
    assert "g4_ofi_sq" in df_full.columns

    # Verify no unexpected NaNs past warm-up in all feature columns
    feat_cols = [c for c in df_full.columns if c.startswith("g") and "_" in c]
    assert len(feat_cols) >= 40

    warmup_df = df_full.slice(300, len(df_full) - 300)
    for col in feat_cols:
        nan_count = warmup_df[col].is_nan().sum()
        null_count = warmup_df[col].is_null().sum()
        assert nan_count == 0, f"Column {col} contains NaNs past warmup"
        assert null_count == 0, f"Column {col} contains Nulls past warmup"


def test_feature_registry_integrity():
    """Verify Feature Registry tracks all groups and metadata properly."""
    summary = REGISTRY.summary()
    assert summary["G1"] >= 5
    assert summary["G2"] >= 5
    assert summary["G3"] >= 10
    assert summary["G4"] >= 10
    assert summary["G5"] >= 5
    assert summary["G6"] >= 5

    for meta in REGISTRY.list_features():
        assert meta.name.startswith("g")
        assert meta.group in ["G1", "G2", "G3", "G4", "G5", "G6"]
        assert meta.citation.startswith("[")
        assert len(meta.description) > 0
        assert len(meta.formula) > 0


def test_feature_pipeline_fit_transform_and_serialize(sample_bars_df: pl.DataFrame, tmp_path: Path):
    """Test stateful FeaturePipeline normalization and serialization."""
    df_full = compute_all_features(sample_bars_df)

    # Train / Val split (first 350 bars train, remaining 150 bars val)
    train_df = df_full.slice(0, 350)
    val_df = df_full.slice(350, 150)

    pipe = FeaturePipeline(clip_sigma=5.0)
    train_norm = pipe.fit_transform(train_df)

    # Verify mean is close to 0 and std is close to 1 on non-constant features
    feat_col = "g1_ofi_1bar"
    mu = float(train_norm[feat_col].mean())
    std = float(train_norm[feat_col].std())
    assert abs(mu) < 1e-4
    assert pytest.approx(std, 0.05) == 1.0

    # Transform val set using frozen train stats
    val_norm = pipe.transform(val_df)
    assert len(val_norm) == 150

    # Test serialization / deserialization
    save_path = tmp_path / "pipeline.pkl"
    pipe.save(save_path)
    loaded_pipe = FeaturePipeline.load(save_path)

    val_norm_loaded = loaded_pipe.transform(val_df)
    assert np.allclose(val_norm[feat_col].to_numpy(), val_norm_loaded[feat_col].to_numpy())


def test_gate3_audit_pass(sample_bars_df: pl.DataFrame, tmp_path: Path):
    """Test Gate 3 validator passes on valid dataset."""
    df_full = compute_all_features(sample_bars_df)
    report = run_gate3_audit(df_full, report_dir=tmp_path, symbol="EURUSD_TEST")

    assert report.gate3_passed is True
    assert len(report.nan_leaking_features) == 0
    assert len(report.zero_variance_features) == 0
    assert report.total_features >= 40
    assert (tmp_path / "gate3_EURUSD_TEST.json").exists()
