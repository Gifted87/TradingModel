"""
Unit and Integration Tests for Phase 7 — Ablation Battery & Universality Testing.

Tests cover:
1. AblationConfig & Suite definitions.
2. VariantNetwork independent heads forward pass.
3. Feature filtering for No-G4 and No-G5 configurations.
4. Raw magnitude vs log-magnitude target fitting.
5. Lookback sequence length variations (L=30, L=60).
6. AblationBatteryRunner execution and Gate 7 evaluation.
7. Cross-instrument transfer test [B2].
8. Surviving configuration profile freezing to JSON.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from src.features.pipeline import compute_all_features
from src.ablations import (
    ABLATION_SUITE,
    AblationBatteryRunner,
    AblationConfig,
    AblationModelVariant,
    BatterySummaryReport,
    TransferTestResult,
    VariantNetwork,
    VariantSession,
    export_surviving_config,
    render_ablation_markdown_report,
    run_cross_instrument_transfer_test,
)


@pytest.fixture
def sample_dataset_df() -> pl.DataFrame:
    """Generate 500 bars of synthetic feature-labeled data for testing."""
    n = 500
    rng = np.random.default_rng(42)
    base = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]

    t = np.linspace(0, 40, n)
    ofi = np.sin(t) * 100.0 + rng.normal(0, 20.0, n)
    spread = 0.00015 + np.abs(rng.normal(0, 0.00002, n))
    mid_price = 1.1000 + np.cumsum(rng.normal(0, 0.0002, n))

    r_lin = 0.00005 * (ofi / 100.0)
    residual = 0.00003 * np.sin(t * 2) + rng.normal(0, 0.0001, n)
    r30 = r_lin + residual
    rv30 = np.log(np.abs(r30) + 1e-5)
    mag30 = np.log(np.abs(r30) + 1e-7)

    high = mid_price + np.abs(rng.normal(0.0003, 0.0001, n))
    low = mid_price - np.abs(rng.normal(0.0003, 0.0001, n))

    df_raw = pl.DataFrame({
        "timestamp": pl.Series(timestamps),
        "open": mid_price + rng.normal(0, 0.0001, n),
        "high": high,
        "low": low,
        "close": mid_price,
        "bid_open": mid_price - spread / 2,
        "bid_close": mid_price - spread / 2,
        "ask_open": mid_price + spread / 2,
        "ask_close": mid_price + spread / 2,
        "spread_close": spread,
        "spread_mean": spread,
        "spread_max": spread * 1.5,
        "volume": rng.uniform(50.0, 500.0, n),
        "bid_volume": rng.uniform(25.0, 260.0, n),
        "ask_volume": rng.uniform(25.0, 260.0, n),
        "r30": r30,
        "rv30": rv30,
        "mag30": mag30,
        "is_dead_zone": np.abs(r30) < spread,
    })

    return compute_all_features(df_raw)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config Suite Definitions
# ─────────────────────────────────────────────────────────────────────────────

def test_ablation_suite_definitions():
    assert "BASELINE_FULL" in ABLATION_SUITE
    assert "NO_G4_INTERACTIONS" in ABLATION_SUITE
    assert "NO_G5_TIME_REGIME" in ABLATION_SUITE
    assert "INDEPENDENT_HEADS" in ABLATION_SUITE
    assert "RAW_MAGNITUDE" in ABLATION_SUITE
    assert "LOOKBACK_30" in ABLATION_SUITE


# ─────────────────────────────────────────────────────────────────────────────
# 2. VariantNetwork Independent Heads Mode
# ─────────────────────────────────────────────────────────────────────────────

def test_variant_network_independent_heads():
    b, l, f_in, g6_dim = 4, 30, 20, 6
    x = torch.randn(b, l, f_in)
    r_hat_lin = torch.randn(b, 1)
    g6 = torch.randn(b, g6_dim)

    # Independent heads mode
    net = VariantNetwork(
        in_features=f_in,
        g6_dim=g6_dim,
        use_cascade=False,
        conv_channels=8,
        lstm_hidden=8,
    )
    outs = net(x, r_hat_lin, g6)
    assert outs.residual_hat.shape == (b, 1)
    assert outs.r30_hat.shape == (b, 1)
    assert outs.mag30_hat.shape == (b, 1)
    assert outs.rv30_hat.shape == (b, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Filtering in Model Variants
# ─────────────────────────────────────────────────────────────────────────────

def test_ablation_feature_filtering(sample_dataset_df: pl.DataFrame):
    feature_cols = [c for c in sample_dataset_df.columns if c.startswith("g") and "_" in c]

    no_g4_cfg = ABLATION_SUITE["NO_G4_INTERACTIONS"]
    session_no_g4 = VariantSession(config=no_g4_cfg)
    cols_no_g4 = session_no_g4._select_columns(feature_cols)
    assert not any(c.startswith("g4_") for c in cols_no_g4)

    no_g5_cfg = ABLATION_SUITE["NO_G5_TIME_REGIME"]
    session_no_g5 = VariantSession(config=no_g5_cfg)
    cols_no_g5 = session_no_g5._select_columns(feature_cols)
    assert not any(c.startswith("g5_") for c in cols_no_g5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Variant Fit and Predict Execution
# ─────────────────────────────────────────────────────────────────────────────

def test_ablation_model_variants_fit_predict(sample_dataset_df: pl.DataFrame):
    train_df = sample_dataset_df.slice(0, 350)
    test_df = sample_dataset_df.slice(350, 150)

    # Test RAW_MAGNITUDE variant
    raw_cfg = ABLATION_SUITE["RAW_MAGNITUDE"]
    model_raw = AblationModelVariant(config=raw_cfg)
    model_raw.trainer_config.epochs = 2
    model_raw.trainer_config.batch_size = 64
    model_raw.fit(train_df, symbol="TEST_RAW")

    r_hat, rv_hat = model_raw.predict(test_df)
    assert len(r_hat) == len(test_df)
    assert len(rv_hat) == len(test_df)
    assert not np.any(np.isnan(r_hat[60:]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ablation Battery Runner
# ─────────────────────────────────────────────────────────────────────────────

def test_ablation_battery_runner_execution(sample_dataset_df: pl.DataFrame, tmp_path: Path):
    # Run a subset of 3 variants for fast unit testing
    runner = AblationBatteryRunner(
        train_bars=250,
        test_bars=80,
        step_bars=80,
        embargo_bars=10,
        epochs=2,
        batch_size=64,
        variants_to_run=["BASELINE_FULL", "NO_G4_INTERACTIONS", "INDEPENDENT_HEADS"],
    )

    report = runner.run_battery(sample_dataset_df, symbol="TEST_SYM")
    assert len(report.variants) == 3
    assert report.frozen_config_name == "BASELINE_FULL"

    # Export surviving config
    out_cfg_path = tmp_path / "surviving_config.json"
    export_surviving_config(report, out_cfg_path)
    assert out_cfg_path.exists()

    md = render_ablation_markdown_report(report)
    assert "Gate 7" in md


# ─────────────────────────────────────────────────────────────────────────────
# 6. Universality Transfer Test [B2]
# ─────────────────────────────────────────────────────────────────────────────

def test_universality_transfer_test(sample_dataset_df: pl.DataFrame):
    # Split synthetic df into source and target
    source_df = sample_dataset_df.slice(0, 300)
    target_df = sample_dataset_df.slice(300, 200)

    result: TransferTestResult = run_cross_instrument_transfer_test(
        source_dfs={"SRC_EURUSD": source_df},
        target_symbol="TGT_XAUUSD",
        target_df=target_df,
        lookback=30,
        epochs=2,
        batch_size=64,
    )

    assert result.target_symbol == "TGT_XAUUSD"
    assert result.n_train_samples > 0
    assert result.n_test_samples > 0
    assert isinstance(result.transfer_successful, bool)
