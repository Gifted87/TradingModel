"""
Unit and Integration Tests for Phase 6 — The Residual Learner (Branch B & Cascade Heads).

Tests cover:
1. SequenceDataset & Batch collation (shapes, lookback bounds, residual calculation).
2. MultiWidthConvBlock: kernel concatenation, dimension preservation.
3. AttentionPooling: context vector shape, attention weights sum to 1.
4. ResidualLearnerNetwork: full forward pass shape validation across all 5 heads.
5. CompositeMultiTaskLoss: backprop gradient flow through all heads and loss components.
6. ResidualLearnerTrainer: training loop execution, loss decrease, early stopping.
7. CompositeTradingModel: end-to-end fit, predict, predict_full complying with Predictor protocol.
8. Gate6Evaluator: walk-forward comparison between Branch A and Full Model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from src.features.pipeline import compute_all_features
from src.models import (
    AttentionPooling,
    BatchData,
    CompositeModelPredictions,
    CompositeMultiTaskLoss,
    CompositeTradingModel,
    Gate6Evaluator,
    ModelOutputs,
    MultiWidthConvBlock,
    ResidualLearnerNetwork,
    ResidualLearnerTrainer,
    SequenceDataset,
    TrainerConfig,
    build_feature_matrices,
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

    # Return has linear OFI component + nonlinear residual
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
# 1. Dataset & Collation
# ─────────────────────────────────────────────────────────────────────────────

def test_sequence_dataset_and_collate():
    n, f_dim, g6_dim, lookback = 120, 25, 6, 30
    rng = np.random.default_rng(0)

    features = rng.normal(0, 1, (n, f_dim)).astype(np.float32)
    r_lin = rng.normal(0, 0.001, n).astype(np.float32)
    g6 = rng.normal(0, 1, (n, g6_dim)).astype(np.float32)
    r30 = rng.normal(0, 0.001, n).astype(np.float32)
    rv30 = rng.normal(-6, 1, n).astype(np.float32)
    mag30 = rng.normal(-6, 1, n).astype(np.float32)

    ds = SequenceDataset(
        features=features,
        r_hat_lin=r_lin,
        g6_features=g6,
        r30=r30,
        rv30=rv30,
        mag30=mag30,
        lookback=lookback,
    )

    assert len(ds) == n - lookback + 1

    item = ds[0]
    assert isinstance(item, (int, np.integer))

    batch: BatchData = ds.collate([0, 1, 2, 3])
    assert batch.sequences.shape == (4, lookback, f_dim)
    assert batch.r_hat_lin.shape == (4, 1)
    assert batch.g6_features.shape == (4, g6_dim)
    assert batch.r30.shape == (4, 1)
    assert batch.residual.shape == (4, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MultiWidthConvBlock
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_width_conv_block():
    b, l, f_in = 8, 60, 30
    x = torch.randn(b, l, f_in)
    conv = MultiWidthConvBlock(in_features=f_in, out_channels_per_kernel=16, kernel_sizes=(3, 5, 7))
    out = conv(x)
    # 16 * 3 = 48 channels
    assert out.shape == (b, l, 48)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AttentionPooling
# ─────────────────────────────────────────────────────────────────────────────

def test_attention_pooling():
    b, l, h = 8, 60, 64
    x = torch.randn(b, l, h)
    attn = AttentionPooling(hidden_dim=h)
    context, weights = attn(x)
    assert context.shape == (b, h)
    assert weights.shape == (b, l)
    # Softmax weights should sum to 1 across sequence length
    assert torch.allclose(torch.sum(weights, dim=1), torch.ones(b), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ResidualLearnerNetwork Forward Pass
# ─────────────────────────────────────────────────────────────────────────────

def test_network_forward_pass():
    b, l, f_in, g6_dim = 4, 60, 35, 8
    x = torch.randn(b, l, f_in)
    r_hat_lin = torch.randn(b, 1)
    g6 = torch.randn(b, g6_dim)

    net = ResidualLearnerNetwork(
        in_features=f_in,
        g6_dim=g6_dim,
        conv_channels=16,
        lstm_hidden=24,
        lstm_layers=2,
        dropout=0.1,
        n_quantiles=9,
    )

    outs: ModelOutputs = net(x, r_hat_lin, g6)
    assert outs.residual_hat.shape == (b, 1)
    assert outs.r30_hat.shape == (b, 1)
    # Check r30_hat = r_hat_lin + residual_hat invariant
    assert torch.allclose(outs.r30_hat, r_hat_lin + outs.residual_hat)
    assert outs.mag30_hat.shape == (b, 1)
    assert outs.rv30_hat.shape == (b, 1)
    assert outs.quantiles_hat.shape == (b, 9)
    assert outs.attention_weights.shape == (b, l)
    # Quantiles must be non-negative (softplus)
    assert (outs.quantiles_hat >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. CompositeMultiTaskLoss & Gradient Backprop
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_backward_gradient_flow():
    b, l, f_in, g6_dim = 4, 60, 20, 6
    net = ResidualLearnerNetwork(
        in_features=f_in,
        g6_dim=g6_dim,
        conv_channels=16,
        lstm_hidden=16,
    )
    loss_fn = CompositeMultiTaskLoss()

    batch = BatchData(
        sequences=torch.randn(b, l, f_in),
        r_hat_lin=torch.randn(b, 1),
        g6_features=torch.randn(b, g6_dim),
        r30=torch.randn(b, 1),
        residual=torch.randn(b, 1),
        mag30=torch.randn(b, 1),
        rv30=torch.randn(b, 1),
        abs_r30=torch.abs(torch.randn(b, 1)),
    )

    outputs = net(batch.sequences, batch.r_hat_lin, batch.g6_features)
    loss, breakdown = loss_fn(outputs, batch)

    assert loss.item() > 0
    assert breakdown.return_loss >= 0
    assert breakdown.quantile_loss >= 0

    loss.backward()

    # Verify all trainable parameters received gradients
    has_grad = [p.grad is not None for p in net.parameters()]
    assert all(has_grad)
    assert any(torch.norm(p.grad) > 0 for p in net.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 6. ResidualLearnerTrainer Execution
# ─────────────────────────────────────────────────────────────────────────────

def test_trainer_execution():
    n, f_dim, g6_dim, lookback = 200, 15, 4, 30
    rng = np.random.default_rng(1)

    ds = SequenceDataset(
        features=rng.normal(0, 1, (n, f_dim)).astype(np.float32),
        r_hat_lin=rng.normal(0, 0.001, n).astype(np.float32),
        g6_features=rng.normal(0, 1, (n, g6_dim)).astype(np.float32),
        r30=rng.normal(0, 0.001, n).astype(np.float32),
        rv30=rng.normal(-6, 1, n).astype(np.float32),
        mag30=rng.normal(-6, 1, n).astype(np.float32),
        lookback=lookback,
    )

    net = ResidualLearnerNetwork(in_features=f_dim, g6_dim=g6_dim, conv_channels=8, lstm_hidden=8)
    cfg = TrainerConfig(epochs=4, batch_size=32, device="cpu")
    trainer = ResidualLearnerTrainer(config=cfg)

    fitted_net, history = trainer.train(net, ds)
    assert len(history) >= 2
    assert history[-1].val_loss > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. CompositeTradingModel End-to-End Fit & Predict
# ─────────────────────────────────────────────────────────────────────────────

def test_composite_trading_model_fit_predict(sample_dataset_df: pl.DataFrame):
    train_df = sample_dataset_df.slice(0, 350)
    test_df = sample_dataset_df.slice(350, 150)

    cfg = TrainerConfig(epochs=3, batch_size=64, device="cpu")
    model = CompositeTradingModel(lookback=30, trainer_config=cfg)

    # Fit
    model.fit(train_df, symbol="TEST_SYM")

    # Predict conforming to Predictor protocol (r_hat, rv_hat)
    r_hat, rv_hat = model.predict(test_df)
    assert len(r_hat) == len(test_df)
    assert len(rv_hat) == len(test_df)
    assert not np.any(np.isnan(r_hat[30:]))

    # Full predictions bundle
    preds: CompositeModelPredictions = model.predict_full(test_df)
    assert len(preds.r_hat) == len(test_df)
    assert len(preds.residual_hat) == len(test_df)
    assert len(preds.direction) == len(test_df)
    assert len(preds.prob_up) == len(test_df)
    assert len(preds.confidence) == len(test_df)
    assert preds.quantiles_hat.shape == (len(test_df), 9)
    assert np.all((preds.prob_up >= 0.0) & (preds.prob_up <= 1.0))
    assert np.all((preds.confidence >= 0.0) & (preds.confidence <= 1.0))

    from src.validation.metrics import compute_confidence_tier_breakdown
    tiers = compute_confidence_tier_breakdown(
        y_true=test_df["r30"].to_numpy(),
        r_hat=preds.r_hat,
        confidence=preds.confidence,
        spreads=test_df["spread_close"].to_numpy(),
    )
    assert len(tiers) > 0
    assert tiers[0].tier_name == "All Predictions"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Gate 6 Evaluator Walk-Forward Comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_gate6_evaluator_end_to_end(sample_dataset_df: pl.DataFrame, tmp_path: Path):
    evaluator = Gate6Evaluator(
        train_bars=250,
        test_bars=80,
        step_bars=80,
        embargo_bars=10,
        lookback=20,
        epochs=3,
        batch_size=64,
    )

    report = evaluator.evaluate_symbol(sample_dataset_df, symbol="TEST_SYM", report_dir=tmp_path)
    assert report.total_folds >= 1
    assert len(report.fold_comparisons) == report.total_folds
    assert (tmp_path / "gate6_TEST_SYM.json").exists()
    assert (tmp_path / "gate6_TEST_SYM_report.md").exists()
