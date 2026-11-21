"""
Phase 7 — Cross-Instrument Universality Transfer Test [B2].

REWRITE (v2) — fixes a critical correctness bug:
  OLD: source DataFrames were concatenated and sorted by timestamp, which
       INTERLEAVES instruments; 60-bar training windows then crossed symbol
       boundaries (EURUSD bars inside a "GBPUSD" window). The [B2]
       universality claim was being tested on corrupted sequences.
  NEW: pipeline / Branch A / TargetScaler are fitted on the merged ROWS
       (row-wise statistics — legitimate), but sequences are built strictly
       PER SYMBOL and combined via ConcatSequenceDataset, whose collate
       gathers each batch within symbol boundaries.

  Protocol (unchanged): train on N-1 source instruments, evaluate zero-shot
  on the held-out target. Transfer succeeds if zero-shot MCC > 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import polars as pl

from ..baselines.linear_core import LinearCoreBranchA
from ..features.pipeline import FeaturePipeline, _SKIP_NORMALISE
from ..models.dataset import (
    ConcatSequenceDataset,
    SequenceDataset,
    TargetScaler,
    run_network_inference,
)
from ..models.network import ResidualLearnerNetwork
from ..models.trainer import ResidualLearnerTrainer, TrainerConfig
from ..validation.metrics import compute_direction_metrics, compute_regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class TransferTestResult:
    source_symbols: List[str]
    target_symbol: str
    n_train_samples: int
    n_test_samples: int
    target_mcc: float
    target_hit_rate_tradable: float
    target_hit_rate_all: float
    target_oos_r2: float
    target_qlike_rv30: float
    target_rmse_r30: float
    transfer_successful: bool
    verdict: str


def run_cross_instrument_transfer_test(
    source_dfs: Dict[str, pl.DataFrame],
    target_symbol: str,
    target_df: pl.DataFrame,
    lookback: int = 60,
    epochs: int = 10,
    batch_size: int = 256,
    decision_bars_only: bool = False,
) -> TransferTestResult:
    logger.info(
        f"Universality transfer test: sources={list(source_dfs.keys())} -> target={target_symbol}"
    )

    # 1. Common schema
    common_cols: set = set()
    for sym, df in source_dfs.items():
        cols = set(df.columns)
        common_cols = cols if not common_cols else (common_cols & cols)
    ordered_cols = sorted(common_cols)
    per_symbol = {
        sym: df.select(ordered_cols).sort("timestamp") for sym, df in source_dfs.items()
    }

    # 2. Fit row-wise components on merged rows (no sequences are built here)
    merged = pl.concat(list(per_symbol.values()))
    feature_cols = [c for c in ordered_cols
                    if c.startswith("g") and "_" in c and c not in _SKIP_NORMALISE]
    g6_cols = [c for c in feature_cols if c.startswith("g6_")]

    pipeline = FeaturePipeline(clip_sigma=5.0).fit(merged)
    branch_a = LinearCoreBranchA(alpha=100.0).fit(merged, symbol="MERGED_SOURCE")
    scaler = TargetScaler.fit(
        merged["r30"].to_numpy(), merged["rv30"].to_numpy(), merged["mag30"].to_numpy()
    )

    # 3. Per-symbol sequence datasets — windows NEVER cross symbol boundaries
    def _matrices(df: pl.DataFrame):
        norm = pipeline.transform(df)
        feats = (norm.select(feature_cols).fill_nan(0.0).fill_null(0.0)
                 .to_numpy().astype(np.float32))
        g6 = (norm.select(g6_cols).fill_nan(0.0).fill_null(0.0)
              .to_numpy().astype(np.float32))
        return feats, branch_a.predict(df), g6

    datasets: List[SequenceDataset] = []
    for sym, df in per_symbol.items():
        feats, r_lin, g6 = _matrices(df)
        ds = SequenceDataset(
            features=feats, r_hat_lin=r_lin, g6_features=g6,
            r30=df["r30"].to_numpy(), rv30=df["rv30"].to_numpy(),
            mag30=df["mag30"].to_numpy(),
            lookback=lookback, scaler=scaler,
        )
        if len(ds) > 0:
            datasets.append(ds)
        logger.info(f"  {sym}: {len(ds):,} training sequences")

    concat = ConcatSequenceDataset(datasets)

    # 4. Train
    network = ResidualLearnerNetwork(in_features=len(feature_cols), g6_dim=len(g6_cols))
    trainer_cfg = TrainerConfig(
        epochs=epochs, batch_size=batch_size, patience=3,
        selection_metric="return",
        val_split="random",   # multi-symbol concat: random val is representative
    )
    trainer = ResidualLearnerTrainer(config=trainer_cfg)
    network, _ = trainer.train(network, concat)

    # 5. Zero-shot prediction on the held-out instrument
    target_df = target_df.sort("timestamp")
    feats_t, r_lin_t, g6_t = _matrices(target_df)
    import torch
    device = torch.device(trainer_cfg.device)
    network.to(device)
    raw = run_network_inference(
        network, feats_t, r_lin_t, g6_t,
        scaler=scaler, lookback=lookback, batch_size=batch_size, device=device,
    )
    r_pred = np.where(np.isnan(raw["r30"]), r_lin_t, raw["r30"])
    rv_pred = np.where(np.isnan(raw["rv"]), scaler.rv30_shift, raw["rv"])

    mask = np.isfinite(target_df["r30"].to_numpy()) & np.isfinite(target_df["rv30"].to_numpy())
    if decision_bars_only and "is_decision_bar" in target_df.columns:
        mask &= target_df["is_decision_bar"].to_numpy().astype(bool)

    y_true = target_df["r30"].to_numpy()[mask]
    rv_true = target_df["rv30"].to_numpy()[mask]
    r_pred = r_pred[mask]
    rv_pred = rv_pred[mask]
    is_dead = (target_df["is_dead_zone"].to_numpy().astype(bool)[mask]
               if "is_dead_zone" in target_df.columns else None)

    dir_m = compute_direction_metrics(y_true, r_pred, is_dead)
    reg_m = compute_regression_metrics(y_true, r_pred, rv_true, rv_pred)

    passed = dir_m.mcc > 0.0
    verdict = (
        f"UNIVERSALITY PASS [B2]: trained on {list(source_dfs.keys())}, zero-shot on "
        f"{target_symbol}: MCC={dir_m.mcc:+.4f}, hit-rate={dir_m.hit_rate_tradable * 100:.2f}%"
        if passed else
        f"UNIVERSALITY WARNING: zero-shot transfer to {target_symbol} yielded "
        f"MCC={dir_m.mcc:+.4f} <= 0"
    )

    return TransferTestResult(
        source_symbols=list(source_dfs.keys()),
        target_symbol=target_symbol,
        n_train_samples=len(concat),
        n_test_samples=int(mask.sum()),
        target_mcc=dir_m.mcc,
        target_hit_rate_tradable=dir_m.hit_rate_tradable,
        target_hit_rate_all=dir_m.hit_rate_all,
        target_oos_r2=reg_m.oos_r2,
        target_qlike_rv30=reg_m.qlike_rv30,
        target_rmse_r30=reg_m.rmse_r30,
        transfer_successful=passed,
        verdict=verdict,
    )