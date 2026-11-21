"""
Phase 6 — Composite Trading Model (Branch A Linear Core + Branch B Residual Learner).

REWRITE (v2):
  - Fits a TargetScaler on the training fold and trains Branch B entirely in
    standardized target space (the fix for the loss-scale training failure).
    All outputs are inverse-transformed back to original units at inference.
  - Warmup rows (first lookback-1) are filled with TRAIN-fold statistics
    (Branch A forecast, train mean rv30/mag30, train |r30| quantiles) instead
    of the old hardcoded -7.0 constants.
  - fit() accepts an optional pre-fitted `branch_a` so Gate 6 no longer fits
    the linear core twice per fold.
  - Optional MLflow context via TrainerConfig (fold_id, registry_version).

Implements the src.validation Predictor protocol:
  fit(df, symbol=...) -> self ; predict(df) -> (r_hat, rv_hat)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

from ..baselines.linear_core import LinearCoreBranchA
from ..features.pipeline import FeaturePipeline, _SKIP_NORMALISE
from .dataset import (
    SequenceDataset,
    TargetScaler,
    build_feature_matrices,
    run_network_inference,
)
from .loss import QUANTILE_GRID
from .network import ResidualLearnerNetwork
from .trainer import ResidualLearnerTrainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class CompositeModelPredictions:
    r_hat: np.ndarray             # (N,) r̂30 = r̂_lin + residual̂, original units
    r_hat_lin: np.ndarray         # (N,) Branch A linear forecast
    residual_hat: np.ndarray      # (N,) Branch B residual, original return units
    direction: np.ndarray         # (N,) sign(r_hat), +1 for up, -1 for down
    mag_hat: np.ndarray           # (N,) log-magnitude forecast
    rv_hat: np.ndarray            # (N,) forward log-RV forecast
    quantiles_hat: np.ndarray     # (N, 9) quantiles of |r30|, original units
    prob_up: Optional[np.ndarray] = None     # (N,) P(up) probability in [0, 1]
    confidence: Optional[np.ndarray] = None  # (N,) Conviction/confidence score in [0, 1]


class CompositeTradingModel:
    """End-to-end composite model: frozen linear core + residual sequence learner."""

    def __init__(
        self,
        lookback: int = 60,
        branch_a_alpha: float = 100.0,
        conv_channels: int = 32,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        trainer_config: Optional[TrainerConfig] = None,
    ):
        self.lookback = lookback
        self.branch_a_alpha = branch_a_alpha
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.trainer_config = trainer_config or TrainerConfig()

        self.branch_a = LinearCoreBranchA(alpha=branch_a_alpha)
        self.pipeline = FeaturePipeline(clip_sigma=5.0)
        self.network: Optional[ResidualLearnerNetwork] = None
        self.scaler: Optional[TargetScaler] = None
        self.feature_cols: List[str] = []
        self.g6_cols: List[str] = []
        self.symbol: str = "UNKNOWN"
        # train-fold statistics for warmup rows
        self._warmup_rv: float = 0.0
        self._warmup_mag: float = 0.0
        self._warmup_quantiles: Optional[np.ndarray] = None
        self.train_history: list = []

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pl.DataFrame,
        symbol: str = "UNKNOWN",
        branch_a: Optional[LinearCoreBranchA] = None,
        fold_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "CompositeTradingModel":
        self.symbol = symbol
        df = df.sort("timestamp")

        # 1. Branch A — reuse a pre-fitted core when supplied (Gate 6 fits once)
        if branch_a is not None:
            self.branch_a = branch_a
        else:
            self.branch_a = LinearCoreBranchA(alpha=self.branch_a_alpha)
            self.branch_a.fit(df, symbol=symbol)

        # 2. Normalizer (train-fold only — zero leakage)
        self.pipeline.fit(df)
        self.feature_cols = [
            c for c in df.columns
            if c.startswith("g") and "_" in c and c not in _SKIP_NORMALISE
        ]
        self.g6_cols = [c for c in self.feature_cols if c.startswith("g6_")]

        # 3. Matrices + target scaler
        features_mat, r_hat_lin, g6_mat, _, _ = build_feature_matrices(
            df=df, pipeline=self.pipeline, linear_core=self.branch_a,
            feature_cols=self.feature_cols,
        )
        r30 = df["r30"].to_numpy() if "r30" in df.columns else np.full(len(df), np.nan)
        rv30 = df["rv30"].to_numpy() if "rv30" in df.columns else np.full(len(df), np.nan)
        mag30 = df["mag30"].to_numpy() if "mag30" in df.columns else np.full(len(df), np.nan)

        self.scaler = TargetScaler.fit(r30, rv30, mag30)
        self._warmup_rv = self.scaler.rv30_shift
        self._warmup_mag = self.scaler.mag30_mean
        abs_r = np.abs(r30[np.isfinite(r30)])
        self._warmup_quantiles = (
            np.quantile(abs_r, QUANTILE_GRID) if len(abs_r) > 10
            else np.full(len(QUANTILE_GRID), self.scaler.abs30_scale)
        )

        # 4. Standardized sequence dataset
        train_dataset = SequenceDataset(
            features=features_mat, r_hat_lin=r_hat_lin, g6_features=g6_mat,
            r30=r30, rv30=rv30, mag30=mag30,
            lookback=self.lookback, scaler=self.scaler,
        )
        if len(train_dataset) == 0:
            raise ValueError(f"0 training sequences (lookback={self.lookback}, rows={len(df)}).")

        # 5. Network + trainer
        self.network = ResidualLearnerNetwork(
            in_features=len(self.feature_cols),
            g6_dim=len(self.g6_cols),
            conv_channels=self.conv_channels,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
        )
        if fold_id is not None:
            self.trainer_config.fold_id = fold_id
        trainer = ResidualLearnerTrainer(config=self.trainer_config)
        self.network, self.train_history = trainer.train(self.network, train_dataset)

        logger.info(
            f"Composite model fitted for {symbol}: {len(self.feature_cols)} features, "
            f"best val {self.trainer_config.selection_metric}={trainer.best_val_score_:.5f}"
        )
        return self

    # ── predict ────────────────────────────────────────────────────────────

    def predict(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.predict_full(df)
        return preds.r_hat, preds.rv_hat

    def predict_full(self, df: pl.DataFrame) -> CompositeModelPredictions:
        if self.network is None or self.scaler is None:
            raise RuntimeError("CompositeTradingModel is not fitted.")

        df = df.sort("timestamp")
        n = len(df)

        features_mat, r_hat_lin, g6_mat, _, _ = build_feature_matrices(
            df=df, pipeline=self.pipeline, linear_core=self.branch_a,
            feature_cols=self.feature_cols,
        )

        device = torch.device(self.trainer_config.device)
        self.network.to(device)
        raw = run_network_inference(
            self.network, features_mat, r_hat_lin, g6_mat,
            scaler=self.scaler, lookback=self.lookback,
            batch_size=self.trainer_config.batch_size, device=device,
        )

        # Warmup rows: Branch A forecast + train-fold statistics (no hardcoded constants)
        warm = slice(0, min(self.lookback - 1, n))
        residual_hat = np.nan_to_num(raw["residual"], nan=0.0)
        r_hat = np.where(np.isnan(raw["r30"]), r_hat_lin, raw["r30"])
        mag_hat = raw["mag"].copy(); mag_hat[warm] = self._warmup_mag
        rv_hat = raw["rv"].copy();   rv_hat[warm] = self._warmup_rv
        quantiles_hat = raw["quantiles"].copy()
        quantiles_hat[warm] = self._warmup_quantiles

        # Direction, probabilistic conviction & confidence
        direction = np.sign(r_hat)
        scale = getattr(self.scaler, "r30_std", 1e-4) if self.scaler is not None else 1e-4
        z_score = r_hat / (scale + 1e-9)
        prob_up = 1.0 / (1.0 + np.exp(-np.clip(z_score, -15.0, 15.0)))
        confidence = np.abs(prob_up - 0.5) * 2.0  # normalized [0, 1]

        return CompositeModelPredictions(
            r_hat=r_hat,
            r_hat_lin=r_hat_lin,
            residual_hat=residual_hat,
            direction=direction,
            mag_hat=mag_hat,
            rv_hat=rv_hat,
            quantiles_hat=quantiles_hat,
            prob_up=prob_up,
            confidence=confidence,
        )