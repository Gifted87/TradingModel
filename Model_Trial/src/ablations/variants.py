"""
Phase 7 — Configurable Model Variants for Ablation Studies.

REWRITE (v2):
  - Variants now train from PRE-COMPUTED fold arrays (FoldArrays) supplied by
    the runner's fold cache. FeaturePipeline transforms, Branch A fits, and
    matrix construction happen ONCE per fold and are shared by all 8 variants
    (previously repeated per variant per fold — the main Phase 7 compute burn).
  - All targets flow through the train-fitted TargetScaler (standardized
    space), so every variant actually learns the direction task.
  - RAW_MAGNITUDE (Condition C7) uses a derived scaler whose magnitude stats
    describe raw |r30|, keeping Huber in its meaningful range.
  - BRANCH_A_ONLY fallback rv30 now uses the train-fold mean (via the scaler)
    instead of the old hardcoded -6.5.
  - AblationModelVariant is kept as a thin DataFrame-level compatibility
    wrapper so existing imports and the harness Predictor protocol keep working.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baselines.linear_core import LinearCoreBranchA
from ..features.pipeline import FeaturePipeline, _SKIP_NORMALISE
from ..models.dataset import (
    SequenceDataset,
    TargetScaler,
    build_feature_matrices,
    run_network_inference,
)
from ..models.network import ModelOutputs, ResidualLearnerNetwork
from ..models.trainer import ResidualLearnerTrainer, TrainerConfig
from .config import AblationConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed per-fold arrays (built once per fold by the runner's cache)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldArrays:
    features: np.ndarray          # (N, F_full) normalized, float32
    feature_cols: List[str]
    r_hat_lin: np.ndarray         # (N,) Branch A forecasts, original space
    g6: np.ndarray                # (N, F_g6) normalized, float32
    g6_cols: List[str]
    r30: np.ndarray               # raw targets (NaN allowed)
    rv30: np.ndarray
    mag30: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Network variant (cascade toggle) — architecture unchanged from v1
# ─────────────────────────────────────────────────────────────────────────────

class VariantNetwork(ResidualLearnerNetwork):
    """ResidualLearnerNetwork with optional independent (non-cascade) heads."""

    def __init__(self, in_features: int, g6_dim: int, use_cascade: bool = True,
                 conv_channels: int = 32, lstm_hidden: int = 32,
                 lstm_layers: int = 2, dropout: float = 0.2, n_quantiles: int = 9):
        super().__init__(in_features=in_features, g6_dim=g6_dim,
                         conv_channels=conv_channels, lstm_hidden=lstm_hidden,
                         lstm_layers=lstm_layers, dropout=dropout, n_quantiles=n_quantiles)
        self.use_cascade = use_cascade
        trunk_dim = lstm_hidden * 2
        if not use_cascade:
            self.cascade_head = nn.Sequential(
                nn.Linear(trunk_dim + g6_dim, 48),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(48, 2),
            )
            self._init_weights()

    def forward(self, x: torch.Tensor, r_hat_lin: torch.Tensor,
                g6_features: torch.Tensor) -> ModelOutputs:
        conv_feat = self.conv_block(x)
        lstm_out, _ = self.lstm(conv_feat)
        h_trunk, attn_weights = self.attention(lstm_out)
        h_trunk = self.trunk_dropout(self.trunk_norm(h_trunk))

        residual_hat = self.residual_head(h_trunk)
        r30_hat = r_hat_lin + residual_hat

        if self.use_cascade:
            cascade_input = torch.cat([h_trunk, r30_hat, g6_features], dim=-1)
        else:
            cascade_input = torch.cat([h_trunk, g6_features], dim=-1)
        cascade_out = self.cascade_head(cascade_input)
        mag30_hat = cascade_out[:, 0:1]
        rv30_hat = cascade_out[:, 1:2]

        quant_input = torch.cat([h_trunk, r30_hat], dim=-1)
        quantiles_hat = F.softplus(self.quantile_head(quant_input))

        return ModelOutputs(
            residual_hat=residual_hat, r30_hat=r30_hat,
            mag30_hat=mag30_hat, rv30_hat=rv30_hat,
            quantiles_hat=quantiles_hat, attention_weights=attn_weights,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Variant session: trains one AblationConfig from cached fold arrays
# ─────────────────────────────────────────────────────────────────────────────

class VariantSession:
    """Trains and scores one ablation variant on pre-computed fold arrays."""

    def __init__(self, config: AblationConfig, trainer_config: Optional[TrainerConfig] = None):
        self.config = config
        self.trainer_config = trainer_config or TrainerConfig(epochs=10, batch_size=256)
        self.network: Optional[VariantNetwork] = None
        self.scaler: Optional[TargetScaler] = None
        self._col_idx: List[int] = []
        self._g6_local_idx: List[int] = []
        self.best_val_return_loss: float = float("nan")

    def _select_columns(self, feature_cols: List[str]) -> List[str]:
        cols = list(feature_cols)
        if not self.config.include_g4:
            cols = [c for c in cols if not c.startswith("g4_")]
        if not self.config.include_g5:
            cols = [c for c in cols if not c.startswith("g5_")]
        return cols

    def fit_arrays(self, train: FoldArrays, scaler: TargetScaler) -> "VariantSession":
        cfg = self.config
        cols = self._select_columns(train.feature_cols)
        self._col_idx = [train.feature_cols.index(c) for c in cols]
        self._g6_local_idx = [i for i, c in enumerate(cols) if c.startswith("g6_")]

        # Condition C7: raw |r30| magnitude target with a derived scaler
        if cfg.log_magnitude_target:
            self.scaler = scaler
            mag_target = train.mag30
        else:
            abs_r = np.abs(train.r30[np.isfinite(train.r30)])
            a_m = float(abs_r.mean()) if len(abs_r) else 0.0
            a_s = float(abs_r.std()) if len(abs_r) and abs_r.std() > 1e-12 else 1.0
            self.scaler = dataclasses.replace(scaler, mag30_mean=a_m, mag30_std=a_s)
            mag_target = np.abs(train.r30)

        if not cfg.branch_b_enabled:
            return self  # Branch-A-only: nothing further to train

        dataset = SequenceDataset(
            features=train.features[:, self._col_idx],
            r_hat_lin=train.r_hat_lin,
            g6_features=train.features[:, self._col_idx][:, self._g6_local_idx]
                if self._g6_local_idx else np.zeros((len(train.r30), 0), dtype=np.float32),
            r30=train.r30, rv30=train.rv30, mag30=mag_target,
            lookback=cfg.lookback, scaler=self.scaler,
        )

        self.network = VariantNetwork(
            in_features=len(cols),
            g6_dim=len(self._g6_local_idx),
            use_cascade=cfg.use_cascade,
        )
        trainer = ResidualLearnerTrainer(config=self.trainer_config)
        self.network, _ = trainer.train(self.network, dataset)
        self.best_val_return_loss = trainer.best_val_score_
        return self

    def predict_arrays(self, test: FoldArrays) -> Dict[str, np.ndarray]:
        n = len(test.r30)
        if not self.config.branch_b_enabled or self.network is None:
            # Branch A only; rv30 = train-fold mean (from scaler), no magic constants
            return {
                "r_hat": test.r_hat_lin.copy(),
                "rv_hat": np.full(n, self.scaler.rv30_shift, dtype=np.float64),
            }
        device = torch.device(self.trainer_config.device)
        self.network.to(device)
        g6_test = (test.features[:, self._col_idx][:, self._g6_local_idx]
                   if self._g6_local_idx else np.zeros((n, 0), dtype=np.float32))
        raw = run_network_inference(
            self.network, test.features[:, self._col_idx], test.r_hat_lin, g6_test,
            scaler=self.scaler, lookback=self.config.lookback,
            batch_size=self.trainer_config.batch_size, device=device,
        )
        r_hat = np.where(np.isnan(raw["r30"]), test.r_hat_lin, raw["r30"])
        rv_hat = np.where(np.isnan(raw["rv"]), self.scaler.rv30_shift, raw["rv"])
        return {"r_hat": r_hat, "rv_hat": rv_hat}


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible DataFrame-level wrapper (harness Predictor protocol)
# ─────────────────────────────────────────────────────────────────────────────

class AblationModelVariant:
    """DataFrame-level wrapper kept for compatibility with existing imports."""

    def __init__(self, config: AblationConfig, trainer_config: Optional[TrainerConfig] = None):
        self.config = config
        self.trainer_config = trainer_config or TrainerConfig(epochs=10, batch_size=256)
        self.branch_a = LinearCoreBranchA(alpha=100.0)
        self.pipeline = FeaturePipeline(clip_sigma=5.0)
        self.session = VariantSession(config, self.trainer_config)
        self.scaler: Optional[TargetScaler] = None
        self.symbol: str = "UNKNOWN"

    def _to_arrays(self, df: pl.DataFrame, fit: bool) -> FoldArrays:
        if fit:
            self.pipeline.fit(df)
        feature_cols = [
            c for c in df.columns
            if c.startswith("g") and "_" in c and c not in _SKIP_NORMALISE
        ]
        g6_cols = [c for c in feature_cols if c.startswith("g6_")]
        feats, r_lin, g6, _, _ = build_feature_matrices(
            df=df, pipeline=self.pipeline, linear_core=self.branch_a,
            feature_cols=feature_cols,
        )
        n = len(df)
        return FoldArrays(
            features=feats, feature_cols=feature_cols, r_hat_lin=r_lin,
            g6=g6, g6_cols=g6_cols,
            r30=df["r30"].to_numpy() if "r30" in df.columns else np.full(n, np.nan),
            rv30=df["rv30"].to_numpy() if "rv30" in df.columns else np.full(n, np.nan),
            mag30=df["mag30"].to_numpy() if "mag30" in df.columns else np.full(n, np.nan),
        )

    def fit(self, df: pl.DataFrame, symbol: str = "UNKNOWN", **kwargs: Any) -> "AblationModelVariant":
        self.symbol = symbol
        df = df.sort("timestamp")
        self.branch_a.fit(df, symbol=symbol)
        train = self._to_arrays(df, fit=True)
        self.scaler = TargetScaler.fit(train.r30, train.rv30, train.mag30)
        self.session.fit_arrays(train, self.scaler)
        return self

    def predict(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.sort("timestamp")
        test = self._to_arrays(df, fit=False)
        out = self.session.predict_arrays(test)
        return out["r_hat"], out["rv_hat"]