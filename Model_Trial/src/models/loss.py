"""
Phase 6 — Multi-Task Composite Loss Function.

Implements the 4-component loss function from Whitepaper §6 and Build Plan Phase 6:
  L = λ₁ · Huber(r30, r̂30)
    + λ₂ · Huber(mag30, m̂30)
    + λ₃ · QLIKE(rv30, RV̂30)
    + λ₄ · PinballLoss(|r30|, q̂)

Loss components:
  1. Huber return loss: robust to fat-tailed financial returns (prevents gradient explosions).
  2. Huber log-magnitude loss: predicts log(|r30| + ε) [E5].
  3. Gaussian QLIKE loss: strictly proper scoring rule for volatility estimators (Patton 2011).
  4. Pinball loss: calibrated quantile interval estimation across q0.1...q0.9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import ModelOutputs
from .dataset import BatchData

QUANTILE_GRID: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class LossBreakdown:
    """Detailed loss component values for tracking & early stopping."""
    total_loss: float
    return_loss: float
    mag_loss: float
    rv_loss: float
    quantile_loss: float


class CompositeMultiTaskLoss(nn.Module):
    """
    Composite multi-task loss combining return, magnitude, volatility (QLIKE),
    and quantile losses.
    """

    def __init__(
        self,
        lambda_return: float = 1.0,
        lambda_mag: float = 0.5,
        lambda_rv: float = 0.5,
        lambda_quantile: float = 0.2,
        huber_beta: float = 1.0,
        quantile_grid: Optional[List[float]] = None,
    ):
        super().__init__()
        self.lambda_return = lambda_return
        self.lambda_mag = lambda_mag
        self.lambda_rv = lambda_rv
        self.lambda_quantile = lambda_quantile
        self.huber_beta = huber_beta
        grid = quantile_grid or QUANTILE_GRID
        self.register_buffer("quantiles", torch.tensor(grid, dtype=torch.float32))

    def compute_qlike_loss(self, y_true_log: torch.Tensor, y_pred_log: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Gaussian QLIKE loss for log-volatility targets:
        QLIKE = mean( exp(diff) - diff - 1 ) where diff = y_true - y_pred.
        """
        diff = torch.clamp(y_true_log - y_pred_log, min=-8.0, max=8.0)
        return torch.mean(torch.exp(diff) - diff - 1.0)

    def compute_pinball_loss(self, y_true_abs: torch.Tensor, q_preds: torch.Tensor) -> torch.Tensor:
        """
        Pinball (quantile) loss across all quantile levels:
        Loss = mean_k( max(q * (y - q_hat), (1 - q) * (q_hat - y)) )
        """
        # y_true_abs: (B, 1) -> expand to (B, K)
        # q_preds: (B, K)
        # quantiles: (K,) -> reshape to (1, K)
        y_exp = y_true_abs.expand_as(q_preds)
        diff = y_exp - q_preds
        q = self.quantiles.view(1, -1)
        loss = torch.max(q * diff, (q - 1.0) * diff)
        return torch.mean(loss)

    def forward(
        self,
        outputs: ModelOutputs,
        batch: BatchData,
    ) -> Tuple[torch.Tensor, LossBreakdown]:
        """
        Compute total multi-task loss and return scalar loss + breakdown.
        """
        # 1. Huber return loss on final predicted return r30_hat
        loss_ret = F.smooth_l1_loss(
            outputs.r30_hat, batch.r30, beta=self.huber_beta
        )

        # 2. Huber log-magnitude loss
        loss_mag = F.smooth_l1_loss(
            outputs.mag30_hat, batch.mag30, beta=self.huber_beta
        )

        # 3. Differentiable QLIKE volatility loss
        loss_rv = self.compute_qlike_loss(
            batch.rv30, outputs.rv30_hat
        )

        # 4. Pinball quantile loss on |r30|
        loss_quant = self.compute_pinball_loss(
            batch.abs_r30, outputs.quantiles_hat
        )

        # Combined weighted loss
        total_loss = (
            self.lambda_return * loss_ret
            + self.lambda_mag * loss_mag
            + self.lambda_rv * loss_rv
            + self.lambda_quantile * loss_quant
        )

        breakdown = LossBreakdown(
            total_loss=float(total_loss.item()),
            return_loss=float(loss_ret.item()),
            mag_loss=float(loss_mag.item()),
            rv_loss=float(loss_rv.item()),
            quantile_loss=float(loss_quant.item()),
        )

        return total_loss, breakdown
