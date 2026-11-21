"""Calibrated Expected Utility Decision Engine (Plan 3 - Hardened).

Institutional Decision Theory:
1. Expected price move: mu_hat = sum_i p_i * m_i
2. Predictive variance: sigma_hat^2 = sum_i p_i * (m_i^2 + v_i) - mu_hat^2
3. Net expected utilities after spread, commission, and slippage hurdles:
   U_long = mu_hat - cost_long
   U_short = -mu_hat - cost_short
4. Conservative Lower Confidence Bounds (LCB) & Signal-to-Noise Ratio (SNR) hurdles:
   LCB_long = mu_hat - alpha * sigma_hat
   LCB_short = -mu_hat - alpha * sigma_hat
   SNR = |mu_hat| / (sigma_hat + epsilon)
5. Position Sizing:
   Supports both SNR-proportional scaling and Fractional Kelly Criterion sizing:
   f_kelly = (mu_hat - cost) / (sigma_hat^2 + epsilon) * kelly_fraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TradeDecision:
    """Institutional Trade Decision Object."""
    action: str                     # "LONG", "SHORT", "FLAT"
    position_size: float           # Scaled position weight in [-1.0, 1.0]
    expected_return: float         # Net expected monetary price move mu_hat
    uncertainty: float             # Standard deviation sigma_hat
    u_long: float                  # Net expected utility for long
    u_short: float                 # Net expected utility for short
    snr: float                     # Signal-to-Noise Ratio
    lcb: float                     # Lower confidence bound
    p_up: float                    # Total probability mass on bullish bins (B5..B8)
    p_down: float                  # Total probability mass on bearish bins (B0..B3)
    p_flat: float                  # Probability mass on neutral bin (B4)


class TemperatureCalibrator(nn.Module):
    """Learns post-hoc temperature scaling parameter T on validation set."""

    def __init__(self, init_temp: float = 1.2) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(init_temp), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class CalibratedDecisionEngine:
    """Institutional Expected Utility Decision Engine with Cost Hurdle & Kelly Sizing."""

    def __init__(
        self,
        bin_conditional_means: np.ndarray,
        bin_conditional_variances: Optional[np.ndarray] = None,
        cost_long: float = 0.50,
        cost_short: float = 0.50,
        hurdle_tau: float = 0.20,
        confidence_alpha: float = 0.50,
        snr_threshold: float = 0.40,
        prob_margin: float = 0.10,
        kelly_fraction: float = 0.25,
        calibrator: Optional[TemperatureCalibrator] = None,
    ) -> None:
        self.bin_means = np.asarray(bin_conditional_means, dtype=np.float32)
        self.num_bins = len(self.bin_means)
        
        if bin_conditional_variances is not None:
            self.bin_vars = np.asarray(bin_conditional_variances, dtype=np.float32)
        else:
            self.bin_vars = np.zeros(self.num_bins, dtype=np.float32)
            
        self.cost_long = float(cost_long)
        self.cost_short = float(cost_short)
        self.hurdle_tau = float(hurdle_tau)
        self.confidence_alpha = float(confidence_alpha)
        self.snr_threshold = float(snr_threshold)
        self.prob_margin = float(prob_margin)
        self.kelly_fraction = float(kelly_fraction)
        self.calibrator = calibrator

    def evaluate_distribution(
        self,
        probs: np.ndarray,
        logits: Optional[torch.Tensor] = None,
    ) -> TradeDecision:
        """Evaluates a 9-bin probability vector and outputs a trading decision."""
        if self.calibrator is not None and logits is not None:
            with torch.no_grad():
                cal_logits = self.calibrator(logits.unsqueeze(0)).squeeze(0)
                probs = F.softmax(cal_logits, dim=-1).cpu().numpy()
                
        probs = np.asarray(probs, dtype=np.float32)
        probs = probs / (probs.sum() + 1e-8)
        
        # 1. Expected Monetary Price Move: mu_hat = sum_i p_i * m_i
        mu_hat = float(np.sum(probs * self.bin_means))
        
        # 2. Predictive Uncertainty: sigma_hat^2 = sum_i p_i * (m_i^2 + v_i) - mu_hat^2
        exp_m2 = np.sum(probs * (self.bin_means ** 2 + self.bin_vars))
        sigma_sq = max(float(exp_m2 - mu_hat ** 2), 1e-6)
        sigma_hat = float(np.sqrt(sigma_sq))
        
        # 3. Directional Mass
        neutral_idx = self.num_bins // 2
        p_down = float(np.sum(probs[:neutral_idx]))
        p_flat = float(probs[neutral_idx])
        p_up = float(np.sum(probs[neutral_idx + 1:]))
        
        # 4. Expected Net Utilities after transaction costs
        u_long = mu_hat - self.cost_long
        u_short = -mu_hat - self.cost_short
        
        # 5. Signal-to-Noise Ratio and Lower Confidence Bounds
        snr = abs(mu_hat) / (sigma_hat + 1e-6)
        lcb_long = mu_hat - self.confidence_alpha * sigma_hat
        lcb_short = -mu_hat - self.confidence_alpha * sigma_hat
        
        # 6. Cost-Aware Decision Logic
        action = "FLAT"
        position_size = 0.0
        active_lcb = 0.0
        
        is_long_valid = (
            u_long > max(0.0, u_short) + self.hurdle_tau
            and lcb_long > 0.0
            and snr >= self.snr_threshold
            and (p_up > p_down + self.prob_margin)
        )
        
        is_short_valid = (
            u_short > max(0.0, u_long) + self.hurdle_tau
            and lcb_short > 0.0
            and snr >= self.snr_threshold
            and (p_down > p_up + self.prob_margin)
        )
        
        if is_long_valid:
            action = "LONG"
            active_lcb = lcb_long
            # Fractional Kelly Sizing
            f_kelly = (u_long / (sigma_sq + 1e-6)) * self.kelly_fraction
            position_size = float(np.clip(f_kelly, 0.1, 1.0))
        elif is_short_valid:
            action = "SHORT"
            active_lcb = lcb_short
            f_kelly = (u_short / (sigma_sq + 1e-6)) * self.kelly_fraction
            position_size = -float(np.clip(f_kelly, 0.1, 1.0))
            
        return TradeDecision(
            action=action,
            position_size=position_size,
            expected_return=mu_hat,
            uncertainty=sigma_hat,
            u_long=u_long,
            u_short=u_short,
            snr=snr,
            lcb=active_lcb,
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
        )

    def evaluate_batch(self, batch_probs: np.ndarray) -> List[TradeDecision]:
        return [self.evaluate_distribution(p) for p in batch_probs]
