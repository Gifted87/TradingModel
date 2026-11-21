"""Ordinal & Continuous Loss Engine for Financial Return Forecasting (Plan 2).

Solves the core structural mismatch of nominal cross-entropy by formulating loss functions
that strictly respect ordinal price distances, preserve continuous return magnitudes,
and support multi-task cost-aware objectives.

Implements:
1. Adaptive Volatility-Scaled Gaussian Soft Ordinal Targets.
2. Ranked Probability Score (RPS) Loss / Discretized CRPS.
3. 1D Earth Mover's Distance (EMD) / Wasserstein-1 Loss.
4. Cumulative-Link / CORAL Ordinal Head with Monotonic Thresholds & Tail Probability Extraction.
5. Parametric Student-t Distribution Head with Exact Moments & Continuous NLL.
6. Nonparametric Log-Concave Unimodality Regularizer.
7. Multi-Task Composite Loss Container (Distribution + Huber + Cost-Aware Dead Zone + Log-Concavity).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_gaussian_soft_targets(
    targets: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: Union[float, torch.Tensor] = 0.5,
) -> torch.Tensor:
    """Generates continuous Gaussian soft-label probability distributions over ordinal bins.
    
    Args:
        targets: (B,) continuous price returns or normalized z-scores.
        bin_centers: (K,) coordinates of the bin centers in target return space.
        sigma: Smoothing bandwidth scalar or (B,) per-sample local volatility vector.
    Returns:
        (B, K) normalized soft probability distribution vectors on the simplex.
    """
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)  # (B, 1)
        
    bin_centers = bin_centers.to(device=targets.device, dtype=targets.dtype)
    if bin_centers.dim() == 1:
        bin_centers = bin_centers.unsqueeze(0)  # (1, K)
        
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(device=targets.device, dtype=targets.dtype)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)  # (B, 1)
            
    diff = targets - bin_centers  # (B, K)
    log_q = -0.5 * (diff / (sigma + 1e-6)) ** 2
    q = torch.softmax(log_q, dim=-1)
    return q


class RankedProbabilityScoreLoss(nn.Module):
    """Ranked Probability Score (RPS) Loss / Discretized Continuous Ranked Probability Score (CRPS).
    
    Penalizes cumulative distribution errors between predicted CDF F(p) and target CDF G(q):
        L_RPS = (1 / (K - 1)) * sum_{k=0}^{K-2} (F_k(p) - G_k(q))^2
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, p_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_pred: (B, K) predicted probabilities (must sum to 1 along last dim).
            q_target: (B, K) target probability distribution or one-hot indicators.
        Returns:
            Scalar RPS loss.
        """
        # Cumulative distributions excluding the last bin (which is always identically 1.0)
        cdf_pred = torch.cumsum(p_pred, dim=-1)[..., :-1]
        cdf_target = torch.cumsum(q_target, dim=-1)[..., :-1]
        
        # Mean squared CDF difference over K-1 cutoffs
        rps_per_sample = torch.mean((cdf_pred - cdf_target) ** 2, dim=-1)
        
        if self.reduction == "mean":
            return rps_per_sample.mean()
        elif self.reduction == "sum":
            return rps_per_sample.sum()
        return rps_per_sample


class EarthMoversDistanceLoss(nn.Module):
    """1D Earth Mover's Distance (EMD) / Wasserstein-1 Loss between discrete distributions:
        L_EMD = (1 / (K - 1)) * sum_{k=0}^{K-2} |F_k(p) - G_k(q)|
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, p_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        cdf_pred = torch.cumsum(p_pred, dim=-1)[..., :-1]
        cdf_target = torch.cumsum(q_target, dim=-1)[..., :-1]
        
        emd_per_sample = torch.mean(torch.abs(cdf_pred - cdf_target), dim=-1)
        
        if self.reduction == "mean":
            return emd_per_sample.mean()
        elif self.reduction == "sum":
            return emd_per_sample.sum()
        return emd_per_sample


class SoftOrdinalCrossEntropyLoss(nn.Module):
    """Kullback-Leibler / Cross-Entropy Loss with continuous soft Gaussian targets."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K) unnormalized network outputs.
            q_target: (B, K) target probability distribution.
        """
        log_p = F.log_softmax(logits, dim=-1)
        loss = -(q_target * log_p).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LogConcavityUnimodalityLoss(nn.Module):
    """Penalizes non-log-concave (multimodal or irregular) discrete probability distributions.
    
    A discrete distribution is log-concave (and thus strictly unimodal) iff:
        log p_i >= (log p_{i-1} + log p_{i+1}) / 2  for all 1 <= i <= K-2
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (B, K) predicted probabilities on simplex.
        """
        log_p = torch.log(p.clamp(min=self.eps))
        mid = 0.5 * (log_p[..., :-2] + log_p[..., 2:])
        actual = log_p[..., 1:-1]
        penalty = F.relu(mid - actual)
        return penalty.sum(dim=-1).mean()


class CumulativeOrdinalHead(nn.Module):
    """CORAL-style Proportional Odds / Cumulative Link Head.
    
    Guarantees monotonically ordered thresholds theta_0 < theta_1 < ... < theta_{K-2}
    and valid non-negative probabilities without ordinal inversion:
        theta_k = theta_0 + sum_{m=1}^k softplus(gamma_m)
        F_k = P(Y <= B_k) = sigmoid(theta_k - s(x))
    """

    def __init__(self, in_features: int, num_bins: int = 9) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.score_layer = nn.Linear(in_features, 1, bias=False)
        
        # Initial threshold parameters
        self.theta_0 = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((num_bins - 2,), 0.5, dtype=torch.float32))

    def get_monotonic_thresholds(self) -> torch.Tensor:
        """Constructs strictly monotonic threshold vector of size K-1."""
        increments = F.softplus(self.gamma) + 1e-4
        thresholds = torch.cat([self.theta_0.unsqueeze(0), self.theta_0 + torch.cumsum(increments, dim=0)])
        return thresholds

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, in_features) latent feature representation.
        Returns:
            p: (B, K) implied discrete bin probabilities on simplex.
            cdf: (B, K-1) cumulative probabilities F_k = P(Y <= B_k).
            scalar_score: (B, 1) latent continuous location score s(x).
        """
        score = self.score_layer(h)  # (B, 1)
        thresholds = self.get_monotonic_thresholds().unsqueeze(0)  # (1, K-1)
        
        # Cumulative logits: eta_k = theta_k - s(x)
        cum_logits = thresholds - score  # (B, K-1)
        cdf = torch.sigmoid(cum_logits)  # (B, K-1)
        
        # Discrete probabilities via CDF differences
        p0 = cdf[..., :1]
        p_mid = F.relu(cdf[..., 1:] - cdf[..., :-1]) + 1e-8
        p_last = F.relu(1.0 - cdf[..., -1:]) + 1e-8
        
        p = torch.cat([p0, p_mid, p_last], dim=-1)
        p = p / p.sum(dim=-1, keepdim=True)
        
        return p, cdf, score

    def get_directional_tail_probabilities(
        self, cdf: torch.Tensor, neutral_idx: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts directional probabilities directly from cumulative CDF:
            p_bearish = P(Y <= B_{neutral-1}) = F_{neutral-1}
            p_bullish = P(Y > B_{neutral}) = 1 - F_{neutral}
            p_flat = 1 - p_bearish - p_bullish
        """
        p_bear = cdf[..., neutral_idx - 1]
        p_bull = 1.0 - cdf[..., neutral_idx]
        p_flat = torch.clamp(1.0 - p_bear - p_bull, min=0.0)
        return p_bear, p_flat, p_bull

    def compute_ordinal_bce_loss(self, cdf: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        """Computes cumulative binary cross-entropy loss against soft/hard targets."""
        target_cdf = torch.cumsum(q_target, dim=-1)[..., :-1]
        loss = F.binary_cross_entropy(cdf.clamp(min=1e-7, max=1.0 - 1e-7), target_cdf, reduction="mean")
        return loss


class StudentTDistributionHead(nn.Module):
    """Parametric Student-t Predictive Distribution Head for Fat-Tailed Return Dynamics.
    
    Predicts continuous parameters:
    - Location mu(x)
    - Scale sigma(x) > 0
    - Degrees of Freedom nu(x) > 2 (ensuring finite mean and variance)
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(in_features, 1)
        self.fc_sigma = nn.Linear(in_features, 1)
        self.fc_nu = nn.Linear(in_features, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, in_features) latent vector.
        Returns:
            mu: (B, 1) location parameter.
            sigma: (B, 1) positive scale parameter.
            nu: (B, 1) degrees of freedom (> 2.0).
        """
        mu = self.fc_mu(h)
        sigma = F.softplus(self.fc_sigma(h)) + 1e-4
        nu = F.softplus(self.fc_nu(h)) + 2.01  # Guarantees finite variance
        return mu, sigma, nu

    def predict_moments(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates theoretical mean and conditional variance of the predicted return:
            E[R] = mu
            Var[R] = sigma^2 * nu / (nu - 2)
        """
        mu, sigma, nu = self.forward(h)
        var = (sigma ** 2) * (nu / (nu - 2.0))
        return mu, var

    def compute_continuous_nll(
        self,
        targets: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        """Computes exact negative log-likelihood under Student-t distribution."""
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
            
        z = (targets - mu) / sigma
        log_norm = (
            torch.lgamma(0.5 * (nu + 1.0))
            - torch.lgamma(0.5 * nu)
            - 0.5 * torch.log(torch.tensor(np.pi, device=targets.device) * nu)
            - torch.log(sigma)
        )
        log_prob = log_norm - 0.5 * (nu + 1.0) * torch.log1p((z ** 2) / nu)
        return -log_prob.mean()


class MultiTaskCompositeLoss(nn.Module):
    """Unified Multi-Task Loss Container (Plan 2 - Hardened Direction Discovery).
    
    Combines:
    1. L_focal: Volatility-Weighted Directional Focal Loss directly on 9-bin probability mass.
    2. L_dir_sign: Directional Sign Alignment Loss on continuous distribution expectation E[y|x].
    3. L_dir_mass: Asymmetric Wrong-Side Probability Mass Penalty on 9-bin distribution.
    4. L_softCE: Gaussian-smoothed ordinal cross entropy with sharpened bandwidth.
    5. L_RPS: Ranked Probability Score CDF penalty.
    6. L_reg: Huber loss on continuous return regression.
    7. L_dir: Cost-aware 3-class Direction Gate (Bearish, Untradable Dead-Zone, Bullish).
    8. L_uni: Unimodality / log-concavity regularizer.
    """

    def __init__(
        self,
        bin_centers: torch.Tensor,
        cost_hurdle: float = 0.50,
        soft_sigma: float = 0.02,
        lambda_focal: float = 2.00,
        lambda_dir_sign: float = 1.50,
        lambda_dir_mass: float = 0.50,
        lambda_rps: float = 0.20,
        lambda_reg: float = 0.20,
        lambda_dir: float = 0.50,
        lambda_uni: float = 0.05,
        margin_tau: float = 0.05,
        focal_gamma: float = 2.0,
        chop_threshold: float = 0.03,
    ) -> None:
        super().__init__()
        self.register_buffer("bin_centers", bin_centers)
        self.cost_hurdle = float(cost_hurdle)
        self.soft_sigma = float(soft_sigma)
        
        self.lambda_focal = float(lambda_focal)
        self.lambda_dir_sign = float(lambda_dir_sign)
        self.lambda_dir_mass = float(lambda_dir_mass)
        self.lambda_rps = float(lambda_rps)
        self.lambda_reg = float(lambda_reg)
        self.lambda_dir = float(lambda_dir)
        self.lambda_uni = float(lambda_uni)
        
        self.margin_tau = float(margin_tau)
        self.focal_gamma = float(focal_gamma)
        self.chop_threshold = float(chop_threshold)
        
        self.soft_ce_loss = SoftOrdinalCrossEntropyLoss()
        self.rps_loss = RankedProbabilityScoreLoss()
        self.huber_loss = nn.SmoothL1Loss(beta=1.0)
        self.dir_loss = nn.CrossEntropyLoss()
        self.uni_loss = LogConcavityUnimodalityLoss()

    def get_cost_aware_direction_labels(self, continuous_targets: torch.Tensor) -> torch.Tensor:
        """Constructs 3-state cost-aware directional labels:
            0: Bearish (r < -cost_hurdle)
            1: Dead-Zone / Flat (-cost_hurdle <= r <= +cost_hurdle)
            2: Bullish (r > +cost_hurdle)
        """
        dir_labels = torch.ones_like(continuous_targets, dtype=torch.long)
        dir_labels[continuous_targets < -self.cost_hurdle] = 0
        dir_labels[continuous_targets > self.cost_hurdle] = 2
        return dir_labels

    def forward(
        self,
        logits: torch.Tensor,
        pred_return: torch.Tensor,
        dir_logits: Optional[torch.Tensor],
        continuous_targets: torch.Tensor,
        local_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (B, K) unnormalized ordinal bin logits.
            pred_return: (B, 1) or (B,) auxiliary continuous return prediction.
            dir_logits: Optional (B, 3) direction gate logits.
            continuous_targets: (B,) true 5-minute price change returns.
            local_sigma: Optional (B,) per-sample local volatility scaling.
        Returns:
            total_loss: scalar tensor for backward().
            loss_dict: dictionary of individual loss magnitudes for monitoring.
        """
        # 1. Continuous Gaussian Soft Targets & Simplex Probabilities
        sigma = local_sigma if local_sigma is not None else self.soft_sigma
        q_target = generate_gaussian_soft_targets(
            continuous_targets, self.bin_centers, sigma=sigma
        )
        p_pred = torch.softmax(logits, dim=-1)  # (B, K)
        
        # 2. Distribution-derived Continuous Expectation E[y|x] = sum_i p_i * mu_i
        dist_expected_return = torch.sum(p_pred * self.bin_centers.unsqueeze(0), dim=-1)  # (B,)
        
        # 3. Volatility-Weighted Directional Focal Loss on Non-Chop Moves
        num_bins = self.bin_centers.shape[0]
        neutral_idx = num_bins // 2
        p_bear = p_pred[:, :neutral_idx].sum(dim=-1).clamp(min=1e-7, max=1.0 - 1e-7)      # (B,)
        p_bull = p_pred[:, neutral_idx + 1:].sum(dim=-1).clamp(min=1e-7, max=1.0 - 1e-7)  # (B,)
        
        is_bullish = (continuous_targets > self.chop_threshold).float()
        is_bearish = (continuous_targets < -self.chop_threshold).float()
        is_actionable = is_bullish + is_bearish
        
        # Price-magnitude scaling (bigger moves matter proportionally more)
        move_weights = torch.clamp(torch.abs(continuous_targets) / 0.50, min=0.2, max=3.0)
        
        focal_bull = -((1.0 - p_bull) ** self.focal_gamma) * torch.log(p_bull) * move_weights
        focal_bear = -((1.0 - p_bear) ** self.focal_gamma) * torch.log(p_bear) * move_weights
        
        focal_sample_loss = (is_bullish * focal_bull) + (is_bearish * focal_bear)
        l_focal = focal_sample_loss.sum() / (is_actionable.sum() + 1e-6)
        
        # 4. Directional Sign Alignment Loss on Distribution Expectation
        target_sign = torch.sign(continuous_targets)
        active_dir_mask = (torch.abs(continuous_targets) > self.chop_threshold).float()
        sign_mismatch = F.relu(-target_sign * dist_expected_return + self.margin_tau) * move_weights
        l_dir_sign = (sign_mismatch * active_dir_mask).sum() / (active_dir_mask.sum() + 1e-6)
        
        # 5. Asymmetric Wrong-Side Probability Mass Penalty
        wrong_mass = (is_bullish * p_bear) + (is_bearish * p_bull)
        l_dir_mass = (wrong_mass * is_actionable).sum() / (is_actionable.sum() + 1e-6)
        
        # 6. Ordinal Distribution Losses
        l_soft_ce = self.soft_ce_loss(logits, q_target)
        l_rps = self.rps_loss(p_pred, q_target)
        l_uni = self.uni_loss(p_pred)
        
        # 7. Continuous Return Huber Loss
        if pred_return.dim() > 1:
            pred_return = pred_return.squeeze(-1)
        l_reg = self.huber_loss(pred_return, continuous_targets)
        
        # 8. Cost-Aware Direction Loss
        l_dir = torch.tensor(0.0, device=logits.device)
        if dir_logits is not None:
            dir_labels = self.get_cost_aware_direction_labels(continuous_targets)
            l_dir = self.dir_loss(dir_logits, dir_labels)
            
        # 9. Composite Objective with Dominant Directional Focal Weights
        total_loss = (
            self.lambda_focal * l_focal
            + self.lambda_dir_sign * l_dir_sign
            + self.lambda_dir_mass * l_dir_mass
            + l_soft_ce
            + self.lambda_rps * l_rps
            + self.lambda_reg * l_reg
            + self.lambda_dir * l_dir
            + self.lambda_uni * l_uni
        )
        
        metrics = {
            "loss_total": float(total_loss.detach().cpu().item()),
            "loss_focal": float(l_focal.detach().cpu().item()),
            "loss_dir_sign": float(l_dir_sign.detach().cpu().item()),
            "loss_dir_mass": float(l_dir_mass.detach().cpu().item()),
            "loss_soft_ce": float(l_soft_ce.detach().cpu().item()),
            "loss_rps": float(l_rps.detach().cpu().item()),
            "loss_reg": float(l_reg.detach().cpu().item()),
            "loss_dir": float(l_dir.detach().cpu().item()),
            "loss_uni": float(l_uni.detach().cpu().item()),
        }
        
        return total_loss, metrics
