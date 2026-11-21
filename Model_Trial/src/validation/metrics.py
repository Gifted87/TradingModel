"""
Phase 5 — Full Metric Suite.

All functions are pure (stateless), NumPy-only.

Metric groups per the build plan (Phase 5, point 2):

    Direction
    ---------
    - MCC (Matthews Correlation Coefficient) on dead-zoned direction
    - Balanced accuracy on dead-zoned direction
    - Hit-rate (overall and conditioned on tradable |r30| >= tau)

    Regression
    ----------
    - MAE on log-magnitude (mag30)
    - QLIKE on rv30 (Gaussian quasi-likelihood, a strictly proper scoring
      rule for volatility forecasters, Patton 2011)
    - OOS R² relative to persistence (r̂=0)

    Quantile Calibration
    --------------------
    - Reliability: for each predicted quantile q_k, compute the empirical
      coverage P(|r30| <= q̂_k) and compare to the nominal level k.
    - Returns mean calibration error (MCE) across the quantile grid.

Whitepaper references: [C2] accuracy ≠ profit; [E5] log-space magnitude;
                        [E1] RV direction forecastable; [A7] V-shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Scalar metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_mcc(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """Matthews Correlation Coefficient.  Inputs must be 0/1 integer arrays."""
    tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = float(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def compute_balanced_accuracy(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """Balanced accuracy = (sensitivity + specificity) / 2."""
    tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = float(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.5
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.5
    return float((sensitivity + specificity) / 2.0)


def compute_qlike(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """
    Gaussian QLIKE for log-scale volatility estimates.

    QLIKE = mean( exp(u - v) - (u - v) - 1 )  where u=true, v=pred.
    Convex and strictly proper (Patton 2011).  Clipped to avoid overflow.
    """
    diff = np.clip(y_true_log - y_pred_log, -10.0, 10.0)
    return float(np.mean(np.exp(diff) - diff - 1.0))


def compute_oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """OOS R² = 1 - MSE(model) / MSE(persistence).  Persistence baseline = r̂=0."""
    mse_model = float(np.mean((y_true - y_pred) ** 2))
    mse_persistence = float(np.mean(y_true ** 2))   # r̂=0 → residual = y_true
    return float(1.0 - mse_model / (mse_persistence + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Directional metric bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DirectionMetrics:
    """All direction-related metrics for one fold / slice."""
    n_total: int
    n_tradable: int          # samples with |r30| >= tau (outside dead-zone)
    hit_rate_all: float      # directional accuracy, all samples
    hit_rate_tradable: float # directional accuracy, tradable only
    mcc: float               # MCC on tradable samples
    balanced_acc: float      # balanced accuracy on tradable samples


def compute_direction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_dead_zone: Optional[np.ndarray] = None,
) -> DirectionMetrics:
    """
    Compute the full direction metric bundle.

    Parameters
    ----------
    y_true : np.ndarray
        True continuous returns (r30).
    y_pred : np.ndarray
        Predicted continuous returns (r̂30); direction = sign(r̂30).
    is_dead_zone : np.ndarray | None
        Boolean mask; True where |r30| < tau (spread threshold).
        If None, all samples treated as tradable.
    """
    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)

    hit_all = float(np.mean(dir_true == dir_pred))

    if is_dead_zone is None:
        tradable_mask = np.ones(len(y_true), dtype=bool)
    else:
        tradable_mask = ~is_dead_zone.astype(bool)

    n_tradable = int(np.sum(tradable_mask))

    if n_tradable > 0:
        hit_trad = float(np.mean(dir_true[tradable_mask] == dir_pred[tradable_mask]))
        y_t_bin = (dir_true[tradable_mask] > 0).astype(int)
        y_p_bin = (dir_pred[tradable_mask] > 0).astype(int)
        mcc = compute_mcc(y_t_bin, y_p_bin)
        bal_acc = compute_balanced_accuracy(y_t_bin, y_p_bin)
    else:
        hit_trad = hit_all
        mcc = 0.0
        bal_acc = 0.5

    return DirectionMetrics(
        n_total=len(y_true),
        n_tradable=n_tradable,
        hit_rate_all=hit_all,
        hit_rate_tradable=hit_trad,
        mcc=mcc,
        balanced_acc=bal_acc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regression metric bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionMetrics:
    """Regression metrics for one fold / slice."""
    rmse_r30: float
    mae_r30: float
    oos_r2: float
    mae_mag30: float         # MAE on log-magnitude [E5]
    qlike_rv30: float        # QLIKE on log-RV [E1]
    rmse_rv30: float
    mae_rv30: float


def compute_regression_metrics(
    y_r30_true: np.ndarray,
    y_r30_pred: np.ndarray,
    y_rv30_true: np.ndarray,
    y_rv30_pred: np.ndarray,
    y_mag30_true: Optional[np.ndarray] = None,
    y_mag30_pred: Optional[np.ndarray] = None,
) -> RegressionMetrics:
    """Compute the full regression metric bundle."""
    r30_resid = y_r30_true - y_r30_pred
    rmse = float(np.sqrt(np.mean(r30_resid ** 2)))
    mae_r = float(np.mean(np.abs(r30_resid)))
    r2 = compute_oos_r2(y_r30_true, y_r30_pred)

    rv_resid = y_rv30_true - y_rv30_pred
    qlike = compute_qlike(y_rv30_true, y_rv30_pred)
    rmse_rv = float(np.sqrt(np.mean(rv_resid ** 2)))
    mae_rv = float(np.mean(np.abs(rv_resid)))

    if y_mag30_true is not None and y_mag30_pred is not None:
        mae_mag = float(np.mean(np.abs(y_mag30_true - y_mag30_pred)))
    else:
        mae_mag = float("nan")

    return RegressionMetrics(
        rmse_r30=rmse,
        mae_r30=mae_r,
        oos_r2=r2,
        mae_mag30=mae_mag,
        qlike_rv30=qlike,
        rmse_rv30=rmse_rv,
        mae_rv30=mae_rv,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quantile calibration
# ─────────────────────────────────────────────────────────────────────────────

QUANTILE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class QuantileCalibration:
    """Quantile calibration reliability report."""
    levels: List[float]              # nominal quantile levels
    empirical_coverage: List[float]  # actual P(|r30| <= q̂_k)
    calibration_errors: List[float]  # |nominal - empirical| per level
    mean_calibration_error: float    # MCE across the grid


def compute_quantile_calibration(
    y_true_abs: np.ndarray,
    q_predictions: Dict[float, np.ndarray],
) -> QuantileCalibration:
    """
    Reliability check for quantile forecasts of |r30|.

    Parameters
    ----------
    y_true_abs : np.ndarray
        Realised absolute returns |r30|.
    q_predictions : dict
        Maps quantile level (e.g. 0.1) → array of predicted quantile values.
        Keys must be a subset of QUANTILE_LEVELS.
    """
    levels = sorted(q_predictions.keys())
    empirical = []
    errors = []

    for level in levels:
        q_hat = q_predictions[level]
        coverage = float(np.mean(y_true_abs <= q_hat))
        empirical.append(coverage)
        errors.append(abs(level - coverage))

    mce = float(np.mean(errors)) if errors else float("nan")

    return QuantileCalibration(
        levels=levels,
        empirical_coverage=empirical,
        calibration_errors=errors,
        mean_calibration_error=mce,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Combined fold metrics & Confidence Tier Slicing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    """All metrics for a single walk-forward fold."""
    fold_idx: int
    n_train: int
    n_test: int
    train_start_ts: str
    train_end_ts: str
    test_start_ts: str
    test_end_ts: str

    direction: DirectionMetrics
    regression: RegressionMetrics
    quantile_cal: Optional[QuantileCalibration] = None
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConfidenceTierMetrics:
    """Performance metrics stratified by model confidence / conviction levels."""
    tier_name: str
    min_conf: float
    max_conf: float
    n_samples: int
    pct_total: float
    hit_rate: float
    tradable_hit_rate: float
    mcc: float
    mean_predicted_mag: float
    mean_realized_mag: float
    net_expectancy_log: float
    net_expectancy_spreads: float
    profit_factor: float
    sharpe_ratio: float


def compute_confidence_tier_breakdown(
    y_true: np.ndarray,
    r_hat: np.ndarray,
    confidence: np.ndarray,
    spreads: np.ndarray,
    is_dead_zone: Optional[np.ndarray] = None,
    econ_k: float = 1.5,
) -> List[ConfidenceTierMetrics]:
    """
    Stratify out-of-sample predictions by confidence levels and calculate
    directional, magnitude, and net economic metrics per tier.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    r_hat = np.asarray(r_hat, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    spreads = np.asarray(spreads, dtype=np.float64)
    
    n_total = len(y_true)
    if n_total == 0:
        return []

    tiers_def = [
        ("All Predictions", 0.0, 1.0, None),
        ("Low Conviction", 0.0, 0.35, None),
        ("Moderate Conviction", 0.35, 0.65, None),
        ("High Conviction", 0.65, 0.85, None),
        ("Elite Conviction (Top)", 0.85, 1.0, None),
        (f"Economic Gate (|r̂|>{econ_k}×spread, conf≥0.5)", 0.50, 1.0, "econ_gate"),
    ]

    results: List[ConfidenceTierMetrics] = []

    for name, c_min, c_max, special in tiers_def:
        if special == "econ_gate":
            mask = (confidence >= c_min) & (np.abs(r_hat) >= (econ_k * spreads))
        else:
            mask = (confidence >= c_min) & (confidence < c_max if c_max < 1.0 else confidence <= c_max)

        n_sub = int(np.sum(mask))
        if n_sub == 0:
            results.append(ConfidenceTierMetrics(
                tier_name=name, min_conf=c_min, max_conf=c_max,
                n_samples=0, pct_total=0.0, hit_rate=0.0, tradable_hit_rate=0.0,
                mcc=0.0, mean_predicted_mag=0.0, mean_realized_mag=0.0,
                net_expectancy_log=0.0, net_expectancy_spreads=0.0,
                profit_factor=0.0, sharpe_ratio=0.0,
            ))
            continue

        y_sub = y_true[mask]
        r_sub = r_hat[mask]
        sp_sub = spreads[mask]
        dead_sub = is_dead_zone[mask] if is_dead_zone is not None else None

        dir_m = compute_direction_metrics(y_sub, r_sub, dead_sub)
        pred_sign = np.sign(r_sub)
        pnl_net = pred_sign * y_sub - sp_sub
        gains = pnl_net[pnl_net > 0].sum()
        losses = np.abs(pnl_net[pnl_net < 0].sum())
        pf = float(gains / losses) if losses > 1e-12 else (float("inf") if gains > 0 else 0.0)

        mean_pnl = float(np.mean(pnl_net))
        std_pnl = float(np.std(pnl_net))
        sharpe = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(252 * 288) if std_pnl > 1e-12 else 0.0
        avg_spread = float(np.mean(sp_sub)) if len(sp_sub) else 1e-4
        exp_spreads = mean_pnl / (avg_spread + 1e-9)

        results.append(ConfidenceTierMetrics(
            tier_name=name,
            min_conf=c_min,
            max_conf=c_max,
            n_samples=n_sub,
            pct_total=(n_sub / n_total) * 100.0,
            hit_rate=dir_m.hit_rate_all * 100.0,
            tradable_hit_rate=dir_m.hit_rate_tradable * 100.0,
            mcc=dir_m.mcc,
            mean_predicted_mag=float(np.mean(np.abs(r_sub))),
            mean_realized_mag=float(np.mean(np.abs(y_sub))),
            net_expectancy_log=mean_pnl,
            net_expectancy_spreads=exp_spreads,
            profit_factor=pf,
            sharpe_ratio=sharpe,
        ))

    return results
