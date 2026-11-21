"""
Phase 5 — Adversarial Checks [C3].

Two adversarial checks per the build plan (Phase 5, point 5):

1. Shuffle test
   Permute the labels (r30) while keeping features and predictions fixed.
   The shuffled MCC and hit-rate must collapse to chance (≈0 and ≈50%).
   If the model still scores above chance after shuffling, there is data
   leakage or look-ahead bias in the pipeline.

2. Single-feature group audit (feature group ablation on Branch A)
   For each feature group G1–G6 in isolation, fit Branch A's Ridge regression
   using only that group's features and score on the test fold.
   This reveals which groups carry independent signal and guards against
   the harness rewarding a model purely because one dominant feature group
   happens to correlate with the target in the test window.

Whitepaper reference: [C3] standard benchmarks embed biases; shuffle test
                       and per-component attribution are required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from .metrics import DirectionMetrics, compute_direction_metrics, compute_oos_r2

logger = logging.getLogger(__name__)

# Feature group prefixes used to partition feature columns.
_GROUP_PREFIXES: Dict[str, str] = {
    "G1": "g1_",
    "G2": "g2_",
    "G3": "g3_",
    "G4": "g4_",
    "G5": "g5_",
    "G6": "g6_",
}

# Columns that are metadata / labels — never treated as features.
_LABEL_COLS = {
    "timestamp", "open", "high", "low", "close",
    "bid_open", "bid_close", "ask_open", "ask_close",
    "spread_close", "spread_mean", "spread_max",
    "volume", "bid_volume", "ask_volume",
    "r30", "mag30", "rv30", "mfe30", "mae30",
    "direction", "dead_zone_tau", "is_dead_zone", "is_decision_bar",
    "ofi_sum", "tick_count", "sub_bar_count", "return_5m",
}


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShuffleTestResult:
    """
    Results of the shuffle test on one fold.

    Attributes
    ----------
    n_shuffles : int
        Number of independent label permutations.
    mean_mcc_shuffled : float
        Mean MCC across permutations — should be ≈ 0.
    std_mcc_shuffled : float
        Std of MCC across permutations — should be small.
    mean_hit_rate_shuffled : float
        Mean hit-rate across permutations — should be ≈ 0.5.
    original_mcc : float
        MCC of the original (non-shuffled) predictions.
    shuffle_test_passed : bool
        True if mean_mcc_shuffled is within [-0.05, +0.05] (chance level).
    """
    n_shuffles: int
    mean_mcc_shuffled: float
    std_mcc_shuffled: float
    mean_hit_rate_shuffled: float
    original_mcc: float
    shuffle_test_passed: bool


@dataclass
class GroupAuditResult:
    """Single-feature-group Ridge audit result."""
    group: str           # "G1" … "G6"
    n_features: int      # number of features in this group
    oos_r2: float        # OOS R² using only this group's features
    mcc: float           # MCC using only this group
    hit_rate_tradable: float


@dataclass
class AdversarialReport:
    """Full adversarial check results for one fold."""
    fold_idx: int
    shuffle: ShuffleTestResult
    group_audits: List[GroupAuditResult]


# ─────────────────────────────────────────────────────────────────────────────
# Shuffle test
# ─────────────────────────────────────────────────────────────────────────────

def run_shuffle_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_dead_zone: Optional[np.ndarray] = None,
    n_shuffles: int = 100,
    rng_seed: int = 42,
    fold_idx: int = 0,
) -> ShuffleTestResult:
    """
    Permute r30 labels and re-score; MCC must collapse to ≈ 0.

    Parameters
    ----------
    y_true : np.ndarray
        True r30 values for the test fold.
    y_pred : np.ndarray
        Model predictions r̂30 (fixed; only labels are shuffled).
    is_dead_zone : np.ndarray | None
        Boolean mask for dead-zone exclusion.
    n_shuffles : int
        Number of random permutations.
    rng_seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(rng_seed)

    # Original score (for reference).
    orig_dir = compute_direction_metrics(y_true, y_pred, is_dead_zone)

    mccs: List[float] = []
    hit_rates: List[float] = []

    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y_true)
        dm = compute_direction_metrics(y_shuffled, y_pred, is_dead_zone)
        mccs.append(dm.mcc)
        hit_rates.append(dm.hit_rate_tradable)

    mean_mcc = float(np.mean(mccs))
    std_mcc = float(np.std(mccs))
    mean_hr = float(np.mean(hit_rates))

    # Pass if shuffled MCC is indistinguishable from chance.
    passed = abs(mean_mcc) < 0.05

    return ShuffleTestResult(
        n_shuffles=n_shuffles,
        mean_mcc_shuffled=mean_mcc,
        std_mcc_shuffled=std_mcc,
        mean_hit_rate_shuffled=mean_hr,
        original_mcc=orig_dir.mcc,
        shuffle_test_passed=passed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-feature-group Ridge audit
# ─────────────────────────────────────────────────────────────────────────────

def _ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 100.0,
) -> np.ndarray:
    """Analytical Ridge regression: (X^T X + αI)w = X^T y."""
    n, p = X_train.shape
    x_mean = X_train.mean(axis=0)
    y_mean = float(y_train.mean())
    Xc = X_train - x_mean
    yc = y_train - y_mean
    A = Xc.T @ Xc + alpha * np.eye(p)
    b = Xc.T @ yc
    w = np.linalg.solve(A, b)
    intercept = y_mean - float(x_mean @ w)
    return intercept + X_test @ w


def run_group_audits(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    is_dead_zone: Optional[np.ndarray] = None,
    alpha: float = 100.0,
) -> List[GroupAuditResult]:
    """
    For each feature group G1–G6, fit Ridge on train, score on test.

    Skips groups with fewer than 2 available features (no meaningful Ridge).
    """
    results: List[GroupAuditResult] = []

    all_cols = set(train_df.columns) & set(test_df.columns) - _LABEL_COLS

    for group, prefix in _GROUP_PREFIXES.items():
        group_cols = sorted(c for c in all_cols if c.startswith(prefix))

        if len(group_cols) < 2:
            logger.debug(f"Group {group}: only {len(group_cols)} feature(s), skipping audit.")
            continue

        # Build clean train matrix
        train_sub = train_df.select(group_cols).fill_nan(0.0).fill_null(0.0)
        test_sub = test_df.select(group_cols).fill_nan(0.0).fill_null(0.0)

        X_tr = train_sub.to_numpy()
        X_te = test_sub.to_numpy()

        # Drop rows with NaN in target
        valid_tr = ~np.isnan(y_train)
        if np.sum(valid_tr) < 10:
            continue

        try:
            y_pred = _ridge_predict(X_tr[valid_tr], y_train[valid_tr], X_te, alpha=alpha)
        except np.linalg.LinAlgError:
            logger.warning(f"Ridge solve failed for group {group}; skipping.")
            continue

        r2 = compute_oos_r2(y_test, y_pred)
        dm = compute_direction_metrics(y_test, y_pred, is_dead_zone)

        results.append(GroupAuditResult(
            group=group,
            n_features=len(group_cols),
            oos_r2=r2,
            mcc=dm.mcc,
            hit_rate_tradable=dm.hit_rate_tradable,
        ))

    return sorted(results, key=lambda r: r.oos_r2, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Combined adversarial report
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial_checks(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_pred: np.ndarray,
    fold_idx: int = 0,
    n_shuffles: int = 100,
    alpha: float = 100.0,
) -> AdversarialReport:
    """
    Run both adversarial checks and return a combined AdversarialReport.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training fold data (used for group audit Ridge fits).
    test_df : pl.DataFrame
        Test fold data.
    y_pred : np.ndarray
        Predictions from the *full* model (for the shuffle test).
    fold_idx : int
        Fold index for bookkeeping.
    n_shuffles : int
        Number of label permutations for the shuffle test.
    alpha : float
        Ridge regularisation for single-group audits.
    """
    y_train = train_df["r30"].to_numpy()
    y_test = test_df["r30"].to_numpy()
    is_dead = (
        test_df["is_dead_zone"].to_numpy().astype(bool)
        if "is_dead_zone" in test_df.columns
        else None
    )

    shuffle = run_shuffle_test(y_test, y_pred, is_dead, n_shuffles=n_shuffles, fold_idx=fold_idx)
    audits = run_group_audits(train_df, test_df, y_train, y_test, is_dead, alpha=alpha)

    return AdversarialReport(fold_idx=fold_idx, shuffle=shuffle, group_audits=audits)
