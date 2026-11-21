"""
Phase 5 — Per-Regime Metric Decomposition.

Decomposes evaluation metrics along two axes (per the build plan, Phase 5,
point 3):

  Time-of-day buckets
  -------------------
  Uses the `g5_tod_bucket` column (or `g5_session_*` flags) already computed
  by the Phase 3 time/regime feature group.  Buckets:
    0 — Asia        (00:00–08:00 UTC)
    1 — London      (08:00–13:00 UTC)
    2 — London/NY   (13:00–17:00 UTC overlap)
    3 — New York    (17:00–21:00 UTC)
    4 — Off-hours   (all other times)

  Volatility terciles
  -------------------
  Computed per fold from `rv30` values in the *test* slice: low (0–33%),
  medium (33–66%), high (66–100%).  Fitted on test data only — they are
  evaluation labels, not features, so there is no leakage concern.

Whitepaper references: [A2] time-of-day as top predictor;
                        [D4] impact coefficient is intraday-seasonal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .metrics import (
    DirectionMetrics,
    RegressionMetrics,
    compute_direction_metrics,
    compute_regression_metrics,
    compute_qlike,
)


# ─────────────────────────────────────────────────────────────────────────────
# Bucket definitions
# ─────────────────────────────────────────────────────────────────────────────

TOD_BUCKET_NAMES: Dict[int, str] = {
    0: "Asia",
    1: "London",
    2: "Lon-NY",
    3: "New York",
    4: "Off-hours",
}

VOL_TERCILE_NAMES: Dict[int, str] = {
    0: "Low-vol",
    1: "Med-vol",
    2: "High-vol",
}


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BucketMetrics:
    """Metrics for one (time-of-day or volatility) bucket."""
    name: str
    n_samples: int
    direction: DirectionMetrics
    regression: RegressionMetrics


@dataclass
class RegimeReport:
    """
    Full per-regime breakdown for one model on one fold.

    Attributes
    ----------
    by_tod : list of BucketMetrics
        One entry per time-of-day bucket that has ≥ min_bucket_size samples.
    by_vol : list of BucketMetrics
        One entry per volatility tercile.
    """
    fold_idx: int
    by_tod: List[BucketMetrics]
    by_vol: List[BucketMetrics]

    def tod_hit_rates(self) -> Dict[str, float]:
        return {b.name: b.direction.hit_rate_tradable for b in self.by_tod}

    def vol_hit_rates(self) -> Dict[str, float]:
        return {b.name: b.direction.hit_rate_tradable for b in self.by_vol}

    def tod_mcc(self) -> Dict[str, float]:
        return {b.name: b.direction.mcc for b in self.by_tod}

    def vol_mcc(self) -> Dict[str, float]:
        return {b.name: b.direction.mcc for b in self.by_vol}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: assign volatility terciles
# ─────────────────────────────────────────────────────────────────────────────

def _assign_vol_tercile(rv30: np.ndarray) -> np.ndarray:
    """
    Assign each sample to a volatility tercile (0=low, 1=mid, 2=high)
    based on rv30 percentile thresholds computed from the same array.
    """
    tercile = np.zeros(len(rv30), dtype=int)
    valid = ~np.isnan(rv30)
    if np.sum(valid) < 3:
        return tercile
    p33 = float(np.percentile(rv30[valid], 33.3))
    p67 = float(np.percentile(rv30[valid], 66.7))
    tercile[valid & (rv30 >= p33) & (rv30 < p67)] = 1
    tercile[valid & (rv30 >= p67)] = 2
    return tercile


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_report(
    df: pl.DataFrame,
    y_pred: np.ndarray,
    fold_idx: int = 0,
    min_bucket_size: int = 20,
    tod_bucket_col: str = "g5_tod_bucket",
    rv30_col: str = "rv30",
    r30_col: str = "r30",
    rv30_pred: Optional[np.ndarray] = None,
) -> RegimeReport:
    """
    Compute per-regime metric breakdown.

    Parameters
    ----------
    df : pl.DataFrame
        Test-fold DataFrame with feature + label columns.
    y_pred : np.ndarray
        Predicted continuous returns r̂30 (same length as df).
    fold_idx : int
        Fold index (for bookkeeping).
    min_bucket_size : int
        Minimum number of samples required for a bucket to appear in results.
    tod_bucket_col : str
        Column name for time-of-day bucket (integer 0–4).
        Falls back to deriving from `g5_hour_of_day` or `timestamp` if absent.
    rv30_col : str
        Column name for forward realized volatility labels.
    r30_col : str
        Column name for forward return labels.
    rv30_pred : np.ndarray | None
        Predicted rv30 values.  If None, a zero-forecast (persistence) is used.
    """
    n = len(df)
    y_true = df[r30_col].to_numpy()
    rv_true = df[rv30_col].to_numpy() if rv30_col in df.columns else np.zeros(n)
    rv_pred = rv30_pred if rv30_pred is not None else np.full(n, float(np.nanmean(rv_true)))

    is_dead = (
        df["is_dead_zone"].to_numpy().astype(bool)
        if "is_dead_zone" in df.columns
        else np.zeros(n, dtype=bool)
    )

    # ── Time-of-day bucket assignment ─────────────────────────────────────
    if tod_bucket_col in df.columns:
        tod_arr = df[tod_bucket_col].to_numpy().astype(int)
    elif "g5_hour_of_day" in df.columns:
        hour = df["g5_hour_of_day"].to_numpy().astype(float)
        tod_arr = _hour_to_bucket(hour)
    elif "timestamp" in df.columns:
        hour = df["timestamp"].dt.hour().to_numpy().astype(float)
        tod_arr = _hour_to_bucket(hour)
    else:
        tod_arr = np.full(n, 4, dtype=int)  # everything → Off-hours

    # ── Volatility tercile assignment ─────────────────────────────────────
    vol_arr = _assign_vol_tercile(rv_true)

    # ── Build regime slices ───────────────────────────────────────────────
    by_tod = _slice_buckets(
        y_true, y_pred, rv_true, rv_pred, is_dead,
        bucket_arr=tod_arr,
        bucket_names=TOD_BUCKET_NAMES,
        min_size=min_bucket_size,
    )

    by_vol = _slice_buckets(
        y_true, y_pred, rv_true, rv_pred, is_dead,
        bucket_arr=vol_arr,
        bucket_names=VOL_TERCILE_NAMES,
        min_size=min_bucket_size,
    )

    return RegimeReport(fold_idx=fold_idx, by_tod=by_tod, by_vol=by_vol)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hour_to_bucket(hour: np.ndarray) -> np.ndarray:
    """Map UTC hour → time-of-day bucket index."""
    bucket = np.full(len(hour), 4, dtype=int)  # default: Off-hours
    bucket[(hour >= 0) & (hour < 8)] = 0       # Asia
    bucket[(hour >= 8) & (hour < 13)] = 1      # London
    bucket[(hour >= 13) & (hour < 17)] = 2     # Lon-NY
    bucket[(hour >= 17) & (hour < 21)] = 3     # New York
    return bucket


def _slice_buckets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rv_true: np.ndarray,
    rv_pred: np.ndarray,
    is_dead: np.ndarray,
    bucket_arr: np.ndarray,
    bucket_names: Dict[int, str],
    min_size: int,
) -> List[BucketMetrics]:
    results = []
    for bucket_id, bucket_name in sorted(bucket_names.items()):
        mask = bucket_arr == bucket_id
        if np.sum(mask) < min_size:
            continue

        dir_m = compute_direction_metrics(y_true[mask], y_pred[mask], is_dead[mask])
        reg_m = compute_regression_metrics(
            y_true[mask], y_pred[mask],
            rv_true[mask], rv_pred[mask],
        )
        results.append(
            BucketMetrics(name=bucket_name, n_samples=int(np.sum(mask)),
                          direction=dir_m, regression=reg_m)
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across folds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_regime_reports(reports: List[RegimeReport]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-fold regime reports into mean metrics per bucket.

    Returns
    -------
    dict
        {"tod": {"Asia": {"hit_rate": ..., "mcc": ...}, ...},
         "vol": {"Low-vol": {...}, ...}}
    """
    def _agg(reports_list: List[List[BucketMetrics]]) -> Dict[str, Dict[str, List[float]]]:
        acc: Dict[str, Dict[str, List[float]]] = {}
        for fold_buckets in reports_list:
            for b in fold_buckets:
                if b.name not in acc:
                    acc[b.name] = {"hit_rate": [], "mcc": [], "oos_r2": [], "qlike": []}
                acc[b.name]["hit_rate"].append(b.direction.hit_rate_tradable)
                acc[b.name]["mcc"].append(b.direction.mcc)
                acc[b.name]["oos_r2"].append(b.regression.oos_r2)
                acc[b.name]["qlike"].append(b.regression.qlike_rv30)
        return acc

    def _mean(acc: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        return {
            name: {k: float(np.mean(v)) for k, v in metrics.items()}
            for name, metrics in acc.items()
        }

    tod_acc = _agg([r.by_tod for r in reports])
    vol_acc = _agg([r.by_vol for r in reports])

    return {"tod": _mean(tod_acc), "vol": _mean(vol_acc)}
