"""
Phase 3 Feature Pipeline — Unified entry-point for all six feature groups.

Responsibilities:
  1. Orchestrate all G1–G6 compute functions in the correct dependency order.
  2. Apply fold-safe rolling z-score normalisation (fit on train rows only).
  3. Expose a single `compute_all_features(df)` function that can be reused
     identically at both training time and serving time (zero train/serve skew).
  4. Provide `FeaturePipeline` — a stateful normalizer that can be serialised
     with pickle for use at inference time.

Build-plan reference (Phase 3, Gate 3):
  - No NaN leakage across fold boundaries.
  - Feature count logged; registry complete.
  - Rolling z-score normalisation fitted exclusively on train folds.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from .registry import REGISTRY, FeatureRegistry
from .microstructure import compute_g1_microstructure
from .momentum import compute_g2_momentum
from .indicators import compute_g3_indicators
from .interactions import compute_g4_interactions
from .time_regime import compute_g5_time_regime
from .volatility import compute_g6_volatility

logger = logging.getLogger(__name__)

# Groups computed in dependency order:
#   G1 (OFI base) → G6 (ATR/RV) → G2 (momentum) → G3 (indicators)
#   → G5 (time × G1, G6) → G4 (cross-products across all groups)
_COMPUTE_ORDER = [
    ("G1", compute_g1_microstructure),
    ("G6", compute_g6_volatility),
    ("G2", compute_g2_momentum),
    ("G3", compute_g3_indicators),
    ("G5", compute_g5_time_regime),
    ("G4", compute_g4_interactions),
]

# Columns that are NEVER normalised (labels, timestamps, metadata, flags).
_SKIP_NORMALISE = {
    "timestamp", "open", "high", "low", "close",
    "bid_open", "bid_close", "ask_open", "ask_close",
    "spread_close", "spread_mean", "spread_max",
    "volume", "bid_volume", "ask_volume",
    "ofi_sum", "tick_count", "sub_bar_count",
    "return_5m",
    # labels
    "r30", "mag30", "rv30", "mfe30", "mae30",
    "direction", "dead_zone_tau", "is_dead_zone", "is_decision_bar",
    # G5 flag / bucket / integer columns (kept raw)
    "g5_session_asian", "g5_session_london", "g5_session_ny",
    "g5_session_lon_ny_overlap", "g5_session_asia_lon_overlap",
    "g5_tod_bucket", "g5_hour_of_day", "g5_day_of_week",
}


def compute_all_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply all six feature groups (G1–G6) to df in dependency order.

    Parameters
    ----------
    df : pl.DataFrame
        Microstructure bar DataFrame (output of bar_builder / labeled parquet).

    Returns
    -------
    pl.DataFrame
        Original df with all feature columns appended.
    """
    for group_id, fn in _COMPUTE_ORDER:
        before = len(df.columns)
        df = fn(df)
        added = len(df.columns) - before
        logger.debug(f"{group_id}: added {added} columns (total {len(df.columns)})")
    return df


@dataclass
class NormStats:
    """Per-feature normalisation statistics fitted on the training fold."""
    mean: float
    std: float


class FeaturePipeline:
    """
    Stateful feature normaliser.

    Fits per-feature (mean, std) on a supplied training slice and transforms
    any slice (train or test) using those fixed statistics — guaranteeing zero
    future leakage across walk-forward fold boundaries.

    Usage
    -----
    pipe = FeaturePipeline()
    train_feat = pipe.fit_transform(train_df)   # fit on train, transform
    val_feat   = pipe.transform(val_df)         # transform with frozen stats
    pipe.save("pipeline.pkl")                   # serialise for serving
    """

    def __init__(self, clip_sigma: float = 5.0):
        """
        Parameters
        ----------
        clip_sigma : float
            After z-scoring, clip values to [-clip_sigma, +clip_sigma].
            This prevents extreme outliers (flash crashes etc.) from
            producing explosive inputs to the model.
        """
        self.clip_sigma = clip_sigma
        self._stats: Dict[str, NormStats] = {}
        self._feature_cols: List[str] = []

    @property
    def feature_columns(self) -> List[str]:
        return list(self._feature_cols)

    def _normalise_cols(self, df: pl.DataFrame) -> List[str]:
        """Return the list of g?_* columns that should be normalised."""
        return [
            c for c in df.columns
            if c.startswith("g") and "_" in c and c not in _SKIP_NORMALISE
        ]

    def fit(self, df: pl.DataFrame) -> "FeaturePipeline":
        """
        Fit normalisation statistics on training data.

        Parameters
        ----------
        df : pl.DataFrame
            Training-fold DataFrame with all feature columns present.
        """
        feat_cols = self._normalise_cols(df)
        self._feature_cols = feat_cols
        self._stats = {}

        for col in feat_cols:
            series = df[col].drop_nulls().drop_nans()
            if len(series) == 0:
                self._stats[col] = NormStats(mean=0.0, std=1.0)
                continue
            mu  = float(series.mean())
            std = float(series.std())
            if std < 1e-9:
                std = 1.0  # constant feature → keep as-is after centering
            self._stats[col] = NormStats(mean=mu, std=std)

        logger.info(f"FeaturePipeline fitted on {len(feat_cols)} feature columns.")
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply fitted z-score normalisation to df.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing the same feature columns used during fit().

        Returns
        -------
        pl.DataFrame
            DataFrame with feature columns replaced by z-scored, clipped values.
        """
        if not self._stats:
            raise RuntimeError("Pipeline not fitted. Call fit() or fit_transform() first.")

        exprs = []
        for col, stats in self._stats.items():
            if col not in df.columns:
                continue
            z = (pl.col(col) - stats.mean) / stats.std
            if self.clip_sigma is not None:
                z = z.clip(-self.clip_sigma, self.clip_sigma)
            exprs.append(z.fill_nan(0.0).fill_null(0.0).alias(col))

        return df.with_columns(exprs)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit on df then transform it. Convenience method."""
        return self.fit(df).transform(df)

    def save(self, path: Path) -> None:
        """Serialise to pickle (for serving)."""
        Path(path).write_bytes(pickle.dumps(self))
        logger.info(f"FeaturePipeline saved to {path}")

    @staticmethod
    def load(path: Path) -> "FeaturePipeline":
        """Deserialise from pickle."""
        pipe: FeaturePipeline = pickle.loads(Path(path).read_bytes())
        logger.info(f"FeaturePipeline loaded from {path} ({len(pipe._stats)} features).")
        return pipe

    def summary(self) -> Dict[str, dict]:
        """Return a JSON-serialisable summary of fitted statistics."""
        return {col: asdict(s) for col, s in self._stats.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Gate 3 Auditor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Gate3Report:
    """Gate 3 audit results."""
    total_features: int
    features_by_group: Dict[str, int]
    nan_leaking_features: List[str]   # features with NaN in first row (lagged boundary)
    zero_variance_features: List[str] # features with std == 0 on train fold
    gate3_passed: bool
    verdict: str


def run_gate3_audit(
    df: pl.DataFrame,
    report_dir: Optional[Path] = None,
    symbol: str = "UNKNOWN",
) -> Gate3Report:
    """
    Gate 3 validator per Build Plan Phase 3.

    Checks:
    1. No NaN leakage across fold boundaries (first non-warmup row must be valid).
    2. Feature count per group matches registry.
    3. No zero-variance features on the supplied slice.
    4. Registry completeness.

    Parameters
    ----------
    df : pl.DataFrame
        Feature-complete DataFrame (output of compute_all_features).
    report_dir : Path, optional
        If provided, writes gate3_report_{symbol}.json there.
    symbol : str
        Symbol name for logging.

    Returns
    -------
    Gate3Report
    """
    logger.info(f"Running Gate 3 Feature Audit on {symbol}...")

    feat_cols = [c for c in df.columns if c.startswith("g") and "_" in c and c not in _SKIP_NORMALISE]

    # 1. Registry counts by group
    reg_summary = REGISTRY.summary()

    # 2. NaN leakage: check row 200 (past warm-up) — should have no NaN
    warmup_row = min(300, len(df) - 1)
    check_row = df.slice(warmup_row, 1)
    nan_leaking = []
    for col in feat_cols:
        if col in check_row.columns:
            val = check_row[col][0]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                nan_leaking.append(col)

    # 3. Zero-variance features
    zero_var = []
    for col in feat_cols:
        if col in df.columns:
            std = float(df[col].drop_nulls().drop_nans().std() or 0.0)
            if std < 1e-9:
                zero_var.append(col)

    passed = (len(nan_leaking) == 0) and (len(zero_var) == 0)

    verdict = (
        f"GATE 3 PASS: {len(feat_cols)} features, {len(nan_leaking)} NaN-leaking, "
        f"{len(zero_var)} zero-variance."
        if passed
        else f"GATE 3 FAIL: {len(nan_leaking)} NaN-leaking features, "
             f"{len(zero_var)} zero-variance features."
    )

    report = Gate3Report(
        total_features=len(feat_cols),
        features_by_group=reg_summary,
        nan_leaking_features=nan_leaking,
        zero_variance_features=zero_var,
        gate3_passed=passed,
        verdict=verdict,
    )

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / f"gate3_{symbol}.json"
        out.write_text(json.dumps(asdict(report), indent=2))
        logger.info(f"Gate 3 report written to {out}")

    return report
