"""
Phase 7 — Ablation Battery Orchestration Engine.

REWRITE (v2) — the Phase 7 compute fix:
  OLD: each of the 8 variants re-ran the full harness per fold — refitting
       FeaturePipeline, Branch A, and rebuilding feature matrices 8× per fold,
       on top of the per-sample collation burn and total-loss early stopping.
  NEW: loop order is inverted. Per fold, a FoldCache is built ONCE (pipeline,
       Branch A, target scaler, feature matrices, spreads, masks) and shared
       by every variant. Variants differ only in column subsets, lookback,
       head wiring, and target transform — exactly what the whitepaper ablates.

  Keep/kill rules now follow the build plan literally: "Kill group if val
  loss unchanged" — each variant's best validation RETURN loss (direction
  task) is recorded and used alongside test MCC with a tolerance margin,
  instead of raw MCC comparisons that flip on noise.

  Gate 7 (C4–C7): surviving configuration frozen; every killed group is a
  documented finding about this market, not a failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from ..baselines.linear_core import LinearCoreBranchA
from ..features.pipeline import FeaturePipeline, _SKIP_NORMALISE
from ..models.dataset import TargetScaler
from ..models.trainer import TrainerConfig
from ..validation.economic_gate import evaluate_economic_gate, extract_spreads
from ..validation.metrics import compute_direction_metrics, compute_regression_metrics
from ..validation.walk_forward import generate_splits
from .config import ABLATION_SUITE
from .variants import FoldArrays, VariantSession

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result containers (fields preserved for report.py compatibility)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantComparison:
    name: str
    condition_tag: str
    description: str
    mean_mcc: float
    mean_hit_rate_tradable: float
    mean_oos_r2: float
    mean_qlike_rv30: float
    mean_mae_mag30: float
    delta_mcc_vs_baseline: float
    delta_qlike_vs_baseline: float
    keep_decision: str
    justification: str
    mean_val_return_loss: float = float("nan")   # plan-literal keep/kill signal


@dataclass
class BatterySummaryReport:
    symbol: str
    baseline_variant: str
    variants: List[VariantComparison]
    condition_c4_verified: bool
    condition_c5_verified: bool
    condition_c6_verified: bool
    condition_c7_verified: bool
    optimal_lookback: int
    frozen_config_name: str
    gate7_passed: bool
    verdict: str


@dataclass
class FoldCache:
    """Everything every variant needs for one fold — built exactly once."""
    scaler: TargetScaler
    train: FoldArrays
    test: FoldArrays
    test_is_dead: Optional[np.ndarray]
    test_spreads: np.ndarray
    test_is_decision: Optional[np.ndarray]


@dataclass
class _FoldRecord:
    mcc: float
    hit_rate_tradable: float
    oos_r2: float
    qlike_rv30: float
    mae_mag30: float
    val_return_loss: float
    net_pnl_positive: bool


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class AblationBatteryRunner:
    def __init__(
        self,
        train_bars: int = 80_000,
        test_bars: int = 20_000,
        step_bars: int = 20_000,
        embargo_bars: int = 30,
        epochs: int = 8,
        batch_size: int = 256,
        variants_to_run: Optional[List[str]] = None,
        decision_bars_only: bool = True,
        economic_gate_k: float = 1.5,
        keep_tol: float = 0.01,               # 1% relative tolerance for keep/kill
        max_folds: Optional[int] = None,      # smoke-test knob
        mlflow_experiment: Optional[str] = None,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.embargo_bars = embargo_bars
        self.epochs = epochs
        self.batch_size = batch_size
        self.variant_keys = variants_to_run or list(ABLATION_SUITE.keys())
        self.decision_bars_only = decision_bars_only
        self.economic_gate_k = economic_gate_k
        self.keep_tol = keep_tol
        self.max_folds = max_folds
        self.mlflow_experiment = mlflow_experiment

    # ── fold cache (built once per fold, shared by all variants) ───────────

    def _build_fold_cache(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> FoldCache:
        pipeline = FeaturePipeline(clip_sigma=5.0).fit(train_df)
        branch_a = LinearCoreBranchA(alpha=100.0).fit(train_df)

        feature_cols = [
            c for c in train_df.columns
            if c.startswith("g") and "_" in c and c not in _SKIP_NORMALISE
        ]
        g6_cols = [c for c in feature_cols if c.startswith("g6_")]

        def _arrays(part: pl.DataFrame) -> FoldArrays:
            norm = pipeline.transform(part)
            feats = (norm.select(feature_cols).fill_nan(0.0).fill_null(0.0)
                     .to_numpy().astype(np.float32))
            g6 = (norm.select(g6_cols).fill_nan(0.0).fill_null(0.0)
                  .to_numpy().astype(np.float32))
            n = len(part)
            return FoldArrays(
                features=feats, feature_cols=feature_cols,
                r_hat_lin=branch_a.predict(part),
                g6=g6, g6_cols=g6_cols,
                r30=part["r30"].to_numpy() if "r30" in part.columns else np.full(n, np.nan),
                rv30=part["rv30"].to_numpy() if "rv30" in part.columns else np.full(n, np.nan),
                mag30=part["mag30"].to_numpy() if "mag30" in part.columns else np.full(n, np.nan),
            )

        train_arrays = _arrays(train_df)
        test_arrays = _arrays(test_df)
        scaler = TargetScaler.fit(train_arrays.r30, train_arrays.rv30, train_arrays.mag30)

        return FoldCache(
            scaler=scaler,
            train=train_arrays,
            test=test_arrays,
            test_is_dead=(test_df["is_dead_zone"].to_numpy().astype(bool)
                          if "is_dead_zone" in test_df.columns else None),
            test_spreads=extract_spreads(test_df),
            test_is_decision=(test_df["is_decision_bar"].to_numpy().astype(bool)
                              if "is_decision_bar" in test_df.columns else None),
        )

    # ── per-variant-per-fold scoring ───────────────────────────────────────

    def _score_variant(self, key: str, cache: FoldCache, fold_idx: int, symbol: str) -> _FoldRecord:
        cfg = ABLATION_SUITE[key]
        trainer_cfg = TrainerConfig(
            epochs=self.epochs, batch_size=self.batch_size,
            patience=3, selection_metric="return",
            mlflow_experiment=self.mlflow_experiment,
            mlflow_run_name=f"{symbol}/{key}/fold{fold_idx}" if self.mlflow_experiment else None,
            fold_id=f"fold{fold_idx}",
        )
        session = VariantSession(cfg, trainer_cfg)
        session.fit_arrays(cache.train, cache.scaler)
        preds = session.predict_arrays(cache.test)

        mask = np.isfinite(cache.test.r30) & np.isfinite(cache.test.rv30)
        if self.decision_bars_only and cache.test_is_decision is not None:
            mask &= cache.test_is_decision
        if mask.sum() < 20:
            raise RuntimeError(f"fold {fold_idx}: only {mask.sum()} scorable rows for {key}")

        y = cache.test.r30[mask]
        rv = cache.test.rv30[mask]
        r_hat = preds["r_hat"][mask]
        rv_hat = preds["rv_hat"][mask]
        is_dead = cache.test_is_dead[mask] if cache.test_is_dead is not None else None

        dir_m = compute_direction_metrics(y, r_hat, is_dead)
        mag_true = cache.test.mag30[mask]
        mag_pred = np.log(np.abs(r_hat) + 1e-7)
        reg_m = compute_regression_metrics(y, r_hat, rv, rv_hat, mag_true, mag_pred)
        gate = evaluate_economic_gate(y, r_hat, cache.test_spreads[mask],
                                      fold_idx=fold_idx, k=self.economic_gate_k)

        return _FoldRecord(
            mcc=dir_m.mcc,
            hit_rate_tradable=dir_m.hit_rate_tradable,
            oos_r2=reg_m.oos_r2,
            qlike_rv30=reg_m.qlike_rv30,
            mae_mag30=reg_m.mae_mag30,
            val_return_loss=float(session.best_val_return_loss),
            net_pnl_positive=gate.positive_net_pnl,
        )

    # ── battery ────────────────────────────────────────────────────────────

    def run_battery(self, df: pl.DataFrame, symbol: str = "EURUSD") -> BatterySummaryReport:
        logger.info(
            f"Phase 7 ablation battery on {symbol}: {len(self.variant_keys)} variants, "
            f"shared fold cache (pipeline + Branch A fitted once per fold)."
        )
        df = df.sort("timestamp")
        splits = generate_splits(
            df, train_bars=self.train_bars, test_bars=self.test_bars,
            step_bars=self.step_bars, embargo_bars=self.embargo_bars,
        )
        if self.max_folds is not None:
            splits = splits[: self.max_folds]
        if not splits:
            raise ValueError(f"No valid walk-forward folds for {symbol}.")

        records: Dict[str, List[_FoldRecord]] = {k: [] for k in self.variant_keys}

        for split in splits:
            train_df = df.slice(split.train_start, split.n_train)
            test_df = df.slice(split.test_start, split.n_test)
            cache = self._build_fold_cache(train_df, test_df)

            for key in self.variant_keys:
                try:
                    rec = self._score_variant(key, cache, split.fold_idx, symbol)
                except Exception as exc:
                    raise RuntimeError(
                        f"[{symbol}] variant {key}, fold {split.fold_idx}: {exc!r}. "
                        "Failing loud per build-plan doctrine."
                    ) from exc
                records[key].append(rec)
            logger.info(f"Fold {split.fold_idx}: {len(self.variant_keys)} variants scored.")

        # ── aggregate ──────────────────────────────────────────────────────
        def _mean(key: str, attr: str) -> float:
            vals = [getattr(r, attr) for r in records[key]]
            return float(np.nanmean(vals)) if vals else float("nan")

        base_key = "BASELINE_FULL" if "BASELINE_FULL" in records else self.variant_keys[0]

        comparisons: List[VariantComparison] = []
        for key in self.variant_keys:
            cfg = ABLATION_SUITE[key]
            d_mcc = _mean(key, "mcc") - _mean(base_key, "mcc")
            d_qlike = _mean(key, "qlike_rv30") - _mean(base_key, "qlike_rv30")
            comparisons.append(VariantComparison(
                name=cfg.name, condition_tag=cfg.condition_tag, description=cfg.description,
                mean_mcc=_mean(key, "mcc"),
                mean_hit_rate_tradable=_mean(key, "hit_rate_tradable"),
                mean_oos_r2=_mean(key, "oos_r2"),
                mean_qlike_rv30=_mean(key, "qlike_rv30"),
                mean_mae_mag30=_mean(key, "mae_mag30"),
                delta_mcc_vs_baseline=d_mcc,
                delta_qlike_vs_baseline=d_qlike,
                keep_decision="PENDING", justification="",
                mean_val_return_loss=_mean(key, "val_return_loss"),
            ))

        comp = {c.name: c for c in comparisons}
        base = comp[base_key]
        base.keep_decision = "REFERENCE"
        base.justification = "Full baseline model specification."
        tol = self.keep_tol

        def _degraded(abl: VariantComparison) -> bool:
            """Plan-literal: did removing the component make validation loss worse?"""
            val_worse = (np.isfinite(abl.mean_val_return_loss)
                         and np.isfinite(base.mean_val_return_loss)
                         and abl.mean_val_return_loss > base.mean_val_return_loss * (1.0 + tol))
            mcc_worse = abl.mean_mcc < base.mean_mcc - 0.005
            return val_worse or mcc_worse

        # C4 — G4 interactions
        if "NO_G4_INTERACTIONS" in comp:
            c4 = _degraded(comp["NO_G4_INTERACTIONS"])
            comp["NO_G4_INTERACTIONS"].keep_decision = "KEEP G4" if c4 else "KILL G4"
            comp["NO_G4_INTERACTIONS"].justification = (
                "Removing G4 degraded validation return loss / MCC [A3, A6, F3]." if c4
                else "Validation loss unchanged without G4 — killed per plan; a finding, not a failure."
            )
        else:
            c4 = False

        # C5 — G5 time regime
        if "NO_G5_TIME_REGIME" in comp:
            c5 = _degraded(comp["NO_G5_TIME_REGIME"])
            comp["NO_G5_TIME_REGIME"].keep_decision = "KEEP G5" if c5 else "KILL G5"
            comp["NO_G5_TIME_REGIME"].justification = (
                "Time-regime conditioning carries independent signal [A2, D4]." if c5
                else "Validation loss unchanged without G5 — killed per plan."
            )
        else:
            c5 = False

        # C6 — cascade heads
        if "INDEPENDENT_HEADS" in comp:
            c6 = base.mean_qlike_rv30 <= comp["INDEPENDENT_HEADS"].mean_qlike_rv30 * (1.0 + tol)
            comp["INDEPENDENT_HEADS"].keep_decision = "KEEP CASCADE" if c6 else "USE INDEPENDENT"
            comp["INDEPENDENT_HEADS"].justification = (
                "Cascade coupling improves magnitude/RV accuracy [E3]." if c6
                else "Independent heads performed better on QLIKE."
            )
        else:
            c6 = False

        # C7 — log magnitude
        if "RAW_MAGNITUDE" in comp:
            c7 = base.mean_qlike_rv30 <= comp["RAW_MAGNITUDE"].mean_qlike_rv30 * (1.0 + tol)
            comp["RAW_MAGNITUDE"].keep_decision = "KEEP LOG-MAG" if c7 else "USE RAW-MAG"
            comp["RAW_MAGNITUDE"].justification = (
                "Log-space magnitude superior [E5]." if c7 else "Raw magnitude target preferred."
            )
        else:
            c7 = False

        if "BRANCH_A_ONLY" in comp:
            comp["BRANCH_A_ONLY"].keep_decision = "BENCHMARK"
            comp["BRANCH_A_ONLY"].justification = "Linear-core floor (confirms Gate 6 / C3)."

        # Lookback selection among {30, 60 (baseline), 120} by QLIKE
        lb_candidates = [comp[n] for n in ("LOOKBACK_30", base_key, "LOOKBACK_120") if n in comp]
        best_lb = min(lb_candidates, key=lambda c: c.mean_qlike_rv30)
        optimal_lb = 30 if "30" in best_lb.name else (120 if "120" in best_lb.name else 60)
        for n in ("LOOKBACK_30", "LOOKBACK_120"):
            if n in comp:
                comp[n].keep_decision = "SELECTED" if comp[n] is best_lb else "REJECTED"
                comp[n].justification = f"Lookback sweep; winner = {optimal_lb} bars by QLIKE."

        gate7_passed = all(len(records[k]) > 0 for k in self.variant_keys)
        verdict = (
            f"GATE 7 PASS: battery complete on {len(splits)} folds; surviving configuration frozen "
            f"(BASELINE_FULL, lookback={optimal_lb}). "
            f"C4={'KEEP G4' if c4 else 'G4 killed'}, C5={'KEEP G5' if c5 else 'G5 killed'}, "
            f"C6={'cascade' if c6 else 'independent'}, C7={'log-mag' if c7 else 'raw-mag'}."
            if gate7_passed else "GATE 7 FAIL: incomplete battery."
        )

        return BatterySummaryReport(
            symbol=symbol,
            baseline_variant=base_key,
            variants=comparisons,
            condition_c4_verified=c4,
            condition_c5_verified=c5,
            condition_c6_verified=c6,
            condition_c7_verified=c7,
            optimal_lookback=optimal_lb,
            frozen_config_name=base_key,   # must remain a key of ABLATION_SUITE (report.py)
            gate7_passed=gate7_passed,
            verdict=verdict,
        )