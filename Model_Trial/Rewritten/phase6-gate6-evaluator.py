"""
Phase 6 — Gate 6 (Condition C3) Evaluator.

REWRITE (v2):
  - Branch A is fitted ONCE per fold and injected into the composite model
    (previously fitted twice per fold).
  - Evaluation is restricted to :00/:30 decision bars by default
    (whitepaper §1: decisions only at decision timestamps); all-bars metrics
    remain available via decision_bars_only=False.
  - Predictor exceptions now RAISE with fold context instead of being
    silently swallowed (silent skipping hid the training collapse).
  - Branch A's rv30 slot uses explicit train-mean persistence instead of the
    old placeholder that leaked rv_true into the metric bundle.

Gate 6 (C3) pass rule (written before execution, per build-plan doctrine):
  PASS iff full model beats Branch A on >= 60% of folds (RMSE or MCC)
  AND full-model mean MCC > 0.
  Fail -> halve trunk capacity, shorten lookback, retry ONCE;
  fail again -> ship Branch A + magnitude head and document why.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl

from ..baselines.linear_core import LinearCoreBranchA
from ..validation.metrics import compute_direction_metrics, compute_regression_metrics
from ..validation.walk_forward import generate_splits
from .full_model import CompositeTradingModel
from .trainer import TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class Gate6FoldComparison:
    fold_idx: int
    test_start_ts: str
    test_end_ts: str
    n_test: int
    branch_a_rmse: float
    branch_a_r2_oos: float
    branch_a_hit_rate_tradable: float
    branch_a_mcc: float
    full_model_rmse: float
    full_model_r2_oos: float
    full_model_hit_rate_tradable: float
    full_model_mcc: float
    full_model_qlike_rv30: float
    rmse_reduction_pct: float
    mcc_lift: float
    full_beats_branch_a: bool


@dataclass
class Gate6Report:
    symbol: str
    total_folds: int
    folds_full_beats_branch_a: int
    win_rate_pct: float
    mean_branch_a_rmse: float
    mean_full_model_rmse: float
    mean_rmse_reduction_pct: float
    mean_branch_a_mcc: float
    mean_full_model_mcc: float
    mean_full_model_hit_rate_tradable: float
    gate6_passed: bool
    verdict: str
    fold_comparisons: List[Gate6FoldComparison]


class Gate6Evaluator:
    """Out-of-sample comparison: Branch A alone vs. full composite model."""

    def __init__(
        self,
        train_bars: int = 80_000,
        test_bars: int = 20_000,
        step_bars: int = 20_000,
        embargo_bars: int = 30,
        lookback: int = 60,
        epochs: int = 12,
        batch_size: int = 256,
        decision_bars_only: bool = True,
        max_folds: Optional[int] = None,      # set small for smoke tests
        conv_channels: int = 32,
        lstm_hidden: int = 32,
        mlflow_experiment: Optional[str] = None,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.embargo_bars = embargo_bars
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.decision_bars_only = decision_bars_only
        self.max_folds = max_folds
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.mlflow_experiment = mlflow_experiment

    def evaluate_symbol(
        self,
        df: pl.DataFrame,
        symbol: str = "EURUSD",
        report_dir: Optional[Path] = None,
    ) -> Gate6Report:
        logger.info(f"Gate 6 (C3) evaluation on {symbol}...")
        df = df.sort("timestamp")
        splits = generate_splits(
            df, train_bars=self.train_bars, test_bars=self.test_bars,
            step_bars=self.step_bars, embargo_bars=self.embargo_bars,
        )
        if self.max_folds is not None:
            splits = splits[: self.max_folds]

        fold_comps: List[Gate6FoldComparison] = []

        for split in splits:
            train_df = df.slice(split.train_start, split.n_train)
            test_df = df.slice(split.test_start, split.n_test)

            valid_test = test_df.filter(
                pl.col("r30").is_not_nan() & pl.col("r30").is_not_null() &
                pl.col("rv30").is_not_nan() & pl.col("rv30").is_not_null()
            )
            if self.decision_bars_only and "is_decision_bar" in valid_test.columns:
                valid_test = valid_test.filter(pl.col("is_decision_bar"))

            if len(valid_test) < max(20, self.lookback // 2):
                logger.warning(f"Fold {split.fold_idx}: only {len(valid_test)} eval rows; skipping.")
                continue

            y_true = valid_test["r30"].to_numpy()
            rv_true = valid_test["rv30"].to_numpy()
            is_dead = (
                valid_test["is_dead_zone"].to_numpy().astype(bool)
                if "is_dead_zone" in valid_test.columns else None
            )

            # ── 1. Branch A, fitted ONCE per fold ──────────────────────────
            branch_a = LinearCoreBranchA(alpha=100.0)
            branch_a.fit(train_df, symbol=symbol)
            r_lin_pred = branch_a.predict(valid_test)

            rv_train_mean = float(np.nanmean(train_df["rv30"].to_numpy()))
            rv_persistence = np.full(len(valid_test), rv_train_mean)

            ba_dir = compute_direction_metrics(y_true, r_lin_pred, is_dead)
            ba_reg = compute_regression_metrics(y_true, r_lin_pred, rv_true, rv_persistence)

            # ── 2. Full composite model reusing the same Branch A ──────────
            trainer_cfg = TrainerConfig(
                epochs=self.epochs, batch_size=self.batch_size,
                learning_rate=1e-3, patience=3,
                selection_metric="return",
                mlflow_experiment=self.mlflow_experiment,
                mlflow_run_name=f"{symbol}/gate6/fold{split.fold_idx}" if self.mlflow_experiment else None,
                fold_id=f"fold{split.fold_idx}",
            )
            full_model = CompositeTradingModel(
                lookback=self.lookback,
                conv_channels=self.conv_channels,
                lstm_hidden=self.lstm_hidden,
                trainer_config=trainer_cfg,
            )
            try:
                full_model.fit(train_df, symbol=symbol, branch_a=branch_a,
                               fold_id=f"fold{split.fold_idx}")
                r_full_pred, rv_full_pred = full_model.predict(valid_test)
            except Exception as exc:
                raise RuntimeError(
                    f"[{symbol}] Gate 6 fold {split.fold_idx}: composite model failed — {exc!r}. "
                    "Failing loud per build-plan doctrine (no silent fold skipping)."
                ) from exc

            fm_dir = compute_direction_metrics(y_true, r_full_pred, is_dead)
            fm_reg = compute_regression_metrics(y_true, r_full_pred, rv_true, rv_full_pred)

            rmse_reduct = float((ba_reg.rmse_r30 - fm_reg.rmse_r30) / (ba_reg.rmse_r30 + 1e-9) * 100.0)
            mcc_lift = float(fm_dir.mcc - ba_dir.mcc)
            beats_ba = (fm_reg.rmse_r30 < ba_reg.rmse_r30) or (fm_dir.mcc > ba_dir.mcc)

            fold_comps.append(Gate6FoldComparison(
                fold_idx=split.fold_idx,
                test_start_ts=split.test_start_ts, test_end_ts=split.test_end_ts,
                n_test=len(valid_test),
                branch_a_rmse=ba_reg.rmse_r30, branch_a_r2_oos=ba_reg.oos_r2,
                branch_a_hit_rate_tradable=ba_dir.hit_rate_tradable, branch_a_mcc=ba_dir.mcc,
                full_model_rmse=fm_reg.rmse_r30, full_model_r2_oos=fm_reg.oos_r2,
                full_model_hit_rate_tradable=fm_dir.hit_rate_tradable, full_model_mcc=fm_dir.mcc,
                full_model_qlike_rv30=fm_reg.qlike_rv30,
                rmse_reduction_pct=rmse_reduct, mcc_lift=mcc_lift,
                full_beats_branch_a=beats_ba,
            ))

        n_folds = len(fold_comps)
        if n_folds == 0:
            raise ValueError(f"No valid folds evaluated for {symbol}.")

        wins = sum(1 for f in fold_comps if f.full_beats_branch_a)
        win_rate = (wins / n_folds) * 100.0
        mean_ba_rmse = float(np.mean([f.branch_a_rmse for f in fold_comps]))
        mean_fm_rmse = float(np.mean([f.full_model_rmse for f in fold_comps]))
        mean_rmse_red = float(np.mean([f.rmse_reduction_pct for f in fold_comps]))
        mean_ba_mcc = float(np.mean([f.branch_a_mcc for f in fold_comps]))
        mean_fm_mcc = float(np.mean([f.full_model_mcc for f in fold_comps]))
        mean_fm_hr = float(np.mean([f.full_model_hit_rate_tradable for f in fold_comps]))

        passed = (win_rate >= 60.0) and (mean_fm_mcc > 0.0)
        verdict = (
            f"GATE 6 (C3) PASS: full model beats Branch A on {wins}/{n_folds} folds "
            f"({win_rate:.1f}%), mean MCC={mean_fm_mcc:+.4f} (vs {mean_ba_mcc:+.4f}), "
            f"tradable hit-rate={mean_fm_hr * 100:.2f}%"
            if passed else
            f"GATE 6 (C3) FAIL: win rate {win_rate:.1f}% (<60%) or MCC {mean_fm_mcc:+.4f} <= 0. "
            "Per plan: halve trunk capacity, shorten lookback, retry once; "
            "if it fails again, ship Branch A + magnitude head."
        )

        report = Gate6Report(
            symbol=symbol, total_folds=n_folds, folds_full_beats_branch_a=wins,
            win_rate_pct=win_rate, mean_branch_a_rmse=mean_ba_rmse,
            mean_full_model_rmse=mean_fm_rmse, mean_rmse_reduction_pct=mean_rmse_red,
            mean_branch_a_mcc=mean_ba_mcc, mean_full_model_mcc=mean_fm_mcc,
            mean_full_model_hit_rate_tradable=mean_fm_hr,
            gate6_passed=passed, verdict=verdict, fold_comparisons=fold_comps,
        )

        if report_dir is not None:
            out = Path(report_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"gate6_{symbol}.json").write_text(json.dumps(asdict(report), indent=2))
            (out / f"gate6_{symbol}_report.md").write_text(_render_gate6_md(report))
            logger.info(f"Gate 6 reports written to {out}/gate6_{symbol}.*")

        return report


def _render_gate6_md(report: Gate6Report) -> str:
    status_emoji = "✅" if report.gate6_passed else "❌"
    rows = ""
    for f in report.fold_comparisons:
        win = "✅" if f.full_beats_branch_a else "❌"
        rows += (
            f"| {f.fold_idx} | `{f.test_start_ts[:10]}` | "
            f"`{f.branch_a_rmse:.6f}` | `{f.full_model_rmse:.6f}` | "
            f"`{f.rmse_reduction_pct:+.2f}%` | `{f.branch_a_mcc:+.4f}` | "
            f"`{f.full_model_mcc:+.4f}` | {win} |\n"
        )
    return f"""# Gate 6 (Condition C3) Residual Learner Verification — {report.symbol}

**Status:** {status_emoji} **{'PASSED' if report.gate6_passed else 'FAILED'}**
**Folds:** `{report.total_folds}` | **Win-rate vs Branch A:** `{report.win_rate_pct:.1f}%`
**Mean RMSE reduction:** `{report.mean_rmse_reduction_pct:+.2f}%`
**Mean MCC:** `{report.mean_full_model_mcc:+.4f}` (Branch A: `{report.mean_branch_a_mcc:+.4f}`)
**Tradable hit-rate:** `{report.mean_full_model_hit_rate_tradable * 100:.2f}%`

| Fold | OOS Window | Branch A RMSE | Full RMSE | RMSE Δ | Branch A MCC | Full MCC | Win |
|---|---|---|---|---|---|---|---|
{rows}

**`{report.verdict}`**
"""