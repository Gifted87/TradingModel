"""
Phase 5 — ValidationHarness: top-level orchestrator.

Accepts any predictor satisfying the `Predictor` protocol and runs the
complete Phase 5 evaluation:

  1. Purged walk-forward splits
  2. Full metric suite per fold (direction, regression, quantile calibration)
  3. Per-regime metric decomposition (time-of-day × volatility terciles)
  4. Economic gate (spread-net PnL)
  5. Adversarial checks (shuffle test + group audits)
  6. Gate 5 check — written before the result is known

Gate 5 pass criterion (written before execution, per build-plan doctrine):
  PASS iff BOTH:
    1. B0 Persistence MCC ≈ 0 on every fold (|mean MCC| < 0.05)
       → harness does not hallucinate signal from r̂=0
    2. B1 HAR-RV QLIKE lift vs B0 > 0 on ≥ 60% of folds
       → harness correctly ranks a known-good volatility model above persistence

  FAIL → the harness is broken (data leakage, label misalignment); stop.

Build-plan reference (Phase 5, §S6 binding-constraint doctrine):
  "The harness must reproduce known results (persistence ≈ chance; HAR beats
   persistence on rv30).  A harness that can't rank obvious baselines can't
   judge a model."
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import polars as pl

from .walk_forward import WalkForwardSplit, generate_splits
from .metrics import (
    FoldMetrics,
    DirectionMetrics,
    RegressionMetrics,
    compute_direction_metrics,
    compute_regression_metrics,
)
from .regime_slicer import RegimeReport, compute_regime_report, aggregate_regime_reports
from .economic_gate import EconomicGateReport, evaluate_economic_gate, extract_spreads
from .adversarial import AdversarialReport, run_adversarial_checks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Predictor protocol — Phase 6 PyTorch model must satisfy this
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Predictor(Protocol):
    """Minimal interface required by ValidationHarness."""

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "Predictor":
        """Fit the model on a training DataFrame."""
        ...

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Return predicted r̂30 as a 1-D NumPy array (same length as df)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HarnessReport:
    """Full harness evaluation result for one predictor on one symbol."""
    predictor_name: str
    symbol: str
    total_folds: int

    # Per-fold metrics
    fold_metrics: List[FoldMetrics]
    regime_reports: List[RegimeReport]
    economic_gate_reports: List[EconomicGateReport]
    adversarial_reports: List[AdversarialReport]

    # Cross-fold aggregates
    mean_mcc: float
    mean_hit_rate_tradable: float
    mean_oos_r2: float
    mean_qlike_rv30: float
    mean_mae_mag30: float
    pct_folds_positive_net_pnl: float

    # Regime aggregate (mean per bucket across folds)
    regime_aggregate: Dict[str, Dict[str, Dict[str, float]]]

    # Gate 5
    gate5_passed: bool
    gate5_verdict: str


@dataclass
class Gate5Report:
    """Gate 5 results for a pair of (B0, B1) baselines on one symbol."""
    symbol: str

    # Criterion 1: B0 MCC ≈ 0
    b0_mean_mcc: float
    b0_mcc_near_zero: bool           # |b0_mean_mcc| < 0.05

    # Criterion 2: B1 QLIKE lift > 0 on ≥ 60% of folds
    b1_qlike_lift_pct_folds: float   # fraction of folds where B1 QLIKE < B0 QLIKE
    b1_beats_b0_majority: bool       # b1_qlike_lift_pct_folds >= 0.60

    gate5_passed: bool
    verdict: str


# ─────────────────────────────────────────────────────────────────────────────
# ValidationHarness
# ─────────────────────────────────────────────────────────────────────────────

class ValidationHarness:
    """
    Model-agnostic purged walk-forward validation harness.

    Parameters
    ----------
    train_bars : int
        Training window length in bars.
    test_bars : int
        Test window length in bars.
    step_bars : int
        Roll step in bars.
    embargo_bars : int
        Purge/embargo buffer between train end and test start.
    economic_gate_k : float
        Spread multiplier for the economic gate decision rule.
    n_shuffle : int
        Number of permutations for the shuffle test.
    run_adversarial : bool
        If False, skip adversarial checks (faster; suitable for quick runs).
    """

    def __init__(
        self,
        train_bars: int = 80_000,
        test_bars: int = 20_000,
        step_bars: int = 20_000,
        embargo_bars: int = 30,
        economic_gate_k: float = 1.5,
        n_shuffle: int = 100,
        run_adversarial: bool = True,
        decision_bars_only: bool = True,
    ) -> None:
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.embargo_bars = embargo_bars
        self.economic_gate_k = economic_gate_k
        self.n_shuffle = n_shuffle
        self.run_adversarial = run_adversarial
        self.decision_bars_only = decision_bars_only

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        df: pl.DataFrame,
        predictor: Predictor,
        predictor_name: str = "model",
        symbol: str = "UNKNOWN",
        report_dir: Optional[Path] = None,
    ) -> HarnessReport:
        """
        Run the full evaluation on df for the given predictor.

        Parameters
        ----------
        df : pl.DataFrame
            Feature + label DataFrame sorted by timestamp.
        predictor : Predictor
            Any object satisfying the Predictor protocol.
        predictor_name : str
            Display name used in reports.
        symbol : str
            Instrument symbol.
        report_dir : Path | None
            If provided, write JSON + Markdown reports there.
        """
        df = df.sort("timestamp")
        splits = generate_splits(
            df,
            train_bars=self.train_bars,
            test_bars=self.test_bars,
            step_bars=self.step_bars,
            embargo_bars=self.embargo_bars,
        )

        if not splits:
            raise ValueError(
                f"No valid walk-forward folds for {symbol} "
                f"({len(df)} rows, train={self.train_bars}, test={self.test_bars})."
            )

        logger.info(
            f"[{symbol}/{predictor_name}] Starting {len(splits)}-fold "
            f"purged walk-forward evaluation."
        )

        fold_metrics_list: List[FoldMetrics] = []
        regime_reports: List[RegimeReport] = []
        gate_reports: List[EconomicGateReport] = []
        adv_reports: List[AdversarialReport] = []

        for split in splits:
            fold_result = self._evaluate_fold(df, predictor, split, symbol)
            if fold_result is None:
                continue
            fm, rr, gr, ar = fold_result
            fold_metrics_list.append(fm)
            regime_reports.append(rr)
            gate_reports.append(gr)
            if ar is not None:
                adv_reports.append(ar)

        if not fold_metrics_list:
            raise ValueError(f"All folds produced no valid results for {symbol}.")

        report = self._aggregate(
            predictor_name=predictor_name,
            symbol=symbol,
            fold_metrics=fold_metrics_list,
            regime_reports=regime_reports,
            gate_reports=gate_reports,
            adv_reports=adv_reports,
        )

        if report_dir is not None:
            self._write_reports(report, Path(report_dir))

        return report

    def check_gate5(
        self,
        b0_report: HarnessReport,
        b1_report: HarnessReport,
        symbol: str,
        report_dir: Optional[Path] = None,
    ) -> Gate5Report:
        """
        Gate 5 check using B0 (persistence) and B1 (HAR-RV) harness reports.

        Criterion 1: B0 MCC ≈ 0 (|mean_mcc| < 0.05) across folds.
        Criterion 2: B1 QLIKE lift vs B0 > 0 on ≥ 60% of folds.
        """
        b0_mccs = [f.direction.mcc for f in b0_report.fold_metrics]
        b0_mean_mcc = float(np.mean(b0_mccs))
        crit1 = abs(b0_mean_mcc) < 0.05

        # Per-fold: does B1 beat B0 on QLIKE?
        n_folds = min(len(b0_report.fold_metrics), len(b1_report.fold_metrics))
        b1_wins = 0
        for i in range(n_folds):
            b1_q = b1_report.fold_metrics[i].regression.qlike_rv30
            b0_q = b0_report.fold_metrics[i].regression.qlike_rv30
            if b1_q < b0_q:
                b1_wins += 1

        b1_lift_frac = float(b1_wins / n_folds) if n_folds > 0 else 0.0
        crit2 = b1_lift_frac >= 0.60

        passed = crit1 and crit2

        verdict_parts = []
        if crit1:
            verdict_parts.append(f"✅ B0 MCC≈0 (mean={b0_mean_mcc:+.4f})")
        else:
            verdict_parts.append(f"❌ B0 MCC={b0_mean_mcc:+.4f} (should be <0.05) — harness leakage suspected")

        if crit2:
            verdict_parts.append(
                f"✅ B1 beats B0 on QLIKE in {b1_wins}/{n_folds} folds ({b1_lift_frac:.0%})"
            )
        else:
            verdict_parts.append(
                f"❌ B1 QLIKE lift only {b1_wins}/{n_folds} folds ({b1_lift_frac:.0%}, need ≥60%)"
            )

        prefix = "GATE 5 PASS" if passed else "GATE 5 FAIL"
        verdict = f"{prefix}: {'; '.join(verdict_parts)}"

        gate5 = Gate5Report(
            symbol=symbol,
            b0_mean_mcc=b0_mean_mcc,
            b0_mcc_near_zero=crit1,
            b1_qlike_lift_pct_folds=b1_lift_frac,
            b1_beats_b0_majority=crit2,
            gate5_passed=passed,
            verdict=verdict,
        )

        if report_dir is not None:
            out = Path(report_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"gate5_{symbol}.json").write_text(
                json.dumps(asdict(gate5), indent=2)
            )
            (out / f"gate5_{symbol}_report.md").write_text(
                _render_gate5_md(gate5)
            )

        return gate5

    # ── Private helpers ──────────────────────────────────────────────────────

    def _evaluate_fold(
        self,
        df: pl.DataFrame,
        predictor: Predictor,
        split: WalkForwardSplit,
        symbol: str,
    ):
        """Evaluate one fold.  Returns (FoldMetrics, RegimeReport, EconGateReport, AdversarialReport|None)."""
        train_df = df.slice(split.train_start, split.n_train)
        test_df = df.slice(split.test_start, split.n_test)

        # Drop rows with missing labels.
        test_df = test_df.filter(
            pl.col("r30").is_not_nan() & pl.col("r30").is_not_null() &
            pl.col("rv30").is_not_nan() & pl.col("rv30").is_not_null()
        )

        # Restrict to :00/:30 decision bars (whitepaper §1).
        if self.decision_bars_only and "is_decision_bar" in test_df.columns:
            test_df = test_df.filter(pl.col("is_decision_bar"))

        if len(test_df) < 20:
            logger.warning(f"Fold {split.fold_idx}: only {len(test_df)} valid test rows; skipping.")
            return None

        # Fit + predict
        try:
            predictor.fit(train_df, symbol=symbol)
            if hasattr(predictor, "predict_both"):
                y_pred, rv_pred = predictor.predict_both(test_df)
            else:
                y_pred_raw = predictor.predict(test_df)
                if hasattr(y_pred_raw, "r_hat") and hasattr(y_pred_raw, "rv_hat"):
                    y_pred = y_pred_raw.r_hat
                    rv_pred = y_pred_raw.rv_hat
                elif isinstance(y_pred_raw, tuple) and len(y_pred_raw) == 2:
                    y_pred, rv_pred = y_pred_raw
                else:
                    y_pred = y_pred_raw
                    if hasattr(predictor, "predict_rv"):
                        rv_pred = predictor.predict_rv(test_df)
                    else:
                        # Persistence baseline rv̂ for QLIKE reference
                        rv_train_mean = float(np.nanmean(train_df["rv30"].to_numpy()))
                        rv_pred = np.full(len(test_df), rv_train_mean)
        except Exception as exc:
            raise RuntimeError(
                f"[{symbol}] fold {split.fold_idx}: predictor failed — {exc!r}. "
                "Failing loud per build-plan doctrine (silent skipping hides training collapse)."
            ) from exc

        y_true = test_df["r30"].to_numpy()
        rv_true = test_df["rv30"].to_numpy()

        is_dead = (
            test_df["is_dead_zone"].to_numpy().astype(bool)
            if "is_dead_zone" in test_df.columns
            else None
        )
        mag_true = (
            test_df["mag30"].to_numpy() if "mag30" in test_df.columns else None
        )
        mag_pred = (
            np.log(np.abs(y_pred) + 1e-7) if mag_true is not None else None
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        dir_m = compute_direction_metrics(y_true, y_pred, is_dead)
        reg_m = compute_regression_metrics(
            y_true, y_pred, rv_true, rv_pred, mag_true, mag_pred
        )

        fm = FoldMetrics(
            fold_idx=split.fold_idx,
            n_train=split.n_train,
            n_test=len(test_df),
            train_start_ts=split.train_start_ts,
            train_end_ts=split.train_end_ts,
            test_start_ts=split.test_start_ts,
            test_end_ts=split.test_end_ts,
            direction=dir_m,
            regression=reg_m,
        )

        # ── Regime ───────────────────────────────────────────────────────────
        rr = compute_regime_report(
            test_df, y_pred, fold_idx=split.fold_idx, rv30_pred=rv_pred
        )

        # ── Economic gate ────────────────────────────────────────────────────
        spreads = extract_spreads(test_df)
        gr = evaluate_economic_gate(
            y_true, y_pred, spreads,
            fold_idx=split.fold_idx, k=self.economic_gate_k
        )

        # ── Adversarial ───────────────────────────────────────────────────────
        ar = None
        if self.run_adversarial:
            try:
                ar = run_adversarial_checks(
                    train_df, test_df, y_pred,
                    fold_idx=split.fold_idx,
                    n_shuffles=self.n_shuffle,
                )
            except Exception as exc:
                logger.warning(f"Fold {split.fold_idx}: adversarial checks failed: {exc!r}")

        return fm, rr, gr, ar

    def _aggregate(
        self,
        predictor_name: str,
        symbol: str,
        fold_metrics: List[FoldMetrics],
        regime_reports: List[RegimeReport],
        gate_reports: List[EconomicGateReport],
        adv_reports: List[AdversarialReport],
    ) -> HarnessReport:
        n = len(fold_metrics)
        mean_mcc = float(np.mean([f.direction.mcc for f in fold_metrics]))
        mean_hr = float(np.mean([f.direction.hit_rate_tradable for f in fold_metrics]))
        mean_r2 = float(np.mean([f.regression.oos_r2 for f in fold_metrics]))
        mean_qlike = float(np.mean([f.regression.qlike_rv30 for f in fold_metrics]))
        mean_mae_mag = float(np.nanmean([f.regression.mae_mag30 for f in fold_metrics]))
        pct_pos = float(np.mean([g.positive_net_pnl for g in gate_reports]))
        reg_agg = aggregate_regime_reports(regime_reports)

        # Gate 5 placeholder — populated by check_gate5() externally.
        gate5_passed = False
        gate5_verdict = "Gate 5 not yet evaluated (run check_gate5 with B0 and B1 reports)."

        return HarnessReport(
            predictor_name=predictor_name,
            symbol=symbol,
            total_folds=n,
            fold_metrics=fold_metrics,
            regime_reports=regime_reports,
            economic_gate_reports=gate_reports,
            adversarial_reports=adv_reports,
            mean_mcc=mean_mcc,
            mean_hit_rate_tradable=mean_hr,
            mean_oos_r2=mean_r2,
            mean_qlike_rv30=mean_qlike,
            mean_mae_mag30=mean_mae_mag,
            pct_folds_positive_net_pnl=pct_pos,
            regime_aggregate=reg_agg,
            gate5_passed=gate5_passed,
            gate5_verdict=gate5_verdict,
        )

    def _write_reports(self, report: HarnessReport, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{report.symbol}_{report.predictor_name}"

        # JSON — convert non-serialisable nested dataclasses manually
        json_data = _report_to_dict(report)
        (out_dir / f"harness_{tag}.json").write_text(json.dumps(json_data, indent=2))

        # Markdown
        (out_dir / f"harness_{tag}_report.md").write_text(_render_harness_md(report))

        logger.info(f"Reports written to {out_dir}/harness_{tag}.*")


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _report_to_dict(report: HarnessReport) -> dict:
    """Convert HarnessReport to a JSON-serialisable dict."""

    def _fold(fm: FoldMetrics) -> dict:
        return {
            "fold_idx": fm.fold_idx,
            "n_train": fm.n_train,
            "n_test": fm.n_test,
            "train_start_ts": fm.train_start_ts,
            "test_start_ts": fm.test_start_ts,
            "mcc": fm.direction.mcc,
            "hit_rate_tradable": fm.direction.hit_rate_tradable,
            "balanced_acc": fm.direction.balanced_acc,
            "oos_r2": fm.regression.oos_r2,
            "qlike_rv30": fm.regression.qlike_rv30,
            "mae_mag30": fm.regression.mae_mag30,
        }

    def _gate(gr: EconomicGateReport) -> dict:
        return {
            "fold_idx": gr.fold_idx,
            "n_trades": gr.n_trades,
            "trade_fraction": gr.trade_fraction,
            "net_pnl_per_bar": gr.net_pnl_per_bar,
            "pnl_sharpe": gr.pnl_sharpe,
            "positive_net_pnl": gr.positive_net_pnl,
        }

    return {
        "predictor_name": report.predictor_name,
        "symbol": report.symbol,
        "total_folds": report.total_folds,
        "mean_mcc": report.mean_mcc,
        "mean_hit_rate_tradable": report.mean_hit_rate_tradable,
        "mean_oos_r2": report.mean_oos_r2,
        "mean_qlike_rv30": report.mean_qlike_rv30,
        "mean_mae_mag30": report.mean_mae_mag30,
        "pct_folds_positive_net_pnl": report.pct_folds_positive_net_pnl,
        "regime_aggregate": report.regime_aggregate,
        "gate5_passed": report.gate5_passed,
        "gate5_verdict": report.gate5_verdict,
        "folds": [_fold(f) for f in report.fold_metrics],
        "economic_gate": [_gate(g) for g in report.economic_gate_reports],
    }


def _render_harness_md(report: HarnessReport) -> str:
    gate_emoji = "✅" if report.gate5_passed else "⏳"
    rows = ""
    for f in report.fold_metrics:
        eg = next((g for g in report.economic_gate_reports if g.fold_idx == f.fold_idx), None)
        net_pnl = f"{eg.net_pnl_per_bar:+.6f}" if eg else "—"
        rows += (
            f"| {f.fold_idx} | `{f.test_start_ts[:10]}` | "
            f"`{f.direction.mcc:+.4f}` | `{f.direction.hit_rate_tradable:.2%}` | "
            f"`{f.regression.oos_r2:+.4%}` | `{f.regression.qlike_rv30:.4f}` | "
            f"`{net_pnl}` |\n"
        )

    return f"""# Validation Harness Report — {report.symbol} / {report.predictor_name}

**Gate 5:** {gate_emoji} {report.gate5_verdict}
**Total Folds:** `{report.total_folds}`

| Metric | Value |
|---|---|
| Mean MCC | `{report.mean_mcc:+.4f}` |
| Mean Tradable Hit-Rate | `{report.mean_hit_rate_tradable:.2%}` |
| Mean OOS R² | `{report.mean_oos_r2:+.4%}` |
| Mean QLIKE (rv30) | `{report.mean_qlike_rv30:.4f}` |
| Mean MAE (mag30) | `{report.mean_mae_mag30:.6f}` |
| % Folds Net PnL > 0 | `{report.pct_folds_positive_net_pnl:.0%}` |

---

## Fold-by-Fold Results

| Fold | Test Start | MCC | Hit-Rate | OOS R² | QLIKE | Net PnL/bar |
|---|---|---|---|---|---|---|
{rows}

---

## Regime Breakdown (mean across folds)

### Time-of-Day
| Bucket | Hit-Rate | MCC | OOS R² | QLIKE |
|---|---|---|---|---|
{"".join(f"| {b} | `{v.get('hit_rate', float('nan')):.2%}` | `{v.get('mcc', float('nan')):+.4f}` | `{v.get('oos_r2', float('nan')):+.4%}` | `{v.get('qlike', float('nan')):.4f}` |{chr(10)}" for b, v in report.regime_aggregate.get("tod", {}).items())}

### Volatility Tercile
| Bucket | Hit-Rate | MCC | OOS R² | QLIKE |
|---|---|---|---|---|
{"".join(f"| {b} | `{v.get('hit_rate', float('nan')):.2%}` | `{v.get('mcc', float('nan')):+.4f}` | `{v.get('oos_r2', float('nan')):+.4%}` | `{v.get('qlike', float('nan')):.4f}` |{chr(10)}" for b, v in report.regime_aggregate.get("vol", {}).items())}
"""


def _render_gate5_md(gate5: Gate5Report) -> str:
    status = "✅ PASSED" if gate5.gate5_passed else "❌ FAILED"
    return f"""# Gate 5 Harness Validation Report — {gate5.symbol}

**Status:** {status}

## Criterion 1 — Persistence (B0) MCC ≈ 0
| | Value | Threshold | Status |
|---|---|---|---|
| B0 Mean MCC | `{gate5.b0_mean_mcc:+.4f}` | `|mcc| < 0.05` | {"✅" if gate5.b0_mcc_near_zero else "❌"} |

## Criterion 2 — HAR-RV (B1) beats B0 on QLIKE in ≥ 60% of folds
| | Value | Threshold | Status |
|---|---|---|---|
| B1 QLIKE-lift folds | `{gate5.b1_qlike_lift_pct_folds:.0%}` | `≥ 60%` | {"✅" if gate5.b1_beats_b0_majority else "❌"} |

## Verdict
**`{gate5.verdict}`**

---

> If Gate 5 FAILS: the harness is broken (data leakage, label misalignment).
> Do not proceed to Phase 6 until this is resolved.
"""
