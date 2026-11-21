"""
Phase 6 Multi-Symbol Residual Learner (Branch B & Cascade Heads) Runner.

Executes Gate 6 (Condition C3) evaluation across walk-forward folds to verify that
the deep residual sequence learner adds significant explanatory power over the linear
core (Branch A) alone across symbols.

Outputs:
  data/reports/phase6/gate6_{SYMBOL}.json
  data/reports/phase6/gate6_{SYMBOL}_report.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import polars as pl

from src.models import Gate6Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
FEATURES_DIR = Path("data/features/5min")
REPORT_DIR = Path("data/reports/phase6")


def _sep(char: str = "=", width: int = 78) -> None:
    print(char * width)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _sep()
    print("PHASE 6 — RESIDUAL LEARNER & GATE 6 (CONDITION C3) VERIFICATION")
    _sep()

    evaluator = Gate6Evaluator(
        train_bars=80_000,
        test_bars=20_000,
        step_bars=20_000,
        embargo_bars=30,
        lookback=60,
        epochs=10,
        batch_size=256,
        decision_bars_only=True,
    )

    summary_rows = []
    any_failure = False

    for symbol in SYMBOLS:
        feat_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"

        _sep()
        print(f"[{symbol}]")

        if not feat_path.exists():
            print(f"  ⚠️  Feature file not found: {feat_path} — SKIPPED")
            summary_rows.append({
                "symbol": symbol,
                "folds": "—",
                "win_rate": "—",
                "branch_a_rmse": "—",
                "full_rmse": "—",
                "rmse_reduction": "—",
                "branch_a_mcc": "—",
                "full_mcc": "—",
                "tradable_hr": "—",
                "status": "SKIPPED",
            })
            continue

        try:
            df = pl.read_parquet(feat_path)
            print(f"  Loaded {len(df):,} rows.")
        except Exception as exc:
            print(f"  ❌ Failed to load dataset: {exc}")
            any_failure = True
            continue

        try:
            report = evaluator.evaluate_symbol(df, symbol=symbol, report_dir=REPORT_DIR)
            status = "PASS ✅" if report.gate6_passed else "FAIL ❌"
            if not report.gate6_passed:
                any_failure = True

            print(f"  Total Folds evaluated   : {report.total_folds}")
            print(f"  Folds Full beats BranchA: {report.folds_full_beats_branch_a}/{report.total_folds} ({report.win_rate_pct:.1f}%)")
            print(f"  Mean Branch A RMSE      : {report.mean_branch_a_rmse:.6f}")
            print(f"  Mean Full Model RMSE    : {report.mean_full_model_rmse:.6f} ({report.mean_rmse_reduction_pct:+.2f}%)")
            print(f"  Mean Branch A MCC       : {report.mean_branch_a_mcc:+.4f}")
            print(f"  Mean Full Model MCC     : {report.mean_full_model_mcc:+.4f}")
            print(f"  Tradable Hit Rate       : {report.mean_full_model_hit_rate_tradable * 100:.2f}%")
            print(f"  Gate 6 Verdict          : {report.verdict}")

            summary_rows.append({
                "symbol": symbol,
                "folds": str(report.total_folds),
                "win_rate": f"{report.win_rate_pct:.1f}%",
                "branch_a_rmse": f"{report.mean_branch_a_rmse:.6f}",
                "full_rmse": f"{report.mean_full_model_rmse:.6f}",
                "rmse_reduction": f"{report.mean_rmse_reduction_pct:+.2f}%",
                "branch_a_mcc": f"{report.mean_branch_a_mcc:+.4f}",
                "full_mcc": f"{report.mean_full_model_mcc:+.4f}",
                "tradable_hr": f"{report.mean_full_model_hit_rate_tradable * 100:.2f}%",
                "status": status,
            })
        except Exception as exc:
            print(f"  ❌ Evaluation failed for {symbol}: {exc}")
            import traceback
            traceback.print_exc()
            any_failure = True

    # ── Summary Matrix ─────────────────────────────────────────────────────
    print()
    _sep()
    print("PHASE 6 (GATE 6 / CONDITION C3) SUMMARY MATRIX")
    _sep()
    col_w = [10, 8, 10, 14, 14, 12, 14, 12, 14, 10]
    headers = [
        "Symbol", "Folds", "Win-Rate", "Branch A RMSE", "Full RMSE",
        "RMSE Reduct", "Branch A MCC", "Full MCC", "Tradable HR", "Status",
    ]
    fmt = "".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-" * sum(col_w))
    for r in summary_rows:
        print(fmt.format(
            r["symbol"], r["folds"], r["win_rate"], r["branch_a_rmse"],
            r["full_rmse"], r["rmse_reduction"], r["branch_a_mcc"],
            r["full_mcc"], r["tradable_hr"], r["status"]
        ))
    _sep()
    print(f"\n✅ Phase 6 evaluation finished. Reports saved to {REPORT_DIR}/")

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
