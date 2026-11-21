"""
Phase 4 Multi-Symbol Baseline & Linear Core Runner.

Executes Purged Walk-Forward Cross-Validation across all feature datasets in
data/features/5min/ to verify Gate 4 (Condition C2):
  - B0 Persistence floor
  - B1 HAR-RV volatility predictor
  - B2 Branch A Linear OFI core

Symbols: EURUSD, GBPUSD, USDJPY, XAUUSD

Gate 4 / Condition C2 criteria:
  - Branch A beats B0 Persistence on >= 60% of walk-forward folds
  - Mean Out-of-Sample R² > 0.0 (strictly positive predictive edge)
  - HAR-RV beats persistence on volatility forecasting (RMSE lift > 0%)

Exits with code 1 if any symbol fails Gate 4.
"""

import sys
from pathlib import Path

import polars as pl

from src.baselines.evaluator import BaselineEvaluator

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
FEATURES_DIR = Path("data/features/5min")
REPORT_DIR = Path("data/reports/phase4")


def print_separator(char: str = "=", width: int = 70) -> None:
    print(char * width)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print_separator()
    print("PHASE 4 — BASELINES & LINEAR CORE (BRANCH A) WALK-FORWARD EVALUATION")
    print_separator()

    evaluator = BaselineEvaluator(
        embargo_bars=30,
        train_bars=80000,    # ~10-12 months
        test_bars=20000,     # ~2.5-3 months
        step_bars=20000,     # ~2.5-3 months roll
    )

    results = []
    any_failure = False

    for symbol in SYMBOLS:
        feat_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"

        print(f"\n[{symbol}]", flush=True)

        if not feat_path.exists():
            print(f"  ⚠️ Feature file not found: {feat_path} — SKIPPED")
            results.append({
                "symbol": symbol,
                "folds": "—",
                "win_rate": "—",
                "oos_r2": "—",
                "hit_rate": "—",
                "har_lift": "—",
                "gate4": "SKIPPED",
            })
            continue

        print(f"  Source : {feat_path}")

        try:
            df = pl.read_parquet(feat_path)
            print(f"  Loaded {len(df):,} rows. Running purged walk-forward CV...")

            report = evaluator.evaluate_symbol(df, symbol=symbol, report_dir=REPORT_DIR)

        except Exception as exc:
            print(f"  ❌ FAILED with exception: {exc}")
            results.append({
                "symbol": symbol,
                "folds": "ERROR",
                "win_rate": "—",
                "oos_r2": "—",
                "hit_rate": "—",
                "har_lift": "—",
                "gate4": f"FAIL ({exc})",
            })
            any_failure = True
            continue

        gate_emoji = "✅" if report.gate4_passed else "❌"
        print(f"  Folds Evaluated    : {report.total_folds}")
        print(f"  Branch A Win-Rate  : {report.branch_a_win_rate_pct:.1f}% ({report.folds_branch_a_beats_b0}/{report.total_folds})")
        print(f"  Median OOS R²      : {report.median_oos_r2:+.4%}")
        print(f"  Trimmed Mean R²    : {report.trimmed_mean_oos_r2:+.4%}")
        print(f"  Tradable Hit-Rate  : {report.mean_hit_rate_tradable * 100:.2f}%")
        print(f"  Mean MCC           : {report.mean_mcc:+.4f}")
        print(f"  HAR-RV Lift vs B0  : {report.har_rv_mean_lift_pct:+.2f}%")
        print(f"  {gate_emoji}  {report.verdict}")

        if not report.gate4_passed:
            any_failure = True

        results.append({
            "symbol": symbol,
            "folds": str(report.total_folds),
            "win_rate": f"{report.branch_a_win_rate_pct:.1f}%",
            "oos_r2": f"{report.median_oos_r2:+.4%}",
            "hit_rate": f"{report.mean_hit_rate_tradable * 100:.1f}%",
            "har_lift": f"{report.har_rv_mean_lift_pct:+.1f}%",
            "gate4": f"{'PASS ✅' if report.gate4_passed else 'FAIL ❌'}",
        })

    # ── Summary Matrix ─────────────────────────────────────────────────────
    print()
    print_separator()
    print("GATE 4 (CONDITION C2) SUMMARY MATRIX")
    print_separator()
    col_widths = [8, 8, 12, 12, 12, 12, 12]
    headers = ["Symbol", "Folds", "Win-Rate", "OOS R²", "Hit-Rate", "HAR Lift", "Gate 4"]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * sum(col_widths))
    for r in results:
        print(fmt.format(r["symbol"], r["folds"], r["win_rate"], r["oos_r2"], r["hit_rate"], r["har_lift"], r["gate4"]))
    print_separator()

    if any_failure:
        print("\n❌ One or more symbols FAILED Gate 4 (Condition C2). Review reports before proceeding.")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} symbols passed Gate 4 (Condition C2). Phase 4 complete.")
        print(f"    Audit reports  → {REPORT_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    main()
