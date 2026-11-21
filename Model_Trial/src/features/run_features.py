"""
Phase 3 Multi-Symbol Feature Pipeline Runner.

Processes all labeled 5-minute datasets in data/labeled/5min/ and generates
complete G1–G6 feature stores for each instrument.

Symbols: EURUSD, GBPUSD, USDJPY, XAUUSD

Gate 3 criteria per symbol:
  - Total feature count >= 40 across G1–G6
  - 0 NaN-leaking features past warm-up
  - 0 zero-variance features on active slice
  - Registry completeness verified

Exits with code 1 if any symbol fails Gate 3.
"""

import sys
from pathlib import Path

import polars as pl

from src.features.pipeline import compute_all_features, run_gate3_audit
from src.features.registry import REGISTRY

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
LABELED_DIR = Path("data/labeled/5min")
FEATURES_DIR = Path("data/features/5min")
REPORT_DIR = Path("data/reports/phase3")


def print_separator(char: str = "=", width: int = 70) -> None:
    print(char * width)


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print_separator()
    print("PHASE 3 — MULTI-SYMBOL FEATURE STORE GENERATION (G1–G6)")
    print_separator()

    results = []
    any_failure = False

    for symbol in SYMBOLS:
        input_path = LABELED_DIR / f"{symbol}_5m_labeled.parquet"
        output_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"

        print(f"\n[{symbol}]", flush=True)

        if not input_path.exists():
            print(f"  ⚠️ Source labeled file not found: {input_path} — SKIPPED")
            results.append({
                "symbol": symbol,
                "rows": "—",
                "features": "—",
                "nan_leak": "—",
                "zero_var": "—",
                "gate3": "SKIPPED",
            })
            continue

        print(f"  Source : {input_path}")
        print(f"  Output : {output_path}")

        try:
            df = pl.read_parquet(input_path)
            print(f"  Loaded {len(df):,} rows. Computing G1–G6 features...")
            feat_df = compute_all_features(df)

            print(f"  Writing {len(feat_df.columns)} columns to {output_path} (zstd)...")
            feat_df.write_parquet(output_path, compression="zstd")

            report = run_gate3_audit(feat_df, report_dir=REPORT_DIR, symbol=symbol)

        except Exception as exc:
            print(f"  ❌ FAILED with exception: {exc}")
            results.append({
                "symbol": symbol,
                "rows": "ERROR",
                "features": "—",
                "nan_leak": "—",
                "zero_var": "—",
                "gate3": f"FAIL ({exc})",
            })
            any_failure = True
            continue

        gate_emoji = "✅" if report.gate3_passed else "❌"
        print(f"  Rows          : {len(feat_df):,}")
        print(f"  Features      : {report.total_features} active across G1–G6")
        print(f"  Group counts  : {report.features_by_group}")
        print(f"  NaN-leaking   : {len(report.nan_leaking_features)}")
        print(f"  Zero-variance : {len(report.zero_variance_features)}")
        print(f"  {gate_emoji}  {report.verdict}")

        if not report.gate3_passed:
            any_failure = True

        results.append({
            "symbol": symbol,
            "rows": f"{len(feat_df):,}",
            "features": str(report.total_features),
            "nan_leak": str(len(report.nan_leaking_features)),
            "zero_var": str(len(report.zero_variance_features)),
            "gate3": f"{'PASS ✅' if report.gate3_passed else 'FAIL ❌'}",
        })

    # ── Summary Matrix ─────────────────────────────────────────────────────
    print()
    print_separator()
    print("GATE 3 SUMMARY MATRIX")
    print_separator()
    col_widths = [8, 14, 12, 12, 12, 12]
    headers = ["Symbol", "Rows", "Features", "NaN-Leak", "Zero-Var", "Gate 3"]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * sum(col_widths))
    for r in results:
        print(fmt.format(r["symbol"], r["rows"], r["features"], r["nan_leak"], r["zero_var"], r["gate3"]))
    print_separator()

    if any_failure:
        print("\n❌ One or more symbols FAILED Gate 3. Review reports before proceeding.")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} symbols passed Gate 3. Phase 3 complete.")
        print(f"    Feature files  → {FEATURES_DIR}")
        print(f"    Audit reports  → {REPORT_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    main()
