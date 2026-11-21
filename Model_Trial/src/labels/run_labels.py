#!/usr/bin/env python3
"""
Phase 2 Multi-Symbol Label Runner.

Processes all available 5-minute bar files in data/bars/5min/ and generates
forward-horizon labels for each symbol. Skips symbols already labeled.

Symbols: EURUSD, GBPUSD, USDJPY, XAUUSD
Horizon: 6 bars × 5 min = 30 min (horizon_bars=6)

Gate 2 criteria per symbol:
  - NaN count == 0
  - |skew(r30)| < 0.60
  - 0.40 <= up_fraction <= 0.60
  - 0.01 <= dead_zone_fraction <= 0.70

Exits with code 1 if any symbol fails Gate 2.
"""

import sys
from pathlib import Path
from src.labels.label_generator import process_dataset_labels


SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
BARS_DIR = Path("data/bars/5min")
LABELED_DIR = Path("data/labeled/5min")
REPORT_DIR = Path("data/reports/phase2")
HORIZON_BARS = 6        # 6 × 5m = 30m horizon
HORIZON_MINUTES = 30


def print_separator(char: str = "=", width: int = 70) -> None:
    print(char * width)


def main() -> None:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print_separator()
    print("PHASE 2 — MULTI-SYMBOL FORWARD LABEL GENERATION")
    print_separator()

    results = []
    any_failure = False

    for symbol in SYMBOLS:
        input_path  = BARS_DIR   / f"{symbol}_5m.parquet"
        output_path = LABELED_DIR / f"{symbol}_5m_labeled.parquet"

        print(f"\n[{symbol}]", flush=True)

        if not input_path.exists():
            print(f"  ⚠️  Source file not found: {input_path} — SKIPPED")
            results.append({
                "symbol": symbol,
                "samples": "—",
                "skew": "—",
                "dead_zone": "—",
                "tradable": "—",
                "gate2": "SKIPPED",
            })
            continue

        if output_path.exists():
            print(f"  ℹ️  Already labeled: {output_path} — re-running audit only")

        print(f"  Source : {input_path}")
        print(f"  Output : {output_path}")
        print(f"  Horizon: {HORIZON_BARS} bars × 5 min = {HORIZON_MINUTES} min")

        try:
            report = process_dataset_labels(
                input_parquet=input_path,
                output_parquet=output_path,
                report_dir=REPORT_DIR,
                symbol=symbol,
                horizon_bars=HORIZON_BARS,
                horizon_minutes=HORIZON_MINUTES,
            )
        except Exception as exc:
            print(f"  ❌  FAILED with exception: {exc}")
            results.append({
                "symbol": symbol,
                "samples": "ERROR",
                "skew": "—",
                "dead_zone": "—",
                "tradable": "—",
                "gate2": f"FAIL ({exc})",
            })
            any_failure = True
            continue

        gate_emoji = "✅" if report.gate2_passed else "❌"
        print(f"  Samples : {report.total_samples:,}")
        print(f"  r30     : mean={report.r30_mean:+.2e}  std={report.r30_std:.6f}  skew={report.r30_skew:+.4f}  kurt={report.r30_kurtosis:+.2f}")
        print(f"  mag30   : mean={report.mag30_mean:.4f}  std={report.mag30_std:.4f}")
        print(f"  rv30    : mean={report.rv30_mean:.4f}   std={report.rv30_std:.4f}")
        print(f"  Dead-zone: {report.dead_zone_fraction * 100:.2f}%   Tradable: {report.tradable_fraction * 100:.2f}%")
        print(f"  {gate_emoji}  {report.verdict}")

        if not report.gate2_passed:
            any_failure = True

        results.append({
            "symbol": symbol,
            "samples": f"{report.total_samples:,}",
            "skew": f"{report.r30_skew:+.4f}",
            "dead_zone": f"{report.dead_zone_fraction * 100:.1f}%",
            "tradable": f"{report.tradable_fraction * 100:.1f}%",
            "gate2": f"{'PASS ✅' if report.gate2_passed else 'FAIL ❌'}",
        })

    # ── Summary Matrix ─────────────────────────────────────────────────────
    print()
    print_separator()
    print("GATE 2 SUMMARY MATRIX")
    print_separator()
    col_widths = [8, 14, 10, 12, 12, 14]
    headers = ["Symbol", "Samples", "Skew", "Dead-Zone", "Tradable", "Gate 2"]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * sum(col_widths))
    for r in results:
        print(fmt.format(r["symbol"], r["samples"], r["skew"], r["dead_zone"], r["tradable"], r["gate2"]))
    print_separator()

    if any_failure:
        print("\n❌  One or more symbols FAILED Gate 2. Review reports before proceeding.")
        sys.exit(1)
    else:
        print(f"\n✅  All {len(results)} symbols passed Gate 2. Phase 2 complete.")
        print(f"    Labeled files  → {LABELED_DIR}")
        print(f"    Audit reports  → {REPORT_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    main()
