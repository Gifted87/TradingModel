#!/usr/bin/env python3
import sys
from pathlib import Path
from src.data.quality_audit import run_quality_gate

def main():
    parquet_path = Path("data/bars/5min/EURUSD_5m.parquet")
    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist.")
        sys.exit(1)
        
    report_dir = Path("data/reports/phase1")
    metrics = run_quality_gate(parquet_path, report_dir, symbol="EURUSD")
    print("\n" + "=" * 70)
    print(metrics.verdict)
    print("=" * 70)
    print(f"Total Bars: {metrics.total_bars:,}")
    print(f"Bad Print Spikes: {metrics.bad_print_spikes}")
    print(f"Trading Gaps: {metrics.total_trading_gaps_gt_5m}")
    print(f"Bad Bar Ratio: {metrics.bad_bar_ratio:.4f}%")
    print(f"Gate 1 Passed: {metrics.gate1_passed}")

if __name__ == "__main__":
    main()
