#!/usr/bin/env python3
"""Run Quality Gate and Partitioning across all downloaded instruments."""
from pathlib import Path
import polars as pl
from src.data.quality_audit import run_quality_gate
from src.data.partitioner import partition_all_symbols

def main():
    bars_dir = Path("data/bars/5min")
    reports_dir = Path("data/reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    audit_summary = []
    
    print("=" * 70)
    print("RUNNING MULTI-SYMBOL PHASE 1 QUALITY AUDIT")
    print("=" * 70)
    
    for sym in symbols:
        p_path = bars_dir / f"{sym}_5m.parquet"
        if not p_path.exists():
            print(f"Skipping {sym} (not found)")
            continue
            
        metrics = run_quality_gate(p_path, reports_dir, symbol=sym)
        audit_summary.append({
            "Symbol": sym,
            "Total Bars": f"{metrics.total_bars:,}",
            "Start": metrics.start_time[:10],
            "End": metrics.end_time[:10],
            "Spikes": metrics.bad_print_spikes,
            "Inverted Spreads": metrics.negative_spread_count,
            "Anomaly Ratio (%)": f"{metrics.bad_bar_ratio:.4f}%",
            "Gate 1 Passed": "PASS ✅" if metrics.gate1_passed else "FAIL ❌",
        })
        print(f"[{sym}] {metrics.verdict}")
        
    summary_df = pl.DataFrame(audit_summary)
    print("\n" + "=" * 70)
    print("MULTI-SYMBOL GATE 1 AUDIT MATRIX:")
    print("=" * 70)
    print(summary_df)
    
    print("\nPartitioning datasets into year/month Hive layout...")
    parts = partition_all_symbols(bars_dir=bars_dir, output_root=Path("data/bars/partitioned"))
    for s, count in parts.items():
        print(f"  {s}: {count} monthly partitions written")
    print("=" * 70)

if __name__ == "__main__":
    main()
