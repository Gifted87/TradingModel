#!/usr/bin/env python3
import polars as pl

df = pl.read_parquet("data/bars/5min/EURUSD_5m.parquet")
print("=" * 70)
print(f"Dataset: EURUSD 5-Minute Archive")
print(f"Total Rows: {df.height:,}")
print(f"Columns ({len(df.columns)}): {df.columns}")
print("=" * 70)
print("First 3 rows:")
print(df.head(3))
print("\nLast 3 rows:")
print(df.tail(3))
print("\nSummary Statistics:")
print(df.select(["open", "high", "low", "close", "spread_close", "volume", "return_5m"]).describe())
