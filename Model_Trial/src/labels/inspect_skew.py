#!/usr/bin/env python3
import polars as pl
import numpy as np
from scipy import stats

df = pl.read_parquet("data/labeled/5min/EURUSD_5m_labeled.parquet")
r30 = df["r30"].drop_nulls().to_numpy()

print(f"Total rows: {len(r30):,}")
print(f"Min: {np.min(r30):.6f}, Max: {np.max(r30):.6f}")
print(f"1st percentile: {np.percentile(r30, 1):.6f}")
print(f"99th percentile: {np.percentile(r30, 99):.6f}")
print(f"Skewness (full): {stats.skew(r30):.4f}")

# Exclude top/bottom 0.01% extreme tails (the black swans like SNB 2015)
p_low, p_high = np.percentile(r30, [0.01, 99.99])
r30_trimmed = r30[(r30 >= p_low) & (r30 <= p_high)]
print(f"Skewness (99.98% mass): {stats.skew(r30_trimmed):.4f}")

# Check the extreme min values
extreme_min = df.filter(pl.col("r30") < -0.02).select(["timestamp", "close", "r30"])
print("\nExtreme negative jumps (r30 < -2% in 30 mins):")
print(extreme_min)
