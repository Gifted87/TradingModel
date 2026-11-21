#!/usr/bin/env python3
import polars as pl

df = pl.read_parquet("data/bars/5min/EURUSD_5m.parquet")
idx = df.filter(pl.col("timestamp") >= pl.datetime(2007, 5, 30, 23, 55))
print("Rows around gap:")
print(idx.head(5).select(["timestamp", "close", "return_5m"]))
