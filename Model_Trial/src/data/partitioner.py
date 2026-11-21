"""
Phase 1 Partitioning Utility.

Partitions bar datasets by symbol, year, and month into Hive-style Parquet layout:
  data/bars/partitioned/symbol={SYMBOL}/year={YYYY}/month={MM}/data.parquet

Ensures fast, memory-efficient, targeted loading for walk-forward validation and backtesting.
"""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def partition_symbol_dataset(
    input_parquet: Path,
    output_root: Path,
    symbol: str,
    timestamp_col: str = "timestamp",
) -> int:
    """
    Partition a single symbol's parquet file into Hive-partitioned directory tree.

    Returns:
        Total number of partition files written.
    """
    input_parquet = Path(input_parquet)
    output_root = Path(output_root)
    
    logger.info(f"Partitioning {symbol} dataset from {input_parquet}...")
    df = pl.read_parquet(input_parquet)
    
    if df[timestamp_col].dtype != pl.Datetime:
        df = df.with_columns(pl.col(timestamp_col).cast(pl.Datetime))
        
    df = df.with_columns([
        pl.col(timestamp_col).dt.year().alias("year"),
        pl.col(timestamp_col).dt.month().alias("month"),
    ])
    
    years = sorted(df["year"].unique().to_list())
    total_partitions = 0
    
    for y in years:
        year_df = df.filter(pl.col("year") == y)
        months = sorted(year_df["month"].unique().to_list())
        
        for m in months:
            month_df = year_df.filter(pl.col("month") == m).drop(["year", "month"])
            
            part_dir = output_root / f"symbol={symbol}" / f"year={y:04d}" / f"month={m:02d}"
            part_dir.mkdir(parents=True, exist_ok=True)
            
            out_file = part_dir / "bars.parquet"
            month_df.write_parquet(out_file, compression="zstd")
            total_partitions += 1
            
    logger.info(f"Successfully partitioned {symbol} into {total_partitions} monthly Parquet files.")
    return total_partitions


def partition_all_symbols(
    bars_dir: Path = Path("data/bars/5min"),
    output_root: Path = Path("data/bars/partitioned"),
) -> dict[str, int]:
    """Partition all symbol parquets found in bars_dir."""
    bars_dir = Path(bars_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(bars_dir.glob("*_5m.parquet"))
    results = {}
    
    for f in parquet_files:
        symbol = f.stem.replace("_5m", "")
        parts = partition_symbol_dataset(f, output_root, symbol)
        results[symbol] = parts
        
    return results
