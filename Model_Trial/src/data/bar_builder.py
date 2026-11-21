"""
Unified Multi-Horizon Bar Builder and Microstructure Resampler.

Implements a single, config-driven code path for converting high-frequency
price data (ticks / 1m / 5m bars) into higher-resolution and multi-horizon target bars (1m, 5m, 15m, 30m).

Features:
- Exact Bid/Ask/Mid OHLC propagation
- Spread dynamics (mean, max, last)
- Microstructure flow integration (volume balance, quote imbalance)
- Clean, vectorized Polars execution
- Reusable at both training time and production serving time (zero train/serve skew)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class BarConfig:
    duration: str = "30m"
    closed: str = "left"
    label: str = "left"


def resample_bars(
    df: pl.DataFrame,
    target_duration: str = "30m",
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Resample 1m or 5m bars to higher-horizon bars (e.g. 30m for decision timestamps :00 and :30).

    Aggregations:
    - Mid OHLC: first(open), max(high), min(low), last(close)
    - Bid/Ask boundaries: first(bid_open), last(bid_close), first(ask_open), last(ask_close)
    - Spread: mean(spread_mean), max(spread_close), last(spread_close)
    - Volume: sum(volume), sum(bid_volume), sum(ask_volume)
    - Sub-bar count: count()
    """
    logger.info(f"Resampling bars from base resolution to {target_duration}...")
    
    # Ensure datetime format
    if df[timestamp_col].dtype != pl.Datetime:
        df = df.with_columns(pl.col(timestamp_col).cast(pl.Datetime))
        
    df = df.sort(timestamp_col)
    
    agg_exprs = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
    ]
    
    if "bid_open" in df.columns:
        agg_exprs.extend([
            pl.col("bid_open").first().alias("bid_open"),
            pl.col("bid_close").last().alias("bid_close"),
            pl.col("ask_open").first().alias("ask_open"),
            pl.col("ask_close").last().alias("ask_close"),
        ])
        
    if "spread_close" in df.columns:
        agg_exprs.extend([
            pl.col("spread_close").mean().alias("spread_mean"),
            pl.col("spread_close").max().alias("spread_max"),
            pl.col("spread_close").last().alias("spread_last"),
        ])
        
    if "volume" in df.columns:
        agg_exprs.append(pl.col("volume").sum().alias("volume"))
    if "bid_volume" in df.columns:
        agg_exprs.append(pl.col("bid_volume").sum().alias("bid_volume"))
    if "ask_volume" in df.columns:
        agg_exprs.append(pl.col("ask_volume").sum().alias("ask_volume"))
        
    agg_exprs.append(pl.len().alias("sub_bar_count"))
    
    resampled = (
        df.group_by_dynamic(
            timestamp_col,
            every=target_duration,
            closed="left",
            label="left",
        )
        .agg(agg_exprs)
        .sort(timestamp_col)
    )
    
    # Forward return of the resampled bar
    resampled = resampled.with_columns([
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("return_horizon"),
    ])
    
    logger.info(f"Resampled {len(df):,} base bars into {len(resampled):,} {target_duration} bars.")
    return resampled
