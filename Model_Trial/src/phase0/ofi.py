"""
OFI (Order Flow Imbalance) proxy construction from tick data.

Implements the Cont, Kukanov & Stoikov (2014) OFI construction adapted for
Dukascopy tick-level bid/ask with volumes.

Formulation (Cont et al. Eq 4):
Bid events:
  - If Bid_t > Bid_{t-1}: e_bid = +BidVol_t (buy pressure lifts bid)
  - If Bid_t < Bid_{t-1}: e_bid = -BidVol_{t-1} (buy support pulled / absorbed)
  - If Bid_t == Bid_{t-1}: e_bid = BidVol_t - BidVol_{t-1} (queue size delta)

Ask events:
  - If Ask_t < Ask_{t-1}: e_ask = +AskVol_t (sell pressure drops ask)
  - If Ask_t > Ask_{t-1}: e_ask = -AskVol_{t-1} (ask lifted / sellers pull back)
  - If Ask_t == Ask_{t-1}: e_ask = AskVol_t - AskVol_{t-1} (queue size delta)

Net Order Flow Imbalance (OFI):
  OFI_t = e_bid_t - e_ask_t
Positive OFI indicates net upward price pressure; Negative OFI indicates downward pressure.
"""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def compute_tick_ofi(ticks: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-tick OFI proxy from tick-level bid/ask/volume data.
    """
    df = ticks.with_columns([
        pl.col("bid").shift(1).alias("prev_bid"),
        pl.col("ask").shift(1).alias("prev_ask"),
        pl.col("bid_vol").shift(1).alias("prev_bid_vol"),
        pl.col("ask_vol").shift(1).alias("prev_ask_vol"),
    ])

    # Bid side event
    e_bid = (
        pl.when(pl.col("bid") > pl.col("prev_bid"))
        .then(pl.col("bid_vol"))
        .when(pl.col("bid") < pl.col("prev_bid"))
        .then(-pl.col("prev_bid_vol"))
        .otherwise(pl.col("bid_vol") - pl.col("prev_bid_vol"))
    )

    # Ask side event (selling pressure is positive e_ask)
    e_ask = (
        pl.when(pl.col("ask") < pl.col("prev_ask"))
        .then(pl.col("ask_vol"))
        .when(pl.col("ask") > pl.col("prev_ask"))
        .then(-pl.col("prev_ask_vol"))
        .otherwise(pl.col("ask_vol") - pl.col("prev_ask_vol"))
    )

    df = df.with_columns([
        e_bid.alias("e_bid"),
        e_ask.alias("e_ask"),
        (e_bid - e_ask).alias("ofi"),
    ])

    return df.drop_nulls(subset=["prev_bid", "prev_ask"])


def aggregate_to_bars(
    tick_ofi: pl.DataFrame,
    bar_duration: str = "1m",
) -> pl.DataFrame:
    """
    Aggregate tick-level data with OFI into time bars.
    """
    bars = tick_ofi.group_by_dynamic(
        "timestamp",
        every=bar_duration,
        closed="left",
        label="left",
    ).agg([
        pl.col("mid").first().alias("open"),
        pl.col("mid").max().alias("high"),
        pl.col("mid").min().alias("low"),
        pl.col("mid").last().alias("close"),
        pl.col("bid").first().alias("bid_open"),
        pl.col("bid").last().alias("bid_close"),
        pl.col("ask").first().alias("ask_open"),
        pl.col("ask").last().alias("ask_close"),
        pl.col("spread").mean().alias("spread_mean"),
        pl.col("spread").max().alias("spread_max"),
        pl.col("mid").count().alias("tick_count"),
        pl.col("bid_vol").sum().alias("bid_vol_sum"),
        pl.col("ask_vol").sum().alias("ask_vol_sum"),
        pl.col("ofi").sum().alias("ofi_sum"),
        pl.col("e_bid").sum().alias("e_bid_sum"),
        pl.col("e_ask").sum().alias("e_ask_sum"),
    ])

    bars = bars.rename({"timestamp": "bar_time"})
    bars = bars.with_columns([
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("mid_return"),
    ])

    return bars.drop_nulls(subset=["mid_return"])


def compute_depth_proxy(bars: pl.DataFrame) -> pl.DataFrame:
    """Add rolling depth proxies and OFI/depth ratio."""
    bars = bars.with_columns([
        (pl.col("bid_vol_sum") + pl.col("ask_vol_sum")).alias("total_vol"),
    ])

    bars = bars.with_columns([
        pl.col("total_vol").rolling_mean(window_size=5).alias("depth_proxy_5"),
        pl.col("total_vol").rolling_mean(window_size=10).alias("depth_proxy_10"),
    ])

    bars = bars.with_columns([
        (pl.col("ofi_sum") / (pl.col("depth_proxy_5") + 1e-6)).alias("ofi_depth_ratio_5"),
        (pl.col("ofi_sum") / (pl.col("depth_proxy_10") + 1e-6)).alias("ofi_depth_ratio_10"),
    ])

    return bars
