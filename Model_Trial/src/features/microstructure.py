"""
G1 — Microstructure Core [D1, D2, D3, D5].

Features:
- OFI proxy: signed quote events × volume.
- Rolling OFI sums at 1-bar, 2-bar, 3-bar, 6-bar (5m, 10m, 15m, 30m).
- Normalized volume / tick intensity.
- Spread level and spread delta (ask - bid).
- OFI / depth-proxy ratio (running impact coefficient estimate [D2]).

This group carries the Branch A linear core.
"""

import numpy as np
import polars as pl

from .registry import REGISTRY


def compute_g1_microstructure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute G1 Microstructure features from bar DataFrame.
    """
    exprs = []
    
    # 1. Spread dynamics
    if "spread_close" in df.columns:
        spread_col = "spread_close"
    elif "spread" in df.columns:
        spread_col = "spread"
    else:
        spread_col = "spread_close"
        df = df.with_columns((pl.col("ask_close") - pl.col("bid_close")).alias(spread_col))
        
    exprs.extend([
        pl.col(spread_col).alias("g1_spread_level"),
        (pl.col(spread_col) - pl.col(spread_col).shift(1)).fill_null(0.0).alias("g1_spread_delta"),
    ])
    
    REGISTRY.register("g1_spread_level", "G1", "[D2, D4]", "Prevailing spread level", "ask_close - bid_close")
    REGISTRY.register("g1_spread_delta", "G1", "[D2]", "1-bar spread change", "spread_t - spread_{t-1}")
    
    # 2. Volume & Order Flow Imbalance (OFI)
    # If explicit tick-calculated ofi_sum exists in df, use it; otherwise compute proxy from bid/ask quote volumes
    if "ofi_sum" in df.columns:
        ofi_base = pl.col("ofi_sum")
    else:
        # Candle-level OFI proxy:
        # e_bid = bid_vol * sign(bid_close - bid_open)
        # e_ask = ask_vol * sign(ask_close - ask_open)
        # ofi = e_bid - e_ask
        bid_delta = pl.col("bid_close") - pl.col("bid_open")
        ask_delta = pl.col("ask_close") - pl.col("ask_open")
        e_bid = pl.col("bid_volume") * bid_delta.sign()
        e_ask = pl.col("ask_volume") * ask_delta.sign()
        ofi_base = e_bid - e_ask

    exprs.append(ofi_base.alias("g1_ofi_1bar"))
    REGISTRY.register("g1_ofi_1bar", "G1", "[D1, D3]", "Order Flow Imbalance (1-bar)", "e_bid - e_ask")
    
    # Rolling OFI sums across multi-bar horizons: 2-bar (10m), 3-bar (15m), 6-bar (30m)
    for w, name in [(2, "2bar"), (3, "3bar"), (6, "6bar")]:
        feat_name = f"g1_ofi_{name}_sum"
        exprs.append(ofi_base.rolling_sum(window_size=w).fill_null(0.0).alias(feat_name))
        REGISTRY.register(feat_name, "G1", "[D1]", f"Rolling OFI sum ({w} bars)", f"sum(OFI, w={w})")
        
    # 3. Depth proxy and impact coefficient ratio (OFI / depth [D2])
    vol_col = "volume" if "volume" in df.columns else "bid_volume"
    total_vol = pl.col(vol_col)
    
    for w, name in [(3, "3bar"), (6, "6bar"), (12, "12bar")]:
        depth_name = f"g1_depth_{name}"
        ratio_name = f"g1_ofi_depth_ratio_{name}"
        
        depth_expr = total_vol.rolling_mean(window_size=w).fill_null(1.0)
        exprs.append(depth_expr.alias(depth_name))
        REGISTRY.register(depth_name, "G1", "[D2]", f"Rolling market depth proxy ({w} bars)", f"mean(volume, w={w})")
        
        ratio_expr = (ofi_base / (depth_expr + 1e-6)).fill_null(0.0)
        exprs.append(ratio_expr.alias(ratio_name))
        REGISTRY.register(ratio_name, "G1", "[D1, D2]", f"OFI-to-depth impact coefficient ratio ({w} bars)", f"OFI / depth_{w}")

    return df.with_columns(exprs)
