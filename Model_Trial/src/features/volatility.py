"""
G6 — Volatility State [E1, E2, E4, E5].

Features:
- HAR-style log-Realized Volatility at 5m, 15m, 30m, 1h, 1d horizons.
- Parkinson range-based volatility estimator.
- Garman-Klass extended OHLC volatility estimator.
- Log-scaled Average True Range ATR(14) and relative ATR (ATR / Close).

Feeds primarily Head 2 (magnitude cascade) and Head 3 (quantiles).
"""

import numpy as np
import polars as pl

from .registry import REGISTRY

EPS = 1e-7


def compute_g6_volatility(df: pl.DataFrame) -> pl.DataFrame:
    """Compute G6 Volatility State features."""
    exprs = []
    
    # 1-step mid-price log return
    log_ret = (pl.col("close").log() - pl.col("close").shift(1).log()).fill_null(0.0)
    ret_sq = log_ret ** 2
    
    # 1. HAR-style Realized Volatilities (in log space per [E5])
    # Windows for 5m bars: 1 (5m), 3 (15m), 6 (30m), 12 (1h), 72 (6h), 288 (1d)
    har_windows = [
        (1, "5m"),
        (3, "15m"),
        (6, "30m"),
        (12, "1h"),
        (72, "6h"),
        (288, "1d"),
    ]
    
    for w, label in har_windows:
        feat_name = f"g6_log_rv_{label}"
        rv_expr = ((ret_sq.rolling_sum(window_size=w).fill_null(EPS) + EPS).sqrt() + EPS).log()
        exprs.append(rv_expr.alias(feat_name))
        REGISTRY.register(
            feat_name, "G6", "[E1, E5]", f"HAR-style log realized volatility ({label})", f"log(sqrt(sum(r^2, w={w})))"
        )

    # 2. Parkinson Range-Based Volatility: sqrt( 1/(4*ln(2)) * ln(H/L)^2 )
    # Constant 1 / (4 * ln 2) ≈ 0.36067376
    log_hl = (pl.col("high").log() - pl.col("low").log()).fill_null(0.0)
    parkinson_1bar = ((0.36067376 * (log_hl ** 2)) + EPS).sqrt()
    
    exprs.append(parkinson_1bar.log().alias("g6_log_parkinson_1bar"))
    REGISTRY.register("g6_log_parkinson_1bar", "G6", "[E1, E2]", "Parkinson range volatility (1-bar)", "log(Parkinson_1)")
    
    parkinson_12bar = (
        ((0.36067376 * (log_hl ** 2)).rolling_mean(window_size=12).fill_null(EPS) + EPS).sqrt()
    ).log()
    exprs.append(parkinson_12bar.alias("g6_log_parkinson_12bar"))
    REGISTRY.register("g6_log_parkinson_12bar", "G6", "[E1, E2]", "Parkinson range volatility (12-bar)", "log(Parkinson_12)")

    # 3. Garman-Klass Volatility: 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2
    log_co = (pl.col("close").log() - pl.col("open").log()).fill_null(0.0)
    gk_stat = 0.5 * (log_hl ** 2) - 0.38629436 * (log_co ** 2)
    gk_1bar = (pl.when(gk_stat > 0).then(gk_stat).otherwise(EPS).sqrt() + EPS).log()
    exprs.append(gk_1bar.alias("g6_log_garman_klass_1bar"))
    REGISTRY.register("g6_log_garman_klass_1bar", "G6", "[E1]", "Garman-Klass OHLC volatility", "log(sqrt(GK_1))")

    # 4. ATR(14) in relative and log terms [E2]
    prev_close = pl.col("close").shift(1)
    tr1 = pl.col("high") - pl.col("low")
    tr2 = (pl.col("high") - prev_close).abs()
    tr3 = (pl.col("low") - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3).fill_null(tr1)
    
    # EWM ATR(14)
    atr14 = tr.ewm_mean(span=14, ignore_nulls=True).fill_null(EPS)
    exprs.append((atr14 + EPS).log().alias("g6_log_atr14"))
    REGISTRY.register("g6_log_atr14", "G6", "[E2, E5]", "Log-scaled 14-period Average True Range", "log(ATR(14))")
    
    # Relative ATR normalized by price
    rel_atr14 = (atr14 / (pl.col("close") + EPS)).fill_null(0.0)
    exprs.append(rel_atr14.alias("g6_rel_atr14"))
    REGISTRY.register("g6_rel_atr14", "G6", "[E2]", "Relative ATR percentage", "ATR(14) / Close")

    return df.with_columns(exprs)
