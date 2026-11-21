"""
G2 — Multi-Scale Momentum [B3, F1].

Features:
- Mid-price continuous log returns across multi-horizon windows (5m, 10m, 15m, 30m, 1h, 2h, 6h, 24h).
- Return momentum acceleration / gradient: r_short - r_long.

Replicates the multi-scale temporal attention observed in DeepLOB [B3].
"""

import polars as pl

from .registry import REGISTRY


def compute_g2_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """Compute G2 Multi-scale Momentum features."""
    exprs = []
    
    # Mid-price log returns over windows
    windows = [
        (1, "5m"),
        (2, "10m"),
        (3, "15m"),
        (6, "30m"),
        (12, "1h"),
        (24, "2h"),
        (72, "6h"),
        (288, "24h"),
    ]
    
    for w, label in windows:
        feat_name = f"g2_return_{label}"
        ret_expr = (pl.col("close").log() - pl.col("close").shift(w).log()).fill_null(0.0)
        exprs.append(ret_expr.alias(feat_name))
        REGISTRY.register(
            feat_name, "G2", "[B3, F1]", f"Multi-horizon log return ({label})", f"log(Close_t / Close_{{t-{w}}})"
        )

    # Momentum Acceleration / Gradient
    # Short-term return vs medium-term return (e.g. 5m return - 30m return / 6)
    accel_1 = (pl.col("close").log() - pl.col("close").shift(1).log()) - (
        (pl.col("close").log() - pl.col("close").shift(6).log()) / 6.0
    )
    exprs.append(accel_1.fill_null(0.0).alias("g2_momentum_accel_5m_30m"))
    REGISTRY.register(
        "g2_momentum_accel_5m_30m", "G2", "[F1]", "Momentum acceleration (5m vs 30m avg)", "r_5m - r_30m/6"
    )

    accel_2 = (pl.col("close").log() - pl.col("close").shift(3).log()) - (
        (pl.col("close").log() - pl.col("close").shift(12).log()) / 4.0
    )
    exprs.append(accel_2.fill_null(0.0).alias("g2_momentum_accel_15m_1h"))
    REGISTRY.register(
        "g2_momentum_accel_15m_1h", "G2", "[F1]", "Momentum acceleration (15m vs 1h avg)", "r_15m - r_1h/4"
    )

    return df.with_columns(exprs)
