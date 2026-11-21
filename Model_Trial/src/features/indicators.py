"""
G3 — Continuous Indicator Core [A1, A4, A5, E2].

Features:
- CCI (Commodity Channel Index) at periods 14, 28 [A4].
- Stochastic Oscillator (%K, %D) at periods 14, 28 [A4].
- Bollinger Bands (%B and Bandwidth) at period 20 [A4].
- RSI (Relative Strength Index) at periods 14, 28 [A5].
- TRIX (Triple Exponential Smoothed Rate of Change) at periods 14, 30 [A5].
- MACD (Moving Average Convergence Divergence) at (12, 26, 9) and (24, 52, 18) [A5, A6].

Strictly outputs continuous normalized values; discrete buy/sell rules are never used per [A1].
"""

import numpy as np
import polars as pl

from .registry import REGISTRY

EPS = 1e-7


def compute_g3_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Compute G3 continuous technical indicators."""
    exprs = []
    
    # 1. Commodity Channel Index (CCI) [A4]
    # CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Absolute Deviation(TP))
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    
    for period in [14, 28]:
        sma_tp = tp.rolling_mean(window_size=period).fill_null(tp)
        # Mean Absolute Deviation
        mad_tp = (tp - sma_tp).abs().rolling_mean(window_size=period).fill_null(EPS)
        cci = ((tp - sma_tp) / (0.015 * mad_tp + EPS)).fill_null(0.0)
        
        feat_name = f"g3_cci_{period}"
        exprs.append(cci.alias(feat_name))
        REGISTRY.register(feat_name, "G3", "[A4]", f"Commodity Channel Index (p={period})", "(TP - SMA(TP)) / (0.015 * MAD)")

    # 2. Stochastic Oscillator (%K, %D) [A4]
    # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    # %D = SMA(%K, 3)
    for period in [14, 28]:
        ll = pl.col("low").rolling_min(window_size=period).fill_null(pl.col("low"))
        hh = pl.col("high").rolling_max(window_size=period).fill_null(pl.col("high"))
        
        stoch_k = (((pl.col("close") - ll) / (hh - ll + EPS)) * 100.0).fill_null(50.0)
        stoch_d = stoch_k.rolling_mean(window_size=3).fill_null(50.0)
        
        k_name = f"g3_stoch_k_{period}"
        d_name = f"g3_stoch_d_{period}"
        exprs.extend([stoch_k.alias(k_name), stoch_d.alias(d_name)])
        REGISTRY.register(k_name, "G3", "[A4]", f"Stochastic %K (p={period})", "(Close - LL) / (HH - LL) * 100")
        REGISTRY.register(d_name, "G3", "[A4]", f"Stochastic %D (p={period}, d=3)", "SMA(Stoch_K, 3)")

    # 3. Bollinger Bands (%B and Bandwidth) [A4]
    # %B = (Price - Lower Band) / (Upper Band - Lower Band)
    # Bandwidth = (Upper Band - Lower Band) / Middle Band
    bb_period = 20
    bb_mid = pl.col("close").rolling_mean(window_size=bb_period).fill_null(pl.col("close"))
    bb_std = pl.col("close").rolling_std(window_size=bb_period).fill_null(0.0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    bb_pct_b = ((pl.col("close") - bb_lower) / (bb_upper - bb_lower + EPS)).fill_null(0.5)
    bb_width = ((bb_upper - bb_lower) / (bb_mid + EPS)).fill_null(0.0)
    
    exprs.extend([
        bb_pct_b.alias("g3_bollinger_pct_b_20"),
        bb_width.alias("g3_bollinger_bandwidth_20"),
    ])
    REGISTRY.register("g3_bollinger_pct_b_20", "G3", "[A4]", "Bollinger %B (p=20, k=2)", "(Close - Lower) / (Upper - Lower)")
    REGISTRY.register("g3_bollinger_bandwidth_20", "G3", "[A4]", "Bollinger Bandwidth (p=20, k=2)", "(Upper - Lower) / Mid")

    # 4. Relative Strength Index (RSI) [A5]
    # Wilder's smoothed RSI = 100 - (100 / (1 + RS))
    diff = pl.col("close") - pl.col("close").shift(1)
    gain = pl.when(diff > 0).then(diff).otherwise(0.0).fill_null(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0).fill_null(0.0)
    
    for period in [14, 28]:
        avg_gain = gain.ewm_mean(span=period, ignore_nulls=True).fill_null(0.0)
        avg_loss = loss.ewm_mean(span=period, ignore_nulls=True).fill_null(0.0)
        rs = avg_gain / (avg_loss + EPS)
        rsi = (100.0 - (100.0 / (1.0 + rs))).fill_null(50.0)
        
        feat_name = f"g3_rsi_{period}"
        exprs.append(rsi.alias(feat_name))
        REGISTRY.register(feat_name, "G3", "[A5]", f"Relative Strength Index (p={period})", "100 - 100 / (1 + RS)")

    # 5. TRIX (Triple Exponential Smoothed Rate of Change) [A5]
    # TRIX = (EMA3_t - EMA3_{t-1}) / EMA3_{t-1} * 10000
    for period in [14, 30]:
        ema1 = pl.col("close").ewm_mean(span=period, ignore_nulls=True)
        ema2 = ema1.ewm_mean(span=period, ignore_nulls=True)
        ema3 = ema2.ewm_mean(span=period, ignore_nulls=True)
        trix = (((ema3 - ema3.shift(1)) / (ema3.shift(1) + EPS)) * 10000.0).fill_null(0.0)
        
        feat_name = f"g3_trix_{period}"
        exprs.append(trix.alias(feat_name))
        REGISTRY.register(feat_name, "G3", "[A5]", f"TRIX indicator (p={period})", "d(EMA(EMA(EMA(Close)))) / EMA3")

    # 6. MACD (Fast, Slow, Signal) [A5, A6]
    for fast, slow, sig in [(12, 26, 9), (24, 52, 18)]:
        fast_ema = pl.col("close").ewm_mean(span=fast, ignore_nulls=True)
        slow_ema = pl.col("close").ewm_mean(span=slow, ignore_nulls=True)
        macd_line = (fast_ema - slow_ema).fill_null(0.0)
        macd_signal = macd_line.ewm_mean(span=sig, ignore_nulls=True).fill_null(0.0)
        macd_hist = (macd_line - macd_signal).fill_null(0.0)
        
        prefix = f"g3_macd_{fast}_{slow}"
        exprs.extend([
            macd_line.alias(f"{prefix}_line"),
            macd_signal.alias(f"{prefix}_signal"),
            macd_hist.alias(f"{prefix}_hist"),
        ])
        REGISTRY.register(f"{prefix}_line", "G3", "[A5]", f"MACD Line ({fast},{slow})", "EMA(fast) - EMA(slow)")
        REGISTRY.register(f"{prefix}_signal", "G3", "[A5]", f"MACD Signal ({fast},{slow},{sig})", "EMA(MACD, sig)")
        REGISTRY.register(f"{prefix}_hist", "G3", "[A6]", f"MACD Histogram ({fast},{slow},{sig})", "MACD_line - MACD_signal")

    return df.with_columns(exprs)
