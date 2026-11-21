"""
G5 — Time / Regime Conditioning [A2, D4].

Features:
- Cyclical time-of-day encoding: sin/cos of bar minute-of-day [A2, D4].
- Session flags: Asian, London, New York, and overlap windows.
- Time-of-day bucket (6 × 4h buckets) for per-regime metric disaggregation.
- Critical cross-term interactions: Time × OFI [A2, D4] and Time × ATR [A2, D4].

Evidence:
  [A2] Time-of-day is the strongest single regime indicator for intraday predictability.
  [D4] OFI impact coefficient varies systematically with time (intraday seasonality).
  Per the Build Plan: G5 must be present before Phase 4 baseline training.
"""

import polars as pl

from .registry import REGISTRY

# Session boundaries in UTC (hour of day, 0-indexed).
# FX market sessions:
_ASIAN_START   = 0    # 00:00 UTC (Tokyo open)
_ASIAN_END     = 9    # 09:00 UTC (Tokyo / Singapore close)
_LONDON_START  = 7    # 07:00 UTC (London open)
_LONDON_END    = 16   # 16:00 UTC (London close)
_NY_START      = 12   # 12:00 UTC (New York open)
_NY_END        = 21   # 21:00 UTC (New York close)

# Number of minutes in a day.
_MINS_IN_DAY = 1440


def compute_g5_time_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute G5 time-of-day / session-regime features.

    Expects df to have a 'timestamp' column (Datetime dtype) and
    already-computed G1/G6 columns (g1_ofi_1bar, g6_rel_atr14) for
    cross-term interactions.
    """
    exprs = []

    # ── 1. Cyclical time-of-day encoding [A2, D4] ──────────────────────────
    # minute_of_day ∈ [0, 1439]; encode as unit-circle coordinates so the
    # model can learn smooth daily patterns without a discontinuity at midnight.
    minute_of_day = (
        pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
    ).cast(pl.Float64)

    tod_sin = (2.0 * 3.141592653589793 * minute_of_day / _MINS_IN_DAY).sin()
    tod_cos = (2.0 * 3.141592653589793 * minute_of_day / _MINS_IN_DAY).cos()

    exprs.extend([
        tod_sin.alias("g5_tod_sin"),
        tod_cos.alias("g5_tod_cos"),
    ])
    REGISTRY.register("g5_tod_sin", "G5", "[A2, D4]",
                      "Cyclical time-of-day (sine)", "sin(2π × minute_of_day / 1440)")
    REGISTRY.register("g5_tod_cos", "G5", "[A2, D4]",
                      "Cyclical time-of-day (cosine)", "cos(2π × minute_of_day / 1440)")

    # ── 2. Hour-of-day (raw, integer) ──────────────────────────────────────
    hour = pl.col("timestamp").dt.hour().cast(pl.Int32)
    exprs.append(hour.alias("g5_hour_of_day"))
    REGISTRY.register("g5_hour_of_day", "G5", "[A2]",
                      "Raw UTC hour of bar", "timestamp.hour")

    # ── 3. Session flags (Boolean → Float 0/1) ─────────────────────────────
    asian_flag   = ((hour >= _ASIAN_START)  & (hour < _ASIAN_END)).cast(pl.Float64)
    london_flag  = ((hour >= _LONDON_START) & (hour < _LONDON_END)).cast(pl.Float64)
    ny_flag      = ((hour >= _NY_START)     & (hour < _NY_END)).cast(pl.Float64)
    # Overlaps: highest-volatility windows per [A2]
    lon_ny_flag  = ((hour >= _NY_START)     & (hour < _LONDON_END)).cast(pl.Float64)
    asia_lon_flag = ((hour >= _LONDON_START) & (hour < _ASIAN_END)).cast(pl.Float64)

    exprs.extend([
        asian_flag.alias("g5_session_asian"),
        london_flag.alias("g5_session_london"),
        ny_flag.alias("g5_session_ny"),
        lon_ny_flag.alias("g5_session_lon_ny_overlap"),
        asia_lon_flag.alias("g5_session_asia_lon_overlap"),
    ])
    for name, desc in [
        ("g5_session_asian",          "Asian session flag (00-09 UTC)"),
        ("g5_session_london",         "London session flag (07-16 UTC)"),
        ("g5_session_ny",             "New York session flag (12-21 UTC)"),
        ("g5_session_lon_ny_overlap", "London–NY overlap flag (12-16 UTC)"),
        ("g5_session_asia_lon_overlap","Asia–London overlap flag (07-09 UTC)"),
    ]:
        REGISTRY.register(name, "G5", "[A2, D4]", desc, "Boolean session window")

    # ── 4. Time-of-day bucket (0–5, corresponding to 4-hour windows) ───────
    # Used for per-regime metric disaggregation in Gate 5 / Phase 5.
    bucket = (hour // 4).cast(pl.Int32)
    exprs.append(bucket.alias("g5_tod_bucket"))
    REGISTRY.register("g5_tod_bucket", "G5", "[A2, D4]",
                      "4-hour time-of-day bucket (0=00-04h … 5=20-24h)", "floor(hour / 4)")

    # ── 5. Day-of-week (0=Mon … 4=Fri) ────────────────────────────────────
    dow = pl.col("timestamp").dt.weekday().cast(pl.Int32)
    exprs.append(dow.alias("g5_day_of_week"))
    REGISTRY.register("g5_day_of_week", "G5", "[A2]",
                      "Day of week (0=Mon … 4=Fri)", "timestamp.weekday()")

    # ── 6. Cross-term: Time × OFI [A2, D4] ─────────────────────────────────
    # The impact coefficient of OFI varies across the day (D4). Multiplying
    # the cyclical encoding directly into OFI lets the model learn this
    # modulation without requiring explicit regime dummies.
    if "g1_ofi_1bar" in df.columns:
        ofi = pl.col("g1_ofi_1bar")
        exprs.extend([
            (ofi * tod_sin).alias("g5_ofi_x_tod_sin"),
            (ofi * tod_cos).alias("g5_ofi_x_tod_cos"),
        ])
        REGISTRY.register("g5_ofi_x_tod_sin", "G5", "[A2, D4]",
                          "OFI × sin(ToD): impact coefficient time modulation (sine)", "g1_ofi_1bar × g5_tod_sin")
        REGISTRY.register("g5_ofi_x_tod_cos", "G5", "[A2, D4]",
                          "OFI × cos(ToD): impact coefficient time modulation (cosine)", "g1_ofi_1bar × g5_tod_cos")
    else:
        # G1 not yet computed; cross-terms will be added downstream
        pass

    # ── 7. Cross-term: Time × ATR [A2, D4] ─────────────────────────────────
    # Volatility is session-dependent; relative ATR conditioned on session
    # gives the model a direct handle on regime-adjusted magnitude.
    if "g6_rel_atr14" in df.columns:
        atr = pl.col("g6_rel_atr14")
        exprs.extend([
            (atr * tod_sin).alias("g5_atr_x_tod_sin"),
            (atr * tod_cos).alias("g5_atr_x_tod_cos"),
            (atr * london_flag).alias("g5_atr_x_london"),
            (atr * ny_flag).alias("g5_atr_x_ny"),
        ])
        REGISTRY.register("g5_atr_x_tod_sin", "G5", "[A2, D4]",
                          "ATR × sin(ToD): session-scaled volatility (sine)", "g6_rel_atr14 × g5_tod_sin")
        REGISTRY.register("g5_atr_x_tod_cos", "G5", "[A2, D4]",
                          "ATR × cos(ToD): session-scaled volatility (cosine)", "g6_rel_atr14 × g5_tod_cos")
        REGISTRY.register("g5_atr_x_london", "G5", "[A2, D4]",
                          "ATR × London session flag", "g6_rel_atr14 × g5_session_london")
        REGISTRY.register("g5_atr_x_ny", "G5", "[A2, D4]",
                          "ATR × NY session flag", "g6_rel_atr14 × g5_session_ny")

    return df.with_columns(exprs)
