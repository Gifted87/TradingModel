"""
G4 — Interaction Features & Tail Transforms [A3, A6, A7, F3].

Features:
- Pairwise cross-products among top single-group features [A3, A6, F3]:
    RSI × Bollinger %B, OFI × Spread, CCI × ATR, MACD-hist × RSI
- Absolute-value tail transforms |x| for key features [A7]:
    |OFI|, |r_5m|, |r_30m|, |MACD hist|, |CCI|
- Distance-from-extreme transforms (how far from overbought/oversold) [A7]:
    RSI distance from 50, Stoch-K distance from 50, %B distance from 0.5
- Squared terms for key non-linear magnitudes:
    OFI², Spread²

Evidence:
  [A3] Combination of indicators outperforms single indicators.
  [A6] Cross-products (RSI × %B, OFI × Spread) capture compound signals.
  [A7] V-shaped (absolute-value) partial dependence; tail-distance transforms.
  [F3] Multi-width kernel effects benefit from cross-feature interactions.

NOTE: This module MUST be called after G1, G2, G3, G5, G6 are computed,
      as it reads columns produced by those groups.
"""

import polars as pl

from .registry import REGISTRY


def compute_g4_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute G4 interaction and tail-transform features.

    Requires the following columns to already be present in df:
        G1: g1_ofi_1bar, g1_spread_level
        G2: g2_ret_5m  (or g2_ret_1bar if 5m not present)
        G3: g3_rsi_14, g3_bollinger_pct_b_20, g3_cci_14,
            g3_macd_12_26_hist, g3_stoch_k_14
        G6: g6_rel_atr14
    Missing columns are silently skipped.
    """
    exprs = []
    cols = set(df.columns)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _have(*names) -> bool:
        return all(n in cols for n in names)

    def _add(expr: pl.Expr, name: str, citation: str, desc: str, formula: str):
        exprs.append(expr.alias(name))
        REGISTRY.register(name, "G4", citation, desc, formula)

    # ── 1. Pairwise cross-products [A3, A6, F3] ─────────────────────────────

    # RSI × Bollinger %B: momentum × band position compound signal
    if _have("g3_rsi_14", "g3_bollinger_pct_b_20"):
        _add(
            (pl.col("g3_rsi_14") * pl.col("g3_bollinger_pct_b_20")),
            "g4_rsi14_x_bb_pct_b",
            "[A3, A6]",
            "RSI(14) × Bollinger %B: momentum-band compound signal",
            "g3_rsi_14 × g3_bollinger_pct_b_20",
        )

    # OFI × Spread: impact-cost compound signal [A6]
    if _have("g1_ofi_1bar", "g1_spread_level"):
        _add(
            (pl.col("g1_ofi_1bar") * pl.col("g1_spread_level")),
            "g4_ofi_x_spread",
            "[A6, D1, D2]",
            "OFI × Spread: order-flow adjusted by transaction cost",
            "g1_ofi_1bar × g1_spread_level",
        )

    # CCI × ATR: trend-strength × volatility [A3]
    if _have("g3_cci_14", "g6_rel_atr14"):
        _add(
            (pl.col("g3_cci_14") * pl.col("g6_rel_atr14")),
            "g4_cci14_x_atr14",
            "[A3, A6]",
            "CCI(14) × Relative ATR(14): trend amplitude compound",
            "g3_cci_14 × g6_rel_atr14",
        )

    # MACD histogram × RSI: divergence × momentum [A6, F3]
    if _have("g3_macd_12_26_hist", "g3_rsi_14"):
        _add(
            (pl.col("g3_macd_12_26_hist") * pl.col("g3_rsi_14")),
            "g4_macd_hist_x_rsi14",
            "[A6, F3]",
            "MACD Histogram × RSI(14): cross-indicator momentum signal",
            "g3_macd_12_26_hist × g3_rsi_14",
        )

    # OFI × short-term return: momentum alignment with order flow [A6, D1]
    ret_col = next(
        (c for c in ["g2_return_5m", "g2_return_10m", "g2_ret_1bar", "g2_ret_5m", "g2_ret_10m"] if c in cols), None
    )
    if ret_col and "g1_ofi_1bar" in cols:
        _add(
            (pl.col("g1_ofi_1bar") * pl.col(ret_col)),
            "g4_ofi_x_ret_short",
            "[A6, D1]",
            f"OFI × short-term return ({ret_col}): flow-momentum alignment",
            f"g1_ofi_1bar × {ret_col}",
        )

    # Stochastic K × RSI: dual momentum confirmation [A3, A6]
    if _have("g3_stoch_k_14", "g3_rsi_14"):
        _add(
            (pl.col("g3_stoch_k_14") * pl.col("g3_rsi_14")),
            "g4_stoch_k14_x_rsi14",
            "[A3, A6]",
            "Stochastic %K(14) × RSI(14): dual momentum confirmation",
            "g3_stoch_k_14 × g3_rsi_14",
        )

    # ── 2. Absolute-value tail transforms [A7] ───────────────────────────────
    # V-shaped partial dependence: large |OFI| (either direction) predicts
    # large subsequent moves.

    if "g1_ofi_1bar" in cols:
        _add(
            pl.col("g1_ofi_1bar").abs(),
            "g4_abs_ofi_1bar",
            "[A7]",
            "|OFI| (1-bar): unsigned order-flow magnitude",
            "|g1_ofi_1bar|",
        )

    if ret_col:
        _add(
            pl.col(ret_col).abs(),
            "g4_abs_ret_short",
            "[A7]",
            f"|Return| short-term ({ret_col}): unsigned momentum magnitude",
            f"|{ret_col}|",
        )

    # |MACD histogram|: unsigned divergence strength
    if "g3_macd_12_26_hist" in cols:
        _add(
            pl.col("g3_macd_12_26_hist").abs(),
            "g4_abs_macd_hist",
            "[A7]",
            "|MACD Histogram (12,26,9)|: unsigned divergence",
            "|g3_macd_12_26_hist|",
        )

    # |CCI|: unsigned trend deviation
    if "g3_cci_14" in cols:
        _add(
            pl.col("g3_cci_14").abs(),
            "g4_abs_cci14",
            "[A7]",
            "|CCI(14)|: unsigned cyclical deviation",
            "|g3_cci_14|",
        )

    # ── 3. Distance-from-extreme transforms [A7] ─────────────────────────────
    # How far is the indicator from the neutral / overbought / oversold centre?

    # RSI distance from 50 (centred)
    if "g3_rsi_14" in cols:
        _add(
            (pl.col("g3_rsi_14") - 50.0).abs(),
            "g4_rsi14_dist_from_50",
            "[A7]",
            "|RSI(14) - 50|: distance from neutral",
            "|g3_rsi_14 - 50|",
        )

    # Stochastic %K distance from 50
    if "g3_stoch_k_14" in cols:
        _add(
            (pl.col("g3_stoch_k_14") - 50.0).abs(),
            "g4_stoch_k14_dist_from_50",
            "[A7]",
            "|Stoch %K(14) - 50|: distance from neutral",
            "|g3_stoch_k_14 - 50|",
        )

    # Bollinger %B distance from 0.5 (centre of band)
    if "g3_bollinger_pct_b_20" in cols:
        _add(
            (pl.col("g3_bollinger_pct_b_20") - 0.5).abs(),
            "g4_bb_pct_b_dist_from_mid",
            "[A7]",
            "|Bollinger %B - 0.5|: distance from band mid-point",
            "|g3_bollinger_pct_b_20 - 0.5|",
        )

    # ── 4. Squared magnitude terms ───────────────────────────────────────────

    # OFI² — captures non-linear impact scaling [D1]
    if "g1_ofi_1bar" in cols:
        _add(
            pl.col("g1_ofi_1bar") ** 2,
            "g4_ofi_sq",
            "[D1, A7]",
            "OFI²: non-linear order-flow impact",
            "g1_ofi_1bar²",
        )

    # Spread² — captures volatility-regime clustering
    if "g1_spread_level" in cols:
        _add(
            pl.col("g1_spread_level") ** 2,
            "g4_spread_sq",
            "[D2, A7]",
            "Spread²: cost-regime clustering term",
            "g1_spread_level²",
        )

    if not exprs:
        return df  # Nothing to add; prerequisite columns not present

    return df.with_columns(exprs)
