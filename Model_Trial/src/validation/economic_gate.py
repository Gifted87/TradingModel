"""
Phase 5 — Economic Gate.

Implements the minimal decision rule from the build plan (Phase 5, point 4)
and whitepaper §7.3 [C2, F2]:

    Act only when the predicted direction is confident AND the predicted
    magnitude exceeds k × actual historical spread.

Rule:
    trade_mask = |r̂30| > k × spread
    pnl_per_bar = sign(r̂30) × r30   (on trade_mask rows only)
    net_pnl     = pnl_per_bar - spread   (deduct one spread as round-trip cost)

The [F2] finding shows that ~2% accuracy differences separate profitable from
unprofitable strategies — so the gate uses *real Dukascopy spreads*, not
assumed ones.  The `spread_col` parameter must reference the actual spread
column from the bar data.

Reports net PnL per bar, trades taken (count + fraction), and a Sharpe ratio
of the trade-by-trade PnL series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class EconomicGateReport:
    """Economic gate results for one fold."""
    fold_idx: int
    k: float                      # magnitude threshold multiplier
    n_total: int
    n_trades: int                 # rows where |r̂30| > k × spread
    trade_fraction: float         # n_trades / n_total
    gross_pnl_per_bar: float      # mean(sign(r̂30) × r30) over trade rows
    net_pnl_per_bar: float        # gross - mean(spread) over trade rows
    pnl_sharpe: float             # mean / std of per-trade net PnL
    positive_net_pnl: bool        # net_pnl_per_bar > 0
    mean_spread_used: float       # mean spread on trade rows


def evaluate_economic_gate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spreads: np.ndarray,
    fold_idx: int = 0,
    k: float = 1.5,
) -> EconomicGateReport:
    """
    Compute economic gate results for one walk-forward test fold.

    Parameters
    ----------
    y_true : np.ndarray
        Realised 30-min log-returns r30.
    y_pred : np.ndarray
        Predicted 30-min log-returns r̂30.
    spreads : np.ndarray
        Actual spread (in log-return units) at each bar — the cost of the
        round trip.  Sourced from `spread_close / mid_price`.
    fold_idx : int
        Fold index for bookkeeping.
    k : float
        Threshold multiplier.  Trade when |r̂30| > k × spread.
    """
    n = len(y_true)
    assert len(y_pred) == n
    assert len(spreads) == n

    # Valid rows (non-NaN)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(spreads))
    y_t = y_true[valid]
    y_p = y_pred[valid]
    sp = spreads[valid]

    # Trade decision
    trade_mask = np.abs(y_p) > k * sp
    n_trades = int(np.sum(trade_mask))

    if n_trades == 0:
        return EconomicGateReport(
            fold_idx=fold_idx,
            k=k,
            n_total=n,
            n_trades=0,
            trade_fraction=0.0,
            gross_pnl_per_bar=0.0,
            net_pnl_per_bar=0.0,
            pnl_sharpe=0.0,
            positive_net_pnl=False,
            mean_spread_used=float(np.mean(sp)) if len(sp) > 0 else 0.0,
        )

    direction_pred = np.sign(y_p[trade_mask])
    realized = y_t[trade_mask]
    trade_spreads = sp[trade_mask]

    gross_pnl = direction_pred * realized          # per-trade gross
    net_pnl = gross_pnl - trade_spreads            # deduct one spread

    gross_mean = float(np.mean(gross_pnl))
    net_mean = float(np.mean(net_pnl))
    net_std = float(np.std(net_pnl))
    sharpe = float(net_mean / net_std) if net_std > 1e-12 else 0.0

    return EconomicGateReport(
        fold_idx=fold_idx,
        k=k,
        n_total=n,
        n_trades=n_trades,
        trade_fraction=float(n_trades / max(1, np.sum(valid))),
        gross_pnl_per_bar=gross_mean,
        net_pnl_per_bar=net_mean,
        pnl_sharpe=sharpe,
        positive_net_pnl=(net_mean > 0.0),
        mean_spread_used=float(np.mean(trade_spreads)),
    )


def extract_spreads(df: pl.DataFrame, spread_col: str = "spread_close") -> np.ndarray:
    """
    Extract spread series in log-return units.

    If `spread_col` contains raw price spreads (e.g. 0.0001 for EURUSD),
    divide by the mid-price (close) to convert to fractional units
    consistent with log-return targets.
    """
    if spread_col not in df.columns:
        # Fall back to a small constant to avoid crashing; emit a warning.
        import warnings
        warnings.warn(
            f"Column '{spread_col}' not found; using constant spread of 0.0001.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(len(df), 0.0001)

    raw_spread = df[spread_col].to_numpy().astype(float)

    # If we have a mid-price column, normalise.
    if "close" in df.columns:
        mid = df["close"].to_numpy().astype(float)
        mid = np.where(mid > 0, mid, np.nan)
        spread = raw_spread / mid
    else:
        spread = raw_spread

    # Replace NaN/inf with the median to keep the gate functional.
    finite = np.isfinite(spread)
    if np.any(finite):
        spread = np.where(finite, spread, float(np.nanmedian(spread[finite])))

    return spread
