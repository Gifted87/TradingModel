"""
src/reflexivity/model.py — Reflexivity Reversal Model (George Soros) for the 1-minute chart.

THE ORDER IS BINDING (per spec and per reflexivity theory):
  1. EXTREME      — price is statistically undervalued/overvalued vs. recent history
                    (z-score of close vs. rolling 4h VWAP, plus a fresh 60-bar low/high:
                    the "bust leg" of the reflexive cycle).
  2. STIMULANT    — a confluence of datapoints indicates perception is shifting at the
                    extreme, BEFORE price confirms:
                      a) OFI divergence      — net buying while price holds the pivot (G1)
                      b) RSI hook            — RSI turning up from oversold (G3)
                      c) Volume climax+dry-up— capitulation bar, then selling dries up
                      d) Spread normalization— stress spike has relaxed (G1)
                      e) Momentum turn       — short-horizon accel > long-horizon drift (G2)
  3. CONFIRMATION — price has ACTUALLY started moving in the reversal direction:
                    micro structure break + >= 0.5*ATR traveled off the pivot +
                    positive OFI follow-through (the feedback loop has turned).
  4. ENTRY        — buy/sell WITH the loop. Stop beyond the reflexive pivot (thesis
                    invalidation), target = fair value (VWAP equilibrium), 30-bar
                    time stop aligned with the r30 label horizon.

We never buy the extreme itself (falling knife). We buy the TURN of the loop.
That is the difference between reflexivity and contrarianism.

Integration:
  - Input: 1m bars in the bar_builder.py schema
    (timestamp, open, high, low, close, [bid_open, bid_close, ask_open, ask_close,
     bid_volume, ask_volume, spread_close, volume]).
  - ReflexivityPredictor implements src.validation.harness.Predictor
    (fit/predict -> (r_hat, rv_hat)) so Gate 5/6 machinery, regime slicing,
    the economic gate, and the adversarial shuffle test all apply unchanged.
    NOTE: signals are event-driven, not clock-driven — evaluate with
    decision_bars_only=False in the harness.
  - Timeframe-agnostic: all windows are in bars. Defaults are tuned for 1m
    (240 bars = 4h). For 5m bars divide window lengths by 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# FSM states
IDLE, ARMED_L, PRIMED_L, ARMED_S, PRIMED_S, IN_POSITION = range(6)
STATE_NAMES = {
    0: "IDLE",
    1: "ARMED_LONG",
    2: "PRIMED_LONG",
    3: "ARMED_SHORT",
    4: "PRIMED_SHORT",
    5: "IN_POSITION",
}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflexivityConfig:
    # ── Stage 1: historical extreme ──────────────────────────────────────
    vwap_window: int = 240          # fair-value anchor: 4h of 1m bars (or 48 bars for 5m)
    z_window: int = 240             # window for std of (close - vwap)
    z_entry: float = 2.5            # |z| required to call price "extreme"
    low_lookback: int = 60          # must also print a fresh 60-bar low/high

    # ── Stage 2: stimulant confluence ────────────────────────────────────
    min_confluence: int = 3         # of 5 components
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    climax_quantile: float = 0.90   # capitulation = volume in top decile...
    climax_window: int = 240
    dryup_ratio: float = 0.6        # ...then current volume < 60% of climax
    spread_relax: float = 1.2       # spread back under 1.2x rolling median

    # ── Stage 3: confirmation (the reflexive turn) ───────────────────────
    structure_lookback: int = 3     # close must break the last 3 bars' high/low
    min_move_atr: float = 0.5       # and have traveled >= 0.5*ATR off the pivot
    econ_gate_k: float = 1.5        # expected reversion must exceed k * spread
    require_confirmation: bool = True   # ABLATION FLAG: False = enter at confluence

    # ── Stage 4: trade management ────────────────────────────────────────
    stop_buffer_atr: float = 0.25   # stop = pivot -/+ 0.25*ATR
    target: str = "vwap"            # "vwap" (equilibrium) | "atr"
    target_atr_mult: float = 1.5    # used when target="atr"
    min_target_r_multiple: float = 1.0  # skip entries whose target pays less than this R
    time_stop_bars: int = 30        # == r30 horizon on 1m

    # ── FSM timing ───────────────────────────────────────────────────────
    max_armed_bars: int = 60        # stimulant must appear within 1h of extreme
    max_primed_bars: int = 20       # confirmation must follow within 20 bars
    cooldown_bars: int = 15         # pause after exit / invalidation

    # ── Session filter: reflexive moves need flow ────────────────────────
    liquid_hours_only: bool = True
    liquid_hours: tuple = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)  # UTC


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation (self-contained; mirrors G1/G3 logic on raw bars)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_arrays(df: pl.DataFrame, cfg: ReflexivityConfig) -> dict:
    df = df.sort("timestamp")
    eps = 1e-7
    cols = set(df.columns)
    has_ba = {"bid_open", "bid_close", "ask_open", "ask_close",
              "bid_volume", "ask_volume"} <= cols
    has_spread = "spread_close" in cols
    has_vol = "volume" in cols

    df = df.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp"),
        (pl.col("volume") if has_vol else pl.lit(1.0)).alias("_vol"),
        pl.col("close").log().alias("_logc"),
    ])
    df = df.with_columns([
        (pl.col("_tp") * pl.col("_vol")).alias("_tpv"),
        pl.col("_vol").rolling_sum(cfg.vwap_window).alias("_vden"),
    ])
    df = df.with_columns([
        pl.when(pl.col("_vden") > 1e-9)
          .then(pl.col("_tpv").rolling_sum(cfg.vwap_window) / pl.col("_vden"))
          .otherwise(pl.col("_tp").rolling_mean(cfg.vwap_window))
          .alias("vwap"),
        (pl.col("close") - pl.col("close").shift(1)).alias("_diff"),
    ])
    df = df.with_columns([(pl.col("close") - pl.col("vwap")).alias("_dev")])
    df = df.with_columns([
        (pl.col("_dev") / (pl.col("_dev").rolling_std(cfg.z_window) + eps)).alias("z"),
    ])

    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    gain = pl.when(pl.col("_diff") > 0).then(pl.col("_diff")).otherwise(0.0)
    loss = pl.when(pl.col("_diff") < 0).then(-pl.col("_diff")).otherwise(0.0)
    if has_ba:
        # Candle-level OFI proxy (same construction as features/microstructure.py)
        ofi = (pl.col("bid_volume") * (pl.col("bid_close") - pl.col("bid_open")).sign()
               - pl.col("ask_volume") * (pl.col("ask_close") - pl.col("ask_open")).sign())
    else:
        ofi = pl.lit(0.0)

    df = df.with_columns([
        tr.ewm_mean(span=14, ignore_nulls=True).alias("atr"),
        (100.0 - 100.0 / (1.0 + gain.ewm_mean(span=cfg.rsi_period, ignore_nulls=True)
                          / (loss.ewm_mean(span=cfg.rsi_period, ignore_nulls=True) + eps))
        ).alias("rsi"),
        ofi.alias("_ofi"),
        (pl.col("_logc") - pl.col("_logc").shift(3)).alias("_r3"),
        (pl.col("_logc") - pl.col("_logc").shift(15)).alias("_r15"),
        pl.col("high").rolling_max(cfg.structure_lookback).shift(1).alias("hh_prev"),
        pl.col("low").rolling_min(cfg.structure_lookback).shift(1).alias("ll_prev"),
        pl.col("low").rolling_min(cfg.low_lookback).alias("ll60"),
        pl.col("high").rolling_max(cfg.low_lookback).alias("hh60"),
        pl.col("_vol").rolling_max(10).alias("_vmax10"),
        pl.col("_vol").rolling_quantile(quantile=cfg.climax_quantile,
                                        interpolation="linear",
                                        window_size=cfg.climax_window).alias("_vq"),
    ])

    if has_spread:
        df = df.with_columns([
            (pl.col("spread_close") / pl.col("close")).alias("spread_frac"),
            pl.col("spread_close").rolling_median(cfg.climax_window).alias("_spmed"),
        ])
    else:
        df = df.with_columns([pl.lit(1e-4).alias("spread_frac"),
                              pl.lit(1e-4).alias("_spmed")])

    df = df.with_columns([
        pl.col("_ofi").rolling_sum(3).alias("_ofi3"),
        pl.col("_ofi").rolling_sum(5).alias("_ofi5"),
        (pl.col("_r3") - pl.col("_r15") / 5.0).alias("_accel"),
        pl.col("rsi").rolling_min(10).alias("_rsi_min10"),
        pl.col("rsi").rolling_max(10).alias("_rsi_max10"),
    ])

    hour = pl.col("timestamp").dt.hour()
    liquid = hour.is_in(list(cfg.liquid_hours)) if cfg.liquid_hours_only else pl.lit(True)

    df = df.with_columns([
        liquid.alias("_liquid"),
        (pl.col("_ofi5") > 0).fill_null(False).alias("ofi_pos5"),
        (pl.col("_ofi5") < 0).fill_null(False).alias("ofi_neg5"),
        (pl.col("_ofi3") > 0).fill_null(False).alias("ofi_pos3"),
        (pl.col("_ofi3") < 0).fill_null(False).alias("ofi_neg3"),
        ((pl.col("rsi") > pl.col("rsi").shift(1))
         & (pl.col("_rsi_min10") < cfg.rsi_oversold)).fill_null(False).alias("rsi_hook_l"),
        ((pl.col("rsi") < pl.col("rsi").shift(1))
         & (pl.col("_rsi_max10") > cfg.rsi_overbought)).fill_null(False).alias("rsi_hook_s"),
        ((pl.col("_vmax10") >= pl.col("_vq"))
         & (pl.col("_vol") <= cfg.dryup_ratio * pl.col("_vmax10"))
         ).fill_null(False).alias("climax_dry"),
        (pl.col("_accel") > 0).fill_null(False).alias("mom_up"),
        (pl.col("_accel") < 0).fill_null(False).alias("mom_dn"),
    ])

    if has_spread:
        df = df.with_columns(
            (pl.col("spread_close") <= pl.col("_spmed") * cfg.spread_relax)
            .fill_null(False).alias("spread_ok"))
    else:
        df = df.with_columns(pl.lit(True).alias("spread_ok"))

    df = df.with_columns([
        ((pl.col("z") <= -cfg.z_entry)
         & (pl.col("low") <= pl.col("ll60") * (1.0 + 1e-9))
         & pl.col("_liquid")).fill_null(False).alias("extreme_l"),
        ((pl.col("z") >= cfg.z_entry)
         & (pl.col("high") >= pl.col("hh60") * (1.0 - 1e-9))
         & pl.col("_liquid")).fill_null(False).alias("extreme_s"),
    ])

    # Pivot-independent confluence base scores (pivot check added in the FSM)
    df = df.with_columns([
        (pl.col("ofi_pos5").cast(pl.Int8) + pl.col("rsi_hook_l").cast(pl.Int8)
         + pl.col("climax_dry").cast(pl.Int8) + pl.col("spread_ok").cast(pl.Int8)
         + pl.col("mom_up").cast(pl.Int8)).alias("conf_l_base"),
        (pl.col("ofi_neg5").cast(pl.Int8) + pl.col("rsi_hook_s").cast(pl.Int8)
         + pl.col("climax_dry").cast(pl.Int8) + pl.col("spread_ok").cast(pl.Int8)
         + pl.col("mom_dn").cast(pl.Int8)).alias("conf_s_base"),
    ])

    # Forward-looking series used ONLY by fit() for calibration (never at inference)
    df = df.with_columns([
        (pl.col("_logc").shift(-cfg.time_stop_bars) - pl.col("_logc")).alias("fwd_ret"),
        (pl.col("_logc") - pl.col("_logc").shift(1)).alias("_r1"),
    ])
    df = df.with_columns([
        (((pl.col("_r1") ** 2).reverse().rolling_sum(cfg.time_stop_bars).reverse()).sqrt()
         + eps).log().alias("rv_fwd"),
    ])

    def f(name: str, default: float = 0.0) -> np.ndarray:
        return np.nan_to_num(df[name].to_numpy().astype(np.float64),
                             nan=default, posinf=default, neginf=default)

    def b(name: str) -> np.ndarray:
        return df[name].to_numpy().astype(bool)

    atr = np.maximum(f("atr", 1e-4), 1e-8)
    return {
        "ts": df["timestamp"].to_numpy(),
        "close": f("close"), "low": f("low"), "high": f("high"),
        "vwap": f("vwap"), "z": f("z"), "atr": atr, "rsi": f("rsi", 50.0),
        # keep NaN in structure levels -> comparisons evaluate False during warm-up
        "hh_prev": df["hh_prev"].to_numpy().astype(np.float64),
        "ll_prev": df["ll_prev"].to_numpy().astype(np.float64),
        "spread_frac": f("spread_frac", 1e-4),
        "ofi_pos3": b("ofi_pos3"), "ofi_neg3": b("ofi_neg3"),
        "ofi_pos5": b("ofi_pos5"), "ofi_neg5": b("ofi_neg5"),
        "conf_l_base": df["conf_l_base"].to_numpy().astype(np.int64),
        "conf_s_base": df["conf_s_base"].to_numpy().astype(np.int64),
        "extreme_l": b("extreme_l"), "extreme_s": b("extreme_s"),
        "fwd_ret": df["fwd_ret"].to_numpy().astype(np.float64),
        "rv_fwd": df["rv_fwd"].to_numpy().astype(np.float64),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 & 3 evaluators (pivot-dependent parts live here, inside the FSM)
# ─────────────────────────────────────────────────────────────────────────────

def _score_long(t: int, a: dict, pivot: float, cfg: ReflexivityConfig) -> int:
    s = int(a["conf_l_base"][t])
    # OFI divergence only counts if price is HOLDING the pivot while flow is buying
    if a["ofi_pos5"][t] and a["low"][t] < pivot - 0.1 * a["atr"][t]:
        s -= 1
    return s


def _score_short(t: int, a: dict, pivot: float, cfg: ReflexivityConfig) -> int:
    s = int(a["conf_s_base"][t])
    if a["ofi_neg5"][t] and a["high"][t] > pivot + 0.1 * a["atr"][t]:
        s -= 1
    return s


def _confirm_long(t: int, a: dict, pivot: float, cfg: ReflexivityConfig) -> bool:
    c = a["close"][t]
    if not (c > a["hh_prev"][t]):                                    # structure break
        return False
    if (c - pivot) < cfg.min_move_atr * a["atr"][t]:                 # real move off low
        return False
    if not a["ofi_pos3"][t]:                                         # flow follow-through
        return False
    exp_move = (a["vwap"][t] - c) / max(c, 1e-12)                    # economic gate
    return exp_move >= cfg.econ_gate_k * a["spread_frac"][t]


def _confirm_short(t: int, a: dict, pivot: float, cfg: ReflexivityConfig) -> bool:
    c = a["close"][t]
    if not (c < a["ll_prev"][t]):
        return False
    if (pivot - c) < cfg.min_move_atr * a["atr"][t]:
        return False
    if not a["ofi_neg3"][t]:
        return False
    exp_move = (c - a["vwap"][t]) / max(c, 1e-12)
    return exp_move >= cfg.econ_gate_k * a["spread_frac"][t]


# ─────────────────────────────────────────────────────────────────────────────
# Trade simulation (conservative: stop checked before target on ambiguous bars)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(a: dict, entry_idx: int, side: int, stop: float, target: float,
              cfg: ReflexivityConfig):
    n = len(a["close"])
    end = min(n - 1, entry_idx + cfg.time_stop_bars)
    for j in range(entry_idx + 1, end + 1):
        if side == 1:
            if a["low"][j] <= stop:
                return j, stop, "stop"
            if a["high"][j] >= target:
                return j, target, "target"
        else:
            if a["high"][j] >= stop:
                return j, stop, "stop"
            if a["low"][j] <= target:
                return j, target, "target"
    return end, a["close"][end], "time"


def _enter(t: int, side: int, conf: int, z_ext: float, a: dict, pivot: float,
           cfg: ReflexivityConfig):
    entry = a["close"][t]
    buf = cfg.stop_buffer_atr * a["atr"][t]
    if side == 1:
        stop = pivot - buf
        target = a["vwap"][t] if cfg.target == "vwap" else entry + cfg.target_atr_mult * a["atr"][t]
        if not np.isfinite(target) or target <= entry:
            return None, t
    else:
        stop = pivot + buf
        target = a["vwap"][t] if cfg.target == "vwap" else entry - cfg.target_atr_mult * a["atr"][t]
        if not np.isfinite(target) or target >= entry:
            return None, t

    risk = abs(np.log(entry) - np.log(stop))
    target_gain = side * (np.log(target) - np.log(entry))
    target_r_multiple = target_gain / risk if risk > 0 else 0.0
    if target_r_multiple < cfg.min_target_r_multiple:
        return None, t

    exit_idx, exit_px, reason = _simulate(a, t, side, stop, target, cfg)
    gross = side * (np.log(exit_px) - np.log(entry))
    cost = a["spread_frac"][t]                       # one round-trip spread
    trade = {
        "entry_idx": t, "exit_idx": exit_idx,
        "entry_time": str(a["ts"][t])[:19],
        "side": side, "entry": entry, "stop": stop, "target": target,
        "exit": exit_px, "exit_reason": reason,
        "confluence": conf, "z_extreme": z_ext,
        "pnl_gross_log": gross, "pnl_net_log": gross - cost,
        "r_multiple": gross / risk if risk > 0 else 0.0,
        "target_r_multiple": target_r_multiple,
        "spread_frac": cost, "bars_held": exit_idx - t,
    }
    return trade, exit_idx


# ─────────────────────────────────────────────────────────────────────────────
# The ordered finite state machine — Stages 1 -> 2 -> 3 -> 4, no shortcuts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FSMResult:
    signal: np.ndarray      # +1 long entry, -1 short entry, 0 flat (per bar)
    state: np.ndarray       # FSM state per bar
    trades: list            # list of trade dicts


def run_reflexivity_fsm(a: dict, cfg: ReflexivityConfig) -> FSMResult:
    n = len(a["close"])
    signal = np.zeros(n, dtype=np.int8)
    state_arr = np.zeros(n, dtype=np.int8)
    trades: list = []

    state = IDLE
    pivot = np.nan
    armed_at = primed_at = -1
    cooldown_until = -1
    z_ext = np.nan
    conf_at_prime = 0

    low, high, atr = a["low"], a["high"], a["atr"]

    t = 0
    while t < n:
        jumped = False

        if state == IDLE:
            if t > cooldown_until:
                if a["extreme_l"][t]:                       # STAGE 1 (long setup)
                    state, pivot, armed_at, z_ext = ARMED_L, low[t], t, a["z"][t]
                elif a["extreme_s"][t]:                     # STAGE 1 (short setup)
                    state, pivot, armed_at, z_ext = ARMED_S, high[t], t, a["z"][t]

        elif state == ARMED_L:
            pivot = min(pivot, low[t])                      # track the bust low
            if t - armed_at > cfg.max_armed_bars:
                state = IDLE
            else:
                s = _score_long(t, a, pivot, cfg)           # STAGE 2
                if s >= cfg.min_confluence:
                    if cfg.require_confirmation:
                        state, primed_at, conf_at_prime = PRIMED_L, t, s
                    else:                                   # ablation: enter at confluence
                        tr, exit_idx = _enter(t, 1, s, z_ext, a, pivot, cfg)
                        if tr is not None:
                            trades.append(tr); signal[t] = 1
                            state_arr[t] = IN_POSITION
                            state_arr[t + 1: exit_idx + 1] = IN_POSITION
                            state = IDLE
                            cooldown_until = exit_idx + cfg.cooldown_bars
                            t, jumped = exit_idx, True
                        else:
                            state = IDLE

        elif state == PRIMED_L:
            if low[t] < pivot - cfg.stop_buffer_atr * atr[t]:
                state = IDLE                                # knife continued; thesis void
                cooldown_until = t + cfg.cooldown_bars
            elif t - primed_at > cfg.max_primed_bars:
                state = IDLE
            elif _confirm_long(t, a, pivot, cfg):           # STAGE 3 -> STAGE 4
                s = max(_score_long(t, a, pivot, cfg), conf_at_prime)
                tr, exit_idx = _enter(t, 1, s, z_ext, a, pivot, cfg)
                if tr is not None:
                    trades.append(tr); signal[t] = 1
                    state_arr[t] = IN_POSITION
                    state_arr[t + 1: exit_idx + 1] = IN_POSITION
                    state = IDLE
                    cooldown_until = exit_idx + cfg.cooldown_bars
                    t, jumped = exit_idx, True
                else:
                    state = IDLE

        elif state == ARMED_S:
            pivot = max(pivot, high[t])
            if t - armed_at > cfg.max_armed_bars:
                state = IDLE
            else:
                s = _score_short(t, a, pivot, cfg)
                if s >= cfg.min_confluence:
                    if cfg.require_confirmation:
                        state, primed_at, conf_at_prime = PRIMED_S, t, s
                    else:
                        tr, exit_idx = _enter(t, -1, s, z_ext, a, pivot, cfg)
                        if tr is not None:
                            trades.append(tr); signal[t] = -1
                            state_arr[t] = IN_POSITION
                            state_arr[t + 1: exit_idx + 1] = IN_POSITION
                            state = IDLE
                            cooldown_until = exit_idx + cfg.cooldown_bars
                            t, jumped = exit_idx, True
                        else:
                            state = IDLE

        elif state == PRIMED_S:
            if high[t] > pivot + cfg.stop_buffer_atr * atr[t]:
                state = IDLE
                cooldown_until = t + cfg.cooldown_bars
            elif t - primed_at > cfg.max_primed_bars:
                state = IDLE
            elif _confirm_short(t, a, pivot, cfg):
                s = max(_score_short(t, a, pivot, cfg), conf_at_prime)
                tr, exit_idx = _enter(t, -1, s, z_ext, a, pivot, cfg)
                if tr is not None:
                    trades.append(tr); signal[t] = -1
                    state_arr[t] = IN_POSITION
                    state_arr[t + 1: exit_idx + 1] = IN_POSITION
                    state = IDLE
                    cooldown_until = exit_idx + cfg.cooldown_bars
                    t, jumped = exit_idx, True
                else:
                    state = IDLE

        if not jumped:
            state_arr[t] = state
        t += 1

    return FSMResult(signal=signal, state=state_arr, trades=trades)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def summarize_trades(trades: pl.DataFrame) -> dict:
    if trades.is_empty():
        return {"n_trades": 0, "verdict": "no signals generated"}
    pnl = trades["pnl_net_log"].to_numpy()
    wins = pnl > 0
    equity = np.cumsum(pnl)
    dd = np.maximum.accumulate(equity) - equity
    gross_win = float(pnl[wins].sum())
    gross_loss = float(abs(pnl[~wins].sum()))
    by_side = trades.group_by("side").agg([
        pl.len().alias("n"),
        (pl.col("pnl_net_log") > 0).mean().alias("hit_rate"),
        pl.col("pnl_net_log").mean().alias("avg_net_log"),
    ]).sort("side")
    return {
        "n_trades": len(trades),
        "hit_rate": float(wins.mean()),
        "avg_r_multiple": float(trades["r_multiple"].mean()),
        "expectancy_net_log": float(pnl.mean()),
        "expectancy_in_spreads": float((trades["pnl_net_log"] / trades["spread_frac"]).mean()),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "max_drawdown_log": float(dd.max()) if len(dd) else 0.0,
        "avg_bars_held": float(trades["bars_held"].mean()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dicts(),
        "by_side": by_side.to_dicts(),
    }


@dataclass
class ReflexivityResult:
    signals: pl.DataFrame   # bars + z, vwap, atr, rsi, state, signal
    trades: pl.DataFrame
    summary: dict


def run_reflexivity(df: pl.DataFrame, cfg: Optional[ReflexivityConfig] = None) -> ReflexivityResult:
    cfg = cfg or ReflexivityConfig()
    a = prepare_arrays(df, cfg)
    res = run_reflexivity_fsm(a, cfg)
    signals = df.with_columns([
        pl.Series("reflex_z", a["z"]),
        pl.Series("reflex_vwap", a["vwap"]),
        pl.Series("reflex_atr", a["atr"]),
        pl.Series("reflex_rsi", a["rsi"]),
        pl.Series("reflex_state", res.state),
        pl.Series("reflex_signal", res.signal),
        pl.Series("reflex_conf_l", a["conf_l_base"]),
        pl.Series("reflex_conf_s", a["conf_s_base"]),
    ])
    trades = pl.DataFrame(res.trades) if res.trades else pl.DataFrame()
    return ReflexivityResult(signals=signals, trades=trades,
                             summary=summarize_trades(trades))


# ─────────────────────────────────────────────────────────────────────────────
# ValidationHarness-compatible predictor
# ─────────────────────────────────────────────────────────────────────────────

def _shrunk_mean(fwd: np.ndarray, idx: list, shrink: int = 25) -> float:
    if not idx:
        return 0.0
    v = fwd[np.asarray(idx, dtype=np.int64)]
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0
    return float(v.mean()) * len(v) / (len(v) + shrink)   # shrink toward 0


class ReflexivityPredictor:
    """
    src.validation.harness.Predictor implementation.

    fit():     runs the FSM on the train fold and calibrates the conditional
               E[r30 | long signal] and E[r30 | short signal] (shrunk toward 0).
    predict(): replays the FSM and returns r_hat = calibrated expectation on
               trigger bars, 0 elsewhere (persistence off-signal), plus a
               constant rv30 forecast for QLIKE comparison.

    Use with decision_bars_only=False — entries are event-driven.
    """

    def __init__(self, config: Optional[ReflexivityConfig] = None):
        self.config = config or ReflexivityConfig()
        self.mu_long_: float = 0.0
        self.mu_short_: float = 0.0
        self.rv_const_: float = -8.0
        self.n_train_trades_: int = 0
        self.symbol_: str = "UNKNOWN"

    def fit(self, df: pl.DataFrame, symbol: str = "UNKNOWN", **kwargs) -> "ReflexivityPredictor":
        self.symbol_ = symbol
        a = prepare_arrays(df, self.config)
        res = run_reflexivity_fsm(a, self.config)
        long_idx = [tr["entry_idx"] for tr in res.trades if tr["side"] == 1]
        short_idx = [tr["entry_idx"] for tr in res.trades if tr["side"] == -1]
        self.mu_long_ = _shrunk_mean(a["fwd_ret"], long_idx)
        self.mu_short_ = _shrunk_mean(a["fwd_ret"], short_idx)
        rv = a["rv_fwd"][np.isfinite(a["rv_fwd"])]
        self.rv_const_ = float(rv.mean()) if len(rv) else -8.0
        self.n_train_trades_ = len(res.trades)
        logger.info(f"[{symbol}] Reflexivity fit: {len(res.trades)} train trades, "
                    f"mu_long={self.mu_long_:+.6f}, mu_short={self.mu_short_:+.6f}")
        return self

    def predict(self, df: pl.DataFrame):
        a = prepare_arrays(df, self.config)
        res = run_reflexivity_fsm(a, self.config)
        r_hat = np.where(res.signal == 1, self.mu_long_,
                         np.where(res.signal == -1, self.mu_short_, 0.0))
        rv_hat = np.full(len(df), self.rv_const_)
        return r_hat, rv_hat


class RLReflexivityPredictor:
    """
    ValidationHarness-compatible predictor using a DQN gate.

    fit(): trains the DQN on train-fold reflexivity candidates only, then
           calibrates return expectations from agent-approved train signals.
    predict(): replays the same FSM on the test fold and only emits signals the
               trained agent chooses to TAKE.
    """

    def __init__(
        self,
        config: Optional[ReflexivityConfig] = None,
        episodes: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        reward_beta: float = 1.5,
        reward_clip: Optional[float] = 10.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.config = config or ReflexivityConfig()
        self.episodes = episodes
        self.batch_size = batch_size
        self.lr = lr
        self.reward_beta = reward_beta
        self.reward_clip = reward_clip
        self.device = device
        self.seed = seed

        self.agent_ = None
        self.env_ = None
        self.training_result_ = None
        self.mu_long_: float = 0.0
        self.mu_short_: float = 0.0
        self.rv_const_: float = -8.0
        self.n_train_candidates_: int = 0
        self.n_train_trades_: int = 0
        self.symbol_: str = "UNKNOWN"

    def fit(self, df: pl.DataFrame, symbol: str = "UNKNOWN", **kwargs) -> "RLReflexivityPredictor":
        from .rl_env import run_reflexivity_fsm_agent
        from .rl_trainer import train_rl_reflexivity

        self.symbol_ = symbol
        result = train_rl_reflexivity(
            df=df,
            config=self.config,
            episodes=self.episodes,
            batch_size=self.batch_size,
            lr=self.lr,
            reward_beta=self.reward_beta,
            reward_clip=self.reward_clip,
            device=self.device,
            seed=self.seed,
        )
        self.agent_ = result.agent
        self.env_ = result.env
        self.training_result_ = result
        self.n_train_candidates_ = result.n_candidates

        a = prepare_arrays(df, self.config)
        gated = run_reflexivity_fsm_agent(
            df,
            self.config,
            self.agent_,
            self.env_.med,
            self.env_.mad,
            greedy=True,
        )
        long_idx = [tr["entry_idx"] for tr in gated.trades if tr["side"] == 1]
        short_idx = [tr["entry_idx"] for tr in gated.trades if tr["side"] == -1]
        self.mu_long_ = _shrunk_mean(a["fwd_ret"], long_idx)
        self.mu_short_ = _shrunk_mean(a["fwd_ret"], short_idx)
        rv = a["rv_fwd"][np.isfinite(a["rv_fwd"])]
        self.rv_const_ = float(rv.mean()) if len(rv) else -8.0
        self.n_train_trades_ = len(gated.trades)

        logger.info(
            "[%s] RL reflexivity fit: candidates=%d approved_train_trades=%d "
            "mu_long=%+.6f mu_short=%+.6f",
            symbol,
            self.n_train_candidates_,
            self.n_train_trades_,
            self.mu_long_,
            self.mu_short_,
        )
        return self

    def predict(self, df: pl.DataFrame):
        from .rl_env import run_reflexivity_fsm_agent

        if self.agent_ is None or self.env_ is None:
            raise RuntimeError("RLReflexivityPredictor is not fitted.")

        res = run_reflexivity_fsm_agent(
            df,
            self.config,
            self.agent_,
            self.env_.med,
            self.env_.mad,
            greedy=True,
        )
        r_hat = np.where(res.signal == 1, self.mu_long_,
                         np.where(res.signal == -1, self.mu_short_, 0.0))
        rv_hat = np.full(len(df), self.rv_const_)
        return r_hat, rv_hat


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "data/bars/1min/EURUSD_1m.parquet"
    df = pl.read_parquet(path)
    result = run_reflexivity(df, ReflexivityConfig())
    print("=" * 70)
    print("REFLEXIVITY REVERSAL MODEL — SUMMARY")
    print("=" * 70)
    for k, v in result.summary.items():
        print(f"  {k}: {v}")
    if not result.trades.is_empty():
        print("\nLast 10 trades:")
        print(result.trades.tail(10))
