"""
src/reflexivity/rl_env.py — RL Trading Environment for the Reflexivity Reversal Model.

The Reflexivity FSM in `model.py` is a fixed rule engine: every valid
Stage-1→2→3→4 setup is taken.  This module wraps the FSM so that a learned
policy (the DQN agent in `rl_agent.py`) decides, at each candidate decision
bar, whether to TAKE the trade or PASS.

Key pieces
----------
* ``ReflexivityTradingEnv`` — extracts a compact, normalized 22-D state vector
  at every candidate setup and simulates the hypothetical trade so we know the
  reward the agent *would* have earned had it taken the trade.  Losing trades
  yield negative reward (amplified), winning trades positive reward; PASS yields
  0 — so the agent is pushed to skip the patterns that lose and take the ones
  that win.
* ``scan_fsm`` — a shared finite-state-machine scanner that mirrors
  ``run_reflexivity_fsm`` but defers the final entry decision to a caller-
  supplied ``decider`` callback.  Both candidate collection (for training) and
  live gating (for inference) reuse the exact same FSM logic, guaranteeing the
  policy is trained on the same decision points it acts on.
* ``run_reflexivity_fsm_agent`` — the inference path: replays the FSM and asks
  the trained agent at each candidate; only entries the agent approves are
  taken, so losing patterns are filtered out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .model import (
    ARMED_L,
    ARMED_S,
    FSMResult,
    IDLE,
    IN_POSITION,
    PRIMED_L,
    PRIMED_S,
    ReflexivityConfig,
    _confirm_long,
    _confirm_short,
    _enter,
    _score_long,
    _score_short,
    prepare_arrays,
)

logger = logging.getLogger(__name__)

# 22-D observation space (see implementation_plan.md §1)
STATE_DIM = 22

FEATURE_NAMES = [
    "z_score", "ofi_1bar", "ofi_3bar", "ofi_5bar", "ofi_depth_ratio",
    "rsi", "rsi_dist_50", "volume_climax_ratio", "volume_dryup_ratio",
    "spread_frac", "spread_vs_median", "mom_accel", "rel_atr",
    "log_rv_5m", "log_rv_30m", "time_sin", "time_cos",
    "is_london", "is_ny", "is_overlap",
    "dist_from_pivot_atr", "bars_since_extreme",
]


@dataclass
class Candidate:
    """A single FSM decision point and the (hypothetical) outcome if taken."""
    idx: int
    side: int                 # +1 long, -1 short
    pivot: float
    z_ext: float
    conf: int
    bars_since_extreme: int
    trade: dict               # result of ``_enter`` at this bar
    reward_take: float        # reward if the agent had TAKEN this trade
    reward_pass: float = 0.0  # reward for PASS is always 0 (capital saved)


# ─────────────────────────────────────────────────────────────────────────────
# Shared FSM scanner — defers the entry decision to ``decider``
# ─────────────────────────────────────────────────────────────────────────────

def scan_fsm(
    a: dict,
    cfg: ReflexivityConfig,
    decider: Callable[[int, int, float, float, int, int], Tuple[bool, Optional[dict]]],
) -> FSMResult:
    """
    Mirror of ``run_reflexivity_fsm`` but the final entry is delegated.

    ``decider(t, side, pivot, z_ext, conf, bars_since_extreme)`` must return
    ``(take: bool, trade_or_None)``.  It is responsible for calling ``_enter``
    when it wants to take the trade.  The scanner handles all state transitions,
    cooldowns, jumps and bookkeeping exactly like the production FSM.
    """
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
                    else:
                        take, tr = decider(t, 1, pivot, z_ext, s, t - armed_at)
                        if take and tr is not None:
                            trades.append(tr); signal[t] = 1
                            state_arr[t] = IN_POSITION
                            state_arr[t + 1: tr["exit_idx"] + 1] = IN_POSITION
                            state = IDLE
                            cooldown_until = tr["exit_idx"] + cfg.cooldown_bars
                            t, jumped = tr["exit_idx"], True
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
                take, tr = decider(t, 1, pivot, z_ext, s, t - armed_at)
                if take and tr is not None:
                    trades.append(tr); signal[t] = 1
                    state_arr[t] = IN_POSITION
                    state_arr[t + 1: tr["exit_idx"] + 1] = IN_POSITION
                    state = IDLE
                    cooldown_until = tr["exit_idx"] + cfg.cooldown_bars
                    t, jumped = tr["exit_idx"], True
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
                        take, tr = decider(t, -1, pivot, z_ext, s, t - armed_at)
                        if take and tr is not None:
                            trades.append(tr); signal[t] = -1
                            state_arr[t] = IN_POSITION
                            state_arr[t + 1: tr["exit_idx"] + 1] = IN_POSITION
                            state = IDLE
                            cooldown_until = tr["exit_idx"] + cfg.cooldown_bars
                            t, jumped = tr["exit_idx"], True
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
                take, tr = decider(t, -1, pivot, z_ext, s, t - armed_at)
                if take and tr is not None:
                    trades.append(tr); signal[t] = -1
                    state_arr[t] = IN_POSITION
                    state_arr[t + 1: tr["exit_idx"] + 1] = IN_POSITION
                    state = IDLE
                    cooldown_until = tr["exit_idx"] + cfg.cooldown_bars
                    t, jumped = tr["exit_idx"], True
                else:
                    state = IDLE

        if not jumped:
            state_arr[t] = state
        t += 1

    return FSMResult(signal=signal, state=state_arr, trades=trades)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class ReflexivityTradingEnv:
    """
    Extracts candidate decision points + 22-D states + simulated rewards.

    Usage
    -----
    >>> env = ReflexivityTradingEnv(cfg, reward_beta=1.5)
    >>> env.collect_candidates(df)
    >>> states = env.state_matrix      # (N, 22), robust-standardized
    >>> rewards = np.array([c.reward_take for c in env.candidates])
    """

    def __init__(
        self,
        cfg: Optional[ReflexivityConfig] = None,
        reward_beta: float = 1.5,
        reward_clip: Optional[float] = None,
    ):
        self.cfg = cfg or ReflexivityConfig()
        self.reward_beta = reward_beta          # >1 penalizes losses more (prospect theory)
        self.reward_clip = reward_clip          # bound reward magnitude for stability
        self.candidates: List[Candidate] = []
        self.state_matrix: np.ndarray = np.zeros((0, STATE_DIM), dtype=np.float32)
        self.med: np.ndarray = np.zeros(STATE_DIM)
        self.mad: np.ndarray = np.ones(STATE_DIM)
        self.n_candidates: int = 0

    # ── extra feature arrays (aligned to ``a`` which is timestamp-sorted) ──
    @staticmethod
    def _compute_extra(df: pl.DataFrame, a: dict, cfg: ReflexivityConfig) -> Dict[str, np.ndarray]:
        cols = set(df.columns)
        has_ba = {"bid_open", "bid_close", "ask_open", "ask_close",
                  "bid_volume", "ask_volume"} <= cols
        has_spread = "spread_close" in cols
        has_vol = "volume" in cols

        if has_ba:
            ofi = (pl.col("bid_volume") * (pl.col("bid_close") - pl.col("bid_open")).sign()
                   - pl.col("ask_volume") * (pl.col("ask_close") - pl.col("ask_open")).sign())
        else:
            ofi = pl.lit(0.0)

        logc = pl.col("close").log()
        r = logc.diff()
        vol_expr = pl.col("volume") if has_vol else pl.lit(1.0)

        exprs = [
            ofi.alias("_ofi"),
            ofi.rolling_sum(3).alias("_ofi3"),
            ofi.rolling_sum(5).alias("_ofi5"),
            (ofi / (ofi.rolling_std(30) + 1e-12)).alias("_ofi_std"),
            r.rolling_std(5).log().alias("_rv5"),
            r.rolling_std(30).log().alias("_rv30"),
            vol_expr.alias("_vol"),
            vol_expr.rolling_max(10).alias("_vmax10"),
            vol_expr.rolling_quantile(quantile=cfg.climax_quantile, interpolation="linear",
                                      window_size=cfg.climax_window).alias("_vq"),
            (logc.diff(3) - logc.diff(15) / 5.0).alias("_accel"),
        ]
        if has_spread:
            sp = pl.col("spread_close") / pl.col("close")
            exprs += [sp.alias("_spread"),
                      sp.rolling_median(cfg.climax_window).alias("_spmed")]
        else:
            exprs += [pl.lit(1e-4).alias("_spread"), pl.lit(1e-4).alias("_spmed")]

        hour = pl.col("timestamp").dt.hour()
        exprs += [
            (hour / 24.0 * 2 * np.pi).sin().alias("_tsin"),
            (hour / 24.0 * 2 * np.pi).cos().alias("_tcos"),
            hour.is_between(8, 16).cast(pl.Float64).alias("_lon"),
            hour.is_between(13, 20).cast(pl.Float64).alias("_ny"),
            hour.is_between(13, 16).cast(pl.Float64).alias("_ovl"),
        ]

        tmp = df.select(exprs)
        out = {c: np.nan_to_num(tmp[c].to_numpy().astype(np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0) for c in tmp.columns}
        return out

    # ── raw (unstandardized) state vector for one bar ──
    def _raw_vec(self, t: int, extra: dict, a: dict, pivot: float, bars_since: int) -> np.ndarray:
        cfg = self.cfg
        close = a["close"][t]
        atr = a["atr"][t]
        z = a["z"][t]
        rsi = a["rsi"][t]
        ofi1 = extra["_ofi"][t]
        ofi3 = extra["_ofi3"][t]
        ofi5 = extra["_ofi5"][t]
        ofi_std = extra["_ofi_std"][t] + 1e-12
        vol = extra["_vol"][t] + 1e-12
        vmax10 = extra["_vmax10"][t] + 1e-12
        vq = extra["_vq"][t] + 1e-12
        spread = a["spread_frac"][t]
        spmed = extra["_spmed"][t] + 1e-12
        accel = extra["_accel"][t]
        rv5 = extra["_rv5"][t]
        rv30 = extra["_rv30"][t]
        vec = np.array([
            z / 5.0,                              # z_score (≈±0.5 at entry)
            ofi1 / ofi_std,                       # ofi_1bar (z-scored)
            ofi3 / (ofi_std * 3 + 1e-12),         # ofi_3bar
            ofi5 / (ofi_std * 5 + 1e-12),         # ofi_5bar
            ofi1 / (abs(ofi5) + 1e-12),           # ofi_depth_ratio
            (rsi - 50.0) / 50.0,                  # rsi
            abs(rsi - 50.0) / 50.0,               # rsi_dist_50
            vol / vq,                             # volume_climax_ratio
            vol / vmax10,                         # volume_dryup_ratio
            spread * 1e4,                         # spread_frac (~1)
            spread / spmed,                       # spread_vs_median
            accel * 1e4,                          # mom_accel
            atr / max(close, 1e-12) * 1000.0,     # rel_atr (scaled)
            rv5,                                  # log_rv_5m
            rv30,                                 # log_rv_30m
            extra["_tsin"][t],                    # time_sin
            extra["_tcos"][t],                    # time_cos
            extra["_lon"][t],                     # is_london
            extra["_ny"][t],                      # is_ny
            extra["_ovl"][t],                     # is_overlap
            (close - pivot) / max(atr, 1e-12),    # dist_from_pivot_atr
            bars_since / max(cfg.max_armed_bars, 1),  # bars_since_extreme
        ], dtype=np.float64)
        return vec

    def standardize(self, raw: np.ndarray) -> np.ndarray:
        """Robust z-score using the stats captured at fit time."""
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        z = (raw - self.med) / (1.4826 * self.mad)
        return np.clip(z, -3.0, 3.0).astype(np.float32)

    # ── reward ──
    def _reward(self, trade: dict) -> float:
        spread = max(float(trade["spread_frac"]), 1e-9)
        r = float(trade["pnl_net_log"]) / spread          # economically normalized
        if trade["exit_reason"] == "stop":
            r = r - 0.0                                   # gamma_stop (kept 0; beta covers loss aversion)
        if r < 0:
            r = r * self.reward_beta                       # loss amplification
        if self.reward_clip is not None:
            r = float(np.clip(r, -self.reward_clip, self.reward_clip))
        return float(r)

    # ── main entry: build candidate set + standardized states ──
    def collect_candidates(self, df: pl.DataFrame) -> "ReflexivityTradingEnv":
        df = df.sort("timestamp")
        a = prepare_arrays(df, self.cfg)
        extra = self._compute_extra(df, a, self.cfg)

        candidates: List[Candidate] = []
        raw_rows: List[np.ndarray] = []

        def decider(t, side, pivot, z_ext, conf, bars_since):
            tr, _ = _enter(t, side, conf, z_ext, a, pivot, self.cfg)
            if tr is None:
                return False, None
            # Record the hypothetical outcome (we never actually enter during training).
            reward = self._reward(tr)
            candidates.append(Candidate(
                idx=t, side=side, pivot=pivot, z_ext=z_ext, conf=conf,
                bars_since_extreme=bars_since, trade=tr, reward_take=reward,
            ))
            raw_rows.append(self._raw_vec(t, extra, a, pivot, bars_since))
            return False, None  # always PASS in the scanner; we only harvest candidates

        scan_fsm(a, self.cfg, decider)

        if raw_rows:
            M = np.stack(raw_rows)
            med = np.nanmedian(M, axis=0)
            mad = np.nanmedian(np.abs(M - med), axis=0) + 1e-9
            Mnorm = np.clip((M - med) / (1.4826 * mad), -3.0, 3.0).astype(np.float32)
        else:
            Mnorm = np.zeros((0, STATE_DIM), dtype=np.float32)
            med = np.zeros(STATE_DIM)
            mad = np.ones(STATE_DIM)

        self.candidates = candidates
        self.state_matrix = Mnorm
        self.med = med
        self.mad = mad
        self.n_candidates = len(candidates)
        logger.info(f"RL env: collected {len(candidates)} candidate setups "
                    f"(state dim={STATE_DIM}).")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Inference path — agent-gated FSM
# ─────────────────────────────────────────────────────────────────────────────

def run_reflexivity_fsm_agent(
    df: pl.DataFrame,
    cfg: ReflexivityConfig,
    agent,                       # DQNReflexivityAgent (must expose select_action)
    med: np.ndarray,
    mad: np.ndarray,
    greedy: bool = True,
) -> FSMResult:
    """
    Replay the FSM and let ``agent`` decide TAKE vs PASS at each candidate.

    The agent is queried with the *same* standardized 22-D state used during
    training, so losing patterns learned to be skipped actually get skipped.
    Returns an ``FSMResult`` whose ``trades`` are only the agent-approved ones.
    """
    df = df.sort("timestamp")
    a = prepare_arrays(df, cfg)
    extra = ReflexivityTradingEnv._compute_extra(df, a, cfg)

    def decider(t, side, pivot, z_ext, conf, bars_since):
        raw = ReflexivityTradingEnv(cfg)._raw_vec(t, extra, a, pivot, bars_since)
        svec = np.clip((raw - med) / (1.4826 * mad), -3.0, 3.0).astype(np.float32)
        action = agent.select_action(svec, greedy=greedy)
        if action == 1:
            tr, _ = _enter(t, side, conf, z_ext, a, pivot, cfg)
            if tr is not None:
                return True, tr
        return False, None

    return scan_fsm(a, cfg, decider)
