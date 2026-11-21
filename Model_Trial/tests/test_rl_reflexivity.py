"""
Unit tests for the RL-gated Reflexivity Reversal Model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from src.reflexivity import (
    DQNReflexivityAgent,
    ReflexivityConfig,
    ReflexivityTradingEnv,
    RLReflexivityPredictor,
    QNetwork,
    ReplayBuffer,
    train_rl_reflexivity,
)


def _sample_bars(n: int = 900) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = [
        datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * i)
        for i in range(n)
    ]

    ret = rng.normal(0, 0.0005, size=n)
    ret[120:125] = -0.006
    ret[126:140] = 0.003
    ret[420:425] = 0.006
    ret[426:440] = -0.003
    ret[700:705] = -0.005
    ret[706:720] = 0.0025

    close = 1.1000 * np.exp(np.cumsum(ret))
    open_px = close * (1 + rng.normal(0, 0.0001, size=n))
    high = np.maximum(open_px, close) * (1 + np.abs(rng.normal(0, 0.0002, size=n)))
    low = np.minimum(open_px, close) * (1 - np.abs(rng.normal(0, 0.0002, size=n)))

    volume = rng.exponential(100.0, size=n) + 10.0
    volume[120:125] *= 10.0
    volume[420:425] *= 10.0
    volume[700:705] *= 10.0

    spread = np.full(n, 0.0001)
    bid_close = close - spread / 2
    ask_close = close + spread / 2
    bid_open = open_px - spread / 2
    ask_open = open_px + spread / 2
    bid_volume = volume * np.where(ret >= 0, 0.65, 0.35)
    ask_volume = volume - bid_volume

    r30 = np.concatenate([np.log(close[6:] / close[:-6]), np.zeros(6)])
    rv30 = np.full(n, -8.0)
    mag30 = np.log(np.abs(r30) + 1e-7)

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_px,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "bid_open": bid_open,
        "bid_close": bid_close,
        "ask_open": ask_open,
        "ask_close": ask_close,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "spread_close": spread,
        "r30": r30,
        "rv30": rv30,
        "mag30": mag30,
        "is_dead_zone": np.abs(r30) < spread,
    })


def _fast_cfg() -> ReflexivityConfig:
    return ReflexivityConfig(
        vwap_window=48,
        z_window=48,
        low_lookback=12,
        min_confluence=1,
        climax_window=48,
        time_stop_bars=6,
        min_target_r_multiple=0.0,
        max_armed_bars=12,
        max_primed_bars=4,
        cooldown_bars=3,
        structure_lookback=2,
        min_move_atr=0.2,
        require_confirmation=False,
        liquid_hours_only=False,
    )


def test_environment_collects_candidate_states_and_rewards():
    env = ReflexivityTradingEnv(_fast_cfg(), reward_clip=10.0).collect_candidates(_sample_bars())

    assert env.n_candidates > 0
    assert env.state_matrix.shape == (env.n_candidates, 22)
    assert np.isfinite(env.state_matrix).all()
    assert all(candidate.reward_pass == 0.0 for candidate in env.candidates)
    assert any(candidate.reward_take != 0.0 for candidate in env.candidates)


def test_q_network_and_replay_buffer_update():
    rng = np.random.default_rng(0)
    net = QNetwork()
    out = net(__import__("torch").zeros(4, 22))
    assert tuple(out.shape) == (4, 2)

    buf = ReplayBuffer(capacity=10)
    for _ in range(8):
        s = rng.normal(size=22).astype(np.float32)
        buf.push(s, 1, 1.0, np.zeros(22, dtype=np.float32), 1.0)
    sample = buf.sample(batch_size=4, rng=rng)
    assert sample[0].shape == (4, 22)
    assert sample[1].shape == (4,)

    agent = DQNReflexivityAgent(seed=0)
    for _ in range(40):
        s = rng.normal(size=22).astype(np.float32)
        agent.push(s, 0, 0.0, np.zeros(22, dtype=np.float32), 1.0)
        agent.push(s, 1, 1.0, np.zeros(22, dtype=np.float32), 1.0)
    loss = agent.update(batch_size=16)
    assert loss is not None
    assert loss >= 0.0
    assert agent.select_action(np.zeros(22, dtype=np.float32), greedy=True) in (0, 1)


def test_train_rl_reflexivity_returns_policy_and_stats():
    result = train_rl_reflexivity(
        _sample_bars(),
        config=_fast_cfg(),
        episodes=2,
        batch_size=8,
        seed=1,
    )

    assert result.n_candidates > 0
    assert result.env.med.shape == (22,)
    assert result.env.mad.shape == (22,)
    assert len(result.agent.buffer) >= result.n_candidates * 2
    assert isinstance(result.losses, list)


def test_rl_reflexivity_predictor_fit_predict_contract():
    df = _sample_bars()
    predictor = RLReflexivityPredictor(
        _fast_cfg(),
        episodes=2,
        batch_size=8,
        seed=2,
    )

    predictor.fit(df.slice(0, 650), symbol="TEST")
    r_hat, rv_hat = predictor.predict(df.slice(650, 200))

    assert r_hat.shape == (200,)
    assert rv_hat.shape == (200,)
    assert np.isfinite(r_hat).all()
    assert np.isfinite(rv_hat).all()
    assert predictor.n_train_candidates_ > 0
