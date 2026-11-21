"""
Training loop for the RL-gated Reflexivity Reversal Model.

The environment supplies fully observed candidate outcomes: each candidate has
an exact PASS reward of 0 and a TAKE reward from the simulated trade.  We train
the DQN on both actions for every candidate as independent one-step episodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from .model import ReflexivityConfig
from .rl_agent import DQNReflexivityAgent
from .rl_env import STATE_DIM, ReflexivityTradingEnv

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingResult:
    agent: DQNReflexivityAgent
    env: ReflexivityTradingEnv
    losses: list[float]
    mean_reward_take: float
    n_candidates: int
    n_positive: int
    n_negative: int


def train_rl_reflexivity(
    df: pl.DataFrame,
    config: Optional[ReflexivityConfig] = None,
    episodes: int = 20,
    batch_size: int = 32,
    updates_per_episode: Optional[int] = None,
    lr: float = 1e-3,
    gamma: float = 0.0,
    reward_beta: float = 1.5,
    reward_clip: Optional[float] = 10.0,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_path: Optional[str | Path] = None,
) -> RLTrainingResult:
    """
    Train a DQN policy to PASS or TAKE reflexivity candidate trades.

    Returns the trained agent plus the environment statistics required to
    standardize future inference states without leaking test-fold data.
    """
    cfg = config or ReflexivityConfig()
    env = ReflexivityTradingEnv(cfg, reward_beta=reward_beta, reward_clip=reward_clip)
    env.collect_candidates(df)

    agent = DQNReflexivityAgent(
        state_dim=STATE_DIM,
        lr=lr,
        gamma=gamma,
        device=device,
        seed=seed,
    )

    if env.n_candidates == 0:
        logger.warning("RL reflexivity training found 0 candidate setups; policy will pass all.")
        return RLTrainingResult(
            agent=agent,
            env=env,
            losses=[],
            mean_reward_take=0.0,
            n_candidates=0,
            n_positive=0,
            n_negative=0,
        )

    zero_next = np.zeros(STATE_DIM, dtype=np.float32)
    rewards = np.array([c.reward_take for c in env.candidates], dtype=np.float32)
    for state, candidate in zip(env.state_matrix, env.candidates):
        agent.push(state, 0, candidate.reward_pass, zero_next, 1.0)
        agent.push(state, 1, candidate.reward_take, zero_next, 1.0)

        # Give economically surprising candidates extra replay presence without
        # introducing a separate prioritized-buffer implementation.
        if abs(candidate.reward_take) >= max(1.0, float(np.nanstd(rewards))):
            agent.push(state, 1, candidate.reward_take, zero_next, 1.0)

    if updates_per_episode is None:
        updates_per_episode = max(1, len(agent.buffer) // max(1, batch_size))

    losses: list[float] = []
    for _ in range(max(1, episodes)):
        for _ in range(updates_per_episode):
            loss = agent.update(batch_size=batch_size)
            if loss is not None:
                losses.append(loss)
        agent.decay_eps()

    if checkpoint_path is not None:
        agent.save(checkpoint_path)

    n_pos = int(np.sum(rewards > 0))
    n_neg = int(np.sum(rewards < 0))
    logger.info(
        "RL reflexivity trained: candidates=%d positive=%d negative=%d mean_take_reward=%.4f",
        env.n_candidates,
        n_pos,
        n_neg,
        float(np.mean(rewards)),
    )

    return RLTrainingResult(
        agent=agent,
        env=env,
        losses=losses,
        mean_reward_take=float(np.mean(rewards)),
        n_candidates=env.n_candidates,
        n_positive=n_pos,
        n_negative=n_neg,
    )
