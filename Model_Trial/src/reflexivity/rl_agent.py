"""
src/reflexivity/rl_agent.py — Double DQN agent for reflexivity trade gating.

The policy is a small MLP that maps the 22-D reflexivity state to Q-values for
two actions: ``PASS`` (0) and ``TAKE`` (1).  We use Double DQN (Hasselt 2015):

    * two networks — online ``q`` and target ``q_target`` (soft-updated);
    * the online net selects the greedy action, the target net evaluates it,
      which removes the maximization bias that single-DQN suffers;
    * an experience replay buffer breaks temporal correlation between updates;
    * epsilon-greedy exploration decays from ``eps_start`` to ``eps_min``.

Because PASS always yields reward 0 and TAKE's reward is fully observable from
the simulated trade, each candidate also supplies an exact (s, PASS, 0) and
(s, TAKE, r_take) transition — so the agent receives exact supervision for both
Q-values rather than a single Monte-Carlo sample.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """MLP: 22 -> 64 (GELU + LayerNorm) -> 64 (GELU) -> 2 (Q-values)."""

    def __init__(self, in_dim: int = 22, hidden: Tuple[int, int] = (64, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.GELU(),
            nn.LayerNorm(hidden[0]),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Linear(hidden[1], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Flat deque replay buffer of (s, a, r, s_next, done) transitions."""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.s: deque = deque(maxlen=capacity)
        self.a: deque = deque(maxlen=capacity)
        self.r: deque = deque(maxlen=capacity)
        self.ns: deque = deque(maxlen=capacity)
        self.d: deque = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d) -> None:
        self.s.append(np.asarray(s, dtype=np.float32))
        self.a.append(int(a))
        self.r.append(float(r))
        self.ns.append(np.asarray(ns, dtype=np.float32))
        self.d.append(float(d))

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = len(self.s)
        idx = rng.integers(0, n, size=min(batch_size, n))
        s = np.stack([self.s[i] for i in idx])
        ns = np.stack([self.ns[i] for i in idx])
        a = np.array([self.a[i] for i in idx], dtype=np.int64)
        r = np.array([self.r[i] for i in idx], dtype=np.float32)
        d = np.array([self.d[i] for i in idx], dtype=np.float32)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return len(self.s)


class DQNReflexivityAgent:
    def __init__(
        self,
        state_dim: int = 22,
        hidden: Tuple[int, int] = (64, 64),
        lr: float = 1e-3,
        gamma: float = 0.0,            # candidates are independent 1-step episodes
        tau: float = 0.005,            # soft target-update coefficient
        buffer_capacity: int = 200_000,
        eps_start: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.985,      # multiplied per training epoch
        device: str = "cpu",
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(state_dim, hidden).to(self.device)
        self.qt = QNetwork(state_dim, hidden).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.AdamW(self.q.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.train_losses_: list = []

    # ── action selection ──
    @torch.no_grad()
    def select_action(self, state, greedy: bool = False) -> int:
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(0, 2))
        x = torch.from_numpy(s).to(self.device)
        q = self.q(x).cpu().numpy()[0]
        return int(np.argmax(q))

    @torch.no_grad()
    def q_values(self, state) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        x = torch.from_numpy(s).to(self.device)
        return self.q(x).cpu().numpy()[0]

    def push(self, s, a, r, ns, d) -> None:
        self.buffer.push(s, a, r, ns, d)

    def decay_eps(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # ── single gradient step (Double DQN) ──
    def update(self, batch_size: int) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(batch_size, self.rng)
        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).long().to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        ns_t = torch.from_numpy(ns).to(self.device)
        d_t = torch.from_numpy(d).to(self.device)

        q_vals = self.q(s_t)
        q_a = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_a = self.q(ns_t).argmax(1)
            next_q = self.qt(ns_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = r_t + self.gamma * next_q * (1.0 - d_t)

        loss = F.smooth_l1_loss(q_a, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        # soft-update target network
        for p, tp in zip(self.q.parameters(), self.qt.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return float(loss.item())

    # ── persistence ──
    def save(self, path) -> None:
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_state": self.q.state_dict(),
            "qt_state": self.qt.state_dict(),
            "cfg": {
                "state_dim": self.state_dim, "lr": self.opt.param_groups[0]["lr"],
                "tau": self.tau, "gamma": self.gamma, "eps_min": self.eps_min,
                "eps_decay": self.eps_decay,
            },
        }, path)
        logger.info(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path, device: str = "cpu") -> "DQNReflexivityAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(state_dim=ckpt["cfg"]["state_dim"], device=device)
        obj.q.load_state_dict(ckpt["q_state"])
        obj.qt.load_state_dict(ckpt["qt_state"])
        obj.q.eval(); obj.qt.eval()
        return obj
