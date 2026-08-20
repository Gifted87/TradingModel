"""Train a conservative offline RL policy on frozen Transformer predictions."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import data_path, load_config
from train_advanced import GoldTransformer


ACTION_SIZES = np.array([0.0, 0.25, 0.50, 1.0, 0.25, 0.50, 1.0], dtype=np.float32)
ACTION_DIRECTIONS = np.array([0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float32)


class Policy(nn.Module):
    def __init__(self, features: int, actions: int = 7) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_state(data: pd.DataFrame, probabilities: np.ndarray, train_end: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    columns = sorted(c for c in data.columns if c.startswith("change_"))
    changes = data[columns].to_numpy(dtype=np.float32)
    train = data.iloc[:train_end]
    class_returns = np.array([train[train.label == i].effect.mean() for i in range(classes)], dtype=np.float32)
    expected = probabilities @ class_returns
    states = [probabilities, expected[:, None], probabilities.max(axis=1)[:, None]]
    for minutes in (5, 15, 30, 60):
        window = changes[:, -minutes:]
        states.extend([window.mean(1, keepdims=True), window.std(1, keepdims=True), window.sum(1, keepdims=True), (window > 0).mean(1, keepdims=True)])
    state = np.concatenate(states, axis=1).astype(np.float32)
    return state, expected


def evaluate(policy: Policy, states: np.ndarray, effects: np.ndarray, cost: float) -> dict:
    with torch.no_grad():
        actions = policy(torch.tensor(states)).argmax(dim=1).numpy()
    pnl = ACTION_DIRECTIONS[actions] * ACTION_SIZES[actions] * effects - (ACTION_SIZES[actions] > 0) * cost
    active = ACTION_SIZES[actions] > 0
    traded = pnl[active]
    if len(traded):
        curve = np.cumsum(traded)
        drawdown = np.maximum.accumulate(curve) - curve
    else:
        drawdown = np.array([0.0])
    return {
        "examples": int(len(effects)),
        "trades": int(active.sum()),
        "coverage": float(active.mean()),
        "total_pnl": float(pnl.sum()),
        "average_trade_pnl": float(traded.mean()) if len(traded) else 0.0,
        "win_rate": float((traded > 0).mean()) if len(traded) else 0.0,
        "max_drawdown": float(drawdown.max()),
        "action_counts": {str(i): int((actions == i).sum()) for i in range(len(ACTION_SIZES))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cost", type=float, default=0.50)
    args = parser.parse_args()
    set_seed()
    config = load_config(args.config)
    data = pd.read_parquet(data_path(config, "processed", config["model_dataset"])).sort_values("target_time").reset_index(drop=True)
    columns = sorted(c for c in data.columns if c.startswith("change_"))
    n = len(data)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    preprocessing = np.load(data_path(config, "models", "gold_transformer_oos_preprocessing.npz"))
    checkpoint = torch.load(data_path(config, "models", "gold_transformer_oos.pt"), map_location="cpu", weights_only=True)
    model = GoldTransformer(checkpoint["sequence_length"], checkpoint["classes"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    x = (data[columns].to_numpy(dtype=np.float32) - preprocessing["mean"]) / preprocessing["std"]
    probabilities = []
    with torch.no_grad():
        for start in range(0, n, 8192):
            logits, _ = model(torch.tensor(x[start:start + 8192], dtype=torch.float32))
            probabilities.append(torch.softmax(logits, dim=1).numpy())
    probabilities = np.concatenate(probabilities)
    states, expected = make_state(data, probabilities, train_end, checkpoint["classes"])
    effects = data.effect.to_numpy(dtype=np.float32)
    rewards = ACTION_DIRECTIONS[None, :] * ACTION_SIZES[None, :] * effects[:, None] - (ACTION_SIZES[None, :] > 0) * args.cost
    state_mean, state_std = states[:train_end].mean(0), states[:train_end].std(0)
    state_std[state_std < 1e-6] = 1.0
    states = (states - state_mean) / state_std
    loader = DataLoader(TensorDataset(torch.tensor(states[:train_end]), torch.tensor(rewards[:train_end])), batch_size=8192, shuffle=True)
    policy = Policy(states.shape[1])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        losses = []
        for batch_state, batch_reward in loader:
            logits = policy(batch_state)
            action_probabilities = torch.softmax(logits, dim=1)
            expected_reward = (action_probabilities * batch_reward).sum(1).mean()
            entropy = -(action_probabilities * torch.log(action_probabilities.clamp_min(1e-8))).sum(1).mean()
            loss = -expected_reward - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        policy.eval()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": evaluate(policy, states[train_end:valid_end], effects[train_end:valid_end], args.cost)})
        print(json.dumps(history[-1]))

    output = data_path(config, "models")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "features": states.shape[1], "actions": len(ACTION_SIZES)}, output / "gold_rl_policy.pt")
    np.savez(output / "gold_rl_preprocessing.npz", mean=state_mean, std=state_std)
    report = {"action_sizes": ACTION_SIZES.tolist(), "action_directions": ACTION_DIRECTIONS.tolist(), "cost": args.cost, "history": history, "unseen_test": evaluate(policy, states[valid_end:], effects[valid_end:], args.cost)}
    with (output / "gold_rl_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"unseen_test": report["unseen_test"]}, indent=2))


if __name__ == "__main__":
    main()
