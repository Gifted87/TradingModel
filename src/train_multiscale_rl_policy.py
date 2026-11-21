"""Train an Offline RL Policy / Contextual Bandit on Frozen Inception-Conformer Predictions.

Integration with Plans 1, 2, 3:
1. Backbone (Plan 3) acts as the frozen Market State Extractor, generating calibrated probability
   vectors, expected returns, predictive uncertainties, direction probabilities, and group transitions.
   The backbone is loaded from the saved Stage 1 checkpoint (gold_multiscale_conformer_oos.pt) and
   its weights are fully frozen — no gradients flow through it.
2. The RL Policy maps the rich multi-task state vector s_t -> 7 discrete sizing actions:
   [No Trade (0.0), Long 0.25x, Long 0.50x, Long 1.0x, Short 0.25x, Short 0.50x, Short 1.0x]
3. The RL objective directly optimizes net PnL after spread and commission costs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import data_path, load_config
from decision_engine import CalibratedDecisionEngine
from hybrid_conformer import HybridInceptionConformer


ACTION_SIZES = np.array([0.0, 0.25, 0.50, 1.0, 0.25, 0.50, 1.0], dtype=np.float32)
ACTION_DIRECTIONS = np.array([0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float32)


class Tee:
    """Write simultaneously to stdout and a persistent log file with immediate flushing."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = log_path.open("w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


class RLPolicyNetwork(nn.Module):
    """Contextual Bandit Policy Network with Entropy Regularization."""

    def __init__(self, state_dim: int, num_actions: int = 7, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_rl_states(
    model: HybridInceptionConformer,
    decision_engine: CalibratedDecisionEngine,
    cause_windows: np.ndarray,
    group_features: np.ndarray,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Extracts rich multi-task state vectors s_t from the frozen Inception-Conformer.
    
    State vector per sample (31 dimensions):
        [9]  ordinal bin probabilities p_0..p_8
        [1]  continuous expected return pred_return
        [1]  conditional volatility pred_vol
        [3]  cost-aware direction probabilities (bearish, flat, bullish)
        [1]  expected price move mu_hat
        [1]  predictive uncertainty sigma_hat
        [1]  signal-to-noise ratio SNR
        [2]  net utilities (u_long, u_short)
        [12] macro group transition features
        = 31 total
    """
    model.eval()
    n = len(cause_windows)
    states = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bx = torch.from_numpy(cause_windows[start:end]).to(device)
            bz = bx[:, :, 0]  # column 0 = z_1m (confirmed from get_feature_names())
            bg = torch.from_numpy(group_features[start:end]).to(device)

            out = model(bx, bz, group_features=bg)
            probs = out["probs"].cpu().numpy()           # (B, 9)
            pred_return = out["pred_return"].cpu().numpy()  # (B, 1)
            dir_probs = out["dir_probs"].cpu().numpy()   # (B, 3)
            pred_vol = out["pred_vol"].cpu().numpy()     # (B, 1)

            batch_states = []
            for i in range(len(probs)):
                p = probs[i]
                d = decision_engine.evaluate_distribution(p)
                s = np.concatenate([
                    p,                                    # 9  ordinal bin probabilities
                    [float(pred_return[i].flat[0])],      # 1  continuous expected return
                    [float(pred_vol[i].flat[0])],         # 1  conditional volatility
                    dir_probs[i],                         # 3  cost-aware direction probabilities
                    [d.expected_return],                  # 1  expected price move mu_hat
                    [d.uncertainty],                      # 1  predictive uncertainty sigma_hat
                    [d.snr],                              # 1  signal-to-noise ratio
                    [d.u_long, d.u_short],                # 2  net utilities
                    group_features[start + i],            # 12 macro group transition features
                ])
                batch_states.append(s)

            states.append(np.array(batch_states, dtype=np.float32))

    return np.concatenate(states, axis=0)


def evaluate_rl_policy(
    policy: RLPolicyNetwork,
    states: np.ndarray,
    effects: np.ndarray,
    cost: float = 0.50,
    device: torch.device = None,
) -> Dict[str, float]:
    policy.eval()
    if device is None:
        device = next(policy.parameters()).device
    with torch.no_grad():
        logits = policy(torch.from_numpy(states).to(device))
        actions = logits.argmax(dim=-1).cpu().numpy()

    sizes = ACTION_SIZES[actions]
    directions = ACTION_DIRECTIONS[actions]
    active_mask = sizes > 0.0

    # Net PnL = Direction * Size * Effect - Cost * (Size > 0)
    pnl = directions * sizes * effects - (active_mask.astype(float)) * cost
    traded_pnl = pnl[active_mask]

    n_trades = int(active_mask.sum())
    coverage = float(active_mask.mean()) * 100.0
    total_pnl = float(pnl.sum())
    mean_pnl = float(traded_pnl.mean()) if n_trades > 0 else 0.0
    win_rate = float((traded_pnl > 0).mean()) * 100.0 if n_trades > 0 else 0.0

    # Max Drawdown on traded P&L curve
    if n_trades > 0:
        cum_pnl = np.cumsum(traded_pnl)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = running_max - cum_pnl
        max_dd = float(drawdown.max())
    else:
        max_dd = 0.0

    return {
        "n_samples": len(effects),
        "n_trades": n_trades,
        "coverage_pct": coverage,
        "total_pnl": total_pnl,
        "mean_pnl_per_trade": mean_pnl,
        "win_rate_pct": win_rate,
        "max_drawdown": max_dd,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_continuous_45m.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cost", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Tee logging to disk (mirrors Stage 1 behaviour)
    # -------------------------------------------------------------------------
    log_path = data_path(config, "reports", "train_multiscale_rl_policy.log")
    tee = Tee(log_path)
    sys.stdout = tee

    print(f"[{time.strftime('%H:%M:%S')}] [Stage 2] Executing on device: {device}")

    # -------------------------------------------------------------------------
    # Load processed dataset
    # -------------------------------------------------------------------------
    dataset_path = data_path(config, "processed", config.get("model_dataset", "gold_multiscale_45m_dataset.npz"))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}.")

    print(f"[{time.strftime('%H:%M:%S')}] Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    cause_windows = data["cause_windows"]
    group_features = data["group_features"]
    targets = data["targets"]
    target_edges = data["target_edges"]

    n_samples = len(cause_windows)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)
    num_bins = len(target_edges) - 1

    print(f"[{time.strftime('%H:%M:%S')}] Dataset: {n_samples:,} examples "
          f"(Train: {train_end:,}, Val: {val_end - train_end:,}, Test: {n_samples - val_end:,})")

    # -------------------------------------------------------------------------
    # Bug 6 FIX: Load the trained Stage 1 Conformer checkpoint.
    # bin_means and bin_vars come from the checkpoint (Bug 6b fix) — they MUST
    # be identical to those used during Stage 1 training so the CalibratedDecisionEngine
    # is consistent between the two stages.
    # -------------------------------------------------------------------------
    conformer_path = data_path(config, "models", "gold_multiscale_conformer_oos.pt")
    if not conformer_path.exists():
        raise FileNotFoundError(
            f"Stage 1 Conformer checkpoint not found at {conformer_path}. "
            f"Run train_multiscale_conformer.py first and ensure it completes successfully."
        )

    print(f"[{time.strftime('%H:%M:%S')}] Loading trained Stage 1 Conformer from {conformer_path}...")
    s1_checkpoint = torch.load(conformer_path, map_location=device, weights_only=False)

    # Load bin statistics from Stage 1 checkpoint (not recomputed — avoids subtle mismatch)
    bin_means = s1_checkpoint["bin_means"]   # np.ndarray (num_bins,)
    bin_vars  = s1_checkpoint["bin_vars"]    # np.ndarray (num_bins,)
    best_epoch_s1 = s1_checkpoint.get("best_epoch", "?")
    print(f"[{time.strftime('%H:%M:%S')}] Stage 1 best epoch: {best_epoch_s1} | "
          f"Bin Means: {np.round(bin_means, 4).tolist()}")

    # Initialise the Conformer and load trained weights
    model = HybridInceptionConformer(
        continuous_dim=cause_windows.shape[-1],
        group_feat_dim=group_features.shape[-1],
        num_target_bins=num_bins,
        d_model=128,
        nhead=4,
        num_conformer_layers=2,
    ).to(device)
    model.load_state_dict(s1_checkpoint["model_state_dict"])
    model.eval()

    # Freeze ALL Conformer weights — no gradients flow through the backbone
    for param in model.parameters():
        param.requires_grad = False

    print(f"[{time.strftime('%H:%M:%S')}] Conformer loaded and frozen ({sum(p.numel() for p in model.parameters()):,} params, all frozen).")

    # -------------------------------------------------------------------------
    # Decision engine uses Stage 1 bin_means/bin_vars (consistent with training)
    # -------------------------------------------------------------------------
    decision_engine = CalibratedDecisionEngine(
        bin_conditional_means=bin_means,
        bin_conditional_variances=bin_vars,
        cost_long=args.cost,
        cost_short=args.cost,
        hurdle_tau=0.20,
    )

    # -------------------------------------------------------------------------
    # Extract RL state vectors from frozen Conformer (one-time forward pass)
    # -------------------------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] Extracting multi-task RL state vectors from frozen Conformer...")
    t_extract = time.time()
    all_states = extract_rl_states(
        model, decision_engine, cause_windows, group_features,
        batch_size=512, device=device,
    )
    print(f"[{time.strftime('%H:%M:%S')}] State matrix shape: {all_states.shape} | "
          f"Mean: {all_states.mean():.4f} | Std: {all_states.std():.4f} | "
          f"Extraction time: {time.time() - t_extract:.1f}s")

    if all_states.std() < 0.001:
        print("WARNING: State std is near zero — Conformer weights may not have loaded correctly.")

    train_states  = all_states[:train_end]
    train_effects = targets[:train_end]
    val_states    = all_states[train_end:val_end]
    val_effects   = targets[train_end:val_end]
    test_states   = all_states[val_end:]
    test_effects  = targets[val_end:]

    # -------------------------------------------------------------------------
    # RL Policy Network
    # -------------------------------------------------------------------------
    state_dim = all_states.shape[-1]
    policy = RLPolicyNetwork(state_dim=state_dim, num_actions=7).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[{time.strftime('%H:%M:%S')}] RL Policy Network: state_dim={state_dim}, "
          f"params={sum(p.numel() for p in policy.parameters()):,}")

    # Precompute action tensors
    action_dirs_t  = torch.from_numpy(ACTION_DIRECTIONS).to(device)
    action_sizes_t = torch.from_numpy(ACTION_SIZES).to(device)

    train_ds = TensorDataset(torch.from_numpy(train_states), torch.from_numpy(train_effects))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # -------------------------------------------------------------------------
    # Policy Gradient Training: maximise E[Net PnL] with entropy regularisation
    # -------------------------------------------------------------------------
    print(f"\n--- Starting Offline RL Policy Training ({args.epochs} epochs) ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_s, batch_eff in train_loader:
            batch_s   = batch_s.to(device)
            batch_eff = batch_eff.to(device)

            logits = policy(batch_s)
            probs  = torch.softmax(logits, dim=-1)

            # Directional Asymmetric Payoff Calculation
            # 1. Base PnL = dir_a * size_a * effect
            raw_pnl = action_dirs_t.unsqueeze(0) * action_sizes_t.unsqueeze(0) * batch_eff.unsqueeze(1)
            is_active = (action_sizes_t.unsqueeze(0) > 0).float()
            
            # 2. Direction Alignment Multipliers
            # When action direction matches effect direction: reward scaled 1.5x + directional discovery bonus
            # When action direction opposes effect direction: penalty scaled 2.0x
            eff_sign = torch.sign(batch_eff).unsqueeze(1)
            act_sign = action_dirs_t.unsqueeze(0)
            
            correct_dir = (act_sign == eff_sign) & (is_active > 0) & (torch.abs(eff_sign) > 0)
            wrong_dir   = (act_sign == -eff_sign) & (is_active > 0) & (torch.abs(eff_sign) > 0)
            
            dir_multiplier = torch.ones_like(raw_pnl)
            dir_multiplier = torch.where(correct_dir, 1.50, dir_multiplier)
            dir_multiplier = torch.where(wrong_dir, 2.00, dir_multiplier)
            
            dir_bonus = torch.where(correct_dir, 0.25 * action_sizes_t.unsqueeze(0), torch.zeros_like(raw_pnl))
            
            payoffs = (raw_pnl * dir_multiplier + dir_bonus) - is_active * args.cost
            expected_pnl = torch.sum(probs * payoffs, dim=-1)

            # Entropy bonus prevents premature policy collapse to always-flat
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            loss = -expected_pnl.mean() - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        elapsed = time.time() - t0
        val_metrics = evaluate_rl_policy(policy, val_states, val_effects, cost=args.cost, device=device)
        print(
            f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {epoch_loss / max(1, n_batches):.4f} | "
            f"Val Trades: {val_metrics['n_trades']:,} ({val_metrics['coverage_pct']:.1f}%) | "
            f"Val Win Rate: {val_metrics['win_rate_pct']:.1f}% | "
            f"Val Mean PnL: {val_metrics['mean_pnl_per_trade']:.4f} | "
            f"Val Total PnL: {val_metrics['total_pnl']:,.2f}"
        )

    # -------------------------------------------------------------------------
    # Bug 9 FIX: Final OOS evaluation and save checkpoint
    # -------------------------------------------------------------------------
    test_metrics = evaluate_rl_policy(policy, test_states, test_effects, cost=args.cost, device=device)

    print("\n==========================================")
    print("    OUT-OF-SAMPLE RL POLICY TEST RESULTS  ")
    print("==========================================")
    print(f"Test Samples:            {test_metrics['n_samples']:,}")
    print(f"Trades Executed:         {test_metrics['n_trades']:,} ({test_metrics['coverage_pct']:.2f}% coverage)")
    print(f"Win Rate:                {test_metrics['win_rate_pct']:.2f}%")
    print(f"Total Net PnL:           {test_metrics['total_pnl']:,.2f} price units")
    print(f"Mean PnL Per Trade:      {test_metrics['mean_pnl_per_trade']:.4f} price units")
    print(f"Max Drawdown (traded):   {test_metrics['max_drawdown']:.2f} price units")
    print(f"Stage 1 Backbone Epoch:  {best_epoch_s1}")

    rl_model_path = data_path(config, "models", "gold_multiscale_rl_policy.pt")
    rl_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "state_dim": state_dim,
        "num_actions": 7,
        "action_sizes": ACTION_SIZES.tolist(),
        "action_directions": ACTION_DIRECTIONS.tolist(),
        "cost": args.cost,
        "test_metrics": test_metrics,
        "conformer_checkpoint": str(conformer_path),
        "stage1_best_epoch": best_epoch_s1,
        "bin_means": bin_means,
        "bin_vars": bin_vars,
    }, rl_model_path)

    print(f"Checkpoint saved to:     {rl_model_path}")
    print("==========================================\n")
    print(f"[{time.strftime('%H:%M:%S')}] Log file saved to {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
