"""End-to-End Training & Evaluation Pipeline for Continuous Inception-Conformer (Plans 1, 2, 3).

Integrates:
1. Multi-scale continuous features, RBF soft bins & macro group transitions (Plan 1).
2. Multi-task composite loss (Soft-CE + RPS + Huber + Cost-Aware Direction Gate + Log-Concavity) (Plan 2).
3. Causal Inception-Conformer backbone & Calibrated Expected Utility Decision Engine with Fractional Kelly sizing (Plan 3).
4. Real-time flushed Tee logging to console and persistent log file in data/reports/.
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
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import data_path, load_config
from decision_engine import CalibratedDecisionEngine, TemperatureCalibrator
from hybrid_conformer import HybridInceptionConformer
from ordinal_losses import MultiTaskCompositeLoss


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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: HybridInceptionConformer,
    loader: DataLoader,
    criterion: MultiTaskCompositeLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    loss_components: Dict[str, float] = {}
    
    for batch_x, batch_z, batch_g, batch_targets, _batch_labels in loader:
        batch_x = batch_x.to(device)
        batch_z = batch_z.to(device)
        batch_g = batch_g.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        out = model(batch_x, batch_z, group_features=batch_g)
        
        loss, metrics = criterion(
            logits=out["logits"],
            pred_return=out["pred_return"],
            dir_logits=out["dir_logits"],
            continuous_targets=batch_targets,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_targets)
        for k, v in metrics.items():
            loss_components[k] = loss_components.get(k, 0.0) + v * len(batch_targets)
            
    n = len(loader.dataset)
    return {k: v / n for k, v in loss_components.items()}


@torch.no_grad()
def evaluate(
    model: HybridInceptionConformer,
    loader: DataLoader,
    criterion: MultiTaskCompositeLoss,
    decision_engine: CalibratedDecisionEngine,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    loss_components: Dict[str, float] = {}
    
    total_correct = 0
    total_top3 = 0
    total_samples = 0
    total_dir_correct = 0
    total_dir_count = 0
    total_mae = 0.0
    
    all_targets = []
    all_decisions = []
    
    for batch_x, batch_z, batch_g, batch_targets, batch_labels in loader:
        batch_x = batch_x.to(device)
        batch_z = batch_z.to(device)
        batch_g = batch_g.to(device)
        batch_targets = batch_targets.to(device)
        batch_labels = batch_labels.to(device)
        
        out = model(batch_x, batch_z, group_features=batch_g)
        loss, metrics = criterion(
            logits=out["logits"],
            pred_return=out["pred_return"],
            dir_logits=out["dir_logits"],
            continuous_targets=batch_targets,
        )
        
        batch_n = len(batch_targets)
        for k, v in metrics.items():
            loss_components[k] = loss_components.get(k, 0.0) + v * batch_n
            
        # Accuracy & Distributional Metrics
        logits = out["logits"]
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == batch_labels).sum().item())
        
        top3 = logits.topk(3, dim=-1).indices
        total_top3 += int((top3 == batch_labels.unsqueeze(-1)).any(dim=-1).sum().item())
        
        probs_tensor = out["probs"]
        bin_means_tensor = criterion.bin_centers.unsqueeze(0)
        dist_expected_returns = torch.sum(probs_tensor * bin_means_tensor, dim=-1)  # E[y|x] = sum p_i * mu_i
        
        total_mae += float(torch.abs(dist_expected_returns - batch_targets).sum().item())
        
        # Distribution-derived Directional Accuracy: sign(E[y|x]) vs sign(true_target)
        pred_dist_dir = torch.sign(dist_expected_returns)
        true_dist_dir = torch.sign(batch_targets)
        valid_dir = (torch.abs(batch_targets) > 1e-4)
        if valid_dir.any():
            total_dir_correct += int((pred_dist_dir[valid_dir] == true_dist_dir[valid_dir]).sum().item())
            total_dir_count += int(valid_dir.sum().item())
            
        total_samples += batch_n
        probs = out["probs"].cpu().numpy()
        targets_np = batch_targets.cpu().numpy()
        all_targets.extend(targets_np)
        
        for p in probs:
            all_decisions.append(decision_engine.evaluate_distribution(p))
            
    n = len(loader.dataset)
    eval_loss = {k: v / n for k, v in loss_components.items()}
    eval_loss["bin_accuracy"] = total_correct / max(1, total_samples)
    eval_loss["top3_accuracy"] = total_top3 / max(1, total_samples)
    eval_loss["directional_accuracy"] = (total_dir_correct / max(1, total_dir_count)) if total_dir_count > 0 else 0.50
    eval_loss["return_mae"] = total_mae / max(1, total_samples)
    
    trade_actions = [d.action for d in all_decisions]
    n_trades = sum(1 for a in trade_actions if a != "FLAT")
    coverage = n_trades / max(1, len(all_decisions))
    
    pnls = []
    correct_dir = 0
    for d, target in zip(all_decisions, all_targets):
        if d.action == "LONG":
            pnl = (target - decision_engine.cost_long) * d.position_size
            pnls.append(pnl)
            if target > 0:
                correct_dir += 1
        elif d.action == "SHORT":
            pnl = (-target - decision_engine.cost_short) * abs(d.position_size)
            pnls.append(pnl)
            if target < 0:
                correct_dir += 1
                
    trading_metrics = {
        "coverage_pct": coverage * 100.0,
        "n_trades": n_trades,
        "total_pnl": float(np.sum(pnls)) if pnls else 0.0,
        "mean_pnl_per_trade": float(np.mean(pnls)) if pnls else 0.0,
        "trade_directional_accuracy": (correct_dir / max(1, n_trades)) * 100.0,
    }
    
    return eval_loss, trading_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_continuous_45m.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    set_seed(42)
    config = load_config(args.config)
    
    log_path = data_path(config, "reports", "train_multiscale_conformer.log")
    tee = Tee(log_path)
    sys.stdout = tee
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Executing on device: {device}")
    
    dataset_path = data_path(config, "processed", config.get("model_dataset", "gold_multiscale_45m_dataset.npz"))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}. Run build_continuous_multiscale_dataset.py first.")
        
    print(f"[{time.strftime('%H:%M:%S')}] Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    cause_windows = data["cause_windows"]
    group_features = data["group_features"]
    targets = data["targets"]
    labels = data["labels"]
    target_edges = data["target_edges"]
    
    n_samples = len(cause_windows)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)
    
    num_bins = len(target_edges) - 1
    bin_means = np.zeros(num_bins, dtype=np.float32)
    bin_vars = np.zeros(num_bins, dtype=np.float32)
    
    train_labels = labels[:train_end]
    train_targets = targets[:train_end]
    for b in range(num_bins):
        mask = train_labels == b
        if np.any(mask):
            bin_means[b] = float(np.mean(train_targets[mask]))
            bin_vars[b] = float(np.var(train_targets[mask]))
            
    print(f"[{time.strftime('%H:%M:%S')}] Dataset loaded: {n_samples:,} examples (Train: {train_end:,}, Val: {val_end - train_end:,}, Test: {n_samples - val_end:,})")
    print(f"[{time.strftime('%H:%M:%S')}] Empirical Training Bin Means: {np.round(bin_means, 4).tolist()}")
    
    z_1m_all = cause_windows[:, :, 0]
    
    train_ds = TensorDataset(
        torch.from_numpy(cause_windows[:train_end]),
        torch.from_numpy(z_1m_all[:train_end]),
        torch.from_numpy(group_features[:train_end]),
        torch.from_numpy(targets[:train_end]),
        torch.from_numpy(labels[:train_end]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(cause_windows[train_end:val_end]),
        torch.from_numpy(z_1m_all[train_end:val_end]),
        torch.from_numpy(group_features[train_end:val_end]),
        torch.from_numpy(targets[train_end:val_end]),
        torch.from_numpy(labels[train_end:val_end]),
    )
    test_ds = TensorDataset(
        torch.from_numpy(cause_windows[val_end:]),
        torch.from_numpy(z_1m_all[val_end:]),
        torch.from_numpy(group_features[val_end:]),
        torch.from_numpy(targets[val_end:]),
        torch.from_numpy(labels[val_end:]),
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    model = HybridInceptionConformer(
        continuous_dim=cause_windows.shape[-1],
        group_feat_dim=group_features.shape[-1],
        num_target_bins=num_bins,
        d_model=128,
        nhead=4,
        num_conformer_layers=2,
    ).to(device)
    
    bin_centers_tensor = torch.from_numpy(bin_means).to(device)
    soft_sigma = float(config.get("soft_sigma", 0.02))
    criterion = MultiTaskCompositeLoss(
        bin_centers=bin_centers_tensor,
        cost_hurdle=float(config.get("cost", 0.50)),
        soft_sigma=soft_sigma,
        lambda_focal=2.00,
        lambda_dir_sign=1.50,
        lambda_dir_mass=0.50,
        lambda_rps=0.20,
        lambda_reg=0.20,
        lambda_dir=0.50,
        lambda_uni=0.05,
        chop_threshold=0.03,
    )
    
    # Entropy sanity check
    from ordinal_losses import generate_gaussian_soft_targets
    with torch.no_grad():
        sample_q = generate_gaussian_soft_targets(
            torch.from_numpy(train_targets[:1000]), bin_centers_tensor.cpu(), sigma=soft_sigma
        )
        sample_entropy = float(-(sample_q * torch.log(sample_q + 1e-8)).sum(dim=-1).mean().item())
        max_ent = float(np.log(num_bins))
    print(f"[{time.strftime('%H:%M:%S')}] Soft Target Bandwidth: sigma={soft_sigma} | Target Entropy: {sample_entropy:.3f} / {max_ent:.3f} ({sample_entropy/max_ent:.1%} of uniform)")
    
    decision_engine = CalibratedDecisionEngine(
        bin_conditional_means=bin_means,
        bin_conditional_variances=bin_vars,
        cost_long=float(config.get("cost", 0.50)),
        cost_short=float(config.get("cost", 0.50)),
        hurdle_tau=0.20,
        snr_threshold=0.35,
        kelly_fraction=0.25,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_dir_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    best_state_dict = None
    
    print("\n--- Starting Inception-Conformer Training ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_losses, val_trade = evaluate(model, val_loader, criterion, decision_engine, device)
        scheduler.step()
        
        val_dir_acc = val_losses["directional_accuracy"]
        if val_dir_acc > best_val_dir_acc:
            best_val_dir_acc = val_dir_acc
            best_val_loss = val_losses["loss_total"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        elapsed = time.time() - t0
        print(
            f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_metrics['loss_total']:.4f} (Focal: {train_metrics.get('loss_focal', 0.0):.4f}) | "
            f"Val Loss: {val_losses['loss_total']:.4f} | "
            f"Val Acc: {val_losses['bin_accuracy']:.2%} | "
            f"Val Top-3: {val_losses['top3_accuracy']:.2%} | "
            f"Val Dir Acc: {val_losses['directional_accuracy']:.2%} | "
            f"Val MAE: {val_losses['return_mae']:.4f}"
        )
        
    # Load best checkpoint for test evaluation
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"\nLoaded Best Checkpoint from Epoch {best_epoch} (Val Dir Acc: {best_val_dir_acc:.2%}) for Out-Of-Sample Evaluation.")
        
    print("\n==========================================")
    print("      OUT-OF-SAMPLE TEST RESULTS          ")
    print("==========================================")
    test_losses, test_trade = evaluate(model, test_loader, criterion, decision_engine, device)
    print(f"9-Bin Range Accuracy:  {test_losses['bin_accuracy']:.2%}")
    print(f"Top-3 Range Accuracy:  {test_losses['top3_accuracy']:.2%}")
    print(f"Directional Accuracy:  {test_losses['directional_accuracy']:.2%}")
    print(f"Continuous MAE:        {test_losses['return_mae']:.4f}")
    print(f"Test RPS Score:        {test_losses['loss_rps']:.4f}")
    print(f"Best Epoch:            {best_epoch}")
    
    # Save checkpoint
    model_path = data_path(config, "models", "gold_multiscale_conformer.pt")
    oos_model_path = data_path(config, "models", "gold_multiscale_conformer_oos.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "model_state_dict": best_state_dict if best_state_dict is not None else model.state_dict(),
        "bin_means": bin_means,
        "bin_vars": bin_vars,
        "target_edges": target_edges,
        "feature_names": data["feature_names"],
        "best_epoch": best_epoch,
        "test_metrics": test_losses,
    }
    torch.save(checkpoint_data, model_path)
    torch.save(checkpoint_data, oos_model_path)
    
    print(f"Checkpoint saved to:   {oos_model_path}")
    print("==========================================\n")
    print(f"[{time.strftime('%H:%M:%S')}] Saved model weights to {model_path} and {oos_model_path}")
    print(f"[{time.strftime('%H:%M:%S')}] Log file saved to {log_path}")
    tee.close()


if __name__ == "__main__":
    main()

