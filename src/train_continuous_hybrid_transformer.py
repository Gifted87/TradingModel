"""Train a Continuous-Input Hybrid Inception-1D + Transformer model on the 45-minute gold sequence.

Architecture:
1. Continuous 1-minute price changes (45 steps) normalized by train split stats (zero leakage).
2. Input Projection: Linear(1, d_model) + 3 Temporal Group Embeddings (15 bars per group).
3. Multi-Scale Inception-1D Conv Block:
   - Branch 1 (k=3): 3-minute micro wicks & reversals
   - Branch 2 (k=5): 5-minute candle structure
   - Branch 3 (k=15): 15-minute temporal group momentum
   - Branch 4: MaxPool1d + 1x1 Conv for volatility spikes
   - Residual connection & LayerNorm
4. Learned Strided Compression (Conv1d k=3, stride=3): 45 continuous steps -> 15 macro tokens.
5. Transformer Multi-Head Self-Attention Encoder (2 layers, nhead=4).
6. Attentive Pooling -> 9-Bin Class Head & Continuous Return Head.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import data_path, load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Tee:
    """Write to both standard stdout and a log file with immediate line-flushing."""

    def __init__(self, stream, log_path: Path) -> None:
        self._stream = stream
        self._file = log_path.open("w", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._stream.flush()
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class Inception1DBlock(nn.Module):
    """Multi-scale 1D convolution block extracting 3m, 5m, and 15m patterns."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        branch_ch = channels // 4
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(channels, branch_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_ch),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(channels, branch_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_ch),
            nn.GELU(),
        )
        self.branch15 = nn.Sequential(
            nn.Conv1d(channels, branch_ch, kernel_size=15, padding=7),
            nn.BatchNorm1d(branch_ch),
            nn.GELU(),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(channels, branch_ch, kernel_size=1),
            nn.BatchNorm1d(branch_ch),
            nn.GELU(),
        )
        
        self.proj_out = nn.Conv1d(branch_ch * 4, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 45, d_model) -> transpose to (B, d_model, 45) for Conv1D
        x_in = x.transpose(1, 2)
        b3 = self.branch3(x_in)
        b5 = self.branch5(x_in)
        b15 = self.branch15(x_in)
        bp = self.branch_pool(x_in)
        
        merged = torch.cat([b3, b5, b15, bp], dim=1)
        out = self.proj_out(merged).transpose(1, 2)  # (B, 45, d_model)
        return self.norm(x + self.drop(out))


class ContinuousHybridInceptionTransformer(nn.Module):
    """
    Hybrid Architecture for Continuous Inputs:
    1. Continuous Linear Projection + 3 Temporal Group Embeddings
    2. Multi-Scale Inception-1D Convolution (3m, 5m, 15m)
    3. Learned Strided Compression (45 -> 15 tokens)
    4. Transformer Multi-Head Self-Attention
    5. Attentive Pooling -> 9-Bin Class Head & Continuous Return Head
    """

    def __init__(
        self,
        sequence_length: int = 45,
        num_target_bins: int = 9,
        num_groups: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.num_groups = num_groups
        self.group_size = sequence_length // num_groups  # 15
        
        # 1. Continuous Input Projection & Group Embeddings
        self.input_proj = nn.Linear(1, d_model)
        self.group_embedding = nn.Embedding(num_groups, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        
        group_indices = torch.tensor([i // self.group_size for i in range(sequence_length)], dtype=torch.long)
        self.register_buffer("group_indices", group_indices)
        
        self.input_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        
        # 2. Multi-scale Inception-1D Convolution Block
        self.inception_block = Inception1DBlock(channels=d_model, dropout=dropout)
        
        # 3. Strided Convolution to compress 45 steps -> 15 macro tokens (3:1 reduction)
        self.num_patches = sequence_length // 3  # 15
        self.strided_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.patch_pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # 4. Transformer Multi-Head Self-Attention Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.transformer_norm = nn.LayerNorm(d_model)
        
        # 5. Attentive Pooling
        self.attn_pool = nn.Linear(d_model, 1)
        
        # 6. Output Heads
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_target_bins),
        )
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_continuous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_continuous shape: (B, 45) -> unsqueeze to (B, 45, 1)
        B = x_continuous.size(0)
        x_proj = self.input_proj(x_continuous.unsqueeze(-1))  # (B, 45, d_model)
        
        groups = self.group_embedding(self.group_indices.unsqueeze(0))
        h = x_proj + groups + self.position_embedding
        h = self.drop(self.input_norm(h))  # (B, 45, d_model)
        
        # Step 2: Multi-Scale Inception-1D Feature Extraction
        h_inception = self.inception_block(h)  # (B, 45, d_model)
        
        # Step 3: Strided compression to 15 patch-tokens
        h_compressed = self.strided_conv(h_inception.transpose(1, 2)).transpose(1, 2)  # (B, 15, d_model)
        h_compressed = h_compressed + self.patch_pos_embedding
        
        # Step 4: Transformer Self-Attention across the 15 macro tokens
        encoded = self.transformer_norm(self.transformer(h_compressed))  # (B, 15, d_model)
        
        # Step 5: Attentive Pooling across tokens
        attn_weights = torch.softmax(self.attn_pool(encoded).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (encoded * attn_weights).sum(dim=1)  # (B, d_model)
        
        # Step 6: Prediction Heads
        logits = self.class_head(pooled)
        predicted_return = self.return_head(pooled).squeeze(-1)
        return logits, predicted_return


def make_loader(
    x: np.ndarray, y_class: np.ndarray, y_return: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y_class, dtype=torch.long),
        torch.tensor(y_return, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    return_scale: float,
    device: torch.device,
    train: bool,
    label: str = "eval",
) -> dict:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_top3 = 0
    total_directional_correct = 0
    total_directional_count = 0
    total_return_error = 0.0
    total_samples = 0
    total_batches = len(loader)
    t0 = time.time()

    for batch_idx, (values, labels, returns) in enumerate(loader, start=1):
        values, labels, returns = values.to(device), labels.to(device), returns.to(device)
        
        with torch.set_grad_enabled(train):
            logits, predicted_return = model(values)
            classification_loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.03)
            regression_loss = nn.functional.huber_loss(predicted_return / return_scale, returns / return_scale)
            loss = classification_loss + 0.25 * regression_loss
            
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = len(labels)
        total_loss += float(loss.item()) * batch_size
        
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        
        top3 = logits.topk(3, dim=1).indices
        total_top3 += int((top3 == labels.unsqueeze(-1)).any(dim=-1).sum().item())
        
        pred_dir = torch.sign(preds.float() - 4.0)
        true_dir = torch.sign(labels.float() - 4.0)
        valid_dir = (true_dir != 0)
        if valid_dir.any():
            total_directional_correct += int((pred_dir[valid_dir] == true_dir[valid_dir]).sum().item())
            total_directional_count += int(valid_dir.sum().item())
            
        total_return_error += float(torch.abs(predicted_return.detach() - returns).sum().item())
        total_samples += batch_size

        if train and (batch_idx % 10 == 0 or batch_idx == total_batches):
            pct = (batch_idx / total_batches) * 100
            cur_acc = (total_correct / total_samples) * 100
            print(
                f"  [{label}] Batch {batch_idx:03d}/{total_batches:03d} ({pct:5.1f}%) | "
                f"Loss: {total_loss/total_samples:.4f} | Acc: {cur_acc:5.2f}% | "
                f"Elapsed: {time.time()-t0:.1f}s",
                flush=True,
            )

    dir_acc = (total_directional_correct / total_directional_count) if total_directional_count > 0 else 0.5
    return {
        "loss": total_loss / total_samples,
        "bin_accuracy": total_correct / total_samples,
        "top3_accuracy": total_top3 / total_samples,
        "directional_accuracy": dir_acc,
        "return_mae": total_return_error / total_samples,
        "time_seconds": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_continuous_45m.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--proto_size", type=int, default=300000, help="Number of most recent rows for fast slice (0 for full)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)

    # Real-time disk-flushed Tee logging
    log_path = Path(__file__).parent.parent / "data" / "reports" / "training_progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee

    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"PyTorch CPU threads: {num_threads}", flush=True)

    device = torch.device("cpu")
    path = data_path(config, "processed", config.get("model_dataset", "gold_1m_continuous_45m_cause_5m_effect.parquet"))
    if not path.exists():
        sys.exit(f"Dataset not found at {path}. Run build_continuous_45m_dataset.py first.")

    print(f"Loading dataset from {path}...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(path).sort_values("target_time").reset_index(drop=True)
    columns = sorted(c for c in df.columns if c.startswith("change_"))

    if args.proto_size > 0 and len(df) > args.proto_size:
        df = df.iloc[-args.proto_size:].reset_index(drop=True)
        print(f"PROTOTYPING SLICE: using most recent {args.proto_size:,} examples.", flush=True)

    x_raw = df[columns].to_numpy(dtype=np.float32)
    y_class = df["label"].to_numpy(dtype=np.int64)
    y_return = df["effect"].to_numpy(dtype=np.float32)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    # Zero leakage z-score normalization computed strictly on training portion
    mean = x_raw[:train_end].mean(axis=0)
    std = x_raw[:train_end].std(axis=0)
    std[std < 1e-8] = 1.0
    x_scaled = (x_raw - mean) / std

    return_scale = float(np.std(y_return[:train_end]) + 1e-6)
    num_target_bins = int(config.get("num_target_bins", 9))
    num_groups = int(config.get("num_groups", 3))
    seq_len = len(columns)

    print(f"Dataset loaded in {time.time()-t0:.1f}s | Total: {n:,} | Train: {train_end:,} | Valid: {valid_end-train_end:,} | Test: {n-valid_end:,}", flush=True)
    print(f"Model: Continuous Hybrid Inception-1D + Transformer | seq_len={seq_len} -> 15 tokens | d_model={args.d_model} | layers={args.layers} | batch={args.batch_size} | epochs={args.epochs}", flush=True)

    model = ContinuousHybridInceptionTransformer(
        sequence_length=seq_len,
        num_target_bins=num_target_bins,
        num_groups=num_groups,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.layers,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_loader = make_loader(x_scaled[:train_end], y_class[:train_end], y_return[:train_end], args.batch_size, True)
    valid_loader = make_loader(x_scaled[train_end:valid_end], y_class[train_end:valid_end], y_return[train_end:valid_end], args.batch_size * 2, False)

    best_state, best_loss, best_epoch = None, float("inf"), 0
    history = []

    print("\n=== TRAINING START ===", flush=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---", flush=True)
        train_metrics = run_epoch(model, train_loader, optimizer, return_scale, device, True, f"Train E{epoch:02d}")
        valid_metrics = run_epoch(model, valid_loader, None, return_scale, device, False, f"Valid E{epoch:02d}")
        scheduler.step()
        
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        print(
            f"==> Epoch {epoch:02d} done in {train_metrics['time_seconds']:.1f}s | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['bin_accuracy']:.2%} | "
            f"Val loss={valid_metrics['loss']:.4f} acc={valid_metrics['bin_accuracy']:.2%} "
            f"top3={valid_metrics['top3_accuracy']:.2%} dir={valid_metrics['directional_accuracy']:.2%}",
            flush=True,
        )

        if valid_metrics["loss"] < best_loss:
            best_loss, best_epoch = valid_metrics["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  *** New best checkpoint at epoch {epoch} (val_loss={best_loss:.4f}) ***", flush=True)

    # Out-of-sample test evaluation
    print("\n=== OUT-OF-SAMPLE TEST EVALUATION ===", flush=True)
    test_loader = make_loader(x_scaled[valid_end:], y_class[valid_end:], y_return[valid_end:], args.batch_size * 2, False)
    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, None, return_scale, device, False, "Test")

    # Save model checkpoint and preprocessing
    output_dir = data_path(config, "models")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "gold_continuous_hybrid_transformer_oos.pt"
    torch.save(
        {
            "model": best_state,
            "sequence_length": seq_len,
            "num_target_bins": num_target_bins,
            "num_groups": num_groups,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "return_scale": return_scale,
        },
        checkpoint_path,
    )
    np.savez(
        output_dir / "gold_continuous_hybrid_transformer_oos_preprocessing.npz",
        mean=mean,
        std=std,
        return_scale=return_scale,
    )

    metrics_path = output_dir / "gold_continuous_hybrid_transformer_metrics.json"
    report = {
        "model_type": "ContinuousHybridInceptionTransformer_45m_3groups",
        "examples_total": n,
        "train_samples": train_end,
        "valid_samples": valid_end - train_end,
        "test_samples": n - valid_end,
        "sequence_length": seq_len,
        "num_target_bins": num_target_bins,
        "best_epoch": best_epoch,
        "validation_history": history,
        "unseen_test_metrics": test_metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n==========================================", flush=True)
    print("      OUT-OF-SAMPLE TEST RESULTS          ", flush=True)
    print("==========================================", flush=True)
    print(f"9-Bin Range Accuracy:  {test_metrics['bin_accuracy']:.2%}", flush=True)
    print(f"Top-3 Range Accuracy:  {test_metrics['top3_accuracy']:.2%}", flush=True)
    print(f"Directional Accuracy:  {test_metrics['directional_accuracy']:.2%}", flush=True)
    print(f"Continuous MAE:        {test_metrics['return_mae']:.4f}", flush=True)
    print(f"Best Epoch:            {best_epoch}", flush=True)
    print(f"Checkpoint saved to:   {checkpoint_path}", flush=True)
    print("==========================================\n", flush=True)


if __name__ == "__main__":
    main()
