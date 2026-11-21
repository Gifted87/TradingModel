"""Train DiscreteGoldTransformer with two speed optimizations:

Output is mirrored in real-time to data/reports/training_progress.log
via a Tee class so the user can watch the file update live.

1. Token Aggregation: 45 bin-tokens -> 15 patch-tokens (3 bars averaged per patch,
   5 patches per temporal group). Attention cost: 45^2 -> 15^2.
2. Fast Prototyping Slice: train/eval on most recent 300k examples only.
   Pass --full to use the complete 1.12M dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time


class Tee:
    """Write to both the original stdout and a log file, flushing after every write."""

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


class DiscreteGoldTransformer(nn.Module):
    """
    Discrete 45-bar input grouped into 15 patch-tokens (3 bars per patch, 5 per group).
    Each bin is embedded, then the 3 embeddings in a patch are averaged -> 15 tokens.
    Three temporal group embeddings mark macro/mid/immediate context.
    """

    def __init__(
        self,
        sequence_length: int = 45,
        num_input_bins: int = 9,
        num_target_bins: int = 9,
        num_groups: int = 3,
        patch_size: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert sequence_length % patch_size == 0, "sequence_length must be divisible by patch_size"
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.num_patches = sequence_length // patch_size   # 45 // 3 = 15
        patches_per_group = self.num_patches // num_groups # 15 // 3 = 5

        # Per-bin embedding (9 possible discrete states)
        self.token_embedding = nn.Embedding(num_input_bins, d_model)

        # Group embedding: 3 groups (macro / mid / immediate)
        self.group_embedding = nn.Embedding(num_groups, d_model)

        # Learnable position embedding over 15 patch positions
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Precompute which group each of the 15 patches belongs to
        group_ids = torch.tensor(
            [p // patches_per_group for p in range(self.num_patches)], dtype=torch.long
        )
        self.register_buffer("group_ids", group_ids)

        self.input_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer encoder over the 15 patch-tokens
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # Attentive pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # Prediction heads
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

    def forward(self, x_bins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_bins: (B, 45) int64
        # 1. Embed 45 tokens -> (B, 45, d_model)
        emb = self.token_embedding(x_bins)

        # 2. Average 3 consecutive embeddings per patch -> (B, 15, d_model)
        B = emb.size(0)
        patches = emb.view(B, self.num_patches, self.patch_size, -1).mean(dim=2)

        # 3. Add group and position context
        h = patches + self.group_embedding(self.group_ids) + self.position_embedding
        h = self.drop(self.input_norm(h))

        # 4. Transformer over 15 patch-tokens
        encoded = self.out_norm(self.encoder(h))

        # 5. Attentive pooling -> single vector
        weights = torch.softmax(self.attn_pool(encoded).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)

        return self.class_head(pooled), self.return_head(pooled).squeeze(-1)


def make_loader(
    x: np.ndarray, y_cls: np.ndarray, y_ret: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y_cls, dtype=torch.long),
        torch.tensor(y_ret, dtype=torch.float32),
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
    tot_loss = tot_correct = tot_top3 = tot_dir_ok = tot_dir = 0
    tot_mae = tot_n = 0
    t0 = time.time()
    n_batches = len(loader)

    for i, (bins, labels, returns) in enumerate(loader, 1):
        bins, labels, returns = bins.to(device), labels.to(device), returns.to(device)

        with torch.set_grad_enabled(train):
            logits, pred_ret = model(bins)
            cls_loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.03)
            reg_loss = nn.functional.huber_loss(pred_ret / return_scale, returns / return_scale)
            loss = cls_loss + 0.20 * reg_loss

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        bs = len(labels)
        tot_loss += float(loss.item()) * bs
        preds = logits.argmax(1)
        tot_correct += int((preds == labels).sum())
        tot_top3 += int((logits.topk(3, 1).indices == labels.unsqueeze(1)).any(1).sum())
        p_dir = torch.sign(preds.float() - 4.0)
        t_dir = torch.sign(labels.float() - 4.0)
        mask = t_dir != 0
        if mask.any():
            tot_dir_ok += int((p_dir[mask] == t_dir[mask]).sum())
            tot_dir += int(mask.sum())
        tot_mae += float(torch.abs(pred_ret.detach() - returns).sum())
        tot_n += bs

        if train and (i % 10 == 0 or i == n_batches):
            print(
                f"  [{label}] Batch {i:03d}/{n_batches:03d} ({100*i/n_batches:5.1f}%) | "
                f"Loss: {tot_loss/tot_n:.4f} | Acc: {100*tot_correct/tot_n:5.2f}% | "
                f"Elapsed: {time.time()-t0:.1f}s",
                flush=True,
            )

    dir_acc = tot_dir_ok / tot_dir if tot_dir > 0 else 0.5
    return {
        "loss": tot_loss / tot_n,
        "bin_accuracy": tot_correct / tot_n,
        "top3_accuracy": tot_top3 / tot_n,
        "directional_accuracy": dir_acc,
        "return_mae": tot_mae / tot_n,
        "time_seconds": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_discrete_45m.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=3,
                        help="Bars per patch token. 45/3=15 patch-tokens fed to Transformer.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--proto_size", type=int, default=300_000,
                        help="Number of most-recent examples to use. Set to 0 for full dataset.")
    parser.add_argument("--full", action="store_true",
                        help="Ignore proto_size and use the complete dataset.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)

    # Mirror all stdout to a log file the user can watch in real-time
    log_path = Path(__file__).parent.parent / "data" / "reports" / "training_progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee  # type: ignore[assignment]

    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"PyTorch CPU threads: {num_threads}", flush=True)

    device = torch.device("cpu")

    path = data_path(
        config, "processed",
        config.get("model_dataset", "gold_1m_discrete_45m_cause_5m_effect.parquet")
    )
    if not path.exists():
        sys.exit(f"Dataset not found: {path}. Run build_discrete_cause_effect_dataset.py first.")

    print(f"Loading dataset from {path}...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(path).sort_values("target_time").reset_index(drop=True)
    bin_cols = sorted(c for c in df.columns if c.startswith("bin_"))

    # Fast prototyping slice: take only the most recent N rows (time-ordered)
    if not args.full and args.proto_size > 0 and len(df) > args.proto_size:
        df = df.iloc[-args.proto_size:].reset_index(drop=True)
        print(f"PROTOTYPING SLICE: using most recent {args.proto_size:,} examples.", flush=True)

    x = df[bin_cols].to_numpy(dtype=np.int64)
    y_cls = df["label"].to_numpy(dtype=np.int64)
    y_ret = df["effect"].to_numpy(dtype=np.float32)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    ret_scale = float(np.std(y_ret[:train_end]) + 1e-6)

    num_input_bins = int(config.get("num_input_bins", 9))
    num_target_bins = int(config.get("num_target_bins", 9))
    num_groups = int(config.get("num_groups", 3))
    seq_len = len(bin_cols)  # 45
    num_patches = seq_len // args.patch_size  # 15

    print(
        f"Dataset loaded in {time.time()-t0:.1f}s | "
        f"Total: {n:,} | Train: {train_end:,} | Valid: {valid_end-train_end:,} | Test: {n-valid_end:,}",
        flush=True,
    )
    print(
        f"Model: {seq_len} bins -> {num_patches} patch-tokens (patch_size={args.patch_size}) | "
        f"d_model={args.d_model} layers={args.layers} nhead={args.nhead} | "
        f"batch={args.batch_size} epochs={args.epochs} lr={args.lr}",
        flush=True,
    )

    model = DiscreteGoldTransformer(
        sequence_length=seq_len,
        num_input_bins=num_input_bins,
        num_target_bins=num_target_bins,
        num_groups=num_groups,
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_loader = make_loader(x[:train_end], y_cls[:train_end], y_ret[:train_end], args.batch_size, True)
    valid_loader = make_loader(
        x[train_end:valid_end], y_cls[train_end:valid_end], y_ret[train_end:valid_end],
        args.batch_size * 2, False
    )

    best_state, best_loss, best_epoch = None, float("inf"), 0
    history = []

    print("\n=== TRAINING START ===", flush=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---", flush=True)
        tr = run_epoch(model, train_loader, optimizer, ret_scale, device, True, f"Train E{epoch:02d}")
        va = run_epoch(model, valid_loader, None, ret_scale, device, False, f"Valid E{epoch:02d}")
        scheduler.step()
        history.append({"epoch": epoch, "train": tr, "valid": va})

        print(
            f"==> E{epoch:02d} done in {tr['time_seconds']:.1f}s | "
            f"Train loss={tr['loss']:.4f} acc={tr['bin_accuracy']:.2%} | "
            f"Val loss={va['loss']:.4f} acc={va['bin_accuracy']:.2%} "
            f"top3={va['top3_accuracy']:.2%} dir={va['directional_accuracy']:.2%}",
            flush=True,
        )

        if va["loss"] < best_loss:
            best_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  *** New best at epoch {epoch} (val_loss={best_loss:.4f}) ***", flush=True)

    # Out-of-sample test
    print("\n=== OUT-OF-SAMPLE TEST ===", flush=True)
    test_loader = make_loader(
        x[valid_end:], y_cls[valid_end:], y_ret[valid_end:], args.batch_size * 2, False
    )
    model.load_state_dict(best_state)
    test = run_epoch(model, test_loader, None, ret_scale, device, False, "Test")

    # Save
    out_dir = data_path(config, "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "gold_discrete_transformer_oos.pt"
    torch.save(
        {
            "model": best_state,
            "sequence_length": seq_len,
            "num_input_bins": num_input_bins,
            "num_target_bins": num_target_bins,
            "num_groups": num_groups,
            "patch_size": args.patch_size,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "return_scale": ret_scale,
        },
        ckpt_path,
    )

    metrics_path = out_dir / "gold_discrete_transformer_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "DiscreteGoldTransformer_45m_3groups_15patches",
                "proto_slice": not args.full,
                "examples_used": n,
                "train_samples": train_end,
                "valid_samples": valid_end - train_end,
                "test_samples": n - valid_end,
                "sequence_length": seq_len,
                "patch_size": args.patch_size,
                "num_patches": num_patches,
                "num_input_bins": num_input_bins,
                "num_target_bins": num_target_bins,
                "best_epoch": best_epoch,
                "validation_history": history,
                "unseen_test_metrics": test,
            },
            f,
            indent=2,
        )

    print("\n==========================================", flush=True)
    print("     OUT-OF-SAMPLE TEST RESULTS           ", flush=True)
    print("==========================================", flush=True)
    print(f"9-Bin Accuracy:      {test['bin_accuracy']:.2%}", flush=True)
    print(f"Top-3 Accuracy:      {test['top3_accuracy']:.2%}", flush=True)
    print(f"Directional Acc:     {test['directional_accuracy']:.2%}", flush=True)
    print(f"Continuous MAE:      {test['return_mae']:.4f}", flush=True)
    print(f"Best epoch:          {best_epoch}", flush=True)
    print(f"Checkpoint:          {ckpt_path}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
