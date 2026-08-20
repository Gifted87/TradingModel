"""Train a multi-task Transformer on every valid gold cause/effect example."""

from __future__ import annotations

import argparse
import json
import random
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


class GoldTransformer(nn.Module):
    def __init__(self, sequence_length: int, classes: int, d_model: int = 32, patch_size: int = 4) -> None:
        super().__init__()
        if sequence_length % patch_size:
            raise ValueError("sequence length must be divisible by patch_size")
        self.patch_size = patch_size
        token_count = sequence_length // patch_size
        self.input_projection = nn.Linear(patch_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, token_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.Linear(d_model, 1)
        self.class_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, classes))
        self.return_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = values.reshape(values.size(0), -1, self.patch_size)
        hidden = self.input_projection(patches) + self.position[:, : patches.size(1)]
        hidden = self.norm(self.encoder(hidden))
        weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1)
        return self.class_head(pooled), self.return_head(pooled).squeeze(-1)


def make_loader(x: np.ndarray, y_class: np.ndarray, y_return: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y_class, dtype=torch.long),
        torch.tensor(y_return, dtype=torch.float32),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, return_scale: float, device: torch.device, train: bool) -> dict:
    model.train(train)
    total_loss = total_correct = total = 0
    return_error = 0.0
    for values, labels, returns in loader:
        values, labels, returns = values.to(device), labels.to(device), returns.to(device)
        with torch.set_grad_enabled(train):
            logits, predicted_return = model(values)
            classification_loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.04)
            regression_loss = nn.functional.huber_loss(predicted_return / return_scale, returns / return_scale)
            loss = classification_loss + 0.35 * regression_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        return_error += float(torch.abs(predicted_return.detach() - returns).sum())
        total += len(labels)
    return {"loss": total_loss / total, "accuracy": total_correct / total, "mae": return_error / total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(42)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")
    path = data_path(config, "processed", config.get("model_dataset", "gold_5m_classification.parquet"))
    data = pd.read_parquet(path).sort_values("target_time").reset_index(drop=True)
    columns = sorted((c for c in data.columns if c.startswith("change_")))
    x = data[columns].to_numpy(dtype=np.float32)
    y_class = data["label"].to_numpy(dtype=np.int64)
    y_return = data["effect"].to_numpy(dtype=np.float32)
    n = len(data)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    mean = x[:train_end].mean(axis=0)
    std = x[:train_end].std(axis=0)
    std[std < 1e-8] = 1.0
    x_scaled = (x - mean) / std
    return_scale = float(np.std(y_return[:train_end]) + 1e-6)
    classes = int(y_class.max() + 1)

    model = GoldTransformer(len(columns), classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    train_loader = make_loader(x_scaled[:train_end], y_class[:train_end], y_return[:train_end], 4096, True)
    valid_loader = make_loader(x_scaled[train_end:valid_end], y_class[train_end:valid_end], y_return[train_end:valid_end], 8192, False)
    best_state, best_valid, best_epoch = None, float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, return_scale, device, True)
        valid_metrics = run_epoch(model, valid_loader, None, return_scale, device, False)
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        print(f"epoch {epoch:02d} train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f}")
        if valid_metrics["loss"] < best_valid:
            best_valid, best_epoch = valid_metrics["loss"], epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    test_loader = make_loader(x_scaled[valid_end:], y_class[valid_end:], y_return[valid_end:], 8192, False)
    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, None, return_scale, device, False)

    # Final fit consumes every example, using the selected validation epoch count.
    final_model = GoldTransformer(len(columns), classes).to(device)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=2e-4, weight_decay=1e-4)
    full_loader = make_loader(x_scaled, y_class, y_return, 4096, True)
    for _ in range(best_epoch):
        run_epoch(final_model, full_loader, final_optimizer, return_scale, device, True)

    output = data_path(config, "models")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "sequence_length": len(columns), "classes": classes}, output / "gold_transformer_oos.pt")
    np.savez(output / "gold_transformer_oos_preprocessing.npz", mean=mean, std=std, return_scale=return_scale)
    torch.save({"model": final_model.state_dict(), "sequence_length": len(columns), "classes": classes}, output / "gold_transformer_full.pt")
    np.savez(output / "gold_transformer_preprocessing.npz", mean=mean, std=std, return_scale=return_scale)
    report = {"examples_used_for_final_training": n, "sequence_length": len(columns), "classes": classes, "selected_epoch_count": best_epoch, "validation": history, "unseen_test": test_metrics}
    with (output / "gold_transformer_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"examples_used_for_final_training": n, "selected_epoch_count": best_epoch, "unseen_test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
