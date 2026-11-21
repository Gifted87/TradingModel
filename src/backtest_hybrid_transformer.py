"""Chronological, cost-aware backtest for the Hybrid Inception-1D + Transformer model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import data_path, load_config
from train_hybrid_inception_transformer import HybridInceptionTransformer


def compute_metrics(pnl: np.ndarray, signals: np.ndarray) -> dict:
    traded = pnl[signals != 0]
    if not len(traded):
        return {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0}
    curve = np.cumsum(traded)
    drawdown = np.maximum.accumulate(curve) - curve
    return {
        "trades": int(len(traded)),
        "total_pnl": float(traded.sum()),
        "average_pnl": float(traded.mean()),
        "win_rate": float((traded > 0).mean()),
        "max_drawdown": float(drawdown.max()),
        "profit_factor": float(traded[traded > 0].sum() / max(1e-9, -traded[traded < 0].sum())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_discrete_45m.yaml")
    parser.add_argument("--cost", type=float, default=0.50, help="Round-trip spread plus slippage in gold price units")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = data_path(config, "processed", config.get("model_dataset", "gold_1m_discrete_45m_cause_5m_effect.parquet"))
    dataset = pd.read_parquet(dataset_path).sort_values("target_time").reset_index(drop=True)
    bin_cols = sorted(c for c in dataset.columns if c.startswith("bin_"))

    model_path = data_path(config, "models", "gold_hybrid_transformer_oos.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    split = int(len(dataset) * 0.85)
    test_data = dataset.iloc[split:].reset_index(drop=True)
    x_test = test_data[bin_cols].to_numpy(dtype=np.int64)
    actual_effects = test_data["effect"].to_numpy(dtype=float)

    model = HybridInceptionTransformer(
        sequence_length=checkpoint["sequence_length"],
        num_input_bins=checkpoint["num_input_bins"],
        num_target_bins=checkpoint["num_target_bins"],
        num_groups=checkpoint.get("num_groups", 3),
        d_model=checkpoint.get("d_model", 64),
        nhead=checkpoint.get("nhead", 4),
        num_transformer_layers=checkpoint.get("layers", 2),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Evaluating {len(test_data):,} unseen test samples...", flush=True)
    with torch.no_grad():
        test_tensor = torch.tensor(x_test, dtype=torch.long)
        logits, predicted_continuous = model(test_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()

    # Training-period conditional class returns
    train_effects = dataset.iloc[:split]
    class_means = train_effects.groupby("label")["effect"].mean()
    centers = np.array([class_means.get(i, train_effects["effect"].mean()) for i in range(checkpoint["num_target_bins"])])

    # Expected effect distribution projection
    expected_effects = probabilities @ centers

    results = {}
    for min_edge in (0.0, 0.25, 0.50, 1.0, 1.5):
        signals = np.where(expected_effects > min_edge, 1, np.where(expected_effects < -min_edge, -1, 0))
        pnl = signals * actual_effects - np.where(signals != 0, args.cost, 0.0)
        res = compute_metrics(pnl, signals)

        traded_actual = actual_effects[signals != 0]
        if len(traded_actual) > 0:
            res["direction_accuracy_before_cost"] = float(np.mean(signals[signals != 0] * traded_actual > 0))
            res["mean_actual_move"] = float(traded_actual.mean())
        else:
            res["direction_accuracy_before_cost"] = 0.0
            res["mean_actual_move"] = 0.0

        results[f"edge_{min_edge:g}"] = res

    report = {
        "model_type": "HybridInceptionTransformer_45m_3groups",
        "test_examples": int(len(test_data)),
        "round_trip_cost": args.cost,
        "expected_return_mean": float(expected_effects.mean()),
        "expected_return_std": float(expected_effects.std()),
        "results": results,
    }

    output_dir = data_path(config, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "gold_hybrid_transformer_backtest.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nBacktest report saved to {report_file}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
