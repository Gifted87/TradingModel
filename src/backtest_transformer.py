"""Chronological, cost-aware backtest for the saved out-of-sample Transformer."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from common import data_path, load_config
from train_advanced import GoldTransformer


def metrics(pnl: np.ndarray, signals: np.ndarray) -> dict:
    traded = pnl[signals != 0]
    if not len(traded):
        return {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
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
    parser.add_argument("--config", default="configs/gold.yaml")
    parser.add_argument("--cost", type=float, default=0.50, help="Round-trip spread plus slippage in gold price units")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = pd.read_parquet(data_path(config, "processed", config["model_dataset"])).sort_values("target_time").reset_index(drop=True)
    columns = sorted(c for c in dataset.columns if c.startswith("change_"))
    checkpoint = torch.load(data_path(config, "models", "gold_transformer_oos.pt"), map_location="cpu", weights_only=True)
    preprocessing = np.load(data_path(config, "models", "gold_transformer_oos_preprocessing.npz"))
    x = dataset[columns].to_numpy(dtype=np.float32)
    x = (x - preprocessing["mean"]) / preprocessing["std"]
    split = int(len(dataset) * 0.85)
    model = GoldTransformer(checkpoint["sequence_length"], checkpoint["classes"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(x[split:], dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()

    # Quantile bands can be very wide at the extremes. Use training-period
    # conditional means instead of geometric midpoints to calibrate returns.
    class_returns = dataset.iloc[:split].groupby("label")["effect"].mean()
    centers = np.array([class_returns.get(i, dataset.iloc[:split]["effect"].mean()) for i in range(checkpoint["classes"])])
    expected = probabilities @ centers
    confidence = probabilities.max(axis=1)
    actual = dataset.iloc[split:]["effect"].to_numpy(dtype=float)
    results = {}
    for minimum_edge in (0.0, 0.25, 0.50, 1.0, 1.5):
        signals = np.where(expected > minimum_edge, 1, np.where(expected < -minimum_edge, -1, 0))
        pnl = signals * actual - np.where(signals != 0, args.cost, 0.0)
        result = metrics(pnl, signals)
        traded_actual = actual[signals != 0]
        result["direction_accuracy_before_cost"] = float(np.mean(signals[signals != 0] * traded_actual > 0)) if len(traded_actual) else 0.0
        result["mean_actual_move"] = float(traded_actual.mean()) if len(traded_actual) else 0.0
        results[f"edge_{minimum_edge:g}"] = result
    report = {
        "test_examples": int(len(actual)),
        "round_trip_cost": args.cost,
        "note": "Out-of-sample checkpoint trained only before the final 15 percent of this dataset.",
        "expected_return_mean": float(expected.mean()),
        "expected_return_std": float(expected.std()),
        "results": results,
    }
    output = data_path(config, "reports")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gold_transformer_backtest.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
