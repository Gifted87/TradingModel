"""Diagnostic backtest using causal 1-minute windows built directly from raw data."""

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
        return {"trades": 0, "total_pnl": 0.0, "average_pnl": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
    curve = np.cumsum(traded)
    drawdown = np.maximum.accumulate(curve) - curve
    return {
        "trades": int(len(traded)),
        "total_pnl": float(traded.sum()),
        "average_pnl": float(traded.mean()),
        "win_rate": float((traded > 0).mean()),
        "max_drawdown": float(drawdown.max()),
    }


def build_causal_arrays(config: dict) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    raw = pd.read_parquet(config["raw_1m_path"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    minute_close = raw["Close"].resample("1min").last()
    minute_change = minute_close.diff()
    five_close = minute_close.resample("5min").last()
    effects = five_close.diff().dropna()
    lookback = int(config["lookback_minutes"])
    gap = int(config["gap_minutes"])
    values = minute_change.to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback)
    end_times = effects.index - pd.Timedelta(minutes=gap)
    end_positions = minute_change.index.get_indexer(end_times)
    start_positions = end_positions - lookback + 1
    valid = (start_positions >= 0) & (start_positions < len(windows))
    valid_positions = np.where(valid)[0]
    valid_positions = valid_positions[~np.isnan(effects.to_numpy(dtype=float)[valid_positions])]
    valid_positions = valid_positions[~np.isnan(windows[start_positions[valid_positions]]).any(axis=1)]
    cause_windows = windows[start_positions[valid_positions]]
    targets = effects.to_numpy(dtype=np.float32)[valid_positions]
    target_times = effects.index[valid_positions]
    return target_times, cause_windows, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    parser.add_argument("--cost", type=float, default=0.50)
    args = parser.parse_args()
    config = load_config(args.config)
    target_times, x, effects = build_causal_arrays(config)
    checkpoint = torch.load(data_path(config, "models", "gold_transformer_oos.pt"), map_location="cpu", weights_only=True)
    preprocessing = np.load(data_path(config, "models", "gold_transformer_oos_preprocessing.npz"))
    model = GoldTransformer(checkpoint["sequence_length"], checkpoint["classes"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    x = (x - preprocessing["mean"]) / preprocessing["std"]
    split = int(len(x) * 0.85)
    with torch.no_grad():
        logits, _ = model(torch.tensor(x[split:], dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()
    edges = np.load(data_path(config, "processed", "gold_1m_cause_5m_effect.bin_edges.npy"))
    labels = np.clip(np.digitize(effects, edges[1:-1], right=False), 0, checkpoint["classes"] - 1)
    class_returns = pd.DataFrame({"label": labels[:split], "effect": effects[:split]}).groupby("label").effect.mean()
    centers = np.array([class_returns.get(i, effects[:split].mean()) for i in range(checkpoint["classes"])])
    expected = probabilities @ centers
    actual = effects[split:].astype(float)
    results = {}
    for minimum_edge in (0.0, 0.25, 0.50, 1.0, 1.5):
        signals = np.where(expected > minimum_edge, 1, np.where(expected < -minimum_edge, -1, 0))
        pnl = signals * actual - np.where(signals != 0, args.cost, 0.0)
        result = metrics(pnl, signals)
        traded_actual = actual[signals != 0]
        result["direction_accuracy_before_cost"] = float(np.mean(signals[signals != 0] * traded_actual > 0)) if len(traded_actual) else 0.0
        results[f"edge_{minimum_edge:g}"] = result
    report = {
        "input": "raw Dukascopy 1-minute data with backward causal windows",
        "window_rule": "windows[end_position - lookback + 1] ending at target_time - gap",
        "checkpoint_warning": "Transformer checkpoint was trained on the prior leaked-window dataset; retrain is required for a clean result.",
        "examples": int(len(x)),
        "test_start": str(target_times[split]),
        "test_end": str(target_times[-1]),
        "round_trip_cost": args.cost,
        "results": results,
    }
    output = data_path(config, "reports")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gold_corrected_cause_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
