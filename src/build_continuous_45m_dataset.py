"""Build the 45-minute continuous-input -> 5-minute 9-bin effect dataset with strict zero lookahead leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import data_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_continuous_45m.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    raw_path = Path(config["raw_1m_path"])
    raw = pd.read_parquet(raw_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    minute_close = raw["Close"].resample("1min").last()
    minute_change = minute_close.diff()
    
    # 5-minute target effects: close(T) - close(T-5m)
    five_close = minute_close.resample("5min", label="right", closed="right").last()
    effects = five_close.diff().dropna()
    
    lookback = int(config.get("lookback_minutes", 45))
    gap = int(config.get("gap_minutes", 5))

    minute_values = minute_change.to_numpy(dtype=np.float32)
    minute_index = minute_change.index
    
    windows = np.lib.stride_tricks.sliding_window_view(minute_values, lookback)
    end_times = effects.index - pd.Timedelta(minutes=gap)
    end_positions = minute_index.get_indexer(end_times)
    
    start_positions = end_positions - lookback + 1
    valid = (start_positions >= 0) & (start_positions < len(windows))
    valid_positions = np.where(valid)[0]
    valid_positions = valid_positions[~np.isnan(effects.to_numpy(dtype=float)[valid_positions])]
    valid_positions = valid_positions[~np.isnan(windows[start_positions[valid_positions]]).any(axis=1)]
    
    targets = effects.to_numpy(dtype=np.float32)[valid_positions]
    cause_windows = windows[start_positions[valid_positions]]
    
    # Strictly zero leakage: Compute target quantiles exclusively on the first 70% (training split)
    n_samples = len(valid_positions)
    train_end = int(n_samples * 0.70)
    
    quantiles = np.asarray(config.get("label_quantiles", [0.0, 0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95, 1.0]), dtype=float)
    target_edges = np.unique(np.quantile(targets[:train_end], quantiles))
    target_labels = np.clip(np.digitize(targets, target_edges[1:-1], right=False), 0, len(target_edges) - 2).astype(np.int64)
    
    # Build continuous dataset
    columns = [f"change_{i:04d}" for i in range(1, lookback + 1)]
    dataset = pd.DataFrame(cause_windows, columns=columns)
    dataset.insert(0, "label", target_labels)
    dataset.insert(0, "effect", targets)
    dataset.insert(0, "target_time", effects.index[valid_positions])
    
    output = data_path(config, "processed", config.get("model_dataset", "gold_1m_continuous_45m_cause_5m_effect.parquet"))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    
    np.save(output.with_suffix(".bin_edges.npy"), target_edges)
    
    print(f"1-minute bars: {len(raw):,}")
    print(f"5-minute target effects: {len(effects):,}")
    print(f"Valid continuous cause/effect examples: {len(dataset):,}")
    print(f"Cause window: {lookback} continuous one-minute changes ending {gap} minutes prior to target")
    print(f"Training split target bin edges: {target_edges.tolist()}")
    print(f"Class distribution:\n{dataset['label'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
