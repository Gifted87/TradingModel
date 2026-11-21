"""Build the 45-minute discrete binned 1m cause -> 5m effect training set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import data_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_discrete_45m.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    raw_path = Path(config["raw_1m_path"])
    raw = pd.read_parquet(raw_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    minute_close = raw["Close"].resample("1min").last()
    minute_change = minute_close.diff()
    
    # 5-minute target effects
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
    cause_windows_continuous = windows[start_positions[valid_positions]]
    
    # Compute Quantiles on the initial 70% (training split) to prevent data leakage
    n_samples = len(valid_positions)
    train_end = int(n_samples * 0.70)
    
    # 1. Target bin edges
    target_quantiles = np.asarray(config.get("label_quantiles", [0.0, 0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95, 1.0]), dtype=float)
    target_edges = np.unique(np.quantile(targets[:train_end], target_quantiles))
    target_labels = np.clip(np.digitize(targets, target_edges[1:-1], right=False), 0, len(target_edges) - 2).astype(np.int64)
    
    # 2. Input 1m change bin edges
    input_quantiles = np.asarray(config.get("input_quantiles", [0.0, 0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95, 1.0]), dtype=float)
    train_input_values = cause_windows_continuous[:train_end].flatten()
    input_edges = np.unique(np.quantile(train_input_values, input_quantiles))
    
    # Discretize input continuous windows into integer bin tokens (0 to num_bins - 1)
    num_input_bins = len(input_edges) - 1
    cause_windows_discrete = np.clip(
        np.digitize(cause_windows_continuous, input_edges[1:-1], right=False),
        0,
        num_input_bins - 1
    ).astype(np.int64)
    
    # Build dataframe
    bin_cols = [f"bin_{i:04d}" for i in range(1, lookback + 1)]
    dataset = pd.DataFrame(cause_windows_discrete, columns=bin_cols)
    dataset.insert(0, "label", target_labels)
    dataset.insert(0, "effect", targets)
    dataset.insert(0, "target_time", effects.index[valid_positions])
    
    output = data_path(config, "processed", config.get("model_dataset", "gold_1m_discrete_45m_cause_5m_effect.parquet"))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    
    # Save both edge sets
    np.savez(
        output.with_suffix(".bin_edges.npz"),
        input_edges=input_edges,
        target_edges=target_edges,
        input_quantiles=input_quantiles,
        target_quantiles=target_quantiles
    )
    
    print(f"1-minute bars: {len(raw):,}")
    print(f"5-minute effects: {len(effects):,}")
    print(f"Valid discrete cause/effect examples: {len(dataset):,}")
    print(f"Cause window: {lookback} discrete tokens grouped into 3 groups of {lookback // 3} bars")
    print(f"Input bin edges (1m changes): {input_edges.tolist()}")
    print(f"Target bin edges (5m effects): {target_edges.tolist()}")
    print(f"Target distribution:\n{dataset['label'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
