"""Build the original 1-minute-cause -> 5-minute-effect training set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import data_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_path = Path(config["raw_1m_path"])
    raw = pd.read_parquet(raw_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    minute_close = raw["Close"].resample("1min").last()
    minute_change = minute_close.diff()
    # Label each five-minute candle by its closing timestamp. This makes an
    # effect at 01:05 equal close(01:05) - close(01:00).
    five_close = minute_close.resample("5min", label="right", closed="right").last()
    effects = five_close.diff().dropna()
    lookback = int(config["lookback_minutes"])
    gap = int(config["gap_minutes"])

    minute_values = minute_change.to_numpy(dtype=np.float32)
    minute_index = minute_change.index
    windows = np.lib.stride_tricks.sliding_window_view(minute_values, lookback)
    end_times = effects.index - pd.Timedelta(minutes=gap)
    end_positions = minute_index.get_indexer(end_times)
    # sliding_window_view[i] starts at i, so convert the desired end index
    # into the corresponding backward-window start index.
    start_positions = end_positions - lookback + 1
    valid = (start_positions >= 0) & (start_positions < len(windows))
    valid_positions = np.where(valid)[0]
    valid_positions = valid_positions[~np.isnan(effects.to_numpy(dtype=float)[valid_positions])]
    valid_positions = valid_positions[~np.isnan(windows[start_positions[valid_positions]]).any(axis=1)]
    targets = effects.to_numpy(dtype=np.float32)[valid_positions]
    cause_windows = windows[start_positions[valid_positions]]

    quantiles = np.asarray(config["label_quantiles"], dtype=float)
    edges = np.unique(np.quantile(targets, quantiles))
    labels = np.clip(np.digitize(targets, edges[1:-1], right=False), 0, len(edges) - 2).astype(np.int64)
    dataset = pd.DataFrame(cause_windows, columns=[f"change_{i:04d}" for i in range(1, lookback + 1)])
    dataset.insert(0, "label", labels)
    dataset.insert(0, "effect", targets)
    dataset.insert(0, "target_time", effects.index[valid_positions])
    output = data_path(config, "processed", "gold_1m_cause_5m_effect.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    np.save(output.with_suffix(".bin_edges.npy"), edges)
    print(f"1-minute bars: {len(raw):,}")
    print(f"5-minute effects: {len(effects):,}")
    print(f"Valid cause/effect examples: {len(dataset):,}")
    print(f"Cause window: {lookback} one-minute changes ending {gap} minutes before target")
    print(f"Labels: {dataset['label'].value_counts().sort_index().to_dict()}")
    print(f"Bin edges: {edges.tolist()}")


if __name__ == "__main__":
    main()
