"""Build a complete 5-minute sequence dataset from the downloaded 1-minute bars."""

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
    raw_path = Path(config.get("raw_5m_path", data_path(config, "raw", f"{config['symbol'].replace('=', '_')}_1m.parquet")))
    raw = pd.read_parquet(raw_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").set_index("timestamp")
    bars = raw[["Open", "High", "Low", "Close", "Volume"]].resample("5min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])
    bars["effect"] = bars["Close"].diff()
    bars = bars.dropna(subset=["effect"]).copy()

    lookback = int(config.get("lookback_bars", 48))
    effects = bars["effect"].to_numpy(dtype=float)
    quantiles = np.asarray(config["label_quantiles"], dtype=float)
    edges = np.unique(np.quantile(effects, quantiles))
    changes = bars["effect"].to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(changes, lookback)[:-1]
    targets = changes[lookback:]
    labels = np.clip(np.digitize(targets, edges[1:-1], right=False), 0, len(edges) - 2).astype(np.int64)
    dataset = pd.DataFrame(windows, columns=[f"change_{i:04d}" for i in range(1, lookback + 1)])
    dataset.insert(0, "label", labels)
    dataset.insert(0, "effect", targets)
    dataset.insert(0, "target_time", bars.index[lookback:])

    output = data_path(config, "processed", "gold_5m_sequence.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    np.save(output.with_suffix(".bin_edges.npy"), edges)
    print(f"Observed 5-minute bars: {len(bars):,}")
    print(f"Training examples using all bars after context: {len(dataset):,}")
    print(f"Labels: {dataset['label'].value_counts().sort_index().to_dict()}")
    print(f"Bin edges: {edges.tolist()}")


if __name__ == "__main__":
    main()
