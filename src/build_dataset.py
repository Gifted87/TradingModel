"""Build 5-minute effects, cause windows, and bounded range labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import data_path, load_config


def make_bins(config: dict, target_values: np.ndarray) -> np.ndarray:
    if "label_quantiles" in config:
        edges = np.quantile(target_values, np.asarray(config["label_quantiles"], dtype=float))
        edges = np.unique(edges)
        if len(edges) < 3:
            raise ValueError("Gold effects do not contain enough unique values for quantile labels")
        return edges
    if "bin_edges" in config:
        edges = np.asarray(config["bin_edges"], dtype=float)
        if len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges must contain strictly increasing values")
        return edges
    minimum = float(config["min_change"])
    maximum = float(config["max_change"])
    width = float(config["bin_width"])
    edges = np.arange(minimum, maximum + width, width)
    if edges[-1] < maximum:
        edges = np.append(edges, maximum)
    edges[0] = minimum
    edges[-1] = maximum
    return edges


def build_dataset(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.set_index("timestamp")
    frame["change_1m"] = frame["Close"].diff()

    # Resample from the same 1-minute source so target and cause use one clock.
    bars = frame[["Open", "High", "Low", "Close", "Volume"]].resample("5min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    bars["effect"] = bars["Close"].diff()

    lookback = int(config["lookback_minutes"])
    gap = int(config["gap_minutes"])
    target_values = bars["effect"].dropna().to_numpy(dtype=float)
    edges = make_bins(config, target_values)
    minimum = float(edges[0])
    maximum = float(edges[-1])
    rows = []

    for target_time, target in bars["effect"].dropna().items():
        cause_end = target_time - pd.Timedelta(minutes=gap)
        cause_start = cause_end - pd.Timedelta(minutes=lookback) + pd.Timedelta(minutes=1)
        window = frame.loc[cause_start:cause_end, "change_1m"]
        if len(window) != lookback or window.isna().any():
            continue
        clipped_target = float(np.clip(target, minimum, maximum))
        label = int(np.clip(np.digitize(clipped_target, edges[1:-1], right=False), 0, len(edges) - 2))
        row = {"target_time": target_time, "effect": float(target), "label": label}
        row.update({f"change_{i:04d}": float(value) for i, value in enumerate(window, 1)})
        rows.append(row)

    return pd.DataFrame(rows), edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_path = data_path(config, "raw", f"{config['symbol'].replace('=', '_')}_1m.parquet")
    if not raw_path.exists():
        raise FileNotFoundError(f"Download raw data first: {raw_path}")
    raw = pd.read_parquet(raw_path)
    dataset, edges = build_dataset(raw, config)
    output = data_path(config, "processed", "gold_5m_classification.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    np.save(output.with_suffix(".bin_edges.npy"), edges)
    print(f"Saved {len(dataset):,} examples to {output}")
    print(f"Labels: {dataset['label'].value_counts().sort_index().to_dict()}")
    print(f"Bin edges: {edges.tolist()}")


if __name__ == "__main__":
    main()
