"""Build Continuous Multi-Scale Feature Dataset for 45-Minute Lookback Sequences (Plan 1 - High Performance).

Institutional Data Engineering:
1. Resamples raw tick/bar stream to a strictly regular 1-minute OHLCV grid.
2. Strictly causal feature extraction via ContinuousFeatureExtractor.
3. Fast vectorized sliding window assembly with zero-leakage training-only quantile bin edges.
4. Immediate line-by-line flushed logging to console and persistent log file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import data_path, load_config
from continuous_features import ContinuousFeatureExtractor, compute_group_features


class Tee:
    """Write simultaneously to stdout and a persistent log file with immediate flushing."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = log_path.open("w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold_continuous_45m.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    log_path = data_path(config, "reports", "build_continuous_multiscale_dataset.log")
    tee = Tee(log_path)
    sys.stdout = tee
    
    t0 = time.time()
    raw_path = Path(config["raw_1m_path"])
    print(f"[{time.strftime('%H:%M:%S')}] Reading raw 1-minute Parquet data from {raw_path}...")
    raw = pd.read_parquet(raw_path)
    
    extractor = ContinuousFeatureExtractor(
        ewma_alpha=float(config.get("ewma_alpha", 0.06)),
        lookback_bars=int(config.get("lookback_minutes", 45)),
        num_rbf_bins=int(config.get("num_rbf_bins", 9)),
    )
    
    # 1. Prepare unbroken 1-minute regular grid
    print(f"[{time.strftime('%H:%M:%S')}] Resampling and aligning to regular 1-minute continuous grid...")
    regular_df = extractor.prepare_regular_grid(raw)
    print(f"[{time.strftime('%H:%M:%S')}] Regular 1-minute grid: {len(regular_df):,} bars from {regular_df.index.min()} to {regular_df.index.max()}")
    
    # 2. Extract continuous multi-scale features
    print(f"[{time.strftime('%H:%M:%S')}] Computing continuous multi-scale features, volatility normalization, and session harmonics...")
    features_df = extractor.extract_bar_features(regular_df)
    feature_cols = extractor.get_feature_names()
    print(f"[{time.strftime('%H:%M:%S')}] Extracted {len(feature_cols)} continuous features per bar.")
    
    # 3. Resample 5-minute target effects: close(T) - close(T-5m)
    minute_close = regular_df["Close"]
    five_close = minute_close.resample("5min", label="right", closed="right").last().dropna()
    effects = five_close.diff().dropna()
    
    lookback = int(config.get("lookback_minutes", 45))
    gap = int(config.get("gap_minutes", 5))

    # Align 45-minute windows with effects
    feature_matrix = features_df[feature_cols].to_numpy(dtype=np.float32)
    time_index = features_df.index
    
    end_times = effects.index - pd.Timedelta(minutes=gap)
    end_positions = time_index.get_indexer(end_times)
    
    start_positions = end_positions - lookback + 1
    valid = (start_positions >= 0) & (end_positions < len(feature_matrix)) & (end_positions != -1)
    valid_positions = np.where(valid)[0]
    
    # Filter valid positions without NaNs in targets
    valid_targets = effects.to_numpy(dtype=float)[valid_positions]
    valid_mask = ~np.isnan(valid_targets)
    valid_positions = valid_positions[valid_mask]
    
    sample_starts = start_positions[valid_positions]
    sample_ends = end_positions[valid_positions]
    
    print(f"[{time.strftime('%H:%M:%S')}] Candidate windows before continuity filtering: {len(valid_positions):,}")
    
    # 4. Fast vectorized time-continuity check
    start_timestamps = time_index[sample_starts]
    end_timestamps = time_index[sample_ends]
    expected_duration = pd.Timedelta(minutes=lookback - 1)
    
    is_contiguous = (end_timestamps - start_timestamps) == expected_duration
    contiguous_indices = np.where(is_contiguous)[0]
    
    sample_starts = sample_starts[contiguous_indices]
    sample_ends = sample_ends[contiguous_indices]
    valid_final_positions = valid_positions[contiguous_indices]
    
    n_clean = len(sample_starts)
    print(f"[{time.strftime('%H:%M:%S')}] Valid contiguous cause/effect examples: {n_clean:,}")
    
    # Extract 3D cause windows: (N, 45, D)
    print(f"[{time.strftime('%H:%M:%S')}] Assembling 3D continuous sequence tensors...")
    cause_windows = np.zeros((n_clean, lookback, len(feature_cols)), dtype=np.float32)
    group_features = np.zeros((n_clean, 12), dtype=np.float32)
    
    for i in range(n_clean):
        s = sample_starts[i]
        e = sample_ends[i]
        w = feature_matrix[s : e + 1]
        cause_windows[i] = w
        group_features[i] = compute_group_features(w, group_size=15)
        
    targets = effects.to_numpy(dtype=np.float32)[valid_final_positions]
    target_times = effects.index[valid_final_positions]
    
    # 5. Strictly Zero Lookahead Leakage: Guarantee 9 symmetric bins with neutral center
    train_end = int(n_clean * 0.70)
    train_tgts = targets[:train_end]
    
    # 4 negative quantiles, 4 positive quantiles
    neg_tgts = train_tgts[train_tgts < -0.01]
    pos_tgts = train_tgts[train_tgts > 0.01]
    
    neg_edges = np.quantile(neg_tgts, [0.0, 0.15, 0.40, 0.75])
    pos_edges = np.quantile(pos_tgts, [0.25, 0.60, 0.85, 1.0])
    
    # Exactly 10 edges -> 9 bins (0..8) with Bin 4 as [-0.03, +0.03] neutral zone
    target_edges = np.array([
        float(train_tgts.min() - 0.01),
        float(neg_edges[1]),
        float(neg_edges[2]),
        float(neg_edges[3]),
        -0.03,
        0.03,
        float(pos_edges[0]),
        float(pos_edges[1]),
        float(pos_edges[2]),
        float(train_tgts.max() + 0.01),
    ], dtype=np.float32)
    
    # Ensure strictly strictly monotonic edges
    for idx in range(1, len(target_edges)):
        if target_edges[idx] <= target_edges[idx - 1]:
            target_edges[idx] = target_edges[idx - 1] + 1e-4
            
    num_bins = len(target_edges) - 1
    assert num_bins == 9, f"Expected 9 target bins, got {num_bins}"
    
    target_labels = np.clip(
        np.digitize(targets, target_edges[1:-1], right=False), 0, num_bins - 1
    ).astype(np.int64)
    
    # 6. Save Processed Artifacts
    output_dir = data_path(config, "processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = config.get("model_dataset", "gold_multiscale_45m_dataset.npz")
    output_path = output_dir / dataset_name
    
    print(f"[{time.strftime('%H:%M:%S')}] Saving compressed dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        cause_windows=cause_windows,
        group_features=group_features,
        targets=targets,
        labels=target_labels,
        feature_names=np.array(feature_cols),
        target_edges=target_edges,
    )
    
    np.save(output_path.with_suffix(".bin_edges.npy"), target_edges)
    
    manifest_path = output_dir / "gold_multiscale_45m_manifest.parquet"
    manifest = pd.DataFrame({
        "target_time": target_times,
        "effect": targets,
        "label": target_labels,
    })
    manifest.to_parquet(manifest_path, index=False)
    
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] Dataset generation complete in {elapsed:.1f}s")
    print(f"  - Total Examples: {n_clean:,}")
    print(f"  - Input Sequence Shape: {cause_windows.shape}")
    print(f"  - Group Features Shape: {group_features.shape}")
    print(f"  - Training Split Size (70%): {train_end:,} examples")
    print(f"  - Zero-Leakage Target Bin Edges: {np.round(target_edges, 4).tolist()}")
    print(f"  - Log written to: {log_path}")
    
    tee.close()


if __name__ == "__main__":
    main()
