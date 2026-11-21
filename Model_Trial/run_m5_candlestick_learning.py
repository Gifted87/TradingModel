#!/usr/bin/env python3
"""
M5 Candlestick Microstructure Probability Learner.

Deconstructs 5-minute candles into five 1-minute steps (M1, M2, M3, M4, M5).
- Input: Direction of first 3 minutes (M1, M2, M3) from {U, D, P} (27 possible prefixes).
- Target: Direction of final 2 minutes (M4, M5) from {UU, UD, DU, DD} (4 directional outcomes).
- Model: Neural Softmax Classifier trained via online error correction (Cross-Entropy Loss).
"""

import argparse
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim


class MicrostructureModel(nn.Module):
    """
    Direct categorical neural model for candlestick prefix transitions.
    Maps each 3-minute prefix to logits over the 4 directional endings (UU, UD, DU, DD).
    Initializes at 0 (representing equal 25% prior probability before learning).
    """
    def __init__(self, num_prefixes: int = 27, num_classes: int = 4):
        super().__init__()
        self.logits = nn.Embedding(num_prefixes, num_classes)
        nn.init.zeros_(self.logits.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    def get_probabilities(self, prefix_id: int) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor([prefix_id], dtype=torch.long)
            probs = torch.softmax(self.logits(x), dim=-1).squeeze().cpu().numpy()
        return probs


def prepare_dataset(parquet_path: str, year: int = 2026, month: int = 7):
    """
    Loads 1-minute XAUUSD data, computes close-to-close directions,
    and sections into 5-minute blocks.
    """
    print(f"\n[1/4] Loading XAUUSD 1-minute data for {year}-{month:02d}...")
    start_dt = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    # End of month
    if month == 12:
        end_dt = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        end_dt = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    df = pl.read_parquet(parquet_path)
    df = (
        df.filter((pl.col('timestamp') >= start_dt) & (pl.col('timestamp') < end_dt))
        .sort('timestamp')
    )
    print(f"      Total 1-minute bars loaded: {df.height:,}")

    # Compute close-to-close change
    df = df.with_columns([
        (pl.col('Close') - pl.col('Close').shift(1)).alias('diff'),
        pl.col('timestamp').dt.truncate('5m').alias('bucket_5m'),
        (pl.col('timestamp').dt.minute() % 5).alias('minute_in_5m')
    ])

    # Direction labeling: U (Up), D (Down), P (Pause)
    df = df.with_columns(
        pl.when(pl.col('diff') > 0).then(pl.lit('U'))
          .when(pl.col('diff') < 0).then(pl.lit('D'))
          .otherwise(pl.lit('P'))
          .alias('dir')
    )

    # Section into 5-minute blocks
    print("[2/4] Sectioning into 5-minute blocks (5 x 1-min candles)...")
    blocks = []
    grouped = df.group_by('bucket_5m', maintain_order=True)

    for bucket, sub_df in grouped:
        if sub_df.height == 5 and sub_df['minute_in_5m'].to_list() == [0, 1, 2, 3, 4]:
            dirs = sub_df['dir'].to_list()
            m1, m2, m3, m4, m5 = dirs[0], dirs[1], dirs[2], dirs[3], dirs[4]
            blocks.append({
                'bucket': bucket[0],
                'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'm5': m5,
                'prefix': m1 + m2 + m3,
                'target': m4 + m5
            })

    # Filter for directional targets (UU, UD, DU, DD)
    valid_targets = ['UU', 'UD', 'DU', 'DD']
    filtered = [b for b in blocks if b['target'] in valid_targets]
    skipped = len(blocks) - len(filtered)
    print(f"      Total 5-min blocks: {len(blocks):,}")
    print(f"      Valid directional target blocks (UU/UD/DU/DD): {len(filtered):,} ({(len(filtered)/len(blocks))*100:.1f}%)")
    print(f"      Filtered out blocks with Pause (P) in M4 or M5: {skipped:,} ({(skipped/len(blocks))*100:.1f}%)")

    return filtered


def run_online_learning(blocks, lr=0.05):
    """
    Simulates sequential candle-by-candle online learning and error correction.
    """
    chars = ['U', 'D', 'P']
    all_prefixes = sorted([a + b + c for a in chars for b in chars for c in chars])
    prefix_to_id = {p: i for i, p in enumerate(all_prefixes)}
    target_to_id = {'UU': 0, 'UD': 1, 'DU': 2, 'DD': 3}
    id_to_target = {v: k for k, v in target_to_id.items()}

    # Frequency tracking
    empirical_counts = defaultdict(lambda: {'UU': 0, 'UD': 0, 'DU': 0, 'DD': 0, 'total': 0})
    for b in blocks:
        empirical_counts[b['prefix']][b['target']] += 1
        empirical_counts[b['prefix']]['total'] += 1

    # Instantiate model
    torch.manual_seed(42)
    model = MicrostructureModel(num_prefixes=len(all_prefixes), num_classes=4)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[3/4] Running sequential step-by-step online learning & error correction...")
    print(f"      Initial model state: Equal probability [25.0% UU, 25.0% UD, 25.0% DU, 25.0% DD]")
    print(f"      Learning Rate: {lr} (SGD with Cross-Entropy Loss)")
    print("-" * 75)

    correct_predictions = 0
    total = len(blocks)
    running_loss = 0.0

    milestones = [500, 1000, 2000, 3000, 4000, 5000, total]

    for step, b in enumerate(blocks):
        x = torch.tensor([prefix_to_id[b['prefix']]], dtype=torch.long)
        y_true = torch.tensor([target_to_id[b['target']]], dtype=torch.long)

        # 1. Model evaluates input (M1, M2, M3) and predicts probabilities
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            pred_class = int(np.argmax(probs))

        if pred_class == y_true.item():
            correct_predictions += 1

        # 2. Market reveals M4, M5 -> compute error and update model weights
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y_true)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        current_step = step + 1
        if current_step in milestones or current_step % 1000 == 0:
            acc = (correct_predictions / current_step) * 100
            avg_loss = running_loss / current_step
            print(f"  Candle Block {current_step:4d}/{total} | Cumulative Accuracy: {acc:5.2f}% | Running Loss: {avg_loss:.4f}")

    print("-" * 75)
    print(f"[4/4] Learning complete across {total:,} 5-minute blocks!")

    # Display comparison table
    print("\n" + "=" * 85)
    print(f"{'3-Min Prefix':<12} {'Count':<7} {'Model P(UU)':<12} {'Model P(UD)':<12} {'Model P(DU)':<12} {'Model P(DD)':<12} {'Top Learned':<12} {'Empirical Top'}")
    print("=" * 85)

    for p in all_prefixes:
        cnt = empirical_counts[p]['total']
        if cnt >= 15:  # Display frequent prefixes
            p_id = prefix_to_id[p]
            probs = model.get_probabilities(p_id) * 100
            model_top = id_to_target[int(np.argmax(probs))]

            emp_top = max(['UU', 'UD', 'DU', 'DD'], key=lambda k: empirical_counts[p][k])
            emp_pct = empirical_counts[p][emp_top] / cnt * 100

            print(
                f"  {p:<10} {cnt:<7} "
                f"{probs[0]:>8.1f}%    "
                f"{probs[1]:>8.1f}%    "
                f"{probs[2]:>8.1f}%    "
                f"{probs[3]:>8.1f}%    "
                f"{model_top} ({probs.max():.1f}%)    "
                f"{emp_top} ({emp_pct:.1f}%)"
            )

    print("=" * 85)
    return model, empirical_counts, blocks


def main():
    parser = argparse.ArgumentParser(description="M5 Candlestick Microstructure Probability Model")
    parser.add_argument("--parquet", type=str, default="/home/gifted/TradingModel/data/raw/XAUUSD_DUKASCOPY_1m.parquet")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.08)
    args = parser.parse_args()

    blocks = prepare_dataset(args.parquet, year=args.year, month=args.month)
    run_online_learning(blocks, lr=args.lr)


if __name__ == "__main__":
    main()

