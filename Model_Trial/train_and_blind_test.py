#!/usr/bin/env python3
"""
M5 Candlestick Microstructure Model — Proper Training Pipeline.

Trains a neural network on 11 months of XAUUSD 1-minute data to predict
the directional pattern of minutes 4-5 given the observed minutes 1-3.

KEY DIFFERENCE FROM PREVIOUS ATTEMPT:
- Uses REAL continuous features: actual price deltas, magnitudes, volumes,
  cumulative displacement, volatility — not just U/D/P labels.
- Proper random weight initialization (PyTorch defaults, no zeroing).
- Proper train/validation split within the 11-month training window.
- Multiple epochs of full batch training with loss/accuracy tracking.
- Saved weights are then loaded for a fully blind test on month 12.
"""

import os
import argparse
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 1. Prefix and Target vocabulary
CHARS = ['U', 'D', 'P']
ALL_PREFIXES = sorted([a + b + c for a in CHARS for b in CHARS for c in CHARS])
PREFIX_TO_ID = {p: i for i, p in enumerate(ALL_PREFIXES)}
TARGET_TO_ID = {'UU': 0, 'UD': 1, 'DU': 2, 'DD': 3}
ID_TO_TARGET = {v: k for k, v in TARGET_TO_ID.items()}
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_NAMES = ['UU', 'UD', 'DU', 'DD']
TARGET_TO_ID = {n: i for i, n in enumerate(TARGET_NAMES)}
ID_TO_TARGET = {i: n for n, i in TARGET_TO_ID.items()}
WEIGHTS_PATH = '/home/gifted/TradingModel/Model_Trial/weights/m5_micro_v2.pt'
PARQUET_PATH = '/home/gifted/TradingModel/data/raw/XAUUSD_DUKASCOPY_1m.parquet'


def encode_minute(m: str):
    # One-hot encoding for a single minute: [is_U, is_D, is_P]
    if m == 'U':
        return [1.0, 0.0, 0.0]
    elif m == 'D':
        return [0.0, 1.0, 0.0]
    else: # 'P'
        return [0.0, 0.0, 1.0]
# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_and_section(parquet_path: str, start_dt: datetime, end_dt: datetime, label: str = ""):
    """Load 1-minute data, compute features, section into 5-minute blocks."""
    print(f"\n{'='*60}")
    print(f"  Loading: {label}")
    print(f"  Range:   {start_dt.date()} → {end_dt.date()}")
    print(f"{'='*60}")


def prepare_dataset(parquet_path: str, start_dt: datetime, end_dt: datetime, label: str = ""):
    print(f"\n--- Loading {label} Dataset ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}) ---")
    df = pl.read_parquet(parquet_path)
    df = (
        df.filter((pl.col('timestamp') >= start_dt) & (pl.col('timestamp') <= end_dt))
        .sort('timestamp')
    )
    print(f"Total 1-minute bars loaded: {df.height:,}")
    print(f"  1-minute bars loaded: {df.height:,}")

    # Compute close-to-close change
    # Compute per-bar features
    df = df.with_columns([
        (pl.col('Close') - pl.col('Close').shift(1)).alias('diff'),
        (pl.col('Close') - pl.col('Close').shift(1)).alias('close_delta'),
        (pl.col('High') - pl.col('Low')).alias('bar_range'),
        (pl.col('Close') - pl.col('Open')).alias('body'),
        pl.col('Volume').alias('volume'),
        pl.col('timestamp').dt.truncate('5m').alias('bucket_5m'),
        (pl.col('timestamp').dt.minute() % 5).alias('minute_in_5m')
        (pl.col('timestamp').dt.minute() % 5).alias('minute_in_5m'),
    ])

    # Classify direction: U, D, P
    # Drop the first row (null delta)
    df = df.drop_nulls(subset=['close_delta'])

    # Direction labels for target filtering
    df = df.with_columns(
        pl.when(pl.col('diff') > 0).then(pl.lit('U'))
          .when(pl.col('diff') < 0).then(pl.lit('D'))
        pl.when(pl.col('close_delta') > 0).then(pl.lit('U'))
          .when(pl.col('close_delta') < 0).then(pl.lit('D'))
          .otherwise(pl.lit('P'))
          .alias('dir')
    )

    # Group into 5-minute blocks
    blocks = []
    features_list = []
    targets_list = []
    prefixes_list = []
    skipped_pause = 0
    total_blocks = 0

    grouped = df.group_by('bucket_5m', maintain_order=True)

    for bucket, sub_df in grouped:
        if sub_df.height == 5 and sub_df['minute_in_5m'].to_list() == [0, 1, 2, 3, 4]:
            dirs = sub_df['dir'].to_list()
            m1, m2, m3, m4, m5 = dirs[0], dirs[1], dirs[2], dirs[3], dirs[4]
            prefix_3 = m1 + m2 + m3
            target_pair = m4 + m5
            if target_pair in TARGET_TO_ID: # Exclude pauses in M4/M5
                feat_vec = encode_minute(m1) + encode_minute(m2) + encode_minute(m3)
                blocks.append({
                    'timestamp': bucket[0],
                    'prefix': prefix_3,
                    'prefix_id': PREFIX_TO_ID[prefix_3],
                    'feats': feat_vec,
                    'target': target_pair,
                    'target_id': TARGET_TO_ID[target_pair]
                })
        if sub_df.height != 5:
            continue
        if sub_df['minute_in_5m'].to_list() != [0, 1, 2, 3, 4]:
            continue

    print(f"Valid 5-minute directional blocks: {len(blocks):,}")
    return blocks
        total_blocks += 1
        dirs = sub_df['dir'].to_list()
        m4, m5 = dirs[3], dirs[4]
        target_pair = m4 + m5

        # Only keep blocks with directional M4-M5
        if target_pair not in TARGET_TO_ID:
            skipped_pause += 1
            continue

class M5MicrostructureNet(nn.Module):
        # ---------- FEATURE ENGINEERING ----------
        deltas = sub_df['close_delta'].to_list()      # 5 values
        ranges = sub_df['bar_range'].to_list()         # 5 values
        bodies = sub_df['body'].to_list()              # 5 values
        vols   = sub_df['volume'].to_list()            # 5 values

        # Features from the FIRST 3 MINUTES (what the model sees):
        d1, d2, d3 = deltas[0], deltas[1], deltas[2]
        r1, r2, r3 = ranges[0], ranges[1], ranges[2]
        b1, b2, b3 = bodies[0], bodies[1], bodies[2]
        v1, v2, v3 = vols[0], vols[1], vols[2]

        # Derived features from M1-M3
        cum_delta = d1 + d2 + d3                       # cumulative displacement after 3 min
        avg_range = (r1 + r2 + r3) / 3.0               # avg volatility of first 3 min
        avg_vol   = (v1 + v2 + v3) / 3.0               # avg volume
        delta_accel = d3 - d1                           # acceleration: is move speeding up?
        max_delta = max(abs(d1), abs(d2), abs(d3))      # max single-minute move
        sign_sum  = (1 if d1>0 else (-1 if d1<0 else 0)) + \
                    (1 if d2>0 else (-1 if d2<0 else 0)) + \
                    (1 if d3>0 else (-1 if d3<0 else 0))  # net direction count (-3 to +3)

        feat_vec = [
            # Raw deltas (price change per minute)
            d1, d2, d3,
            # Bar bodies (open-to-close)
            b1, b2, b3,
            # Bar ranges (high-low, volatility proxy)
            r1, r2, r3,
            # Volumes
            v1, v2, v3,
            # Derived / aggregate
            cum_delta,
            avg_range,
            avg_vol,
            delta_accel,
            max_delta,
            float(sign_sum),
        ]

        features_list.append(feat_vec)
        targets_list.append(TARGET_TO_ID[target_pair])
        prefixes_list.append(dirs[0] + dirs[1] + dirs[2])

    print(f"  Total 5-min blocks formed: {total_blocks:,}")
    print(f"  Valid directional blocks:  {len(features_list):,} ({len(features_list)/max(total_blocks,1)*100:.1f}%)")
    print(f"  Skipped (Pause in M4/M5):  {skipped_pause:,}")
    print(f"  Feature dimension:         {len(features_list[0]) if features_list else 0}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets_list, dtype=np.int64)

    return X, y, prefixes_list


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class M5Net(nn.Module):
    """
    Neural pattern model for candlestick microstructure.
    Learns non-linear interactions between sequential minute directions and predicts
    the calibrated probability distribution across {UU, UD, DU, DD}.
    Feedforward network for candlestick microstructure prediction.
    Uses standard PyTorch default initialization (Kaiming uniform).
    NO manual zeroing. NO uniform prior bias.
    """
    def __init__(self, num_prefixes=27, feature_dim=9, num_classes=4):
    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.prefix_emb = nn.Embedding(num_prefixes, 16)
        self.fc = nn.Sequential(
            nn.Linear(16 + feature_dim, 64),
            nn.LayerNorm(64),
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)

            nn.Linear(32, num_classes),
        )

    def forward(self, prefix_id, feats):
        emb = self.prefix_emb(prefix_id)
        x = torch.cat([emb, feats], dim=-1)
        return self.fc(x)
    def forward(self, x):
        return self.net(x)


def train_model(train_blocks, weights_path: str, epochs: int = 25, lr: float = 0.003, batch_size: int = 128):
    print(f"\n=======================================================")
    print(f"       TRAINING PATTERN MODEL (11-MONTH DATASET)       ")
    print(f"=======================================================")
    print(f"Training samples: {len(train_blocks):,} blocks")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")
# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------
def normalize_features(X_train, X_test):
    """Z-score normalize using training set statistics only (no data leakage)."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std < 1e-8] = 1.0  # prevent division by zero

    prefix_tensor = torch.tensor([b['prefix_id'] for b in train_blocks], dtype=torch.long)
    feats_tensor = torch.tensor([b['feats'] for b in train_blocks], dtype=torch.float32)
    target_tensor = torch.tensor([b['target_id'] for b in train_blocks], dtype=torch.long)
    X_train_norm = (X_train - mean) / std
    X_test_norm  = (X_test - mean) / std

    dataset = TensorDataset(prefix_tensor, feats_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return X_train_norm, X_test_norm, mean, std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, input_dim, epochs=80, lr=0.001, batch_size=256):
    print(f"\n{'='*60}")
    print(f"  MODEL TRAINING — 11 MONTHS OF PATTERN LEARNING")
    print(f"{'='*60}")

    # Split: 90% train, 10% validation (temporal — last 10% of training data)
    split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    print(f"  Train samples:      {len(X_tr):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Input features:     {input_dim}")
    print(f"  Output classes:     4 (UU, UD, DU, DD)")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    # Class distribution
    for i, name in ID_TO_TARGET.items():
        tr_pct = (y_tr == i).mean() * 100
        val_pct = (y_val == i).mean() * 100
        print(f"    {name}: Train {tr_pct:.1f}% | Val {val_pct:.1f}%")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_X_t = torch.from_numpy(X_val)
    val_y_t = torch.from_numpy(y_val)

    torch.manual_seed(42)
    np.random.seed(42)
    model = M5MicrostructureNet(num_prefixes=27, feature_dim=9, num_classes=4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model = M5Net(input_dim=input_dim, num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 20

    print(f"\n  {'Epoch':<7} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<12} {'Val Acc':<11} {'LR':<10} {'Status'}")
    print(f"  {'-'*75}")

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for b_prefix, b_feats, b_target in loader:
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(b_prefix, b_feats)
            loss = criterion(logits, b_target)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_target)
            preds = logits.argmax(dim=-1)
            correct += (preds == b_target).sum().item()
            total += len(b_target)
            train_loss += loss.item() * len(by)
            train_correct += (logits.argmax(dim=-1) == by).sum().item()
            train_total += len(by)

        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100
        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:5.2f}% (Baseline: 25.0%)")
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_t)
            val_loss = criterion(val_logits, val_y_t).item()
            val_acc = (val_logits.argmax(dim=-1) == val_y_t).float().mean().item() * 100

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save best weights
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(model.state_dict(), weights_path)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

    print(f"\n>>> Best training weights stored successfully to: {weights_path}")
        # Track best
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'epoch': epoch,
            }, WEIGHTS_PATH)
            status = "✓ SAVED"
        else:
            patience_counter += 1

        if epoch <= 5 or epoch % 10 == 0 or status or epoch == epochs:
            print(f"  {epoch:<7} {train_loss:<12.4f} {train_acc:<11.2f} {val_loss:<12.4f} {val_acc:<11.2f} {current_lr:<10.6f} {status}")

        if patience_counter >= max_patience:
            print(f"\n  Early stopping at epoch {epoch} (no val improvement for {max_patience} epochs)")
            break

    print(f"\n  >>> Best Validation Loss: {best_val_loss:.4f} | Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  >>> Weights saved to: {WEIGHTS_PATH}")
    return model


def run_blind_test(test_blocks, weights_path: str):
    print(f"\n=======================================================")
    print(f"      BLIND PREDICTION EVALUATION (12th UNSEEN MONTH)  ")
    print(f"=======================================================")
    print(f"Loading frozen trained weights from: {weights_path}")
# ---------------------------------------------------------------------------
# Blind Test
# ---------------------------------------------------------------------------
def blind_test(X_test, y_test, prefixes_test, input_dim):
    print(f"\n{'='*60}")
    print(f"  BLIND TEST — 12TH UNSEEN MONTH (FROZEN WEIGHTS)")
    print(f"{'='*60}")

    model = M5MicrostructureNet(num_prefixes=27, feature_dim=9, num_classes=4)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    # Load saved checkpoint
    checkpoint = torch.load(WEIGHTS_PATH, weights_only=False)
    model = M5Net(input_dim=input_dim, num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    prefix_tensor = torch.tensor([b['prefix_id'] for b in test_blocks], dtype=torch.long)
    feats_tensor = torch.tensor([b['feats'] for b in test_blocks], dtype=torch.float32)
    target_tensor = torch.tensor([b['target_id'] for b in test_blocks], dtype=torch.long)
    print(f"  Loaded weights from epoch {checkpoint['epoch']} (val_loss={checkpoint['best_val_loss']:.4f}, val_acc={checkpoint['best_val_acc']:.2f}%)")
    print(f"  Blind test samples: {len(X_test):,}")

    criterion = nn.CrossEntropyLoss()
    X_t = torch.from_numpy(X_test)
    y_t = torch.from_numpy(y_test)

    with torch.no_grad():
        logits = model(prefix_tensor, feats_tensor)
        loss = criterion(logits, target_tensor).item()
        logits = model(X_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        loss = nn.CrossEntropyLoss()(logits, y_t).item()

    targets = target_tensor.cpu().numpy()
    correct_mask = (preds == targets)
    total_samples = len(test_blocks)
    accuracy = (correct_mask.sum() / total_samples) * 100
    correct = (preds == y_test)
    accuracy = correct.mean() * 100

    print(f"Blind Test Samples: {total_samples:,}")
    print(f"Blind Test Accuracy: {accuracy:.2f}% (vs 25.0% random baseline)")
    print(f"Blind Test Cross-Entropy Loss: {loss:.4f}")
    print(f"\n  BLIND TEST ACCURACY: {accuracy:.2f}%  (Random baseline: 25.00%)")
    print(f"  BLIND TEST LOSS:     {loss:.4f}")

    # Empirical test distribution vs Model predictions
    counts = defaultdict(lambda: {'count': 0, 'correct': 0, 'UU': 0, 'UD': 0, 'DU': 0, 'DD': 0})
    for i, b in enumerate(test_blocks):
        p = b['prefix']
        t = b['target']
        counts[p]['count'] += 1
        counts[p][t] += 1
        if correct_mask[i]:
            counts[p]['correct'] += 1
    # Class distribution in test set
    print(f"\n  Actual class distribution in unseen month:")
    for i, name in ID_TO_TARGET.items():
        pct = (y_test == i).mean() * 100
        print(f"    {name}: {(y_test == i).sum():,} ({pct:.1f}%)")

    print("\n" + "=" * 90)
    print(f"{'Prefix':<8} {'Count':<7} {'Model P(UU)':<12} {'Model P(UD)':<12} {'Model P(DU)':<12} {'Model P(DD)':<12} {'Model Pick':<11} {'Actual Top':<11} {'Test Acc'}")
    print("=" * 90)
    # Per-class precision (when model predicts X, how often is it right?)
    print(f"\n  {'Predicted':<12} {'Times':<8} {'Precision':<12} {'Interpretation'}")
    print(f"  {'-'*55}")
    for i, name in ID_TO_TARGET.items():
        mask = (preds == i)
        count = mask.sum()
        if count > 0:
            prec = (y_test[mask] == i).mean() * 100
            edge = prec - 25.0
            interp = f"+{edge:.1f}% edge" if edge > 0 else f"{edge:.1f}% (no edge)"
            print(f"  {name:<12} {count:<8} {prec:<12.2f} {interp}")
        else:
            print(f"  {name:<12} {'0':<8} {'N/A':<12} Never predicted")

    # Get sample prediction probabilities for each prefix
    with torch.no_grad():
        sample_prefixes = torch.arange(27, dtype=torch.long)
        sample_feats = []
        for p_id in range(27):
            p_str = ALL_PREFIXES[p_id]
            f = encode_minute(p_str[0]) + encode_minute(p_str[1]) + encode_minute(p_str[2])
            sample_feats.append(f)
        sample_feats = torch.tensor(sample_feats, dtype=torch.float32)
        sample_probs = torch.softmax(model(sample_prefixes, sample_feats), dim=-1).cpu().numpy() * 100
    # Per-prefix analysis
    print(f"\n  Per-Prefix Blind Performance:")
    print(f"  {'Prefix':<8} {'N':<6} {'Acc%':<8} {'Model Top':<11} {'Actual Top':<11} {'Avg P(top)':<10}")
    print(f"  {'-'*60}")

    for p in ALL_PREFIXES:
        p_id = PREFIX_TO_ID[p]
        cnt = counts[p]['count']
        if cnt >= 15: # Common prefixes in test set
            pr = sample_probs[p_id]
            pred_pick = ID_TO_TARGET[int(np.argmax(pr))]
            actual_top = max(['UU', 'UD', 'DU', 'DD'], key=lambda k: counts[p][k])
            actual_top_pct = counts[p][actual_top] / cnt * 100
            p_acc = (counts[p]['correct'] / cnt) * 100
    prefix_stats = defaultdict(lambda: {'count': 0, 'correct': 0, 'targets': defaultdict(int)})
    for i in range(len(y_test)):
        p = prefixes_test[i]
        prefix_stats[p]['count'] += 1
        prefix_stats[p]['targets'][y_test[i]] += 1
        if correct[i]:
            prefix_stats[p]['correct'] += 1

            print(
                f"{p:<8} {cnt:<7} "
                f"{pr[0]:>8.1f}%    "
                f"{pr[1]:>8.1f}%    "
                f"{pr[2]:>8.1f}%    "
                f"{pr[3]:>8.1f}%    "
                f"{pred_pick:<11} "
                f"{actual_top} ({actual_top_pct:.1f}%) "
                f"{p_acc:>6.1f}%"
            )
    for prefix in sorted(prefix_stats.keys()):
        st = prefix_stats[prefix]
        if st['count'] < 10:
            continue
        pacc = st['correct'] / st['count'] * 100
        actual_top_id = max(st['targets'], key=st['targets'].get)
        actual_top = ID_TO_TARGET[actual_top_id]
        actual_top_pct = st['targets'][actual_top_id] / st['count'] * 100

    print("=" * 90)
        # Get model's average prediction for this prefix
        idxs = [j for j, pp in enumerate(prefixes_test) if pp == prefix]
        avg_probs = probs[idxs].mean(axis=0) * 100
        model_top = ID_TO_TARGET[int(avg_probs.argmax())]

    # Performance per predicted class
    print("\n--- Breakdown by Model's Predicted Class (Precision) ---")
    for c_id, c_name in ID_TO_TARGET.items():
        predicted_as_c = (preds == c_id)
        if predicted_as_c.sum() > 0:
            c_acc = (targets[predicted_as_c] == c_id).mean() * 100
            print(f"When model predicted '{c_name}': {predicted_as_c.sum():4d} times | Win Rate: {c_acc:5.2f}%")
        print(f"  {prefix:<8} {st['count']:<6} {pacc:<8.1f} {model_top:<11} {actual_top} ({actual_top_pct:.0f}%)   {avg_probs.max():<.1f}%")

    print("\n--- Confusion Matrix (Actual vs Predicted) ---")
    print(f"{'Actual \\ Pred':<14} {'UU':<8} {'UD':<8} {'DU':<8} {'DD':<8}")
    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':14}", end="")
    for name in TARGET_NAMES:
        print(f"{name:<8}", end="")
    print()
    for act_id, act_name in ID_TO_TARGET.items():
        row = [(targets[preds == pred_id] == act_id).sum() for pred_id in range(4)]
        print(f"{act_name:<14} {row[0]:<8} {row[1]:<8} {row[2]:<8} {row[3]:<8}")
        print(f"  {act_name:<14}", end="")
        for pred_id in range(4):
            cnt = ((y_test == act_id) & (preds == pred_id)).sum()
            print(f"{cnt:<8}", end="")
        print()

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train on 11 months, blind predict on 12th month")
    parser.add_argument("--parquet", type=str, default="/home/gifted/TradingModel/data/raw/XAUUSD_DUKASCOPY_1m.parquet")
    parser.add_argument("--weights", type=str, default="/home/gifted/TradingModel/Model_Trial/weights/m5_microstructure_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # Define 11-month training window: 2025-09-01 to 2026-07-31
    # Time windows
    train_start = datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    train_end = datetime(2026, 7, 31, 23, 59, 59, tzinfo=timezone.utc)
    train_end   = datetime(2026, 7, 31, 23, 59, 59, tzinfo=timezone.utc)
    test_start  = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    test_end    = datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)

    # Define 12th month blind test window: 2026-08-01 to 2026-08-31
    test_start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    test_end = datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)
    # Load & section data
    X_train_raw, y_train, pfx_train = load_and_section(PARQUET_PATH, train_start, train_end, "11-Month Training Set")
    X_test_raw,  y_test,  pfx_test  = load_and_section(PARQUET_PATH, test_start,  test_end,  "12th Month Blind Test")

    # 1. Prepare 11 months for training
    train_blocks = prepare_dataset(args.parquet, train_start, train_end, label="11-Month Training")
    # Normalize features using ONLY training statistics
    X_train, X_test, feat_mean, feat_std = normalize_features(X_train_raw, X_test_raw)

    # 2. Prepare 12th month for blind test
    test_blocks = prepare_dataset(args.parquet, test_start, test_end, label="12th Month Blind Test")
    input_dim = X_train.shape[1]
    print(f"\n  Feature normalization complete (mean/std from training set only, no leakage)")
    print(f"  Input dimension: {input_dim}")

    # 3. Train model and save weights
    train_model(train_blocks, args.weights, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    # Train
    train_model(X_train, y_train, input_dim, epochs=80, lr=0.001, batch_size=256)

    # 4. Perform blind test
    run_blind_test(test_blocks, args.weights)
    # Blind test
    blind_test(X_test, y_test, pfx_test, input_dim)


if __name__ == "__main__":
    main()

