#!/usr/bin/env python3
"""
M5 Candlestick Microstructure Model — Binary Net 2-Minute Displacement Predictor.

Task:
At minute 3 (after observing M1, M2, M3), predict whether the NET price change over the
remaining 2 minutes (Close_M5 - Close_M3) will be:
  - UP (1): Close_M5 > Close_M3
  - DOWN (0): Close_M5 < Close_M3

Trained on 11 months (Sep 2025 - Jul 2026), evaluated blindly on the 12th month (Aug 2026).
"""

import os
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

WEIGHTS_PATH = '/home/gifted/TradingModel/Model_Trial/weights/m5_net2m_binary.pt'
PARQUET_PATH = '/home/gifted/TradingModel/data/raw/XAUUSD_DUKASCOPY_1m.parquet'


def load_and_prepare_binary_data(parquet_path: str, start_dt: datetime, end_dt: datetime, label: str = ""):
    print(f"\n{'='*65}")
    print(f"  Loading: {label}")
    print(f"  Range:   {start_dt.date()} → {end_dt.date()}")
    print(f"{'='*65}")

    df = pl.read_parquet(parquet_path)
    df = (
        df.filter((pl.col('timestamp') >= start_dt) & (pl.col('timestamp') <= end_dt))
        .sort('timestamp')
    )
    print(f"  1-minute bars loaded: {df.height:,}")

    # Compute bar metrics
    df = df.with_columns([
        (pl.col('Close') - pl.col('Close').shift(1)).alias('close_delta'),
        (pl.col('High') - pl.col('Low')).alias('bar_range'),
        (pl.col('Close') - pl.col('Open')).alias('body'),
        pl.col('Volume').alias('volume'),
        pl.col('timestamp').dt.truncate('5m').alias('bucket_5m'),
        (pl.col('timestamp').dt.minute() % 5).alias('minute_in_5m'),
        pl.col('timestamp').dt.hour().alias('hour_utc'),
    ])

    df = df.drop_nulls(subset=['close_delta'])

    features_list = []
    targets_list = []
    net_deltas = []
    prefixes_list = []
    skipped_flat = 0

    grouped = df.group_by('bucket_5m', maintain_order=True)
    for bucket, sub_df in grouped:
        if sub_df.height != 5:
            continue
        if sub_df['minute_in_5m'].to_list() != [0, 1, 2, 3, 4]:
            continue

        closes = sub_df['Close'].to_list()
        deltas = sub_df['close_delta'].to_list()
        ranges = sub_df['bar_range'].to_list()
        bodies = sub_df['body'].to_list()
        vols   = sub_df['volume'].to_list()
        hour   = sub_df['hour_utc'][0]

        # Net 2-minute displacement: Close_M5 - Close_M3 (closes[4] - closes[2])
        net_2m = closes[4] - closes[2]

        if net_2m == 0:
            skipped_flat += 1
            continue

        target = 1 if net_2m > 0 else 0  # 1 = UP, 0 = DOWN

        # Observed features from M1 to M3:
        d1, d2, d3 = deltas[0], deltas[1], deltas[2]
        r1, r2, r3 = ranges[0], ranges[1], ranges[2]
        b1, b2, b3 = bodies[0], bodies[1], bodies[2]
        v1, v2, v3 = vols[0], vols[1], vols[2]

        cum_delta_m3 = closes[2] - sub_df['Open'][0] # displacement from candle open to M3 close
        avg_range_m3 = (r1 + r2 + r3) / 3.0
        avg_vol_m3   = (v1 + v2 + v3) / 3.0
        accel        = d3 - d1
        max_abs_move = max(abs(d1), abs(d2), abs(d3))

        # Direction signs
        s1 = 1 if d1 > 0 else (-1 if d1 < 0 else 0)
        s2 = 1 if d2 > 0 else (-1 if d2 < 0 else 0)
        s3 = 1 if d3 > 0 else (-1 if d3 < 0 else 0)
        sign_sum = s1 + s2 + s3

        # Hour sin/cos cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        feat_vec = [
            d1, d2, d3,
            b1, b2, b3,
            r1, r2, r3,
            v1, v2, v3,
            cum_delta_m3,
            avg_range_m3,
            avg_vol_m3,
            accel,
            max_abs_move,
            float(sign_sum),
            hour_sin,
            hour_cos
        ]

        # Text prefix for grouping analysis
        pfx = ("U" if s1>0 else ("D" if s1<0 else "P")) + \
              ("U" if s2>0 else ("D" if s2<0 else "P")) + \
              ("U" if s3>0 else ("D" if s3<0 else "P"))

        features_list.append(feat_vec)
        targets_list.append(target)
        net_deltas.append(net_2m)
        prefixes_list.append(pfx)

    print(f"  Valid 5-minute blocks:  {len(features_list):,}")
    print(f"  Skipped (exact 0 delta): {skipped_flat}")
    print(f"  Feature dimension:      {len(features_list[0]) if features_list else 0}")
    print(f"  Class balance: UP={sum(targets_list)/len(targets_list)*100:.2f}% | DOWN={(1-sum(targets_list)/len(targets_list))*100:.2f}%")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets_list, dtype=np.float32) # For BCE loss
    return X, y, prefixes_list, np.array(net_deltas, dtype=np.float32)


class Net2MClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, 1)  # Logits for binary outcome
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def train_binary_model(X_train, y_train, epochs=60, batch_size=256, lr=0.001):
    print(f"\n{'='*65}")
    print(f"  TRAINING BINARY 2-MINUTE NET PREDICTOR (11 MONTHS)")
    print(f"{'='*65}")

    split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    print(f"  Train samples: {len(X_tr):,} | Val samples: {len(X_val):,}")
    print(f"  Random Guessing Baseline: 50.00%")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_X_t = torch.from_numpy(X_val)
    val_y_t = torch.from_numpy(y_val)

    torch.manual_seed(42)
    model = Net2MClassifier(input_dim=X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 0
    max_patience = 15

    print(f"\n  {'Epoch':<7} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<12} {'Val Acc':<11} {'LR':<10} {'Status'}")
    print(f"  {'-'*75}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == by).sum().item()
            total += len(by)

        train_loss = total_loss / total
        train_acc = (correct / total) * 100

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_t)
            val_loss = criterion(val_logits, val_y_t).item()
            val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = (val_preds == val_y_t).float().mean().item() * 100

        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]['lr']

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience = 0
            os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': X_train.shape[1],
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'epoch': epoch
            }, WEIGHTS_PATH)
            status = "✓ BEST"
        else:
            patience += 1

        if epoch <= 5 or epoch % 5 == 0 or status or epoch == epochs:
            print(f"  {epoch:<7} {train_loss:<12.4f} {train_acc:<11.2f} {val_loss:<12.4f} {val_acc:<11.2f} {cur_lr:<10.6f} {status}")

        if patience >= max_patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  >>> Best Validation Loss: {best_val_loss:.4f} | Accuracy: {best_val_acc:.2f}%")
    print(f"  >>> Weights saved to: {WEIGHTS_PATH}")
    return model


def blind_test_binary(X_test, y_test, prefixes_test, net_deltas_test):
    print(f"\n{'='*65}")
    print(f"  BLIND OUT-OF-SAMPLE TEST (12th MONTH — FROZEN WEIGHTS)")
    print(f"{'='*65}")

    checkpoint = torch.load(WEIGHTS_PATH, weights_only=False)
    model = Net2MClassifier(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['best_val_acc']:.2f}%)")
    print(f"  Blind test blocks: {len(X_test):,}")

    X_t = torch.from_numpy(X_test)
    y_t = torch.from_numpy(y_test)

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.float32)
        bce = nn.BCEWithLogitsLoss()(logits, y_t).item()

    acc = (preds == y_test).mean() * 100
    up_cnt = (y_test == 1).sum()
    down_cnt = (y_test == 0).sum()

    print(f"\n  BLIND TEST OVERALL ACCURACY: {acc:.2f}%  (Baseline: 50.00%)")
    print(f"  BLIND TEST BCE LOSS:        {bce:.4f}")
    print(f"  Actual Test Distribution:   UP={up_cnt:,} ({up_cnt/len(y_test)*100:.1f}%) | DOWN={down_cnt:,} ({down_cnt/len(y_test)*100:.1f}%)")

    # High-confidence subsets (edge analysis)
    print(f"\n  Confidence Threshold Slicing:")
    print(f"  {'Threshold':<14} {'Trades':<8} {'Win Rate':<12} {'Edge vs 50%':<12}")
    print(f"  {'-'*50}")

    for conf in [0.50, 0.52, 0.54, 0.56, 0.58]:
        long_mask = probs >= conf
        short_mask = probs <= (1.0 - conf)
        total_mask = long_mask | short_mask

        if total_mask.sum() > 0:
            win_count = (preds[total_mask] == y_test[total_mask]).sum()
            wr = win_count / total_mask.sum() * 100
            edge = wr - 50.0
            print(f"  P >= {conf*100:.0f}%      {total_mask.sum():<8} {wr:<12.2f} {edge:+.2f}%")

    # Prefix breakdown
    print(f"\n  Blind Accuracy Grouped by 3-Minute Prefix (M1-M3):")
    print(f"  {'Prefix':<8} {'Count':<7} {'Actual UP%':<12} {'Model Pred UP%':<16} {'Model Acc':<10}")
    print(f"  {'-'*60}")

    pfx_data = defaultdict(lambda: {'count': 0, 'actual_up': 0, 'pred_up': 0, 'correct': 0})
    for i, p in enumerate(prefixes_test):
        pfx_data[p]['count'] += 1
        if y_test[i] == 1:
            pfx_data[p]['actual_up'] += 1
        if preds[i] == 1:
            pfx_data[p]['pred_up'] += 1
        if preds[i] == y_test[i]:
            pfx_data[p]['correct'] += 1

    for p in sorted(pfx_data.keys()):
        cnt = pfx_data[p]['count']
        if cnt >= 20:
            act_up = pfx_data[p]['actual_up'] / cnt * 100
            pr_up  = pfx_data[p]['pred_up'] / cnt * 100
            pacc   = pfx_data[p]['correct'] / cnt * 100
            print(f"  {p:<8} {cnt:<7} {act_up:>8.1f}%     {pr_up:>10.1f}%       {pacc:>6.2f}%")

    print(f"{'='*65}\n")


def main():
    train_start = datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    train_end   = datetime(2026, 7, 31, 23, 59, 59, tzinfo=timezone.utc)
    test_start  = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    test_end    = datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)

    X_train_raw, y_train, pfx_train, _ = load_and_prepare_binary_data(
        PARQUET_PATH, train_start, train_end, "11-Month Training Set"
    )
    X_test_raw, y_test, pfx_test, net_deltas_test = load_and_prepare_binary_data(
        PARQUET_PATH, test_start, test_end, "12th Month Blind Test"
    )

    X_train, X_test = normalize(X_train_raw, X_test_raw)

    train_binary_model(X_train, y_train, epochs=60, batch_size=256, lr=0.001)
    blind_test_binary(X_test, y_test, pfx_test, net_deltas_test)


if __name__ == "__main__":
    main()

