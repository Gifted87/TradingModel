import time, torch, numpy as np, polars as pl
from src.features.pipeline import FeaturePipeline
from src.baselines.linear_core import LinearCoreBranchA
from src.models.network import ResidualLearnerNetwork, ModelOutputs
from src.models.loss import CompositeMultiTaskLoss
from src.models.dataset import BatchData

# Load 80k rows from real dataset
df = pl.read_parquet('data/features/5min/EURUSD_5m_features.parquet').slice(0, 80000)

t0 = time.time()
feature_cols = [c for c in df.columns if c.startswith('g') and '_' in c]
g6_cols = [c for c in feature_cols if c.startswith('g6_')]

pipe = FeaturePipeline(clip_sigma=5.0).fit(df)
norm_df = pipe.transform(df)

branch_a = LinearCoreBranchA(alpha=100.0).fit(df, symbol="EURUSD")
r_hat_lin = branch_a.predict(df)

feat_np = norm_df.select(feature_cols).fill_nan(0.0).fill_null(0.0).to_numpy()
g6_np = norm_df.select(g6_cols).fill_nan(0.0).fill_null(0.0).to_numpy()
r30 = df['r30'].to_numpy()
rv30 = df['rv30'].to_numpy()
mag30 = df['mag30'].to_numpy()

lookback = 60
feat_t = torch.tensor(feat_np, dtype=torch.float32)
windows = feat_t.unfold(0, lookback, 1).permute(0, 2, 1).contiguous()

r_lin_t = torch.tensor(r_hat_lin[lookback-1:], dtype=torch.float32).unsqueeze(1)
g6_t = torch.tensor(g6_np[lookback-1:], dtype=torch.float32)
r30_t = torch.tensor(r30[lookback-1:], dtype=torch.float32).unsqueeze(1)
mag30_t = torch.tensor(mag30[lookback-1:], dtype=torch.float32).unsqueeze(1)
rv30_t = torch.tensor(rv30[lookback-1:], dtype=torch.float32).unsqueeze(1)
abs_r30_t = torch.abs(r30_t)

net = ResidualLearnerNetwork(in_features=len(feature_cols), g6_dim=len(g6_cols))
loss_fn = CompositeMultiTaskLoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

n_samples = len(r30_t)
n_val = int(n_samples * 0.15)
n_tr = n_samples - n_val

BS = 512
epochs = 8

for ep in range(epochs):
    net.train()
    perm = torch.randperm(n_tr)
    for i in range(0, n_tr, BS):
        idx = perm[i:i+BS]
        opt.zero_grad()
        outs = net(windows[idx], r_lin_t[idx], g6_t[idx])
        batch = BatchData(
            sequences=windows[idx], r_hat_lin=r_lin_t[idx], g6_features=g6_t[idx],
            r30=r30_t[idx], residual=r30_t[idx] - r_lin_t[idx],
            mag30=mag30_t[idx], rv30=rv30_t[idx], abs_r30=abs_r30_t[idx]
        )
        l, _ = loss_fn(outs, batch)
        l.backward()
        opt.step()

print(f"COMPLETE 8-EPOCH FOLD TRAINING TIME ON CPU: {time.time() - t0:.2f} seconds!")
