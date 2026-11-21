import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, polars as pl
from src.features.pipeline import FeaturePipeline
from src.baselines.linear_core import LinearCoreBranchA
from src.models.network import ResidualLearnerNetwork, ModelOutputs
from src.models.loss import CompositeMultiTaskLoss
from src.models.dataset import BatchData
from src.validation.metrics import compute_regression_metrics, compute_direction_metrics

df = pl.read_parquet('data/features/5min/EURUSD_5m_features.parquet').slice(0, 80000)
test_df = pl.read_parquet('data/features/5min/EURUSD_5m_features.parquet').slice(80030, 20000)

feature_cols = [c for c in df.columns if c.startswith('g') and '_' in c]
g6_cols = [c for c in feature_cols if c.startswith('g6_')]

pipe = FeaturePipeline(clip_sigma=5.0).fit(df)
norm_train = pipe.transform(df)
norm_test = pipe.transform(test_df)

branch_a = LinearCoreBranchA(alpha=100.0).fit(df, symbol="EURUSD")
r_lin_train = branch_a.predict(df)
r_lin_test = branch_a.predict(test_df)

y_test = test_df['r30'].to_numpy()
rv_test = test_df['rv30'].to_numpy()
ba_reg = compute_regression_metrics(y_test, r_lin_test, rv_test, rv_test)
print(f"Branch A Test RMSE: {ba_reg.rmse_r30:.6f}")

LOOKBACK = 60
feat_t = torch.tensor(norm_train.select(feature_cols).fill_nan(0.0).to_numpy(), dtype=torch.float32)
windows_tr = feat_t.unfold(0, LOOKBACK, 1).permute(0, 2, 1).contiguous()

r_lin_t = torch.tensor(r_lin_train[LOOKBACK-1:], dtype=torch.float32).unsqueeze(1)
g6_t = torch.tensor(norm_train.select(g6_cols).fill_nan(0.0).to_numpy()[LOOKBACK-1:], dtype=torch.float32)
r30_t = torch.tensor(df['r30'].to_numpy()[LOOKBACK-1:], dtype=torch.float32).unsqueeze(1)
mag30_t = torch.tensor(df['mag30'].to_numpy()[LOOKBACK-1:], dtype=torch.float32).unsqueeze(1)
rv30_t = torch.tensor(df['rv30'].to_numpy()[LOOKBACK-1:], dtype=torch.float32).unsqueeze(1)
abs_r30_t = torch.abs(r30_t)

# Proper Residual Network with ZERO-initialized residual output head
net = ResidualLearnerNetwork(in_features=len(feature_cols), g6_dim=len(g6_cols))
nn.init.zeros_(net.residual_head[-1].weight)
nn.init.zeros_(net.residual_head[-1].bias)

# Scaled return loss to match financial return scale (e.g. 1e4 scaling for ~0.001 returns)
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

BS = 512
n_samples = len(r30_t)
n_tr = int(n_samples * 0.85)

for ep in range(1, 9):
    net.train()
    perm = torch.randperm(n_tr)
    for i in range(0, n_tr, BS):
        idx = perm[i:i+BS]
        optimizer.zero_grad()
        outs = net(windows_tr[idx], r_lin_t[idx], g6_t[idx])
        
        # Balance return loss with magnitude loss
        loss_ret = F.mse_loss(outs.r30_hat, r30_t[idx]) * 10000.0  # Scale returns to balance gradients
        loss_mag = F.smooth_l1_loss(outs.mag30_hat, mag30_t[idx])
        loss_rv  = F.smooth_l1_loss(outs.rv30_hat, rv30_t[idx])
        loss_total = loss_ret + 0.5 * loss_mag + 0.5 * loss_rv
        
        loss_total.backward()
        optimizer.step()

# Test inference
net.eval()
feat_test_t = torch.tensor(norm_test.select(feature_cols).fill_nan(0.0).to_numpy(), dtype=torch.float32)
windows_test = feat_test_t.unfold(0, LOOKBACK, 1).permute(0, 2, 1).contiguous()
r_lin_test_t = torch.tensor(r_lin_test[LOOKBACK-1:], dtype=torch.float32).unsqueeze(1)
g6_test_t = torch.tensor(norm_test.select(g6_cols).fill_nan(0.0).to_numpy()[LOOKBACK-1:], dtype=torch.float32)

with torch.no_grad():
    outs = net(windows_test, r_lin_test_t, g6_test_t)
    res_pred = outs.residual_hat.numpy().squeeze(1)

r_full_pred = np.zeros(len(test_df))
r_full_pred[LOOKBACK-1:] = r_lin_test[LOOKBACK-1:] + res_pred

fm_reg = compute_regression_metrics(y_test, r_full_pred, rv_test, rv_test)
print(f"Full Model Test RMSE:  {fm_reg.rmse_r30:.6f}")
print(f"RMSE Reduction:        {(ba_reg.rmse_r30 - fm_reg.rmse_r30)/ba_reg.rmse_r30 * 100:+.2f}%")
