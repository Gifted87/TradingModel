"""
Phase 6 — Sequence Dataset, Target Scaler, and Fast Batch Collation.

REWRITE (v2) — fixes the defects that caused training failure and compute burn:

1. TARGET SCALE MISMATCH (root cause of training failure).
   On 5m bars r30 ~ N(0, ~5e-4), so Huber(r30) ~= 1e-7 while the magnitude /
   QLIKE / pinball terms are O(1) — the direction task received ~0.000002% of
   the total loss and early stopping was blind to it. All targets are now
   standardized with TRAIN-fitted statistics (TargetScaler):
     - r30, mag30 : z-scored       (Huber is scale-sensitive via beta)
     - rv30       : centered only  (QLIKE is shift-invariant:
                     exp((u-c)-(v-c)) - ((u-c)-(v-c)) - 1 == QLIKE(u, v);
                     scaling would break the proper scoring rule)
     - |r30|      : scaled only    (pinball is equivariant under positive
                     scaling: pinball(q, a*y, a*yhat) = a*pinball(q, y, yhat);
                     centering would break quantile positivity)
   r_hat_lin is standardized with the r30 stats before entering the network,
   so r30_hat_std = r_hat_lin_std + residual_hat stays in standardized space;
   predictions are inverse-transformed at inference (run_network_inference).

2. PER-SAMPLE PYTHON COLLATION (root cause of data-loading burn).
   Windows are a single zero-copy strided view; collation gathers one batch
   at a time instead of building 8 tensors per sample.

3. MULTI-SYMBOL SAFETY (transfer test). ConcatSequenceDataset builds
   sequences strictly within one symbol — no window crosses symbol boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from ..features.pipeline import FeaturePipeline, _SKIP_NORMALISE


# ─────────────────────────────────────────────────────────────────────────────
# Target scaler (train-fitted; the fix for the loss-scale mismatch)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetScaler:
    r30_mean: float
    r30_std: float
    mag30_mean: float
    mag30_std: float
    rv30_shift: float      # center ONLY (QLIKE shift-invariance)
    abs30_scale: float     # scale ONLY (pinball equivariance, positivity)

    @classmethod
    def fit(cls, r30: np.ndarray, rv30: np.ndarray, mag30: np.ndarray) -> "TargetScaler":
        def _ms(x: np.ndarray) -> Tuple[float, float]:
            x = np.asarray(x, dtype=np.float64)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                return 0.0, 1.0
            s = float(x.std())
            return float(x.mean()), (s if s > 1e-12 else 1.0)

        r_m, r_s = _ms(r30)
        m_m, m_s = _ms(mag30)
        rv_m, _ = _ms(rv30)
        abs_r = np.abs(np.asarray(r30, dtype=np.float64))
        abs_r = abs_r[np.isfinite(abs_r)]
        a_s = float(abs_r.std()) if len(abs_r) and abs_r.std() > 1e-12 else 1.0
        return cls(r_m, r_s, m_m, m_s, rv_m, a_s)

    # forward (numpy in, numpy out)
    def std_r30(self, x):      return (x - self.r30_mean) / self.r30_std
    def std_rhat_lin(self, x): return (x - self.r30_mean) / self.r30_std
    def std_mag30(self, x):    return (x - self.mag30_mean) / self.mag30_std
    def ctr_rv30(self, x):     return x - self.rv30_shift
    def scl_abs30(self, x):    return x / self.abs30_scale
    # inverse
    def inv_r30(self, x):      return x * self.r30_std + self.r30_mean
    def inv_mag30(self, x):    return x * self.mag30_std + self.mag30_mean
    def inv_rv30(self, x):     return x + self.rv30_shift
    def inv_abs30(self, x):    return x * self.abs30_scale


# ─────────────────────────────────────────────────────────────────────────────
# Batch container (unchanged field set — loss.py and trainer stay compatible)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchData:
    sequences: torch.Tensor       # (B, L, F) float32, standardized features
    r_hat_lin: torch.Tensor       # (B, 1) float32, STANDARDIZED Branch A forecast
    g6_features: torch.Tensor     # (B, F_g6) float32
    r30: torch.Tensor             # (B, 1) float32, standardized
    residual: torch.Tensor        # (B, 1) float32, standardized
    mag30: torch.Tensor           # (B, 1) float32, standardized
    rv30: torch.Tensor            # (B, 1) float32, centered
    abs_r30: torch.Tensor         # (B, 1) float32, scaled


# ─────────────────────────────────────────────────────────────────────────────
# Fast sequence dataset
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Trailing L-bar sequences over standardized features.

    __getitem__ returns the integer end-bar index; `collate` performs a single
    batched gather from the strided window view. Pass `collate_fn=ds.collate`
    to the DataLoader (works through torch Subset, since subset items are the
    same end indices).
    """

    def __init__(
        self,
        features: np.ndarray,          # (N, F) normalized feature matrix
        r_hat_lin: np.ndarray,         # (N,) Branch A forecasts, ORIGINAL space
        g6_features: np.ndarray,       # (N, F_g6)
        r30: np.ndarray,               # (N,) raw targets (NaN allowed; filtered)
        rv30: np.ndarray,
        mag30: np.ndarray,
        lookback: int = 60,
        scaler: Optional[TargetScaler] = None,
        end_indices: Optional[np.ndarray] = None,
    ):
        features = np.asarray(features)
        assert features.ndim == 2
        n = len(features)
        if n < lookback:
            raise ValueError(f"SequenceDataset: n={n} < lookback={lookback}")

        self.lookback = int(lookback)
        self.scaler = scaler or TargetScaler.fit(r30, rv30, mag30)

        feats32 = np.ascontiguousarray(features, dtype=np.float32)
        # (N-L+1, L, F) zero-copy strided view
        self.windows = sliding_window_view(feats32, (self.lookback, feats32.shape[1])).squeeze(1)

        s = self.scaler
        self.r_hat_lin = s.std_rhat_lin(np.nan_to_num(r_hat_lin, nan=0.0)).astype(np.float32)
        self.g6 = np.nan_to_num(g6_features, nan=0.0).astype(np.float32)
        r30s = s.std_r30(r30).astype(np.float32)
        self.r30 = r30s
        self.residual = (r30s - self.r_hat_lin).astype(np.float32)
        self.mag30 = s.std_mag30(mag30).astype(np.float32)
        self.rv30 = s.ctr_rv30(rv30).astype(np.float32)
        self.abs30 = s.scl_abs30(np.abs(np.asarray(r30, dtype=np.float64))).astype(np.float32)

        if end_indices is None:
            valid = np.isfinite(r30) & np.isfinite(rv30) & np.isfinite(mag30)
            idx = np.flatnonzero(valid)
        else:
            idx = np.asarray(end_indices, dtype=np.int64)
        self.end_indices = idx[idx >= (self.lookback - 1)].astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.end_indices))

    def __getitem__(self, i: int) -> int:
        return int(self.end_indices[i])

    def gather(self, ends: np.ndarray) -> BatchData:
        ends = np.asarray(ends, dtype=np.int64)
        w = ends - (self.lookback - 1)
        seqs = torch.from_numpy(np.ascontiguousarray(self.windows[w]))

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(arr[ends]))

        return BatchData(
            sequences=seqs,
            r_hat_lin=t(self.r_hat_lin).unsqueeze(-1),
            g6_features=t(self.g6),
            r30=t(self.r30).unsqueeze(-1),
            residual=t(self.residual).unsqueeze(-1),
            mag30=t(self.mag30).unsqueeze(-1),
            rv30=t(self.rv30).unsqueeze(-1),
            abs_r30=t(self.abs30).unsqueeze(-1),
        )

    def collate(self, batch: Sequence[int]) -> BatchData:
        return self.gather(np.asarray(batch, dtype=np.int64))


class ConcatSequenceDataset(Dataset):
    """
    Concatenates per-symbol SequenceDatasets WITHOUT ever building a window
    that crosses a symbol boundary (fixes the transfer-test interleaving bug).
    Items are (ds_id, end_bar_index); `collate` groups by symbol and gathers.
    """

    def __init__(self, datasets: Sequence[SequenceDataset]):
        assert len(datasets) > 0
        self.datasets = list(datasets)
        self._offsets = np.cumsum([0] + [len(d) for d in self.datasets])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, i: int) -> Tuple[int, int]:
        ds_id = int(np.searchsorted(self._offsets, i, side="right") - 1)
        local = i - int(self._offsets[ds_id])
        return (ds_id, int(self.datasets[ds_id].end_indices[local]))

    def collate(self, batch: Sequence[Tuple[int, int]]) -> BatchData:
        by_ds: dict[int, List[int]] = {}
        for ds_id, end in batch:
            by_ds.setdefault(ds_id, []).append(end)
        parts = [self.datasets[d].gather(np.asarray(e, dtype=np.int64))
                 for d, e in sorted(by_ds.items())]
        if len(parts) == 1:
            return parts[0]
        return BatchData(*[
            torch.cat([getattr(p, f.name) for p in parts], dim=0)
            for f in fields(BatchData)
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Shared inference helper (used by full_model, ablation variants, transfer)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_network_inference(
    network,
    features: np.ndarray,          # (N, F) normalized
    r_hat_lin_raw: np.ndarray,     # (N,) Branch A forecasts, ORIGINAL space
    g6: np.ndarray,                # (N, F_g6)
    scaler: TargetScaler,
    lookback: int,
    batch_size: int,
    device,
) -> dict:
    """
    Batched forward pass over all valid end-bars. Returns dict of arrays in
    ORIGINAL target space; rows < lookback-1 are NaN (caller fills warmup).
    """
    n = len(features)
    nq = int(getattr(network, "n_quantiles", 9))
    out = {
        "residual": np.full(n, np.nan, dtype=np.float64),
        "r30": np.full(n, np.nan, dtype=np.float64),
        "mag": np.full(n, np.nan, dtype=np.float64),
        "rv": np.full(n, np.nan, dtype=np.float64),
        "quantiles": np.full((n, nq), np.nan, dtype=np.float64),
    }
    if n < lookback:
        return out

    feats32 = np.ascontiguousarray(features, dtype=np.float32)
    windows = sliding_window_view(feats32, (lookback, feats32.shape[1])).squeeze(1)
    r_lin_std = scaler.std_rhat_lin(np.nan_to_num(r_hat_lin_raw, nan=0.0)).astype(np.float32)
    g6_32 = np.nan_to_num(g6, nan=0.0).astype(np.float32)

    network.eval()
    ends_all = np.arange(lookback - 1, n)
    for i in range(0, len(ends_all), batch_size):
        ends = ends_all[i : i + batch_size]
        w = ends - (lookback - 1)
        x = torch.from_numpy(np.ascontiguousarray(windows[w])).to(device)
        rl = torch.from_numpy(r_lin_std[ends, None]).to(device)
        g6t = torch.from_numpy(g6_32[ends]).to(device)

        o = network(x, rl, g6t)
        res_std = o.residual_hat.squeeze(-1).cpu().numpy()
        r30_std = r_lin_std[ends] + res_std

        out["residual"][ends] = res_std * scaler.r30_std          # residual in return units
        out["r30"][ends] = scaler.inv_r30(r30_std)
        out["mag"][ends] = scaler.inv_mag30(o.mag30_hat.squeeze(-1).cpu().numpy())
        out["rv"][ends] = scaler.inv_rv30(o.rv30_hat.squeeze(-1).cpu().numpy())
        out["quantiles"][ends] = scaler.inv_abs30(o.quantiles_hat.cpu().numpy())

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder (unchanged semantics; float32 output)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrices(
    df: pl.DataFrame,
    pipeline: FeaturePipeline,
    linear_core=None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract normalized feature matrices, Branch A predictions, and G6 features.

    Returns (features_matrix float32, r_hat_lin, g6_matrix float32,
             active_feature_cols, g6_cols).
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c.startswith("g") and "_" in c and c not in _SKIP_NORMALISE
        ]
    g6_cols = [c for c in feature_cols if c.startswith("g6_")]

    norm_df = pipeline.transform(df)
    features_mat = (
        norm_df.select(feature_cols).fill_nan(0.0).fill_null(0.0)
        .to_numpy().astype(np.float32)
    )
    g6_mat = (
        norm_df.select(g6_cols).fill_nan(0.0).fill_null(0.0)
        .to_numpy().astype(np.float32)
    )

    if linear_core is not None:
        r_hat_lin = linear_core.predict(df)
    else:
        r_hat_lin = np.zeros(len(df), dtype=np.float64)

    return features_mat, r_hat_lin, g6_mat, feature_cols, g6_cols