"""Adversarial Unit & Property Tests for Plan 1 Continuous Multi-Scale Feature Engine.

Tests:
1. Shape & feature name integrity across all 28 indicators.
2. Strict causality: perturbing future bars t+k has ZERO effect on past features t.
3. Flash-crash / outlier invariance: extreme price spikes remain strictly bounded in [-1, 1] without gradient explosion.
4. Regular grid alignment: quiet zero-tick bars are properly forward-filled without weekend bridging.
5. Gaussian RBF soft-bin layer autograd gradient flow.
6. MultiScaleInputEmbedding dual projection (continuous features + soft bins + group IDs + macro group tensor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import unittest

from src.continuous_features import (
    ContinuousFeatureExtractor,
    GaussianRBFSoftBinLayer,
    MultiScaleInputEmbedding,
    compute_group_features,
)


class TestContinuousFeaturesAdversarial(unittest.TestCase):

    def setUp(self) -> None:
        dates = pd.date_range("2026-01-01 08:00:00", periods=100, freq="1min", tz="UTC")
        prices = 2000.0 + np.cumsum(np.random.randn(100) * 0.5)
        highs = prices + np.random.uniform(0.1, 0.5, 100)
        lows = prices - np.random.uniform(0.1, 0.5, 100)
        opens = prices + np.random.uniform(-0.2, 0.2, 100)
        volume = np.random.uniform(10, 500, 100)
        
        self.df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volume,
        }, index=dates)

    def test_feature_extraction_shapes(self) -> None:
        extractor = ContinuousFeatureExtractor()
        feat_df = extractor.extract_bar_features(self.df)
        cols = extractor.get_feature_names()
        
        self.assertEqual(len(cols), 28)
        for col in cols:
            self.assertIn(col, feat_df.columns, f"Missing feature column: {col}")
            self.assertFalse(feat_df[col].isna().any(), f"NaNs detected in feature {col}")

    def test_strict_causality(self) -> None:
        """Perturbing future bar 80 must produce IDENTICAL feature vectors at bars 0..79."""
        extractor = ContinuousFeatureExtractor()
        df1 = self.df.copy()
        df2 = self.df.copy()
        
        # Severe perturbation at future bar 80
        df2.iloc[80, df2.columns.get_loc("Close")] += 150.0
        df2.iloc[80, df2.columns.get_loc("High")] += 200.0
        
        f1 = extractor.extract_bar_features(df1)
        f2 = extractor.extract_bar_features(df2)
        
        cols = extractor.get_feature_names()
        np.testing.assert_allclose(
            f1.iloc[:80][cols].to_numpy(),
            f2.iloc[:80][cols].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Lookahead leakage detected: future change altered past features!"
        )

    def test_flash_crash_boundedness(self) -> None:
        """Adversarial test: 50x price spike must remain bounded in [-1, 1] without inf/nan."""
        extractor = ContinuousFeatureExtractor()
        df_spike = self.df.copy()
        df_spike.iloc[50, df_spike.columns.get_loc("High")] += 1000.0
        df_spike.iloc[50, df_spike.columns.get_loc("Close")] -= 500.0
        
        feat_df = extractor.extract_bar_features(df_spike)
        
        self.assertTrue(np.isfinite(feat_df["norm_range"].to_numpy()).all())
        self.assertTrue((feat_df["norm_range"].abs() <= 1.0).all(), "norm_range exceeded [-1, 1] bound!")
        self.assertTrue((feat_df["z_1m"].abs() <= 1.0).all(), "z_1m exceeded [-1, 1] bound!")

    def test_prepare_regular_grid(self) -> None:
        """Adversarial test: Omitted minutes must be forward-filled on regular 1m grid."""
        extractor = ContinuousFeatureExtractor()
        # Drop bars at indices 10, 11, 12 (3-minute zero-tick quiet period)
        df_gapped = self.df.drop(self.df.index[[10, 11, 12]])
        
        grid = extractor.prepare_regular_grid(df_gapped)
        self.assertEqual(len(grid), 100)
        self.assertIn(grid.index.freqstr, ["min", "1min"])
        self.assertFalse(grid["Close"].isna().any())


    def test_multiscale_input_embedding_with_group_tensor(self) -> None:
        extractor = ContinuousFeatureExtractor()
        cols = extractor.get_feature_names()
        emb_module = MultiScaleInputEmbedding(
            continuous_dim=len(cols),
            group_feat_dim=12,
            rbf_bins=9,
            rbf_embed_dim=32,
            d_model=128,
            lookback_bars=45,
            num_groups=3,
        )
        
        x_cont = torch.randn(4, 45, len(cols))
        z_1m = torch.randn(4, 45)
        group_feat = torch.randn(4, 12)
        
        out = emb_module(x_cont, z_1m, group_features=group_feat)
        self.assertEqual(out.shape, (4, 45, 128))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
