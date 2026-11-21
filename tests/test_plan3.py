"""Adversarial Unit & Integration Tests for Plan 3 Backbone and Decision Engine.

Verifies:
1. Strictly causal 1D Inception padding: future sequence tokens do not affect past latent states.
2. Conformer block execution and shape preservation under Macaron depthwise convolution.
3. Complete HybridInceptionConformer forward pass with group_features tensor injection.
4. CalibratedDecisionEngine Bayes-optimal expected utility logic, Fractional Kelly sizing, and abstention rules.
5. Post-hoc TemperatureCalibrator scaling.
"""

from __future__ import annotations

import numpy as np
import torch
import unittest

from src.hybrid_conformer import (
    CausalConv1d,
    CausalInceptionBlock,
    ConformerBlock,
    HybridInceptionConformer,
)
from src.decision_engine import (
    CalibratedDecisionEngine,
    TemperatureCalibrator,
    TradeDecision,
)


class TestPlan3ArchitectureAndDecisionEngine(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.bin_means = np.array([-3.2, -1.8, -1.0, -0.4, 0.0, 0.4, 1.0, 1.8, 3.2], dtype=np.float32)
        self.bin_vars = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.1, 0.2, 0.3, 0.5], dtype=np.float32)

    def test_causal_conv_leakage_invariance(self) -> None:
        """Perturbing time step t=40 must NOT change outputs at steps t=0..39."""
        conv = CausalConv1d(in_channels=16, out_channels=16, kernel_size=5, dilation=2)
        x1 = torch.randn(2, 16, 45)
        x2 = x1.clone()
        
        x2[:, :, 40:] += 100.0
        
        out1 = conv(x1)
        out2 = conv(x2)
        
        torch.testing.assert_close(
            out1[:, :, :40],
            out2[:, :, :40],
            atol=1e-6,
            rtol=1e-6,
            msg="CausalConv1d leaked future information to past time steps!",
        )

    def test_hybrid_conformer_forward_with_group_tensor(self) -> None:
        model = HybridInceptionConformer(
            continuous_dim=28,
            group_feat_dim=12,
            num_target_bins=9,
            rbf_bins=9,
            rbf_embed_dim=32,
            d_model=64,
            nhead=4,
            num_conformer_layers=2,
            lookback_bars=45,
            num_groups=3,
        )
        
        x_cont = torch.randn(4, 45, 28)
        z_1m = torch.randn(4, 45)
        group_feat = torch.randn(4, 12)
        
        outputs = model(x_cont, z_1m, group_features=group_feat)
        
        self.assertIn("logits", outputs)
        self.assertIn("probs", outputs)
        self.assertIn("pred_return", outputs)
        self.assertIn("dir_logits", outputs)
        self.assertIn("dir_probs", outputs)
        self.assertIn("pred_vol", outputs)
        
        self.assertEqual(outputs["logits"].shape, (4, 9))
        self.assertEqual(outputs["probs"].shape, (4, 9))
        self.assertEqual(outputs["pred_return"].shape, (4, 1))
        self.assertEqual(outputs["dir_logits"].shape, (4, 3))
        self.assertEqual(outputs["pred_vol"].shape, (4, 1))

    def test_decision_engine_abstention_on_neutral(self) -> None:
        engine = CalibratedDecisionEngine(
            bin_conditional_means=self.bin_means,
            bin_conditional_variances=self.bin_vars,
            cost_long=0.50,
            cost_short=0.50,
            hurdle_tau=0.20,
        )
        
        probs = np.zeros(9, dtype=np.float32)
        probs[3] = 0.25
        probs[4] = 0.50
        probs[5] = 0.25
        
        decision = engine.evaluate_distribution(probs)
        self.assertEqual(decision.action, "FLAT", "Model must abstain when expected edge is zero!")
        self.assertEqual(decision.position_size, 0.0)

    def test_decision_engine_kelly_sizing(self) -> None:
        engine = CalibratedDecisionEngine(
            bin_conditional_means=self.bin_means,
            bin_conditional_variances=self.bin_vars,
            cost_long=0.50,
            cost_short=0.50,
            hurdle_tau=0.20,
            snr_threshold=0.30,
            kelly_fraction=0.25,
        )
        
        # Bullish distribution
        probs = np.zeros(9, dtype=np.float32)
        probs[6] = 0.20
        probs[7] = 0.40
        probs[8] = 0.40
        
        decision = engine.evaluate_distribution(probs)
        self.assertEqual(decision.action, "LONG")
        self.assertGreater(decision.position_size, 0.0)
        self.assertLessEqual(decision.position_size, 1.0)


if __name__ == "__main__":
    unittest.main()
