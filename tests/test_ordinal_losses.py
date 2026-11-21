"""Unit & Mathematical Property Tests for Plan 2 Ordinal & Continuous Loss Functions.

Verifies:
1. Ordinal transport penalty: Ranked Probability Score (RPS) penalizes far-away errors strictly more than adjacent errors.
2. Soft Gaussian target generator with fixed and per-sample dynamic volatility bandwidths.
3. Monotonic threshold invariants, probability simplex validity, and directional tail probabilities in CumulativeOrdinalHead.
4. Student-t parametric distribution NLL stability, moments estimation, and finite variance constraint (nu > 2).
5. Unimodality / log-concavity regularizer behavior on unimodal vs bimodal distributions.
6. MultiTaskCompositeLoss autograd gradient flow across all multi-task heads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import unittest

from src.ordinal_losses import (
    generate_gaussian_soft_targets,
    RankedProbabilityScoreLoss,
    EarthMoversDistanceLoss,
    SoftOrdinalCrossEntropyLoss,
    LogConcavityUnimodalityLoss,
    CumulativeOrdinalHead,
    StudentTDistributionHead,
    MultiTaskCompositeLoss,
)


class TestOrdinalLosses(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.bin_centers = torch.tensor([-2.5, -1.8, -1.0, -0.4, 0.0, 0.4, 1.0, 1.8, 2.5], dtype=torch.float32)

    def test_soft_gaussian_targets(self) -> None:
        targets = torch.tensor([0.0, 1.8, -2.5], dtype=torch.float32)
        q = generate_gaussian_soft_targets(targets, self.bin_centers, sigma=0.5)
        
        # Must be shape (3, 9) and sum to 1.0 (valid probability distribution)
        self.assertEqual(q.shape, (3, 9))
        torch.testing.assert_close(q.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
        
        # Max probability must align with the closest bin center
        self.assertEqual(int(q[0].argmax().item()), 4)  # 0.0 is center bin (index 4)
        self.assertEqual(int(q[1].argmax().item()), 7)  # 1.8 is index 7
        self.assertEqual(int(q[2].argmax().item()), 0)  # -2.5 is index 0

    def test_dynamic_volatility_soft_targets(self) -> None:
        targets = torch.tensor([0.0, 0.0], dtype=torch.float32)
        # Sample 0 has low vol (narrow target), Sample 1 has high vol (broad target)
        local_sigmas = torch.tensor([0.2, 1.5], dtype=torch.float32)
        q = generate_gaussian_soft_targets(targets, self.bin_centers, sigma=local_sigmas)
        
        # Low vol sample must be more concentrated on center bin than high vol sample
        self.assertGreater(q[0, 4].item(), q[1, 4].item())

    def test_rps_ordinal_penalty_monotonicity(self) -> None:
        """Core mathematical test: RPS must penalize predictions proportional to ordinal distance."""
        rps = RankedProbabilityScoreLoss()
        
        # True target is Bin 4 (neutral)
        target = torch.zeros(1, 9)
        target[0, 4] = 1.0
        
        # Prediction A: adjacent error (Bin 5)
        pred_adjacent = torch.zeros(1, 9)
        pred_adjacent[0, 5] = 1.0
        
        # Prediction B: far error (Bin 8 - extreme bullish)
        pred_far = torch.zeros(1, 9)
        pred_far[0, 8] = 1.0
        
        # Prediction C: opposite extreme error (Bin 0 - extreme bearish)
        pred_opposite = torch.zeros(1, 9)
        pred_opposite[0, 0] = 1.0
        
        loss_adj = rps(pred_adjacent, target).item()
        loss_far = rps(pred_far, target).item()
        loss_opp = rps(pred_opposite, target).item()
        
        self.assertLess(loss_adj, loss_far, "RPS failed: adjacent error scored worse than far error!")
        self.assertLess(loss_adj, loss_opp, "RPS failed: adjacent error scored worse than opposite error!")
        self.assertAlmostEqual(loss_far, loss_opp, places=4, msg="Symmetric distance should yield identical RPS.")

    def test_cumulative_ordinal_head_monotonicity(self) -> None:
        head = CumulativeOrdinalHead(in_features=32, num_bins=9)
        h = torch.randn(4, 32)
        
        p, cdf, score = head(h)
        
        # Output shapes
        self.assertEqual(p.shape, (4, 9))
        self.assertEqual(cdf.shape, (4, 8))
        self.assertEqual(score.shape, (4, 1))
        
        # Monotonicity check on thresholds
        thresholds = head.get_monotonic_thresholds()
        diffs = thresholds[1:] - thresholds[:-1]
        self.assertTrue((diffs > 0).all(), "Thresholds are not strictly monotonic!")
        
        # Probabilities sum to 1.0
        torch.testing.assert_close(p.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)
        
        # Directional tail probabilities
        p_bear, p_flat, p_bull = head.get_directional_tail_probabilities(cdf, neutral_idx=4)
        torch.testing.assert_close(p_bear + p_flat + p_bull, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_student_t_head(self) -> None:
        head = StudentTDistributionHead(in_features=32)
        h = torch.randn(4, 32)
        targets = torch.randn(4)
        
        mu, sigma, nu = head(h)
        self.assertTrue((sigma > 0).all(), "Sigma must be strictly positive!")
        self.assertTrue((nu > 2.0).all(), "Nu must be > 2.0 for finite variance!")
        
        exp_mu, exp_var = head.predict_moments(h)
        self.assertTrue((exp_var > 0).all(), "Theoretical variance must be positive!")
        
        nll = head.compute_continuous_nll(targets, mu, sigma, nu)
        self.assertFalse(torch.isnan(nll))
        self.assertFalse(torch.isinf(nll))
        
        nll.backward()
        self.assertIsNotNone(head.fc_mu.weight.grad)

    def test_unimodality_regularizer(self) -> None:
        uni_loss = LogConcavityUnimodalityLoss()
        
        # Smooth unimodal Gaussian distribution over 9 bins
        unimodal = torch.softmax(-0.5 * (self.bin_centers - 0.0) ** 2 / 0.5 ** 2, dim=-1).unsqueeze(0)
        
        # Bimodal distribution (two distinct peaks at index 1 and index 7)
        bimodal = torch.zeros(1, 9)
        bimodal[0, 1] = 0.45
        bimodal[0, 2] = 0.05
        bimodal[0, 3] = 0.00
        bimodal[0, 4] = 0.00
        bimodal[0, 5] = 0.00
        bimodal[0, 6] = 0.05
        bimodal[0, 7] = 0.45
        
        loss_uni = uni_loss(unimodal).item()
        loss_bi = uni_loss(bimodal).item()
        
        self.assertAlmostEqual(loss_uni, 0.0, places=4, msg="Unimodal Gaussian should have zero log-concavity penalty.")
        self.assertGreater(loss_bi, 0.1, "Bimodal distribution should receive significant log-concavity penalty.")

    def test_multi_task_composite_loss(self) -> None:
        composite = MultiTaskCompositeLoss(
            bin_centers=self.bin_centers,
            cost_hurdle=0.50,
            lambda_rps=0.5,
            lambda_reg=0.2,
            lambda_dir=0.2,
            lambda_uni=0.05,
        )
        
        logits = torch.randn(8, 9, requires_grad=True)
        pred_return = torch.randn(8, 1, requires_grad=True)
        dir_logits = torch.randn(8, 3, requires_grad=True)
        continuous_targets = torch.tensor([0.8, -0.6, 0.1, -0.1, 1.2, -1.5, 0.0, 0.45])
        
        total_loss, metrics = composite(logits, pred_return, dir_logits, continuous_targets)
        
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_rps", metrics)
        self.assertIn("loss_soft_ce", metrics)
        self.assertIn("loss_reg", metrics)
        self.assertIn("loss_dir", metrics)
        
        total_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(pred_return.grad)
        self.assertIsNotNone(dir_logits.grad)


if __name__ == "__main__":
    unittest.main()
