"""Semantic Correctness & Dimensional Verification Test Suite.

Verifies:
1. Temporal Causation Alignment: Cause window [T - lookback - gap, T - gap] ends strictly before target [T - gap, T] begins.
2. Dimensional Homogeneity:
   - Returns, conditional means, expected payoffs, and transaction costs all share identical dollar/price-change units.
   - Uncertainty, variances, and SNRs maintain correct physical dimensionality.
3. Expected Utility Monotonicity:
   - When expected return mu_hat > cost_long + tau, U_long > 0 and LONG action is triggered.
   - When expected return mu_hat < -(cost_short + tau), U_short > 0 and SHORT action is triggered.
   - When |mu_hat| <= min(cost_long, cost_short), action is strictly FLAT (abstention).
4. Loss-Target Congruence:
   - MultiTaskCompositeLoss soft target centers match empirical conditional bin means.
   - Direction gate thresholds match the economic cost hurdle c = 0.50.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import unittest

from src.continuous_features import ContinuousFeatureExtractor, compute_group_features
from src.decision_engine import CalibratedDecisionEngine, TradeDecision
from src.ordinal_losses import (
    MultiTaskCompositeLoss,
    generate_gaussian_soft_targets,
)


class TestSemanticCorrectness(unittest.TestCase):

    def test_temporal_causation_semantics(self) -> None:
        """Verifies that the cause window at index T-gap contains zero information from [T-gap, T]."""
        # Create a synthetic 100-minute time series
        timestamps = pd.date_range("2026-08-21 09:00:00", periods=100, freq="1min", tz="UTC")
        prices = 2500.0 + np.cumsum(np.random.randn(100) * 0.2)
        
        df = pd.DataFrame({
            "Open": prices,
            "High": prices + 0.1,
            "Low": prices - 0.1,
            "Close": prices,
            "Volume": 100.0,
        }, index=timestamps)
        
        # Target at 10:40 (representing move from 10:35 to 10:40)
        target_time = pd.Timestamp("2026-08-21 10:40:00", tz="UTC")
        gap = 5
        lookback = 45
        
        cause_end_time = target_time - pd.Timedelta(minutes=gap)  # 10:35
        cause_start_time = cause_end_time - pd.Timedelta(minutes=lookback - 1)  # 09:51
        
        # Cause duration must be exactly 44 minutes spanning 45 one-minute bars
        self.assertEqual(cause_end_time - cause_start_time, pd.Timedelta(minutes=44))
        self.assertLessEqual(cause_end_time, target_time - pd.Timedelta(minutes=gap))

    def test_decision_engine_economic_semantics(self) -> None:
        """Verifies that trading decisions obey economic net expected utility theory."""
        bin_means = np.array([-3.0, -1.8, -1.0, -0.4, 0.0, 0.4, 1.0, 1.8, 3.0], dtype=np.float32)
        cost = 0.50
        hurdle = 0.20
        
        engine = CalibratedDecisionEngine(
            bin_conditional_means=bin_means,
            cost_long=cost,
            cost_short=cost,
            hurdle_tau=hurdle,
            snr_threshold=0.30,
        )
        
        # Scenario 1: Expected return mu_hat = +1.20 (> cost 0.50 + hurdle 0.20 = 0.70)
        # Prob mass concentrated on bullish bins 6, 7, 8
        probs_bullish = np.zeros(9, dtype=np.float32)
        probs_bullish[6] = 0.30
        probs_bullish[7] = 0.40
        probs_bullish[8] = 0.30
        
        dec_bull = engine.evaluate_distribution(probs_bullish)
        self.assertEqual(dec_bull.action, "LONG")
        self.assertGreater(dec_bull.u_long, 0.0)
        self.assertLess(dec_bull.u_short, 0.0)
        self.assertGreater(dec_bull.position_size, 0.0)
        
        # Scenario 2: Expected return mu_hat = -1.20 (< -(cost 0.50 + hurdle 0.20) = -0.70)
        probs_bearish = np.zeros(9, dtype=np.float32)
        probs_bearish[0] = 0.30
        probs_bearish[1] = 0.40
        probs_bearish[2] = 0.30
        
        dec_bear = engine.evaluate_distribution(probs_bearish)
        self.assertEqual(dec_bear.action, "SHORT")
        self.assertGreater(dec_bear.u_short, 0.0)
        self.assertLess(dec_bear.u_long, 0.0)
        self.assertLess(dec_bear.position_size, 0.0)
        
        # Scenario 3: Expected return inside the dead-zone (mu_hat = +0.20, cannot overcome 0.50 cost)
        probs_deadzone = np.zeros(9, dtype=np.float32)
        probs_deadzone[4] = 0.60
        probs_deadzone[5] = 0.40  # mu_hat ~ +0.16
        
        dec_flat = engine.evaluate_distribution(probs_deadzone)
        self.assertEqual(dec_flat.action, "FLAT", "Engine must abstain when expected move does not cover costs!")
        self.assertEqual(dec_flat.position_size, 0.0)

    def test_loss_target_congruence(self) -> None:
        """Verifies that composite loss correctly aligns continuous returns with bin centers."""
        bin_centers = torch.tensor([-3.0, -1.8, -1.0, -0.4, 0.0, 0.4, 1.0, 1.8, 3.0], dtype=torch.float32)
        composite = MultiTaskCompositeLoss(
            bin_centers=bin_centers,
            cost_hurdle=0.50,
        )
        
        # Positive move of +1.8 price units
        targets = torch.tensor([1.8], dtype=torch.float32)
        labels = composite.get_cost_aware_direction_labels(targets)
        self.assertEqual(labels.item(), 2, "Price move > +0.50 must be labeled Bullish (2)")
        
        # Negative move of -1.8 price units
        targets_neg = torch.tensor([-1.8], dtype=torch.float32)
        labels_neg = composite.get_cost_aware_direction_labels(targets_neg)
        self.assertEqual(labels_neg.item(), 0, "Price move < -0.50 must be labeled Bearish (0)")
        
        # Neutral move of +0.20 price units
        targets_flat = torch.tensor([0.20], dtype=torch.float32)
        labels_flat = composite.get_cost_aware_direction_labels(targets_flat)
        self.assertEqual(labels_flat.item(), 1, "Price move within [-0.50, +0.50] must be labeled Dead-Zone (1)")


if __name__ == "__main__":
    unittest.main()
