"""
Phase 7 — Ablation Configuration & Switch Definitions.

Maps each ablation experiment to its whitepaper condition tag and architectural toggle flags:
  1. BASELINE_FULL: Full model (L=60, G1-G6, cascade, log-magnitude)
  2. NO_G4_INTERACTIONS: Condition C4 ([A3, A6, F3]) — remove cross-indicator interaction features
  3. NO_G5_TIME_REGIME: Condition C5 ([A2, D4]) — remove cyclical time & time×regime features
  4. INDEPENDENT_HEADS: Condition C6 ([E3, A7]) — decouple magnitude head from return estimate r̂30
  5. RAW_MAGNITUDE: Condition C7 ([E5]) — linear |r30| target instead of log(|r30| + ε)
  6. LOOKBACK_30: Short temporal lookback (L=30 bars = 2.5h)
  7. LOOKBACK_120: Extended temporal lookback (L=120 bars = 10h)
  8. BRANCH_A_ONLY: Standalone Ridge linear core (benchmark floor)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AblationConfig:
    """Configuration definition for an individual ablation experiment."""
    name: str
    condition_tag: str
    description: str
    include_g4: bool = True
    include_g5: bool = True
    use_cascade: bool = True
    log_magnitude_target: bool = True
    lookback: int = 60
    branch_b_enabled: bool = True


# Canonical Ablation Battery Table per Build Plan Phase 7
ABLATION_SUITE: Dict[str, AblationConfig] = {
    "BASELINE_FULL": AblationConfig(
        name="BASELINE_FULL",
        condition_tag="Reference",
        description="Full Composite Model with all 6 groups, cascade heads, log-mag target, L=60 lookback.",
        include_g4=True,
        include_g5=True,
        use_cascade=True,
        log_magnitude_target=True,
        lookback=60,
        branch_b_enabled=True,
    ),
    "NO_G4_INTERACTIONS": AblationConfig(
        name="NO_G4_INTERACTIONS",
        condition_tag="Condition C4 [A3, A6, F3]",
        description="Remove G4 explicit cross-indicator interactions and tail transforms.",
        include_g4=False,
        include_g5=True,
        use_cascade=True,
        log_magnitude_target=True,
        lookback=60,
        branch_b_enabled=True,
    ),
    "NO_G5_TIME_REGIME": AblationConfig(
        name="NO_G5_TIME_REGIME",
        condition_tag="Condition C5 [A2, D4]",
        description="Remove G5 cyclical time-of-day encodings and time×OFI interactions.",
        include_g4=True,
        include_g5=False,
        use_cascade=True,
        log_magnitude_target=True,
        lookback=60,
        branch_b_enabled=True,
    ),
    "INDEPENDENT_HEADS": AblationConfig(
        name="INDEPENDENT_HEADS",
        condition_tag="Condition C6 [E3, A7]",
        description="Independent output heads (magnitude head does NOT receive r̂30 directional estimate).",
        include_g4=True,
        include_g5=True,
        use_cascade=False,
        log_magnitude_target=True,
        lookback=60,
        branch_b_enabled=True,
    ),
    "RAW_MAGNITUDE": AblationConfig(
        name="RAW_MAGNITUDE",
        condition_tag="Condition C7 [E5]",
        description="Train magnitude head on raw linear |r30| instead of log-transformed target.",
        include_g4=True,
        include_g5=True,
        use_cascade=True,
        log_magnitude_target=False,
        lookback=60,
        branch_b_enabled=True,
    ),
    "LOOKBACK_30": AblationConfig(
        name="LOOKBACK_30",
        condition_tag="Lookback Tuning",
        description="Short lookback window L=30 bars (2.5 hours).",
        include_g4=True,
        include_g5=True,
        use_cascade=True,
        log_magnitude_target=True,
        lookback=30,
        branch_b_enabled=True,
    ),
    "LOOKBACK_120": AblationConfig(
        name="LOOKBACK_120",
        condition_tag="Lookback Tuning",
        description="Long lookback window L=120 bars (10 hours).",
        include_g4=True,
        include_g5=True,
        use_cascade=True,
        log_magnitude_target=True,
        lookback=120,
        branch_b_enabled=True,
    ),
    "BRANCH_A_ONLY": AblationConfig(
        name="BRANCH_A_ONLY",
        condition_tag="Condition C3 [D1, C1]",
        description="Linear Ridge core only without Branch B residual sequence model.",
        include_g4=False,
        include_g5=False,
        use_cascade=False,
        log_magnitude_target=True,
        lookback=60,
        branch_b_enabled=False,
    ),
}
