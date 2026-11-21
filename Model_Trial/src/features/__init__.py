# Phase 3 Feature Store: G1-G6 Microstructure, Technical, Regime, and Interaction Features

from .registry import REGISTRY, FeatureMeta, FeatureRegistry
from .microstructure import compute_g1_microstructure
from .momentum import compute_g2_momentum
from .indicators import compute_g3_indicators
from .interactions import compute_g4_interactions
from .time_regime import compute_g5_time_regime
from .volatility import compute_g6_volatility
from .pipeline import (
    compute_all_features,
    FeaturePipeline,
    run_gate3_audit,
    Gate3Report,
)

__all__ = [
    "REGISTRY",
    "FeatureMeta",
    "FeatureRegistry",
    "compute_g1_microstructure",
    "compute_g2_momentum",
    "compute_g3_indicators",
    "compute_g4_interactions",
    "compute_g5_time_regime",
    "compute_g6_volatility",
    "compute_all_features",
    "FeaturePipeline",
    "run_gate3_audit",
    "Gate3Report",
]
