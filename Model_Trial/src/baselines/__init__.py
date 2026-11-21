# Phase 4 Baselines and Linear Core (Branch A)

from .persistence import PersistenceBaseline, PersistencePredictions
from .har_rv import HARRVBaseline, HARModelSummary
from .linear_core import LinearCoreBranchA, LinearCoreSummary
from .evaluator import (
    BaselineEvaluator,
    FoldMetrics,
    Gate4Report,
    generate_walk_forward_splits,
    compute_qlike,
)

__all__ = [
    "PersistenceBaseline",
    "PersistencePredictions",
    "HARRVBaseline",
    "HARModelSummary",
    "LinearCoreBranchA",
    "LinearCoreSummary",
    "BaselineEvaluator",
    "FoldMetrics",
    "Gate4Report",
    "generate_walk_forward_splits",
    "compute_qlike",
]
