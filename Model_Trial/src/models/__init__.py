# Phase 6 — The Residual Learner (Branch B & Cascade Heads)

from .dataset import (
    BatchData,
    TargetScaler,
    SequenceDataset,
    ConcatSequenceDataset,
    build_feature_matrices,
    run_network_inference,
)
from .network import (
    ModelOutputs,
    MultiWidthConvBlock,
    AttentionPooling,
    ResidualLearnerNetwork,
)
from .loss import (
    LossBreakdown,
    CompositeMultiTaskLoss,
    QUANTILE_GRID,
)
from .trainer import (
    TrainerConfig,
    EpochMetrics,
    ResidualLearnerTrainer,
)
from .full_model import (
    CompositeModelPredictions,
    CompositeTradingModel,
)
from .evaluator import (
    Gate6FoldComparison,
    Gate6Report,
    Gate6Evaluator,
)

__all__ = [
    # dataset
    "BatchData",
    "TargetScaler",
    "SequenceDataset",
    "ConcatSequenceDataset",
    "build_feature_matrices",
    "run_network_inference",
    # network
    "ModelOutputs",
    "MultiWidthConvBlock",
    "AttentionPooling",
    "ResidualLearnerNetwork",
    # loss
    "LossBreakdown",
    "CompositeMultiTaskLoss",
    "QUANTILE_GRID",
    # trainer
    "TrainerConfig",
    "EpochMetrics",
    "ResidualLearnerTrainer",
    # full_model
    "CompositeModelPredictions",
    "CompositeTradingModel",
    # evaluator
    "Gate6FoldComparison",
    "Gate6Report",
    "Gate6Evaluator",
]
