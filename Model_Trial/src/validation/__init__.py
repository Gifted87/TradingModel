# Phase 5 — Purged Walk-Forward Validation Harness

from .walk_forward import WalkForwardSplit, generate_splits
from .metrics import (
    FoldMetrics,
    DirectionMetrics,
    RegressionMetrics,
    QuantileCalibration,
    compute_mcc,
    compute_balanced_accuracy,
    compute_qlike,
    compute_oos_r2,
    compute_direction_metrics,
    compute_regression_metrics,
    compute_quantile_calibration,
    compute_confidence_tier_breakdown,
    ConfidenceTierMetrics,
    QUANTILE_LEVELS,
)
from .regime_slicer import (
    RegimeReport,
    BucketMetrics,
    compute_regime_report,
    aggregate_regime_reports,
    TOD_BUCKET_NAMES,
    VOL_TERCILE_NAMES,
)
from .economic_gate import (
    EconomicGateReport,
    evaluate_economic_gate,
    extract_spreads,
)
from .adversarial import (
    AdversarialReport,
    ShuffleTestResult,
    GroupAuditResult,
    run_shuffle_test,
    run_group_audits,
    run_adversarial_checks,
)
from .harness import (
    Predictor,
    HarnessReport,
    Gate5Report,
    ValidationHarness,
)

__all__ = [
    # walk_forward
    "WalkForwardSplit",
    "generate_splits",
    # metrics
    "FoldMetrics",
    "DirectionMetrics",
    "RegressionMetrics",
    "QuantileCalibration",
    "compute_mcc",
    "compute_balanced_accuracy",
    "compute_qlike",
    "compute_oos_r2",
    "compute_direction_metrics",
    "compute_regression_metrics",
    "compute_quantile_calibration",
    "compute_confidence_tier_breakdown",
    "ConfidenceTierMetrics",
    "QUANTILE_LEVELS",
    # regime_slicer
    "RegimeReport",
    "BucketMetrics",
    "compute_regime_report",
    "aggregate_regime_reports",
    "TOD_BUCKET_NAMES",
    "VOL_TERCILE_NAMES",
    # economic_gate
    "EconomicGateReport",
    "evaluate_economic_gate",
    "extract_spreads",
    # adversarial
    "AdversarialReport",
    "ShuffleTestResult",
    "GroupAuditResult",
    "run_shuffle_test",
    "run_group_audits",
    "run_adversarial_checks",
    # harness
    "Predictor",
    "HarnessReport",
    "Gate5Report",
    "ValidationHarness",
]
