# Phase 7 — Ablation Battery & Universality Testing

from .config import AblationConfig, ABLATION_SUITE
from .variants import VariantNetwork, VariantSession, AblationModelVariant
from .transfer import TransferTestResult, run_cross_instrument_transfer_test
from .runner import (
    VariantComparison,
    BatterySummaryReport,
    AblationBatteryRunner,
)
from .report import (
    render_ablation_markdown_report,
    export_surviving_config,
)

__all__ = [
    "AblationConfig",
    "ABLATION_SUITE",
    "VariantNetwork",
    "VariantSession",
    "AblationModelVariant",
    "TransferTestResult",
    "run_cross_instrument_transfer_test",
    "VariantComparison",
    "BatterySummaryReport",
    "AblationBatteryRunner",
    "render_ablation_markdown_report",
    "export_surviving_config",
]
