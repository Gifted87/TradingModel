"""
Phase 7 — Report Generator and Surviving Configuration Freezer.

Produces:
  1. JSON report of all ablation battery comparisons.
  2. Markdown report formatted for review.
  3. surviving_config.json: Frozen optimal model profile for Phase 8 serving.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import ABLATION_SUITE
from .runner import BatterySummaryReport
from .transfer import TransferTestResult


def render_ablation_markdown_report(
    report: BatterySummaryReport,
    transfer_result: Optional[TransferTestResult] = None,
) -> str:
    """Generate comprehensive Markdown report for Gate 7 verification."""
    status_emoji = "✅" if report.gate7_passed else "❌"

    rows = ""
    for v in report.variants:
        mcc_delta = f"{v.delta_mcc_vs_baseline:+.4f}" if v.name != "BASELINE_FULL" else "—"
        rows += (
            f"| `{v.name}` | {v.condition_tag} | `{v.mean_mcc:+.4f}` | "
            f"`{v.mean_hit_rate_tradable * 100:.2f}%` | `{v.mean_qlike_rv30:.4f}` | "
            f"`{mcc_delta}` | **{v.keep_decision}** |\n"
        )

    transfer_section = ""
    if transfer_result is not None:
        t_emoji = "✅" if transfer_result.transfer_successful else "⚠️"
        transfer_section = f"""
---

## 2. Universality & Cross-Instrument Transfer Test [B2]

**Transfer Status:** {t_emoji} **{'PASSED' if transfer_result.transfer_successful else 'WARNING'}**
- **Source Training Instruments:** `{', '.join(transfer_result.source_symbols)}` ({transfer_result.n_train_samples:,} samples)
- **Target Held-Out Instrument:** `{transfer_result.target_symbol}` ({transfer_result.n_test_samples:,} samples)
- **Zero-Shot Target MCC:** `{transfer_result.target_mcc:+.4f}`
- **Zero-Shot Tradable Hit-Rate:** `{transfer_result.target_hit_rate_tradable * 100:.2f}%`
- **Zero-Shot QLIKE (rv30):** `{transfer_result.target_qlike_rv30:.4f}`
- **Verdict:** `{transfer_result.verdict}`
"""

    return f"""# Gate 7 (Conditions C4–C7) Ablation Battery & Universality Report — {report.symbol}

**Gate Status:** {status_emoji} **{'PASSED' if report.gate7_passed else 'FAILED'}**
**Frozen Configuration Profile:** `{report.frozen_config_name}` (Lookback = `{report.optimal_lookback}` Bars)

---

## 1. Ablation Battery Matrix

| Variant Name | Condition / Hypothesis | Mean MCC | Tradable Hit-Rate | QLIKE (rv30) | $\\Delta$ MCC vs Base | Decision |
|---|---|---|---|---|---|---|
{rows}

---

## 2. Theoretical Invariant Confirmations [C4–C7]

- **Condition C4 (G4 Interaction Features [A3, A6, F3]):** {'✅ Verified' if report.condition_c4_verified else 'Neutral'}
- **Condition C5 (G5 Time-Regime Conditioning [A2, D4]):** {'✅ Verified' if report.condition_c5_verified else 'Neutral'}
- **Condition C6 (Cascade Output Architecture [E3, A7]):** {'✅ Verified' if report.condition_c6_verified else 'Neutral'}
- **Condition C7 (Log-Space Magnitude Transformation [E5]):** {'✅ Verified' if report.condition_c7_verified else 'Neutral'}
{transfer_section}
---

## 3. Verdict
**`{report.verdict}`**
"""


def export_surviving_config(
    report: BatterySummaryReport,
    output_path: Path,
) -> None:
    """Freeze optimal surviving configuration profile to JSON."""
    chosen_cfg = ABLATION_SUITE[report.frozen_config_name]
    config_dict = {
        "frozen_at_phase": "Phase 7 (Gate 7 Accepted)",
        "config_name": chosen_cfg.name,
        "symbol": report.symbol,
        "lookback_bars": report.optimal_lookback,
        "feature_groups": {
            "include_g1_microstructure": True,
            "include_g2_momentum": True,
            "include_g3_indicators": True,
            "include_g4_interactions": chosen_cfg.include_g4,
            "include_g5_time_regime": chosen_cfg.include_g5,
            "include_g6_volatility": True,
        },
        "architecture": {
            "use_branch_a_linear_core": True,
            "use_branch_b_sequence_model": chosen_cfg.branch_b_enabled,
            "use_cascade_heads": chosen_cfg.use_cascade,
            "log_magnitude_target": chosen_cfg.log_magnitude_target,
            "conv_channels": 32,
            "lstm_hidden": 32,
            "lstm_layers": 2,
            "dropout": 0.2,
        },
        "conditions_verified": {
            "C4_g4_interactions": report.condition_c4_verified,
            "C5_g5_time_regime": report.condition_c5_verified,
            "C6_cascade_heads": report.condition_c6_verified,
            "C7_log_magnitude": report.condition_c7_verified,
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config_dict, indent=2))
