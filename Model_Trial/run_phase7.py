"""
Phase 7 Multi-Symbol Ablation Battery & Transfer Test Runner.

Executes the complete Phase 7 ablation battery across walk-forward folds to verify
Conditions C4-C7, test cross-instrument universality [B2], and freeze the optimal
surviving model configuration for Phase 8 serving.

Outputs:
  data/reports/phase7/ablation_{SYMBOL}.json
  data/reports/phase7/ablation_{SYMBOL}_report.md
  data/reports/phase7/surviving_config.json
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import polars as pl

from src.ablations import (
    AblationBatteryRunner,
    export_surviving_config,
    render_ablation_markdown_report,
    run_cross_instrument_transfer_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
FEATURES_DIR = Path("data/features/5min")
REPORT_DIR = Path("data/reports/phase7")


def _sep(char: str = "=", width: int = 72) -> None:
    print(char * width)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _sep()
    print("PHASE 7 — ABLATION BATTERY & UNIVERSALITY TESTING (CONDITIONS C4–C7)")
    _sep()

    loaded_dfs = {}
    for sym in SYMBOLS:
        feat_path = FEATURES_DIR / f"{sym}_5m_features.parquet"
        if feat_path.exists():
            loaded_dfs[sym] = pl.read_parquet(feat_path)
            print(f"  Loaded {sym}: {len(loaded_dfs[sym]):,} rows.")
        else:
            print(f"  ⚠️ {feat_path} not found.")

    if not loaded_dfs:
        print("❌ No feature datasets found in data/features/5min/. Exiting.")
        sys.exit(1)

    runner = AblationBatteryRunner(
        train_bars=80_000,
        test_bars=20_000,
        step_bars=20_000,
        embargo_bars=30,
        epochs=8,
        batch_size=256,
    )

    primary_symbol = "EURUSD" if "EURUSD" in loaded_dfs else list(loaded_dfs.keys())[0]
    print(f"\n[Running Full Ablation Battery on {primary_symbol}]...")

    battery_report = runner.run_battery(loaded_dfs[primary_symbol], symbol=primary_symbol)

    # ── Cross-Instrument Transfer Test [B2] ───────────────────────────────
    transfer_result = None
    if len(loaded_dfs) >= 2:
        target_sym = "XAUUSD" if "XAUUSD" in loaded_dfs else list(loaded_dfs.keys())[-1]
        source_dict = {k: v for k, v in loaded_dfs.items() if k != target_sym}
        print(f"\n[Running Universality Transfer Test: {list(source_dict.keys())} -> {target_sym}]...")

        transfer_result = run_cross_instrument_transfer_test(
            source_dfs=source_dict,
            target_symbol=target_sym,
            target_df=loaded_dfs[target_sym],
            lookback=battery_report.optimal_lookback,
            epochs=8,
            batch_size=256,
        )

    # ── Write Reports & Export Surviving Config ────────────────────────────
    md_content = render_ablation_markdown_report(battery_report, transfer_result)
    (REPORT_DIR / f"ablation_{primary_symbol}_report.md").write_text(md_content)
    (REPORT_DIR / f"ablation_{primary_symbol}.json").write_text(
        json.dumps(asdict(battery_report), indent=2)
    )

    export_surviving_config(battery_report, REPORT_DIR / "surviving_config.json")

    print("\n" + md_content)
    print(f"\n✅ Phase 7 Complete. Optimal profile frozen to {REPORT_DIR / 'surviving_config.json'}")


if __name__ == "__main__":
    main()
