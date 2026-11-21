"""
Reflexivity Reversal Model Runner (George Soros FSM & RL Agent).

Executes the 4-Stage Reflexivity Reversal Model and Purged Walk-Forward Evaluation
across all feature/bar datasets in data/features/5min/ (or data/bars/1min/):
  1. Extreme: Statistical undervaluation/overvaluation (|z| >= 2.5 vs 4h VWAP + 60-bar pivot)
  2. Stimulant: Confluence of 5 perception-shift signals (OFI divergence, RSI hook, volume climax+dryup, spread normalization, momentum acceleration)
  3. Confirmation: Microstructure break + >= 0.5*ATR move off pivot + OFI follow-through + Economic gate expectancy
  4. Entry & Management: Stop at pivot +/- 0.25*ATR, target = VWAP equilibrium, 30-min time stop

Symbols: EURUSD, GBPUSD, USDJPY, XAUUSD
Reports: data/reports/reflexivity/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import polars as pl

from src.reflexivity import (
    RLReflexivityPredictor,
    ReflexivityConfig,
    run_reflexivity,
)
from src.validation.harness import ValidationHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
FEATURES_DIR = Path("data/features/5min")
REPORT_DIR = Path("data/reports/reflexivity")
RL_EPISODES = 20
RL_BATCH_SIZE = 64


def _sep(char: str = "=", width: int = 78) -> None:
    print(char * width)


def get_config_for_df(df: pl.DataFrame) -> ReflexivityConfig:
    """Auto-scale window lengths based on estimated bar interval."""
    # Check if data is 5m (approx. 5 min between bars) or 1m
    is_5m = True
    if "timestamp" in df.columns and len(df) > 1:
        diffs = (df["timestamp"].slice(1, 10) - df["timestamp"].slice(0, 10)).dt.total_seconds()
        median_sec = float(diffs.median() or 300.0)
        if median_sec <= 90:
            is_5m = False

    if is_5m:
        # 5m bars: 48 bars = 4h VWAP, 12 bars = 1h extreme lookback, 6 bars = 30m time stop
        return ReflexivityConfig(
            vwap_window=48,
            z_window=48,
            z_entry=2.5,
            low_lookback=12,
            min_confluence=3,
            climax_window=48,
            time_stop_bars=6,
            min_target_r_multiple=1.0,
            max_armed_bars=12,
            max_primed_bars=4,
            cooldown_bars=3,
            structure_lookback=2,
            min_move_atr=0.4,
            econ_gate_k=1.2,
        )
    else:
        # 1m default config
        return ReflexivityConfig()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _sep()
    print("GEORGE SOROS REFLEXIVITY REVERSAL MODEL (FSM & RL EVALUATION)")
    _sep()

    summary_rows = []
    any_failure = False

    for symbol in SYMBOLS:
        feat_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"

        _sep()
        print(f"[{symbol}]")

        if not feat_path.exists():
            print(f"  ⚠️  Feature file not found: {feat_path} — SKIPPED")
            summary_rows.append({
                "symbol": symbol,
                "n_trades": "—",
                "hit_rate": "—",
                "profit_factor": "—",
                "expectancy_log": "—",
                "expectancy_spreads": "—",
                "max_dd": "—",
                "status": "SKIPPED",
            })
            continue

        try:
            df = pl.read_parquet(feat_path)
            print(f"  Loaded {len(df):,} rows.")
        except Exception as exc:
            print(f"  ❌ Failed to load dataset: {exc}")
            any_failure = True
            continue

        cfg = get_config_for_df(df)
        sym_dir = REPORT_DIR / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)

        # 1. Full dataset Reflexivity Simulation
        try:
            result = run_reflexivity(df, cfg)
            summary = result.summary
            trades = result.trades

            # Save full signals and trades
            if not trades.is_empty():
                trades.write_parquet(sym_dir / f"reflexivity_{symbol}_trades.parquet")
            (sym_dir / f"reflexivity_{symbol}_summary.json").write_text(
                json.dumps(summary, indent=2)
            )

            n_trades = summary.get("n_trades", 0)
            hit_rate = summary.get("hit_rate", 0.0)
            profit_factor = summary.get("profit_factor", 0.0)
            exp_net_log = summary.get("expectancy_net_log", 0.0)
            exp_spreads = summary.get("expectancy_in_spreads", 0.0)
            max_dd = summary.get("max_drawdown_log", 0.0)
            pf_str = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"

            print(f"  Trades Generated   : {n_trades:,}")
            print(f"  Hit Rate (Win %)   : {hit_rate * 100:.1f}%")
            print(f"  Profit Factor      : {pf_str}")
            print(f"  Net Expectancy     : {exp_net_log:+.6f} log ({exp_spreads:+.2f} spreads)")
            print(f"  Max Drawdown (log) : {max_dd:.4f}")

            status = "PASS ✅" if n_trades > 0 and exp_net_log > 0 else ("ACTIVE ⚡" if n_trades > 0 else "NO SIGNALS ⚠️")

            print("  RL Walk-Forward  : training DQN gate...")
            harness = ValidationHarness(
                train_bars=80_000,
                test_bars=20_000,
                step_bars=20_000,
                embargo_bars=cfg.time_stop_bars,
                economic_gate_k=cfg.econ_gate_k,
                run_adversarial=False,
                decision_bars_only=False,
            )
            rl_predictor = RLReflexivityPredictor(
                cfg,
                episodes=RL_EPISODES,
                batch_size=RL_BATCH_SIZE,
                seed=42,
            )
            rl_report = harness.evaluate(
                df,
                rl_predictor,
                predictor_name="RLReflexivityDQN",
                symbol=symbol,
                report_dir=sym_dir,
            )
            print(f"  RL Folds          : {rl_report.total_folds}")
            print(f"  RL Mean MCC       : {rl_report.mean_mcc:+.4f}")
            print(f"  RL Net-PnL Folds  : {rl_report.pct_folds_positive_net_pnl:.0%}")

            summary_rows.append({
                "symbol": symbol,
                "n_trades": f"{n_trades:,}",
                "hit_rate": f"{hit_rate * 100:.1f}%",
                "profit_factor": pf_str,
                "expectancy_log": f"{exp_net_log:+.6f}",
                "expectancy_spreads": f"{exp_spreads:+.2f}x",
                "max_dd": f"{max_dd:.4f}",
                "rl_folds": str(rl_report.total_folds),
                "rl_mcc": f"{rl_report.mean_mcc:+.4f}",
                "rl_pnl_folds": f"{rl_report.pct_folds_positive_net_pnl:.0%}",
                "status": status,
            })
        except Exception as exc:
            print(f"  ❌ Reflexivity evaluation failed: {exc}")
            import traceback
            traceback.print_exc()
            any_failure = True

    # ── Summary Matrix ─────────────────────────────────────────────────────
    print()
    _sep()
    print("REFLEXIVITY REVERSAL MODEL SUMMARY MATRIX")
    _sep()
    col_w = [10, 10, 10, 15, 16, 12, 10, 10, 12, 14]
    headers = [
        "Symbol", "Trades", "Hit-Rate", "Profit Factor", "Net Exp (Log)",
        "In Spreads", "RL Folds", "RL MCC", "RL PnL+", "Status",
    ]
    fmt = "".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-" * sum(col_w))
    for r in summary_rows:
        print(fmt.format(
            r["symbol"], r["n_trades"], r["hit_rate"], r["profit_factor"],
            r["expectancy_log"], r["expectancy_spreads"], r.get("rl_folds", "—"),
            r.get("rl_mcc", "—"), r.get("rl_pnl_folds", "—"), r["status"]
        ))
    _sep()
    print(f"\n✅ Reflexivity evaluation finished. Reports saved to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
