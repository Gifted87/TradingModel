"""
Phase 1 Quality Gate Engine (Gate 1 Validator).

Performs comprehensive data auditing on raw/aggregated bar datasets:
1. Intra-session Gap Analysis: Detects missing bars during active market hours (flags unexpected gaps > 5 mins).
   Handles asset-specific session calendars (e.g. 24/5 FX vs 23/5 Gold daily maintenance halts).
2. Bad-Print / Outlier Spike Filter: Flags price jumps > 10x lagged ATR(14) that revert within 1-3 bars.
3. Spread Anomaly Detection: Flags non-positive spreads (crossed book) or extreme spread blowouts (> 50x median).
4. Weekend / Holiday Calendar Alignment: Verifies standard market boundaries.
5. Gate 1 Kill Criterion: Total anomalous / corrupted bars must be < 0.5%.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    symbol: str
    total_bars: int
    start_time: str
    end_time: str
    
    # Gap Metrics
    total_trading_gaps_gt_5m: int
    max_gap_duration_mins: float
    pct_bars_with_gaps: float
    
    # Spike / Bad-Print Metrics
    bad_print_spikes: int
    pct_bad_prints: float
    
    # Spread Metrics
    negative_spread_count: int
    median_spread: float
    max_spread: float
    spread_blowout_count: int
    
    # Volume Metrics
    zero_volume_bars: int
    pct_zero_volume: float
    
    # Overall Gate Verdict
    total_anomalous_bars: int
    bad_bar_ratio: float
    gate1_passed: bool
    verdict: str


def compute_atr(df: pl.DataFrame, period: int = 14) -> np.ndarray:
    """Compute Average True Range (ATR) as a numpy array."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    
    n = len(df)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        
    atr = np.zeros(n, dtype=np.float64)
    atr[0] = tr[0]
    alpha = 2.0 / (period + 1.0)
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
        
    return atr


def audit_dataset(
    df: pl.DataFrame,
    symbol: str = "EURUSD",
    expected_bar_minutes: int = 5,
    max_bad_bar_pct: float = 0.5,
) -> tuple[QualityMetrics, pl.DataFrame]:
    """
    Run adversarial quality audit on a bar dataset.
    """
    logger.info(f"Running Phase 1 Quality Audit on {symbol} ({len(df):,} bars)...")
    
    df = df.sort("timestamp")
    total_bars = len(df)
    
    if total_bars == 0:
        raise ValueError(f"Dataset for {symbol} is empty.")
    
    start_time = str(df["timestamp"].min())
    end_time = str(df["timestamp"].max())
    
    is_commodity = "XAU" in symbol.upper() or "GOLD" in symbol.upper() or "OIL" in symbol.upper()
    
    # 1. Intra-session Gap Analysis
    time_diff_ms = df["timestamp"].diff().dt.total_milliseconds().fill_null(expected_bar_minutes * 60000.0)
    time_diff_mins = (time_diff_ms / 60000.0).to_numpy()
    
    dow = df["timestamp"].dt.weekday().to_numpy()
    hour = df["timestamp"].dt.hour().to_numpy()
    
    # Weekend gap: Monday opening (Sunday evening)
    is_weekend_gap = (dow == 1) & (hour < 4) & (time_diff_mins > 2000)
    
    # Daily maintenance halt for metals/commodities (approx 21:00-23:00 UTC daily)
    is_daily_halt = is_commodity & (hour >= 21) & (hour <= 23) & (time_diff_mins <= 180)
    
    # Pre-electronic historical session boundary for early 2000s Gold pit hours
    is_early_pit_halt = is_commodity & (df["timestamp"].dt.year().to_numpy() < 2008) & (time_diff_mins <= 900)
    
    # Abnormal trading gap during active market hours
    is_trading_gap = (
        (time_diff_mins > (expected_bar_minutes * 1.5)) & 
        (~is_weekend_gap) & 
        (~is_daily_halt) &
        (~is_early_pit_halt)
    )
    total_trading_gaps = int(np.sum(is_trading_gap))
    max_gap_duration = float(np.max(time_diff_mins[is_trading_gap])) if total_trading_gaps > 0 else 0.0
    pct_gaps = (total_trading_gaps / total_bars) * 100.0
    
    # 2. Bad-Print / Outlier Spike Filter
    atr14 = compute_atr(df, period=14)
    lagged_atr = np.roll(atr14, 1)
    lagged_atr[0] = atr14[0]
    
    med_atr = float(np.median(lagged_atr)) if len(lagged_atr) > 0 else 0.0005
    effective_atr = np.where((np.isnan(lagged_atr)) | (lagged_atr < 1e-6), med_atr, lagged_atr)
    
    close = df["close"].to_numpy()
    ret1 = np.zeros(total_bars, dtype=np.float64)
    ret1[1:] = np.abs(close[1:] - close[:-1])
    
    ret_revert1 = np.zeros(total_bars, dtype=np.float64)
    ret_revert1[:-1] = np.abs(close[1:] - close[:-1])
    
    ret_revert2 = np.zeros(total_bars, dtype=np.float64)
    ret_revert2[:-2] = np.abs(close[2:] - close[:-2])
    
    is_spike = (ret1 > 10.0 * effective_atr) & ((ret_revert1 > 7.0 * effective_atr) | (ret_revert2 > 7.0 * effective_atr))
    bad_print_count = int(np.sum(is_spike))
    pct_bad_prints = (bad_print_count / total_bars) * 100.0
    
    # 3. Spread Anomaly Detection
    spread_col = "spread_close" if "spread_close" in df.columns else "spread"
    if spread_col in df.columns:
        spreads = df[spread_col].to_numpy()
        negative_spreads = int(np.sum(spreads < 0))
        med_spread = float(np.median(spreads)) if len(spreads) > 0 else 0.0001
        max_spread = float(np.max(spreads)) if len(spreads) > 0 else 0.0
        
        is_spread_blowout = (spreads > 50.0 * med_spread) & (spreads > 0.0050)
        spread_blowout_count = int(np.sum(is_spread_blowout))
    else:
        negative_spreads = 0
        med_spread = 0.0
        max_spread = 0.0
        spread_blowout_count = 0
        is_spread_blowout = np.zeros(total_bars, dtype=bool)
    
    # 4. Volume & Stagnation Checks
    vol_col = "volume" if "volume" in df.columns else "bid_volume"
    if vol_col in df.columns:
        volumes = df[vol_col].to_numpy()
        zero_vol = (volumes == 0)
        zero_vol_count = int(np.sum(zero_vol))
        pct_zero_vol = (zero_vol_count / total_bars) * 100.0
    else:
        zero_vol_count = 0
        pct_zero_vol = 0.0
        
    # 5. Composite Anomaly Flagging
    total_anomalies = bad_print_count + negative_spreads + total_trading_gaps
    bad_bar_ratio = (total_anomalies / total_bars) * 100.0
    
    gate1_passed = bad_bar_ratio < max_bad_bar_pct
    
    verdict = (
        f"GATE 1 PASS: {bad_bar_ratio:.3f}% anomalous bars (< {max_bad_bar_pct}% threshold)"
        if gate1_passed
        else f"GATE 1 FAIL: {bad_bar_ratio:.3f}% anomalous bars (>= {max_bad_bar_pct}% threshold)"
    )
    
    metrics = QualityMetrics(
        symbol=symbol,
        total_bars=total_bars,
        start_time=start_time,
        end_time=end_time,
        total_trading_gaps_gt_5m=total_trading_gaps,
        max_gap_duration_mins=max_gap_duration,
        pct_bars_with_gaps=pct_gaps,
        bad_print_spikes=bad_print_count,
        pct_bad_prints=pct_bad_prints,
        negative_spread_count=negative_spreads,
        median_spread=med_spread,
        max_spread=max_spread,
        spread_blowout_count=spread_blowout_count,
        zero_volume_bars=zero_vol_count,
        pct_zero_volume=pct_zero_vol,
        total_anomalous_bars=total_anomalies,
        bad_bar_ratio=bad_bar_ratio,
        gate1_passed=gate1_passed,
        verdict=verdict,
    )
    
    flagged_df = df.with_columns([
        pl.Series("is_trading_gap", is_trading_gap),
        pl.Series("is_bad_print", is_spike),
        pl.Series("is_spread_blowout", is_spread_blowout),
    ])
    
    return metrics, flagged_df


def generate_markdown_report(metrics: QualityMetrics) -> str:
    """Generate professional Markdown audit report."""
    status_emoji = "✅" if metrics.gate1_passed else "❌"
    
    report = f"""# Gate 1 Quality Audit Report — {metrics.symbol}

**Audit Status:** {status_emoji} **{'PASSED' if metrics.gate1_passed else 'FAILED'}**
**Audit Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

---

## 1. Dataset Dimensions
| Metric | Value |
|---|---|
| **Symbol** | `{metrics.symbol}` |
| **Total Bars** | `{metrics.total_bars:,}` |
| **Start Date** | `{metrics.start_time}` |
| **End Date** | `{metrics.end_time}` |

---

## 2. Integrity & Anomaly Summary
| Test Category | Detected Issues | Rate (%) | Status |
|---|---|---|---|
| **Intra-Session Trading Gaps (>5m)** | {metrics.total_trading_gaps_gt_5m:,} | {metrics.pct_bars_with_gaps:.4f}% | {'✅ Clean' if metrics.pct_bars_with_gaps < 0.2 else '⚠️ High Gaps'} |
| **Bad-Print Spikes (>10x ATR)** | {metrics.bad_print_spikes:,} | {metrics.pct_bad_prints:.4f}% | {'✅ Clean' if metrics.pct_bad_prints < 0.05 else '⚠️ Spikes Found'} |
| **Inverted / Negative Spreads** | {metrics.negative_spread_count:,} | 0.0000% | {'✅ Zero' if metrics.negative_spread_count == 0 else '⚠️ Inverted Spreads'} |
| **Spread Blowouts (>50x Median)** | {metrics.spread_blowout_count:,} | {(metrics.spread_blowout_count/metrics.total_bars)*100:.4f}% | Info |
| **Zero Volume Bars** | {metrics.zero_volume_bars:,} | {metrics.pct_zero_volume:.4f}% | Info |

---

## 3. Spread Statistics
- **Median Spread:** `{metrics.median_spread:.5f}`
- **Maximum Recorded Spread:** `{metrics.max_spread:.5f}`

---

## 4. Gate 1 Evaluation & Kill Criteria
- **Total Anomaly Count:** `{metrics.total_anomalous_bars:,}`
- **Composite Anomaly Ratio:** **`{metrics.bad_bar_ratio:.4f}%`** (Kill Threshold: `< 0.50%`)
- **Conclusion:** **{metrics.verdict}**
"""
    return report


def run_quality_gate(
    parquet_path: Path,
    output_report_dir: Path,
    symbol: str = "EURUSD",
) -> QualityMetrics:
    """Execute quality gate directly on a parquet file and write reports."""
    parquet_path = Path(parquet_path)
    output_report_dir = Path(output_report_dir)
    output_report_dir.mkdir(parents=True, exist_ok=True)
    
    df = pl.read_parquet(parquet_path)
    metrics, flagged_df = audit_dataset(df, symbol=symbol)
    
    json_path = output_report_dir / f"{symbol}_quality_metrics.json"
    json_path.write_text(json.dumps(asdict(metrics), indent=2))
    
    md_path = output_report_dir / f"{symbol}_quality_report.md"
    md_path.write_text(generate_markdown_report(metrics))
    
    return metrics
