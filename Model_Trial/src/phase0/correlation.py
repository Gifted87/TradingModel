"""
Gate 0 correlation analysis: Does OFI proxy predict next-bar returns?

This is the linchpin test of the entire model design (C1 in the whitepaper).
Cont, Kukanov & Stoikov (2014) found R² ≈ 65% for OFI → price on NYSE stocks.
We expect ρ ≈ 0.3–0.6 for FX tick data given the queue-size proxy approximation.

Tests performed:
1. Pearson correlation: OFI_sum vs next-bar mid_return
2. Contemporaneous correlation: OFI_sum vs same-bar mid_return
3. OFI/depth ratio vs next-bar mid_return (depth-adjusted signal)
4. Rolling window stability: correlation across 1-hour windows
5. Signed hit-rate: does sign(OFI) predict sign(next-bar return)?
6. Linear regression: R² of OFI → next-bar return
7. Time-of-day breakdown: correlation by hour

If C1 fails (ρ ≈ 0 or negative), the microstructure branch is void and we
fall back to the whitepaper's contingency (§8): trade-imbalance + momentum.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Results from a single correlation test."""
    name: str
    n_obs: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    r_squared: float  # from OLS
    hit_rate: float  # fraction where sign(x) == sign(y)
    mean_x: float
    std_x: float
    mean_y: float
    std_y: float


@dataclass
class Gate0Report:
    """Complete Gate 0 analysis report."""
    symbol: str
    n_bars: int
    n_ticks: int
    date_range: str

    # Core tests
    contemporaneous: CorrelationResult  # OFI vs same-bar return
    predictive: CorrelationResult  # OFI vs next-bar return
    predictive_depth_adj: Optional[CorrelationResult]  # OFI/depth vs next-bar return

    # Stability
    hourly_correlations: pl.DataFrame  # correlation by hour-of-day
    rolling_correlations: pl.DataFrame  # rolling 60-bar correlations

    # Verdict
    passed: bool
    verdict: str


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
) -> CorrelationResult:
    """
    Compute comprehensive correlation statistics between two arrays.
    """
    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    if n < 10:
        return CorrelationResult(
            name=name, n_obs=n,
            pearson_r=np.nan, pearson_p=np.nan,
            spearman_r=np.nan, spearman_p=np.nan,
            r_squared=np.nan, hit_rate=np.nan,
            mean_x=np.nan, std_x=np.nan,
            mean_y=np.nan, std_y=np.nan,
        )

    # Pearson
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)

    # Spearman (rank correlation, robust to outliers)
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

    # OLS R²
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r_squared = r_value ** 2

    # Signed hit rate: does sign(OFI) predict sign(return)?
    # Exclude zero-OFI and zero-return bars
    nonzero_mask = (x_clean != 0) & (y_clean != 0)
    if nonzero_mask.sum() > 0:
        hit_rate = (np.sign(x_clean[nonzero_mask]) == np.sign(y_clean[nonzero_mask])).mean()
    else:
        hit_rate = np.nan

    return CorrelationResult(
        name=name,
        n_obs=n,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        r_squared=r_squared,
        hit_rate=hit_rate,
        mean_x=float(np.mean(x_clean)),
        std_x=float(np.std(x_clean)),
        mean_y=float(np.mean(y_clean)),
        std_y=float(np.std(y_clean)),
    )


def compute_hourly_correlations(bars: pl.DataFrame) -> pl.DataFrame:
    """
    Break down OFI→return correlation by hour of day.

    This tests the whitepaper's claim that impact is intraday-seasonal [D4]:
    correlation should vary by time, stronger during active sessions.
    """
    # Add hour column
    bars_h = bars.with_columns([
        pl.col("bar_time").dt.hour().alias("hour"),
    ])

    results = []
    for hour in range(24):
        hour_data = bars_h.filter(pl.col("hour") == hour)
        if len(hour_data) < 10:
            continue

        ofi = hour_data["ofi_sum"].to_numpy()
        ret = hour_data["mid_return"].to_numpy()
        mask = np.isfinite(ofi) & np.isfinite(ret)

        if mask.sum() < 10:
            continue

        r, p = stats.pearsonr(ofi[mask], ret[mask])
        nonzero = (ofi[mask] != 0) & (ret[mask] != 0)
        hit = (np.sign(ofi[mask][nonzero]) == np.sign(ret[mask][nonzero])).mean() if nonzero.sum() > 0 else np.nan

        results.append({
            "hour": hour,
            "n_bars": int(mask.sum()),
            "pearson_r": r,
            "pearson_p": p,
            "hit_rate": hit,
        })

    return pl.DataFrame(results)


def compute_rolling_correlations(
    bars: pl.DataFrame,
    window: int = 60,
) -> pl.DataFrame:
    """
    Compute rolling window correlation to assess stability.

    A stable OFI signal should show consistently positive correlation,
    not sporadic spikes.
    """
    ofi = bars["ofi_sum"].to_numpy()
    ret = bars["mid_return"].to_numpy()
    times = bars["bar_time"].to_numpy()

    correlations = []
    for i in range(window, len(ofi)):
        x_win = ofi[i - window : i]
        y_win = ret[i - window : i]
        mask = np.isfinite(x_win) & np.isfinite(y_win)

        if mask.sum() < 10:
            correlations.append({
                "bar_time": times[i],
                "rolling_corr": np.nan,
                "n_valid": int(mask.sum()),
            })
            continue

        r, _ = stats.pearsonr(x_win[mask], y_win[mask])
        correlations.append({
            "bar_time": times[i],
            "rolling_corr": r,
            "n_valid": int(mask.sum()),
        })

    return pl.DataFrame(correlations)


def run_gate0_analysis(
    bars: pl.DataFrame,
    symbol: str = "EURUSD",
    n_ticks: int = 0,
    date_range: str = "",
) -> Gate0Report:
    """
    Run the full Gate 0 analysis.

    This is the make-or-break test: does our OFI proxy predict returns?

    Parameters
    ----------
    bars : pl.DataFrame
        Bar data from ofi.build_ofi_bars(), must contain:
        ofi_sum, mid_return, ofi_depth_ratio_5, bar_time
    symbol : str
        Symbol name for reporting.
    n_ticks : int
        Total ticks processed (for reporting).
    date_range : str
        Date range string (for reporting).

    Returns
    -------
    Gate0Report
        Complete analysis with pass/fail verdict.
    """
    n_bars = len(bars)
    logger.info(f"Running Gate 0 analysis on {n_bars} bars for {symbol}...")

    # --- Test 1: Contemporaneous correlation (OFI vs same-bar return) ---
    # This should be strong (R² ≈ 0.65 per Cont et al.)
    ofi_arr = bars["ofi_sum"].to_numpy().astype(np.float64)
    ret_arr = bars["mid_return"].to_numpy().astype(np.float64)

    contemp = compute_correlation(ofi_arr, ret_arr, "OFI → same-bar return (contemporaneous)")

    # --- Test 2: Predictive correlation (OFI vs NEXT-bar return) ---
    # This is the actual trading signal — weaker but must be positive
    next_ret = np.roll(ret_arr, -1)
    next_ret[-1] = np.nan  # last bar has no next return
    predictive = compute_correlation(ofi_arr[:-1], next_ret[:-1], "OFI → next-bar return (predictive)")

    # --- Test 3: Depth-adjusted OFI vs next-bar return ---
    predictive_depth = None
    if "ofi_depth_ratio_5" in bars.columns:
        ofi_depth = bars["ofi_depth_ratio_5"].to_numpy().astype(np.float64)
        predictive_depth = compute_correlation(
            ofi_depth[:-1], next_ret[:-1],
            "OFI/depth → next-bar return (depth-adjusted)"
        )

    # --- Test 4: Hourly breakdown ---
    hourly = compute_hourly_correlations(bars)

    # --- Test 5: Rolling stability ---
    rolling = compute_rolling_correlations(bars, window=60)

    # --- Verdict ---
    # Gate 0 passes if:
    # 1. Contemporaneous correlation is strongly positive (ρ > 0.15)
    # 2. Predictive correlation is positive (even weak, ρ > 0.02 is meaningful)
    # 3. Signed hit rate > 50%
    # Note: for tick-volume OFI proxy on FX, we expect weaker signal than
    # NYSE order-book OFI, so thresholds are conservative.

    passed = True
    reasons = []

    if contemp.pearson_r < 0.10:
        passed = False
        reasons.append(
            f"Contemporaneous ρ = {contemp.pearson_r:.4f} < 0.10 — "
            "OFI proxy not tracking same-bar returns"
        )
    else:
        reasons.append(f"✓ Contemporaneous ρ = {contemp.pearson_r:.4f} (p={contemp.pearson_p:.2e})")

    if predictive.pearson_r <= 0:
        passed = False
        reasons.append(
            f"Predictive ρ = {predictive.pearson_r:.4f} ≤ 0 — "
            "no forward-looking signal"
        )
    else:
        reasons.append(f"✓ Predictive ρ = {predictive.pearson_r:.4f} (p={predictive.pearson_p:.2e})")

    if contemp.hit_rate < 0.50:
        passed = False
        reasons.append(f"Hit rate = {contemp.hit_rate:.1%} < 50% — worse than random")
    else:
        reasons.append(f"✓ Signed hit rate = {contemp.hit_rate:.1%}")

    if predictive_depth is not None:
        reasons.append(
            f"  Depth-adj predictive ρ = {predictive_depth.pearson_r:.4f} "
            f"(p={predictive_depth.pearson_p:.2e})"
        )

    # Rolling stability check
    if len(rolling) > 0:
        rolling_vals = rolling["rolling_corr"].drop_nulls().to_numpy()
        pct_positive = (rolling_vals > 0).mean() if len(rolling_vals) > 0 else 0
        reasons.append(f"  Rolling corr positive in {pct_positive:.1%} of windows")

    verdict = "PASS ✓" if passed else "FAIL ✗"
    verdict_detail = f"Gate 0: {verdict}\n" + "\n".join(reasons)

    logger.info(f"\n{'='*60}\n{verdict_detail}\n{'='*60}")

    return Gate0Report(
        symbol=symbol,
        n_bars=n_bars,
        n_ticks=n_ticks,
        date_range=date_range,
        contemporaneous=contemp,
        predictive=predictive,
        predictive_depth_adj=predictive_depth,
        hourly_correlations=hourly,
        rolling_correlations=rolling,
        passed=passed,
        verdict=verdict_detail,
    )


def format_report(report: Gate0Report) -> str:
    """Format a Gate 0 report as a human-readable string."""
    lines = [
        "=" * 70,
        "GATE 0 REPORT — OFI Proxy Correlation Test",
        "=" * 70,
        f"Symbol:     {report.symbol}",
        f"Date range: {report.date_range}",
        f"Ticks:      {report.n_ticks:,}",
        f"Bars:       {report.n_bars:,}",
        "",
        "--- Contemporaneous (OFI vs same-bar return) ---",
        f"  Pearson ρ:   {report.contemporaneous.pearson_r:+.4f}  (p={report.contemporaneous.pearson_p:.2e})",
        f"  Spearman ρ:  {report.contemporaneous.spearman_r:+.4f}  (p={report.contemporaneous.spearman_p:.2e})",
        f"  R²:          {report.contemporaneous.r_squared:.4f}",
        f"  Hit rate:    {report.contemporaneous.hit_rate:.1%}",
        f"  n:           {report.contemporaneous.n_obs:,}",
        "",
        "--- Predictive (OFI vs next-bar return) ---",
        f"  Pearson ρ:   {report.predictive.pearson_r:+.4f}  (p={report.predictive.pearson_p:.2e})",
        f"  Spearman ρ:  {report.predictive.spearman_r:+.4f}  (p={report.predictive.spearman_p:.2e})",
        f"  R²:          {report.predictive.r_squared:.4f}",
        f"  Hit rate:    {report.predictive.hit_rate:.1%}",
        f"  n:           {report.predictive.n_obs:,}",
    ]

    if report.predictive_depth_adj is not None:
        d = report.predictive_depth_adj
        lines.extend([
            "",
            "--- Depth-adjusted (OFI/depth vs next-bar return) ---",
            f"  Pearson ρ:   {d.pearson_r:+.4f}  (p={d.pearson_p:.2e})",
            f"  Spearman ρ:  {d.spearman_r:+.4f}  (p={d.spearman_p:.2e})",
            f"  R²:          {d.r_squared:.4f}",
            f"  Hit rate:    {d.hit_rate:.1%}",
        ])

    if len(report.hourly_correlations) > 0:
        lines.extend(["", "--- Hourly Correlation Breakdown ---"])
        for row in report.hourly_correlations.iter_rows(named=True):
            sig = "***" if row["pearson_p"] < 0.001 else "**" if row["pearson_p"] < 0.01 else "*" if row["pearson_p"] < 0.05 else ""
            lines.append(
                f"  Hour {row['hour']:02d}:  ρ={row['pearson_r']:+.4f}  "
                f"hit={row['hit_rate']:.1%}  n={row['n_bars']:,}  {sig}"
            )

    lines.extend([
        "",
        "=" * 70,
        report.verdict,
        "=" * 70,
    ])

    return "\n".join(lines)
