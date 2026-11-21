#!/usr/bin/env python3
"""
Phase 0 Runner — Test the linchpin before building anything.

Downloads a few days of EURUSD tick data from Dukascopy, decodes the bi5 files,
computes the OFI proxy, builds 1-minute bars, and runs the Gate 0 correlation test.

Gate 0 criterion: OFI proxy must correlate positively with next-bar mid returns.
Expected: ρ ≈ 0.3–0.6 contemporaneous (based on Cont et al. R²≈65%),
          ρ > 0 predictive.

Usage:
    python -m src.phase0.run_phase0
    python -m src.phase0.run_phase0 --symbol GBPUSD --days 5
    python -m src.phase0.run_phase0 --start-date 2024-06-10 --end-date 2024-06-14
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .decoder import decode_symbol_range
from .downloader import POINT_VALUES, download_range
from .ofi import build_ofi_bars
from .correlation import run_gate0_analysis, format_report

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results" / "phase0"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: OFI proxy linchpin test"
    )
    parser.add_argument(
        "--symbol", type=str, default="EURUSD",
        help=f"Symbol to test. Available: {list(POINT_VALUES.keys())}",
    )
    parser.add_argument(
        "--days", type=int, default=3,
        help="Number of recent weekdays to download (used if --start-date not given)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date YYYY-MM-DD (inclusive). Overrides --days.",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date YYYY-MM-DD (inclusive). Defaults to start + days.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, use existing bi5 files in data/raw/",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def pick_weekday_range(n_days: int) -> tuple[datetime, datetime]:
    """
    Pick a range of n recent weekdays, avoiding weekends.
    Uses dates ~2 weeks ago to avoid hitting data that might not be available yet.
    """
    # Start from 2 weeks ago to be safe
    end = datetime.now() - timedelta(days=14)
    # Walk backward to find a Friday
    while end.weekday() > 4:  # 5=Sat, 6=Sun
        end -= timedelta(days=1)

    # Walk back n weekdays
    start = end
    counted = 0
    while counted < n_days - 1:
        start -= timedelta(days=1)
        if start.weekday() <= 4:
            counted += 1

    return (
        datetime(start.year, start.month, start.day),
        datetime(end.year, end.month, end.day),
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("phase0")

    symbol = args.symbol.upper()
    if symbol not in POINT_VALUES:
        logger.error(f"Unknown symbol: {symbol}. Available: {list(POINT_VALUES.keys())}")
        return 1

    point_value = POINT_VALUES[symbol]

    # Determine date range
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        else:
            end_date = start_date + timedelta(days=args.days - 1)
    else:
        start_date, end_date = pick_weekday_range(args.days)

    date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    logger.info(f"Phase 0: {symbol}, {date_range_str}")
    logger.info(f"Point value: {point_value}")

    # ── Step 1: Download ──
    if not args.skip_download:
        logger.info("Step 1: Downloading tick data from Dukascopy...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = download_range(symbol, start_date, end_date, RAW_DIR)
        logger.info(f"Downloaded {len(downloaded)} hourly files")

        if len(downloaded) == 0:
            logger.error(
                "No data downloaded. Possible causes:\n"
                "  - Weekend/holiday dates selected\n"
                "  - Network issues\n"
                "  - Dukascopy CDN blocked\n"
                "Try different dates or check your network."
            )
            return 1
    else:
        logger.info("Step 1: Skipping download (--skip-download)")

    # ── Step 2: Decode ──
    logger.info("Step 2: Decoding bi5 files...")
    ticks = decode_symbol_range(RAW_DIR, symbol, start_date, end_date, point_value)

    if ticks is None or len(ticks) == 0:
        logger.error("No ticks decoded. Check data/raw/ directory.")
        return 1

    n_ticks = len(ticks)
    logger.info(f"Decoded {n_ticks:,} ticks")

    # Sanity check: print price range
    bid_min = ticks["bid"].min()
    bid_max = ticks["bid"].max()
    ask_min = ticks["ask"].min()
    ask_max = ticks["ask"].max()
    spread_mean = ticks["spread"].mean()
    logger.info(
        f"Price sanity check: bid [{bid_min:.5f}, {bid_max:.5f}], "
        f"ask [{ask_min:.5f}, {ask_max:.5f}], "
        f"mean spread {spread_mean:.6f}"
    )

    # Quick sanity: EURUSD should be roughly 1.0–1.5
    if symbol == "EURUSD" and (bid_min < 0.5 or bid_max > 2.0):
        logger.warning(
            f"EURUSD bid range [{bid_min:.5f}, {bid_max:.5f}] looks wrong! "
            "Check point_value or data integrity."
        )

    # ── Step 3: Build OFI bars ──
    logger.info("Step 3: Computing OFI and building 1-minute bars...")
    bars = build_ofi_bars(ticks, bar_duration="1m")
    logger.info(f"Built {len(bars):,} bars")

    # Save bars for inspection
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bars_path = RESULTS_DIR / f"{symbol}_bars.parquet"
    bars.write_parquet(bars_path)
    logger.info(f"Bars saved to {bars_path}")

    # ── Step 4: Gate 0 analysis ──
    logger.info("Step 4: Running Gate 0 correlation analysis...")
    report = run_gate0_analysis(
        bars,
        symbol=symbol,
        n_ticks=n_ticks,
        date_range=date_range_str,
    )

    # Print report
    report_text = format_report(report)
    print("\n" + report_text)

    # Save report
    report_path = RESULTS_DIR / f"{symbol}_gate0_report.txt"
    report_path.write_text(report_text)
    logger.info(f"Report saved to {report_path}")

    # Save key metrics as JSON for programmatic access
    metrics = {
        "symbol": report.symbol,
        "date_range": report.date_range,
        "n_ticks": report.n_ticks,
        "n_bars": report.n_bars,
        "contemporaneous_pearson_r": report.contemporaneous.pearson_r,
        "contemporaneous_r_squared": report.contemporaneous.r_squared,
        "contemporaneous_hit_rate": report.contemporaneous.hit_rate,
        "predictive_pearson_r": report.predictive.pearson_r,
        "predictive_r_squared": report.predictive.r_squared,
        "predictive_hit_rate": report.predictive.hit_rate,
        "passed": report.passed,
    }
    if report.predictive_depth_adj is not None:
        metrics["predictive_depth_adj_pearson_r"] = report.predictive_depth_adj.pearson_r

    metrics_path = RESULTS_DIR / f"{symbol}_gate0_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved to {metrics_path}")

    # ── Verdict ──
    if report.passed:
        logger.info("🟢 Gate 0 PASSED — OFI proxy has predictive signal. Proceed to Phase 1.")
        return 0
    else:
        logger.warning(
            "🔴 Gate 0 FAILED — OFI proxy does not show expected correlation.\n"
            "Contingency: fall back to trade-imbalance + momentum features.\n"
            "The linear core becomes HAR-style volatility + momentum instead of OFI.\n"
            "Review the report for details."
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
