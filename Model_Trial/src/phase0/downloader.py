"""
Dukascopy tick-data downloader.

Downloads LZMA-compressed bi5 tick files from Dukascopy's public CDN.
URL pattern: https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1}/{DD}/{HH}h_ticks.bi5
Note: Dukascopy months are 0-indexed (January = 00).

Each bi5 file contains ticks for one hour, LZMA-compressed, with 20-byte
big-endian records: (ms_offset: i32, ask: i32, bid: i32, ask_vol: f32, bid_vol: f32)
Prices are stored as integer multiples of the pip (1e-5 for most FX, 1e-3 for JPY pairs, 1e-2 for gold).
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Dukascopy CDN base
CDN_BASE = "https://datafeed.dukascopy.com/datafeed"

# Price point values — prices stored as integers, divide by this to get real price.
# Most FX = 1e5 (5 decimal places), JPY pairs = 1e3, Gold = 1e2
POINT_VALUES = {
    "EURUSD": 1e5,
    "GBPUSD": 1e5,
    "USDJPY": 1e3,
    "USDCHF": 1e5,
    "AUDUSD": 1e5,
    "XAUUSD": 1e2,
}


def build_url(symbol: str, dt: datetime, hour: int) -> str:
    """
    Build Dukascopy CDN URL for a specific symbol, date, and hour.
    Months are 0-indexed on Dukascopy (Jan=00, Feb=01, ..., Dec=11).
    """
    month_0idx = dt.month - 1  # Dukascopy uses 0-indexed months
    return (
        f"{CDN_BASE}/{symbol}/"
        f"{dt.year:04d}/{month_0idx:02d}/{dt.day:02d}/"
        f"{hour:02d}h_ticks.bi5"
    )


def download_hour(
    symbol: str,
    dt: datetime,
    hour: int,
    output_dir: Path,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[Path]:
    """
    Download a single hour's bi5 tick file.

    Returns the path to the downloaded file, or None if the hour has no data
    (weekends, holidays — Dukascopy returns empty or 404).
    """
    url = build_url(symbol, dt, hour)

    # Output path: data/raw/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
    out_path = (
        output_dir
        / symbol
        / f"{dt.year:04d}"
        / f"{dt.month:02d}"
        / f"{dt.day:02d}"
        / f"{hour:02d}h_ticks.bi5"
    )

    # Skip if already downloaded and non-empty
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.debug(f"Already exists: {out_path}")
        return out_path

    sess = session or requests.Session()

    for attempt in range(1, max_retries + 1):
        try:
            resp = sess.get(url, timeout=30)

            if resp.status_code == 404:
                logger.debug(f"No data (404): {url}")
                return None

            resp.raise_for_status()

            # Dukascopy returns empty content for non-trading hours (weekends etc.)
            if len(resp.content) == 0:
                logger.debug(f"Empty response: {url}")
                return None

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(resp.content)
            logger.info(f"Downloaded {out_path.name} ({len(resp.content)} bytes) from {url}")
            return out_path

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt}/{max_retries} failed for {url}: {e}")
                time.sleep(retry_delay * attempt)
            else:
                logger.error(f"Failed after {max_retries} attempts: {url}: {e}")
                raise

    return None


def download_day(
    symbol: str,
    date: datetime,
    output_dir: Path,
    session: Optional[requests.Session] = None,
) -> list[Path]:
    """
    Download all 24 hourly tick files for a given symbol and date.
    Returns list of paths to successfully downloaded (non-empty) files.
    """
    sess = session or requests.Session()
    paths = []

    for hour in range(24):
        result = download_hour(symbol, date, hour, output_dir, session=sess)
        if result is not None:
            paths.append(result)

    logger.info(f"{symbol} {date.strftime('%Y-%m-%d')}: downloaded {len(paths)}/24 hours")
    return paths


def download_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> list[Path]:
    """
    Download tick data for a date range [start_date, end_date] inclusive.
    Politely rate-limits to avoid hammering the CDN.
    """
    if symbol not in POINT_VALUES:
        raise ValueError(
            f"Unknown symbol '{symbol}'. Known: {list(POINT_VALUES.keys())}. "
            f"Add its point value to POINT_VALUES before downloading."
        )

    output_dir = Path(output_dir)
    all_paths: list[Path] = []

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (trading-model research downloader)",
        "Accept-Encoding": "gzip, deflate",
    })

    current = start_date
    while current <= end_date:
        day_paths = download_day(symbol, current, output_dir, session=session)
        all_paths.extend(day_paths)

        # Polite delay between days
        time.sleep(0.5)
        current += timedelta(days=1)

    logger.info(
        f"Download complete: {symbol} "
        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, "
        f"{len(all_paths)} files"
    )
    return all_paths
