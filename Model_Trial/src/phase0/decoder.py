"""
Dukascopy bi5 tick decoder.

Decodes LZMA-compressed bi5 files into structured tick data.

Binary format per record (20 bytes, big-endian):
  - ms_offset  (int32):  milliseconds since the start of the hour
  - ask_pips   (int32):  ask price in pip units (divide by point_value for real price)
  - bid_pips   (int32):  bid price in pip units
  - ask_vol    (float32): ask volume (lots, rounded)
  - bid_vol    (float32): bid volume (lots, rounded)

Reference: Dukascopy tick data specification.
"""

import lzma
import logging
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Each tick record is 20 bytes: i32 + i32 + i32 + f32 + f32
RECORD_SIZE = 20
RECORD_FORMAT = ">iiiff"  # big-endian: 3 signed int32 + 2 float32


def decode_bi5_bytes(
    compressed_data: bytes,
    base_dt: datetime,
    point_value: float,
) -> Optional[pl.DataFrame]:
    """
    Decode raw LZMA-compressed bi5 bytes into a Polars DataFrame.

    Parameters
    ----------
    compressed_data : bytes
        Raw bi5 file content (LZMA-compressed).
    base_dt : datetime
        The base datetime for this file (year/month/day/hour at minute=0, second=0).
        ms_offset in each record is added to this.
    point_value : float
        Divisor to convert integer pip prices to real prices.
        E.g., 1e5 for EURUSD (5 decimal places).

    Returns
    -------
    pl.DataFrame or None
        Columns: [timestamp, bid, ask, bid_vol, ask_vol, mid, spread]
        None if the data is empty or cannot be decoded.
    """
    try:
        raw = lzma.decompress(compressed_data)
    except lzma.LZMAError as e:
        logger.warning(f"LZMA decompression failed for {base_dt}: {e}")
        return None

    if len(raw) == 0:
        return None

    n_records = len(raw) // RECORD_SIZE
    remainder = len(raw) % RECORD_SIZE

    if remainder != 0:
        logger.warning(
            f"Truncated data for {base_dt}: {len(raw)} bytes, "
            f"{remainder} bytes leftover after {n_records} records"
        )

    if n_records == 0:
        return None

    # Unpack all records at once for speed
    # struct.iter_unpack is faster than repeated struct.unpack
    timestamps = np.empty(n_records, dtype="datetime64[ms]")
    asks = np.empty(n_records, dtype=np.float64)
    bids = np.empty(n_records, dtype=np.float64)
    ask_vols = np.empty(n_records, dtype=np.float64)
    bid_vols = np.empty(n_records, dtype=np.float64)

    base_ts_ms = int(base_dt.timestamp() * 1000)

    for i, (ms_offset, ask_pips, bid_pips, ask_vol, bid_vol) in enumerate(
        struct.iter_unpack(RECORD_FORMAT, raw[: n_records * RECORD_SIZE])
    ):
        timestamps[i] = np.datetime64(base_ts_ms + ms_offset, "ms")
        asks[i] = ask_pips / point_value
        bids[i] = bid_pips / point_value
        ask_vols[i] = ask_vol
        bid_vols[i] = bid_vol

    df = pl.DataFrame({
        "timestamp": timestamps,
        "bid": bids,
        "ask": asks,
        "bid_vol": bid_vols,
        "ask_vol": ask_vols,
    })

    # Derived columns
    df = df.with_columns([
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
    ])

    return df


def decode_bi5_file(
    filepath: Path,
    base_dt: datetime,
    point_value: float,
) -> Optional[pl.DataFrame]:
    """
    Decode a bi5 file from disk.

    Parameters
    ----------
    filepath : Path
        Path to the .bi5 file.
    base_dt : datetime
        Base datetime for this hour (e.g., 2024-01-15 14:00:00 UTC).
    point_value : float
        Price point value for the symbol.

    Returns
    -------
    pl.DataFrame or None
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None

    data = filepath.read_bytes()
    if len(data) == 0:
        logger.debug(f"Empty file: {filepath}")
        return None

    return decode_bi5_bytes(data, base_dt, point_value)


def decode_day(
    day_dir: Path,
    date: datetime,
    point_value: float,
) -> Optional[pl.DataFrame]:
    """
    Decode all hourly bi5 files for a single day and concatenate.

    Parameters
    ----------
    day_dir : Path
        Directory containing {HH}h_ticks.bi5 files.
    date : datetime
        The date (year, month, day). Hours are inferred from filenames.
    point_value : float
        Price point value.

    Returns
    -------
    pl.DataFrame or None
        Concatenated ticks for the full day, sorted by timestamp.
    """
    day_dir = Path(day_dir)
    frames = []

    for hour in range(24):
        bi5_path = day_dir / f"{hour:02d}h_ticks.bi5"
        if not bi5_path.exists():
            continue

        base_dt = datetime(date.year, date.month, date.day, hour, 0, 0)
        df = decode_bi5_file(bi5_path, base_dt, point_value)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        return None

    result = pl.concat(frames).sort("timestamp")
    logger.info(f"Decoded {date.strftime('%Y-%m-%d')}: {len(result)} ticks from {len(frames)} hours")
    return result


def decode_symbol_range(
    raw_dir: Path,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    point_value: float,
) -> Optional[pl.DataFrame]:
    """
    Decode all downloaded bi5 files for a symbol across a date range.

    Looks for files in: raw_dir/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5

    Returns
    -------
    pl.DataFrame or None
        All ticks across the date range, sorted by timestamp.
    """
    raw_dir = Path(raw_dir)
    frames = []
    current = start_date

    while current <= end_date:
        day_path = (
            raw_dir
            / symbol
            / f"{current.year:04d}"
            / f"{current.month:02d}"
            / f"{current.day:02d}"
        )

        if day_path.exists():
            df = decode_day(day_path, current, point_value)
            if df is not None:
                frames.append(df)

        current += timedelta(days=1)

    if not frames:
        logger.warning(f"No data decoded for {symbol} {start_date} to {end_date}")
        return None

    result = pl.concat(frames).sort("timestamp")
    logger.info(
        f"Decoded {symbol}: {len(result)} total ticks, "
        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    return result
