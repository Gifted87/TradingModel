"""Resumable monthly downloader for Dukascopy XAU/USD 5-minute bars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from dukascopy_python import fetch

from common import data_path, load_config


def download_month(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    frame = fetch(instrument, "5MIN", "B", start, end, max_retries=7, limit=30_000)
    if frame.empty:
        return frame
    frame = frame.reset_index()
    frame = frame.rename(columns={"timestamp": "timestamp", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame[["timestamp", "Open", "High", "Low", "Close", "Volume"]].drop_duplicates("timestamp").sort_values("timestamp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", choices=("1m", "5m"), default="5m")
    args = parser.parse_args()
    config = load_config(args.config)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)
    interval = "1MIN" if args.timeframe == "1m" else "5MIN"
    cache = data_path(config, "raw", f"dukascopy_xauusd_{args.timeframe}_monthly")
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor < end:
        next_month = cursor + relativedelta(months=1)
        month_end = min(next_month, end)
        output = cache / f"xauusd_{args.timeframe}_bid_{cursor:%Y_%m}.parquet"
        if output.exists():
            frame = pd.read_parquet(output)
            print(f"Cached {cursor:%Y-%m}: {len(frame):,} rows")
        else:
            print(f"Downloading {cursor:%Y-%m} ...", flush=True)
            frame = fetch("XAU/USD", interval, "B", cursor, month_end, max_retries=7, limit=30_000).reset_index()
            frame = frame.rename(columns={"timestamp": "timestamp", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame[["timestamp", "Open", "High", "Low", "Close", "Volume"]].drop_duplicates("timestamp").sort_values("timestamp")
            frame.to_parquet(output, index=False)
            print(f"Saved {len(frame):,} rows", flush=True)
        parts.append(frame)
        cursor = next_month

    combined = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    combined_path = data_path(config, "raw", f"XAUUSD_DUKASCOPY_{args.timeframe}.parquet")
    combined.to_parquet(combined_path, index=False)
    print(f"Combined {len(combined):,} bars: {combined['timestamp'].min()} -> {combined['timestamp'].max()}")
    print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
