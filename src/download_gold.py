"""Download the maximum available 1-minute gold history from Yahoo Finance."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from common import data_path, load_config


def download_history(config: dict) -> pd.DataFrame:
    symbol = config["symbol"]
    interval = config["interval"]
    chunk_days = int(config["download_chunk_days"])
    stop_after = int(config["empty_chunks_to_stop"])
    end = datetime.now(timezone.utc)
    pieces: list[pd.DataFrame] = []
    empty_chunks = 0

    # Yahoo restricts intraday history, so request bounded windows backwards.
    for _ in range(400):
        start = end - timedelta(days=chunk_days)
        print(f"Requesting {symbol} {start.date()} to {end.date()} ({interval})")
        try:
            frame = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            print(f"Request failed: {exc}")
            empty_chunks += 1
            if empty_chunks >= stop_after:
                break
            end = start
            continue

        if frame.empty:
            empty_chunks += 1
            if empty_chunks >= stop_after:
                break
        else:
            empty_chunks = 0
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame.reset_index()
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})
            pieces.append(frame)
        end = start

    if not pieces:
        raise RuntimeError("Yahoo Finance returned no data.")

    result = pd.concat(pieces, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.drop_duplicates("timestamp").sort_values("timestamp")
    result = result[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    result = result.dropna(subset=["Close"])
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output = data_path(config, "raw", f"{config['symbol'].replace('=', '_')}_1m.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = download_history(config)
    frame.to_parquet(output, index=False)
    print(f"Saved {len(frame):,} rows to {output}")
    print(f"Range: {frame['timestamp'].min()} -> {frame['timestamp'].max()}")


if __name__ == "__main__":
    main()
