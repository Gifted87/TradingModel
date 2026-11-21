#!/usr/bin/env python3
"""Test pagination and throughput for 5m data download."""
import requests
import json
import time
from datetime import datetime, timezone

url = "https://freeserv.dukascopy.com/2.0/index.php"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://freeserv.dukascopy.com/",
}

def test_fetch_batch(limit=5000, last_update=1167584400000): # 2006-12-31 17:00:00
    params = {
        "path": "chart/json3",
        "instrument": "EUR/USD",
        "offer_side": "B",
        "interval": "5MIN",
        "splits": "true",
        "stocks": "true",
        "time_direction": "N",
        "last_update": last_update,
        "limit": limit,
        "jsonp": "_cb",
    }
    t0 = time.time()
    r = requests.get(url, headers=headers, params=params, timeout=20)
    elapsed = time.time() - t0
    if r.status_code == 200 and r.text.startswith("_cb("):
        data = json.loads(r.text[4:-2])
        print(f"Limit {limit}: received {len(data)} bars in {elapsed:.2f}s ({len(data)/elapsed:.0f} bars/sec)")
        if data:
            first_dt = datetime.fromtimestamp(data[0][0]/1000, tz=timezone.utc)
            last_dt = datetime.fromtimestamp(data[-1][0]/1000, tz=timezone.utc)
            print(f"  Range: {first_dt} -> {last_dt}")
            return data
    else:
        print(f"Failed: {r.status_code}, {r.text[:100]}")
    return None

test_fetch_batch(5000)
test_fetch_batch(10000)
