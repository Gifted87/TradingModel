#!/usr/bin/env python3
"""Test freeserv endpoint for EURUSD 5m data."""
import requests
import json
from datetime import datetime

url = "https://freeserv.dukascopy.com/2.0/index.php"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://freeserv.dukascopy.com/",
}

# 5 min interval, EUR/USD
# Note instrument name might be EUR/USD or EURUSD
params = {
    "path": "chart/json3",
    "instrument": "EUR/USD",
    "offer_side": "B",
    "interval": "5MIN",
    "splits": "true",
    "stocks": "true",
    "time_direction": "N",
    "last_update": int(datetime(2023, 1, 2, 0, 0).timestamp() * 1000),
    "limit": 500,
    "jsonp": "_cb",
}

r = requests.get(url, headers=headers, params=params, timeout=15)
print("Status:", r.status_code)
print("Text preview:", r.text[:200])

if r.status_code == 200 and r.text.startswith("_cb("):
    data = json.loads(r.text[4:-2])
    print("Received records:", len(data))
    if len(data) > 0:
        print("First record:", data[0])
        print("Last record:", data[-1])
        t0 = datetime.fromtimestamp(data[0][0]/1000)
        t1 = datetime.fromtimestamp(data[-1][0]/1000)
        print(f"Time range: {t0} to {t1}")
