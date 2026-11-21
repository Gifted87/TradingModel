#!/usr/bin/env python3
"""Probe the earliest date for EUR/USD on Dukascopy freeserv API."""
import requests
import json
from datetime import datetime

url = "https://freeserv.dukascopy.com/2.0/index.php"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://freeserv.dukascopy.com/",
}

def check_date(year, month=1, day=1, instrument="EUR/USD", interval="5MIN"):
    dt = datetime(year, month, day, 0, 0)
    ts = int(dt.timestamp() * 1000)
    params = {
        "path": "chart/json3",
        "instrument": instrument,
        "offer_side": "B",
        "interval": interval,
        "splits": "true",
        "stocks": "true",
        "time_direction": "N",
        "last_update": ts,
        "limit": 10,
        "jsonp": "_cb",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200 and r.text.startswith("_cb("):
            data = json.loads(r.text[4:-2])
            if len(data) > 0:
                first_ts = data[0][0]
                first_dt = datetime.fromtimestamp(first_ts / 1000)
                return True, first_dt, len(data)
    except Exception as e:
        return False, str(e), 0
    return False, None, 0

print("Probing years for EUR/USD 5m data...")
for y in [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2010, 2015, 2020]:
    has_data, dt, count = check_date(y, 1, 1)
    if has_data:
        print(f"Year {y}: HAS DATA -> First timestamp returned: {dt}")
    else:
        print(f"Year {y}: No data")
