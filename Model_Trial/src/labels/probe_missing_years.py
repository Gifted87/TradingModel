#!/usr/bin/env python3
import requests
import json
from datetime import datetime, timezone

url = "https://freeserv.dukascopy.com/2.0/index.php"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://freeserv.dukascopy.com/",
}

def probe_timestamp(dt: datetime):
    ts = int(dt.timestamp() * 1000)
    params = {
        "path": "chart/json3",
        "instrument": "EUR/USD",
        "offer_side": "B",
        "interval": "5MIN",
        "splits": "true",
        "stocks": "true",
        "time_direction": "N",
        "last_update": str(ts),
        "limit": 10,
        "jsonp": "_cb",
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code == 200 and r.text.startswith("_cb("):
        data = json.loads(r.text[4:-2])
        if data:
            first_dt = datetime.fromtimestamp(data[0][0]/1000, tz=timezone.utc)
            last_dt = datetime.fromtimestamp(data[-1][0]/1000, tz=timezone.utc)
            return True, first_dt, last_dt
    return False, None, None

for y in [2007, 2008, 2009, 2010, 2011, 2012]:
    for m in [1, 6]:
        dt = datetime(y, m, 1, 0, 0, tzinfo=timezone.utc)
        ok, t0, t1 = probe_timestamp(dt)
        print(f"Probe {dt.strftime('%Y-%m-%d')}: {'Found -> ' + str(t0) if ok else 'No data'}")
