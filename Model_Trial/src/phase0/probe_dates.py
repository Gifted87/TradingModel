#!/usr/bin/env python3
"""
Debug Dukascopy CDN — try with various headers including Referer.
Dukascopy blocks bare requests; their web app sends specific headers.
"""
import requests
import lzma

# The Dukascopy web platform uses these headers
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed",
    "Origin": "https://www.dukascopy.com",
})

# Test with the Referer header
url = "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/02/12h_ticks.bi5"
r = session.get(url, timeout=15)
print(f"With Referer: [{r.status_code}] {len(r.content)} bytes")
if len(r.content) > 0 and len(r.content) < 200:
    print(f"  Content: {r.content[:100]}")
if len(r.content) > 0:
    try:
        raw = lzma.decompress(r.content)
        print(f"  LZMA decoded: {len(raw)} bytes = {len(raw)//20} ticks")
    except:
        pass

# Try without Referer but with cookie
session2 = requests.Session()
session2.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
})
# First visit the main page to get cookies
try:
    main = session2.get("https://www.dukascopy.com/swiss/english/marketwatch/historical/", timeout=15)
    print(f"\nMain page: [{main.status_code}] {len(main.content)} bytes, cookies: {dict(session2.cookies)}")
except Exception as e:
    print(f"\nMain page error: {e}")

r2 = session2.get(url, timeout=15)
print(f"With cookies: [{r2.status_code}] {len(r2.content)} bytes")
if len(r2.content) > 0:
    try:
        raw = lzma.decompress(r2.content)
        print(f"  LZMA decoded: {len(raw)} bytes = {len(raw)//20} ticks")
    except:
        if len(r2.content) < 200:
            print(f"  Content: {r2.content[:100]}")

# Try the 503 content to see what it says
print(f"\n503 response body: {r.content}")
