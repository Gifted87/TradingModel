#!/usr/bin/env python3
"""Test dukascopy_python fetch for 5-minute data."""
from datetime import datetime
import dukascopy_python as dk

print("Available intervals:")
print("INTERVAL_MIN_5:", dk.INTERVAL_MIN_5)

# Test fetch for EURUSD 5m data from a known date range
start = datetime(2023, 1, 1)
end = datetime(2023, 1, 5)

print(f"Fetching EURUSD BID 5m from {start} to {end}...")
df_bid = dk.fetch("EURUSD", dk.INTERVAL_MIN_5, dk.OFFER_SIDE_BID, start, end)
print("BID shape:", df_bid.shape if df_bid is not None else "None")
if df_bid is not None:
    print(df_bid.head())

print(f"Fetching EURUSD ASK 5m from {start} to {end}...")
df_ask = dk.fetch("EURUSD", dk.INTERVAL_MIN_5, dk.OFFER_SIDE_ASK, start, end)
print("ASK shape:", df_ask.shape if df_ask is not None else "None")
if df_ask is not None:
    print(df_ask.head())
