"""
Adversarial Unit & Integration Tests for Phase 0 and Phase 1.

Tests cover:
1. bi5 binary parsing correctness, endianness, and truncation handling.
2. Cont et al. (2014) OFI proxy mathematical invariance.
3. Quality Gate engine under synthetic stress (injected bad prints, gaps, inverted spreads).
4. Multi-horizon bar builder and microstructure aggregation.
"""

import lzma
import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.data.bar_builder import resample_bars
from src.data.quality_audit import audit_dataset, compute_atr
from src.phase0.decoder import decode_bi5_bytes
from src.phase0.ofi import aggregate_to_bars, compute_depth_proxy, compute_tick_ofi


# =====================================================================
# 1. Binary bi5 Decoder Tests
# =====================================================================

def test_bi5_decoder_valid_record():
    """Verify correct unpacking of 20-byte big-endian bi5 records (>iiiff)."""
    record = struct.pack(">iiiff", 5000, 105000, 104990, 1.5, 2.5)
    compressed = lzma.compress(record)
    
    base_dt = datetime(2024, 1, 15, 12, 0, 0)
    df = decode_bi5_bytes(compressed, base_dt, point_value=1e5)
    
    assert df is not None
    assert len(df) == 1
    assert pytest.approx(df["ask"][0], 1e-6) == 1.05000
    assert pytest.approx(df["bid"][0], 1e-6) == 1.04990
    assert pytest.approx(df["mid"][0], 1e-6) == 1.04995
    assert pytest.approx(df["spread"][0], 1e-6) == 0.00010
    assert pytest.approx(df["ask_vol"][0], 1e-6) == 1.5
    assert pytest.approx(df["bid_vol"][0], 1e-6) == 2.5


def test_bi5_decoder_empty_and_corrupted():
    """Decoder must handle empty buffers and corrupt LZMA gracefully."""
    base_dt = datetime(2024, 1, 15, 12, 0, 0)
    assert decode_bi5_bytes(b"", base_dt, 1e5) is None
    assert decode_bi5_bytes(b"corrupted_payload_12345", base_dt, 1e5) is None
    partial_comp = lzma.compress(b"1234567890")
    assert decode_bi5_bytes(partial_comp, base_dt, 1e5) is None


# =====================================================================
# 2. OFI Mathematics & Mechanics Tests
# =====================================================================

def test_ofi_directional_sign():
    """
    Test that OFI is positive when bid steps up and negative when ask drops,
    strictly following Cont, Kukanov & Stoikov (2014).
    """
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    ticks = pl.DataFrame({
        "timestamp": [
            t0,
            t0 + timedelta(seconds=1),
            t0 + timedelta(seconds=2),
        ],
        "bid": [1.0000, 1.0001, 1.0001],
        "ask": [1.0002, 1.0002, 1.0001],
        "bid_vol": [1.0, 10.0, 5.0],
        "ask_vol": [1.0, 5.0, 8.0],
        "mid": [1.0001, 1.00015, 1.0001],
        "spread": [0.0002, 0.0001, 0.0000],
    })
    
    df_ofi = compute_tick_ofi(ticks)
    assert len(df_ofi) == 2
    assert pytest.approx(df_ofi["ofi"][0]) == 6.0
    assert df_ofi["ofi"][1] < 0


# =====================================================================
# 3. Quality Audit & Gate 1 Engine Tests
# =====================================================================

def test_quality_audit_clean_data():
    """Verify clean linear sequence passes Gate 1 with 0% anomalies."""
    n = 1000
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]
    prices = 1.1000 + np.sin(np.linspace(0, 10, n)) * 0.0050
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 0.0002,
        "low": prices - 0.0002,
        "close": prices,
        "spread_close": np.full(n, 0.0001),
        "volume": np.full(n, 100.0),
    })
    
    metrics, flagged = audit_dataset(df, symbol="EURUSD_TEST", expected_bar_minutes=5)
    assert metrics.gate1_passed is True
    assert metrics.bad_print_spikes == 0
    assert metrics.negative_spread_count == 0
    assert metrics.bad_bar_ratio < 0.1


def test_quality_audit_synthetic_anomalies():
    """Verify detection of injected bad prints, inverted spreads, and gaps."""
    n = 500
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]
    close = np.full(n, 1.1000)
    high = close + 0.0002
    low = close - 0.0002
    
    close[100] = 1.2500
    high[100] = 1.2502
    low[100] = 1.2498
    
    spreads = np.full(n, 0.0001)
    spreads[200] = -0.0005
    
    timestamps[301] = timestamps[300] + timedelta(minutes=45)
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "spread_close": spreads,
        "volume": np.full(n, 50.0),
    })
    
    metrics, flagged = audit_dataset(df, symbol="ANOMALY_TEST", expected_bar_minutes=5)
    
    assert metrics.bad_print_spikes >= 1
    assert metrics.negative_spread_count == 1
    assert metrics.total_trading_gaps_gt_5m >= 1


# =====================================================================
# 4. Multi-Horizon Bar Resampler Tests
# =====================================================================

def test_bar_resampling_integrity():
    """Verify 5-minute to 30-minute bar resampling preserves exact boundaries."""
    n = 12  # Exactly two 30-minute bars (6 bars each)
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]
    
    opens = [1.1000 + i * 0.0001 for i in range(n)]
    highs = [o + 0.0005 for o in opens]
    lows = [o - 0.0005 for o in opens]
    closes = [o + 0.0002 for o in opens]
    
    df_5m = pl.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "spread_close": np.full(n, 0.00015),
        "volume": np.full(n, 100.0),
    })
    
    df_30m = resample_bars(df_5m, target_duration="30m")
    assert len(df_30m) == 2
    
    # Bar 0 (00:00 to 00:30)
    assert pytest.approx(df_30m["open"][0]) == opens[0]
    assert pytest.approx(df_30m["high"][0]) == max(highs[:6])
    assert pytest.approx(df_30m["low"][0]) == min(lows[:6])
    assert pytest.approx(df_30m["close"][0]) == closes[5]
    assert df_30m["volume"][0] == 600.0
    assert df_30m["sub_bar_count"][0] == 6
    
    # Bar 1 (00:30 to 01:00)
    assert pytest.approx(df_30m["open"][1]) == opens[6]
    assert pytest.approx(df_30m["high"][1]) == max(highs[6:])
    assert pytest.approx(df_30m["low"][1]) == min(lows[6:])
    assert pytest.approx(df_30m["close"][1]) == closes[11]
    assert df_30m["volume"][1] == 600.0
    assert df_30m["sub_bar_count"][1] == 6
