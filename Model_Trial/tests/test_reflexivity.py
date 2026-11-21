"""
Tests for George Soros Reflexivity Reversal Model (src.reflexivity).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.reflexivity import (
    ReflexivityConfig,
    ReflexivityPredictor,
    prepare_arrays,
    run_reflexivity,
    run_reflexivity_fsm,
    summarize_trades,
)
from src.validation.harness import ValidationHarness


@pytest.fixture
def sample_bars() -> pl.DataFrame:
    """Generate synthetic 1m/5m microstructure bars with realistic schema."""
    np.random.seed(42)
    n = 2000
    dates = pl.datetime_range(
        pl.datetime(2023, 1, 1, 0, 0),
        pl.datetime(2023, 1, 8, 0, 0),
        interval="5m",
        eager=True,
    ).slice(0, n)

    ret = np.random.normal(0, 0.0005, size=n)
    # Inject an extreme bust and reversal
    ret[200:205] = -0.005  # crash
    ret[206:215] = +0.003  # turnaround
    ret[800:805] = +0.005  # spike
    ret[806:815] = -0.003  # turnaround

    close = 1.1000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(np.random.normal(0, 0.0002, size=n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0002, size=n)))
    open_px = close * (1 + np.random.normal(0, 0.0001, size=n))

    volume = np.random.exponential(100.0, size=n) + 10.0
    volume[200:205] *= 10.0  # volume climax
    volume[800:805] *= 10.0

    bid_close = close - 0.00005
    ask_close = close + 0.00005
    bid_open = open_px - 0.00005
    ask_open = open_px + 0.00005
    spread_close = ask_close - bid_close
    bid_vol = volume * 0.5
    ask_vol = volume * 0.5

    # Labels for walk-forward harness
    r30 = np.concatenate([np.log(close[30:] / close[:-30]), np.zeros(30)])
    rv30 = np.full(n, -8.0)
    mag30 = np.log(np.abs(r30) + 1e-6)

    df = pl.DataFrame({
        "timestamp": dates,
        "open": open_px,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "bid_open": bid_open,
        "bid_close": bid_close,
        "ask_open": ask_open,
        "ask_close": ask_close,
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "spread_close": spread_close,
        "r30": r30,
        "rv30": rv30,
        "mag30": mag30,
    })
    return df


def test_prepare_arrays(sample_bars: pl.DataFrame):
    cfg = ReflexivityConfig(vwap_window=48, z_window=48, low_lookback=12, climax_window=48)
    a = prepare_arrays(sample_bars, cfg)
    assert len(a["close"]) == len(sample_bars)
    assert "vwap" in a
    assert "z" in a
    assert "atr" in a
    assert "rsi" in a
    assert "extreme_l" in a
    assert "extreme_s" in a


def test_run_reflexivity(sample_bars: pl.DataFrame):
    cfg = ReflexivityConfig(vwap_window=48, z_window=48, low_lookback=12, climax_window=48)
    result = run_reflexivity(sample_bars, cfg)
    assert "reflex_signal" in result.signals.columns
    assert "reflex_state" in result.signals.columns
    assert isinstance(result.summary, dict)


def test_reflexivity_predictor_harness_fit_predict(sample_bars: pl.DataFrame):
    cfg = ReflexivityConfig(vwap_window=48, z_window=48, low_lookback=12, climax_window=48)
    predictor = ReflexivityPredictor(cfg)

    train_df = sample_bars.slice(0, 1500)
    test_df = sample_bars.slice(1500, 500)

    predictor.fit(train_df, symbol="TEST")
    r_hat, rv_hat = predictor.predict(test_df)

    assert len(r_hat) == len(test_df)
    assert len(rv_hat) == len(test_df)
    assert isinstance(r_hat, np.ndarray)
    assert isinstance(rv_hat, np.ndarray)
