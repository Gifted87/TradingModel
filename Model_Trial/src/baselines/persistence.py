"""
B0 — Persistence Baseline [C2, E1].

Formulation:
- Return forecast: r_hat = 0 (martingale assumption for price)
- Magnitude forecast: trailing rolling mean of |r30| or log(|r30|)
- Realized Volatility forecast: trailing rolling mean of rv30 (random walk / naive persistence)

Role in Architecture:
- Defines the absolute floor for direction, magnitude, and volatility prediction.
- Gate 4 / Condition C2 requires Branch A (linear core) to strictly beat B0 out-of-sample.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class PersistencePredictions:
    r_hat: np.ndarray          # Expected return: zeros
    mag_hat: np.ndarray        # Magnitude estimate (log-space)
    rv_hat: np.ndarray         # Realized volatility estimate (log-space)


class PersistenceBaseline:
    """
    Naive persistence benchmark (B0).
    Produces zero return forecast and trailing historical means for magnitude & volatility.
    """

    def __init__(self, trailing_window: int = 72):
        """
        Parameters
        ----------
        trailing_window : int
            Number of trailing bars to average for magnitude / RV baseline (default: 72 bars = 6h).
        """
        self.trailing_window = trailing_window
        self.mean_mag_train: float = 0.0
        self.mean_rv_train: float = 0.0

    def fit(self, df: pl.DataFrame) -> "PersistenceBaseline":
        """Fit global training mean for fallback / initial rows."""
        if "mag30" in df.columns:
            m = df["mag30"].drop_nulls().drop_nans()
            self.mean_mag_train = float(m.mean()) if len(m) > 0 else -8.0

        if "rv30" in df.columns:
            rv = df["rv30"].drop_nulls().drop_nans()
            self.mean_rv_train = float(rv.mean()) if len(rv) > 0 else -8.0

        return self

    def predict(self, df: pl.DataFrame) -> PersistencePredictions:
        """
        Generate B0 persistence forecasts.
        r_hat is identically 0.0.
        mag_hat & rv_hat use trailing historical rolling averages.
        """
        n = len(df)
        r_hat = np.zeros(n, dtype=np.float64)

        if "mag30" in df.columns:
            mag_series = df["mag30"].fill_null(self.mean_mag_train).fill_nan(self.mean_mag_train)
            mag_hat = (
                mag_series.rolling_mean(window_size=self.trailing_window)
                .fill_null(self.mean_mag_train)
                .to_numpy()
            )
        else:
            mag_hat = np.full(n, self.mean_mag_train, dtype=np.float64)

        if "rv30" in df.columns:
            rv_series = df["rv30"].fill_null(self.mean_rv_train).fill_nan(self.mean_rv_train)
            rv_hat = (
                rv_series.rolling_mean(window_size=self.trailing_window)
                .fill_null(self.mean_rv_train)
                .to_numpy()
            )
        else:
            rv_hat = np.full(n, self.mean_rv_train, dtype=np.float64)

        return PersistencePredictions(r_hat=r_hat, mag_hat=mag_hat, rv_hat=rv_hat)
