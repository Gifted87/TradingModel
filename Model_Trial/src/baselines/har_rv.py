"""
B1 — Linear HAR-RV Baseline [E1, E2, E5].

Formulation:
- Standard Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009; MDPI JRFM 2018 [E1]).
- Target: rv30 = log(sqrt(sum_{i=1}^h r_i^2) + eps) forward realized volatility in log space [E5].
- Inputs: G6 volatility state features (HAR-style log-RV at 5m, 15m, 30m, 1h, 1d + Parkinson & ATR).
- Estimator: Regularized Ridge regression (analytical numpy solution) to prevent multicollinearity.

Role in Architecture:
- Sets the literature's empirical gold standard for volatility forecasting.
- Component attribution (§7.5) requires the full cascade model's magnitude head to outperform B1.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import polars as pl


@dataclass
class HARModelSummary:
    intercept: float
    coefficients: Dict[str, float]
    alpha: float
    n_samples: int
    r2_train: float


class HARRVBaseline:
    """
    Linear HAR-RV volatility predictor (B1).
    Fits log-space forward realized volatility (rv30) from multi-horizon G6 volatility features.
    """

    DEFAULT_FEATURES = [
        "g6_log_rv_5m",
        "g6_log_rv_15m",
        "g6_log_rv_30m",
        "g6_log_rv_1h",
        "g6_log_rv_6h",
        "g6_log_rv_1d",
        "g6_log_parkinson_1bar",
        "g6_log_parkinson_12bar",
        "g6_log_garman_klass_1bar",
        "g6_log_atr14",
    ]

    def __init__(
        self,
        alpha: float = 1.0,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "rv30",
    ):
        self.alpha = alpha
        self.feature_cols = feature_cols or self.DEFAULT_FEATURES
        self.target_col = target_col
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self._summary: Optional[HARModelSummary] = None
        self._active_cols: List[str] = []

    def fit(self, df: pl.DataFrame) -> "HARRVBaseline":
        """Fit HAR-RV model on training DataFrame using analytical Ridge."""
        self._active_cols = [c for c in self.feature_cols if c in df.columns]
        if not self._active_cols:
            raise ValueError(f"None of the specified feature columns {self.feature_cols} found in DataFrame.")

        filter_expr = pl.col(self.target_col).is_not_nan() & pl.col(self.target_col).is_not_null()
        for c in self._active_cols:
            filter_expr = filter_expr & pl.col(c).is_not_nan() & pl.col(c).is_not_null()

        valid_df = df.filter(filter_expr)
        if len(valid_df) == 0:
            raise ValueError("No valid rows available to fit HAR-RV model.")

        X = valid_df.select(self._active_cols).to_numpy()
        y = valid_df[self.target_col].to_numpy()

        n, p = X.shape
        x_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_c = X - x_mean
        y_c = y - y_mean

        # Analytical Ridge: (X^T X + alpha * I) w = X^T y
        A = X_c.T @ X_c + self.alpha * np.eye(p)
        b = X_c.T @ y_c
        w = np.linalg.solve(A, b)
        intercept = float(y_mean - np.dot(x_mean, w))

        self.intercept_ = intercept
        self.coef_ = w

        # Training R²
        y_pred = intercept + X @ w
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

        coef_dict = {col: float(coef) for col, coef in zip(self._active_cols, w)}

        self._summary = HARModelSummary(
            intercept=intercept,
            coefficients=coef_dict,
            alpha=self.alpha,
            n_samples=len(valid_df),
            r2_train=r2,
        )
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Generate forward log realized volatility forecasts rv_hat."""
        if self.coef_ is None or not self._active_cols:
            raise RuntimeError("HAR-RV model is not fitted.")

        X_df = df.select(self._active_cols).fill_nan(0.0).fill_null(0.0)
        X = X_df.to_numpy()
        return self.intercept_ + X @ self.coef_

    @property
    def summary(self) -> HARModelSummary:
        if self._summary is None:
            raise RuntimeError("Model is not fitted.")
        return self._summary
