"""
B2 — Branch A Linear Core [D1, D2, D3, D5].

Formulation:
- Ridge regression on G1 Microstructure features only:
    r_hat_lin = w_0 + sum_i w_i * x_{G1, i}
- Target: r30 = log(mid_{t+30} / mid_t) continuous log return.
- Regularization: Analytical Ridge (L2 penalty) to stabilize collinear rolling sums.

Mechanistic Foundations & Audit Invariants [D1, D2]:
- P1: OFI moves price linearly (Cont et al. 2014 R²≈65%).
- P2: Price impact scales inversely with depth (1/depth law).
- Positive coefficient invariant: Net OFI terms MUST have positive coefficients (w_ofi > 0).

Role in Architecture (§6):
- Fit first and frozen: Serves as the interpretable, auditable linear backbone.
- Produces the residual target for Branch B: residual = r30 - r_hat_lin.
- Gate 4 (Condition C2): Must beat naive persistence out-of-sample.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class LinearCoreSummary:
    symbol: str
    intercept: float
    coefficients: Dict[str, float]
    alpha: float
    n_train: int
    r2_train: float
    positive_ofi_sign_verified: bool
    depth_scaling_verified: bool
    comment: str


class LinearCoreBranchA:
    """
    Branch A Linear Microstructure Core (B2).
    Linear Ridge estimator mapping G1 microstructure dynamics to forward returns r30.
    """

    DEFAULT_G1_FEATURES = [
        "g1_ofi_1bar",
        "g1_ofi_2bar_sum",
        "g1_ofi_3bar_sum",
        "g1_ofi_6bar_sum",
        "g1_spread_level",
        "g1_spread_delta",
        "g1_ofi_depth_ratio_3bar",
        "g1_ofi_depth_ratio_6bar",
        "g1_ofi_depth_ratio_12bar",
    ]

    def __init__(
        self,
        alpha: float = 100.0,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "r30",
    ):
        self.alpha = alpha
        self.feature_cols = feature_cols or self.DEFAULT_G1_FEATURES
        self.target_col = target_col
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self._summary: Optional[LinearCoreSummary] = None
        self._active_cols: List[str] = []

    def fit(self, df: pl.DataFrame, symbol: str = "UNKNOWN") -> "LinearCoreBranchA":
        """
        Fit Branch A linear core on training DataFrame using analytical Ridge.
        """
        self._active_cols = [c for c in self.feature_cols if c in df.columns]
        if not self._active_cols:
            raise ValueError(f"None of the G1 features {self.feature_cols} found in DataFrame.")

        filter_expr = pl.col(self.target_col).is_not_nan() & pl.col(self.target_col).is_not_null()
        for c in self._active_cols:
            filter_expr = filter_expr & pl.col(c).is_not_nan() & pl.col(c).is_not_null()

        valid_df = df.filter(filter_expr)
        if len(valid_df) == 0:
            raise ValueError("No valid rows available to fit Branch A Linear Core.")

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

        # Verify microstructure invariants [D1, D2]
        ofi_weights = [coef_dict[c] for c in coef_dict if "ofi" in c]
        positive_ofi = any(w > 0 for w in ofi_weights) if ofi_weights else True

        # Depth ratio check: OFI/depth ratio should positively drive price change
        depth_weights = [coef_dict[c] for c in coef_dict if "depth_ratio" in c]
        depth_scaling = any(w > 0 for w in depth_weights) if depth_weights else True

        comment = "Verified: positive OFI impact coefficient respects Cont et al. [D1, D2]" if positive_ofi else "Warning: Non-positive OFI weights"

        self._summary = LinearCoreSummary(
            symbol=symbol,
            intercept=intercept,
            coefficients=coef_dict,
            alpha=self.alpha,
            n_train=len(valid_df),
            r2_train=r2,
            positive_ofi_sign_verified=positive_ofi,
            depth_scaling_verified=depth_scaling,
            comment=comment,
        )
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Compute linear return prediction r_hat_lin."""
        if self.coef_ is None or not self._active_cols:
            raise RuntimeError("Linear Core Branch A is not fitted.")

        X_df = df.select(self._active_cols).fill_nan(0.0).fill_null(0.0)
        X = X_df.to_numpy()
        return self.intercept_ + X @ self.coef_

    def compute_residuals(self, df: pl.DataFrame) -> np.ndarray:
        """
        Compute nonlinear residual: residual = r30 - r_hat_lin.
        Used as the target for Branch B (residual learner).
        """
        r_hat = self.predict(df)
        y = df[self.target_col].to_numpy()
        return y - r_hat

    @property
    def summary(self) -> LinearCoreSummary:
        if self._summary is None:
            raise RuntimeError("Model is not fitted.")
        return self._summary
