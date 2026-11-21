"""
Phase 5 — Purged Walk-Forward Split Engine.

Upgrades the bare-tuple split generator from Phase 4's evaluator into a
typed, self-documenting `WalkForwardSplit` dataclass.  The purge / embargo
logic is unchanged — this module is the canonical implementation going
forward; Phase 4's `generate_walk_forward_splits` remains for backward
compatibility but delegates here.

Build-plan reference (Phase 5, point 1):
  - Train 24m → purge 30 bars → embargo → validate 1m → roll.
  - No random CV anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import polars as pl


@dataclass(frozen=True)
class WalkForwardSplit:
    """
    One fold of a purged walk-forward cross-validation.

    Indices are row-positions into the full (sorted) DataFrame.
    The gap [train_end, test_start) is the combined purge + embargo buffer
    — no row in that range is used for either training or testing.
    """

    fold_idx: int
    train_start: int
    train_end: int       # exclusive
    test_start: int      # exclusive of embargo — first usable test row
    test_end: int        # exclusive
    embargo_bars: int    # bars actually dropped between train_end and test_start

    # ── Human-readable timestamps (set after slicing against the DataFrame) ──
    train_start_ts: str = ""
    train_end_ts: str = ""
    test_start_ts: str = ""
    test_end_ts: str = ""

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start


def generate_splits(
    df: pl.DataFrame,
    train_bars: int = 80_000,
    test_bars: int = 20_000,
    step_bars: int = 20_000,
    embargo_bars: int = 30,
) -> List[WalkForwardSplit]:
    """
    Generate chronological, non-overlapping purged walk-forward splits.

    Parameters
    ----------
    df : pl.DataFrame
        The full dataset, **must be sorted by timestamp** before calling.
    train_bars : int
        Number of bars in each training window.
    test_bars : int
        Number of bars in each test window.
    step_bars : int
        How many bars to advance the window each fold.
    embargo_bars : int
        Bars dropped between train_end and test_start to prevent label-overlap
        leakage (adjacent samples share horizon_bars - 1 forward bars).

    Returns
    -------
    List[WalkForwardSplit]
        Ordered list of folds; at least one fold is always returned if the
        dataset is large enough.
    """
    n = len(df)

    # Grab timestamp column for human-readable labels.
    ts_col = "timestamp" if "timestamp" in df.columns else None

    def _ts(idx: int) -> str:
        if ts_col is None or idx >= n or idx < 0:
            return ""
        return str(df[ts_col][idx])[:19]

    splits: List[WalkForwardSplit] = []
    start_idx = 0
    fold_idx = 1

    while True:
        train_start = start_idx
        train_end = train_start + train_bars
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars

        if test_start >= n:
            break

        # Accept a short final fold if it has at least half the target length.
        actual_test_end = min(test_end, n)
        if actual_test_end - test_start < test_bars // 2:
            break

        splits.append(
            WalkForwardSplit(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=min(train_end, n),
                test_start=test_start,
                test_end=actual_test_end,
                embargo_bars=embargo_bars,
                train_start_ts=_ts(train_start),
                train_end_ts=_ts(min(train_end, n) - 1),
                test_start_ts=_ts(test_start),
                test_end_ts=_ts(actual_test_end - 1),
            )
        )

        start_idx += step_bars
        fold_idx += 1

    # Fallback: if the dataset is too small for even one full fold, use a
    # single 75/25 split with the embargo buffer.
    if not splits:
        n_train = max(1, int(n * 0.75))
        test_start = n_train + embargo_bars
        if test_start < n:
            splits.append(
                WalkForwardSplit(
                    fold_idx=1,
                    train_start=0,
                    train_end=n_train,
                    test_start=test_start,
                    test_end=n,
                    embargo_bars=embargo_bars,
                    train_start_ts=_ts(0),
                    train_end_ts=_ts(n_train - 1),
                    test_start_ts=_ts(test_start),
                    test_end_ts=_ts(n - 1),
                )
            )

    return splits
