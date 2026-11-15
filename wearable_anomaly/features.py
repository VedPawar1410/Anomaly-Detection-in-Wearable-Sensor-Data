"""Causal (past-only) rolling features, computed within each subject.

"Causal" means each row only uses samples before it, so the same code works on a
live stream: nothing here peeks at the future.
"""

import numpy as np
import pandas as pd


def subject_blocks(df: pd.DataFrame):
    """Yield (row slice, block) per subject. Rows of one subject are contiguous."""
    if "subject" not in df:
        yield slice(0, len(df)), df
        return
    subj = df["subject"].to_numpy()
    starts = np.flatnonzero(np.r_[True, subj[1:] != subj[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends, strict=True):
        yield slice(s, e), df.iloc[s:e]


def run_length(a: np.ndarray) -> np.ndarray:
    """For each cell, how many consecutive identical values end here (1 = value just changed)."""
    n = len(a)
    change = np.ones(a.shape, dtype=bool)
    change[1:] = a[1:] != a[:-1]
    rows = np.arange(n)[:, None]
    last_change = np.maximum.accumulate(np.where(change, rows, 0), axis=0)
    return (rows - last_change + 1).astype(np.float32)


def past_rolling(block: pd.DataFrame, window: int, stat: str, min_periods: int) -> pd.DataFrame:
    """Rolling statistic over the `window` samples *before* each row."""
    return getattr(block.shift(1).rolling(window, min_periods=min_periods), stat)()
