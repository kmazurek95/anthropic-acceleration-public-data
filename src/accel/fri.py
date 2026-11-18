"""FRI's "fast year": the largest rise of the all-developer step frontier over a
12-month window.

Primary: complete 12-calendar-month windows (start + DateOffset(months=12))
that end on or before the cutoff, one window per start day, carry-forward
frontier values, no interpolation, no extrapolation past the cutoff.
Sensitivity: 365-day windows, reported separately.

A window counts only if the frontier is defined at its start (on or after the
first frontier point). Calendar months handle leap days the pandas way:
2024-02-29 + 12 months = 2025-02-28.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frontier import step_frontier


def _values_on(frontier: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    dates = frontier["date"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(dates, days.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    score = np.where(idx >= 0, frontier["score"].to_numpy(float)[np.clip(idx, 0, None)], np.nan)
    model = np.where(idx >= 0, frontier["model"].to_numpy(object)[np.clip(idx, 0, None)], None)
    return score, model


def window_rises(records: pd.DataFrame, cutoff, kind: str = "calendar12",
                 first_start=None) -> pd.DataFrame:
    """One row per admissible start day with the rise, end date and endpoint models."""
    cutoff = pd.Timestamp(cutoff).normalize()
    f = step_frontier(records, cutoff=cutoff)
    if f.empty:
        return pd.DataFrame()
    start0 = f["date"].iloc[0] if first_start is None else max(pd.Timestamp(first_start), f["date"].iloc[0])
    starts = pd.date_range(start0, cutoff, freq="D")
    if kind == "calendar12":
        ends = pd.DatetimeIndex([s + pd.DateOffset(months=12) for s in starts])
    elif kind == "days365":
        ends = starts + pd.Timedelta(days=365)
    else:
        raise ValueError(kind)
    keep = ends <= cutoff
    starts, ends = starts[keep], ends[keep]
    v0, m0 = _values_on(f, starts)
    v1, m1 = _values_on(f, ends)
    return pd.DataFrame({"start": starts, "end": ends, "start_score": v0, "end_score": v1,
                         "rise": v1 - v0, "start_model": m0, "end_model": m1, "window": kind})


def summarise_rises(rises: pd.DataFrame, threshold: float = 30.0) -> dict:
    if rises.empty:
        return {"n_windows": 0}
    i = rises["rise"].idxmax()
    top = rises.loc[i]
    ties = rises[np.isclose(rises["rise"], top["rise"])]
    return {
        "window": top["window"],
        "n_windows": int(len(rises)),
        "max_rise": float(top["rise"]),
        "first_max_start": top["start"].date().isoformat(),
        "first_max_end": top["end"].date().isoformat(),
        "max_start_range": f"{ties['start'].min().date()}..{ties['start'].max().date()}",
        "start_model": top["start_model"],
        "end_model": top["end_model"],
        "start_score": float(top["start_score"]),
        "end_score": float(top["end_score"]),
        "any_window_ge_threshold": bool((rises["rise"] >= threshold).any()),
        "share_windows_ge_threshold": float((rises["rise"] >= threshold).mean()),
        "threshold": threshold,
    }
