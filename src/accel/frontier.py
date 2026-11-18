"""Release-date frontiers built from full model records.

A record is one fitted model: columns model, date, organization, score.
Every function here starts from records, never from an already-selected set
of frontier rows, so deleting or resampling models rebuilds the frontier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("model", "date", "organization", "score")


def _check(records: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(records.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")
    r = records.loc[:, list(REQUIRED)].copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    if r["model"].duplicated().any():
        raise ValueError(f"duplicate models: {sorted(r.loc[r.model.duplicated(), 'model'])}")
    if r[["date", "score"]].isna().any().any():
        raise ValueError("records contain missing dates or scores")
    return r


def daily_best(records: pd.DataFrame) -> pd.DataFrame:
    """One row per release day: the highest score that day.

    Ties on score are broken by model name, so the result does not depend on
    input row order. All models released that day are kept in `same_day_models`
    (sorted) for the audit file.
    """
    r = _check(records)
    r = r.sort_values(["date", "score", "model"], ascending=[True, False, True], kind="mergesort")
    same_day = r.groupby("date")["model"].apply(lambda s: ";".join(sorted(s)))
    best = r.groupby("date", sort=True).head(1).set_index("date")
    best["same_day_models"] = same_day
    best["n_same_day"] = r.groupby("date").size()
    return best.reset_index()


def step_frontier(records: pd.DataFrame, cutoff=None) -> pd.DataFrame:
    """Frontier points: release days whose best score strictly exceeds every
    earlier day's best. Records dated after `cutoff` are excluded first, so no
    later information enters (scores themselves are one data vintage)."""
    r = _check(records)
    if cutoff is not None:
        r = r[r["date"] <= pd.Timestamp(cutoff)]
    if r.empty:
        return pd.DataFrame(columns=list(REQUIRED) + ["same_day_models", "n_same_day"])
    d = daily_best(r)
    prior_max = d["score"].astype(float).cummax().shift(1, fill_value=-np.inf)
    return d[d["score"] > prior_max].reset_index(drop=True)


def frontier_in_window(records: pd.DataFrame, start, end) -> pd.DataFrame:
    """Frontier points dated in [start, end], computed on the full history up
    to `end`. Trimming early rows never promotes a model that an earlier model
    dominated (the frontier is built before the window is applied)."""
    f = step_frontier(records, cutoff=end)
    return f[f["date"] >= pd.Timestamp(start)].reset_index(drop=True)


def frontier_value(frontier: pd.DataFrame, when) -> tuple[float, str | None]:
    """Carry-forward step value of the frontier at `when` (NaN before the first
    point). Returns (score, model)."""
    when = pd.Timestamp(when)
    f = frontier[frontier["date"] <= when]
    if f.empty:
        return float("nan"), None
    last = f.iloc[-1]
    return float(last["score"]), str(last["model"])


def frontier_window_only(records: pd.DataFrame, start, end) -> pd.DataFrame:
    """Frontier built from the records dated in [start, end] only. Earlier
    models are dropped before the running maximum, so a model that an earlier
    release dominated can become the first point. This is not the construction
    used for the main results (see frontier_in_window); it exists to reproduce
    outside estimates that trim first."""
    r = _check(records)
    r = r[(r["date"] >= pd.Timestamp(start)) & (r["date"] <= pd.Timestamp(end))]
    return step_frontier(r)


def running_max_at(records: pd.DataFrame, dates, cutoff=None) -> pd.DataFrame:
    """Carry-forward frontier value (built on full history up to `cutoff`) on
    each of `dates`, sorted; NaN before the first frontier point. Columns:
    date, score, model."""
    f = step_frontier(records, cutoff=cutoff)[["date", "score", "model"]].sort_values("date")
    d = pd.DataFrame({"date": pd.to_datetime(pd.Series(list(dates))).dt.normalize()})
    d = d.sort_values("date").reset_index(drop=True)
    f["date"] = f["date"].astype(d["date"].dtype)
    return pd.merge_asof(d, f, on="date", direction="backward")
