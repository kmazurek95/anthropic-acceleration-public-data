"""Catch-up diagnostics: one developer's frontier against the whole field.

Both frontiers are carry-forward step frontiers built on full history (the
same construction as frontier.step_frontier), evaluated daily. The field
includes the developer, so the gap (developer minus field) is never positive;
a gap of zero means the developer holds the field's frontier that day.
Descriptive only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frontier import step_frontier

LEAD_TOL = 1e-9


def _daily(frontier: pd.DataFrame, days: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    f = frontier[["date", "score", "model"]].sort_values("date")
    out = pd.merge_asof(pd.DataFrame({"date": days}), f, on="date", direction="backward")
    return out.rename(columns={"score": f"{prefix}_best", "model": f"{prefix}_model"})


def gap_series(org_records: pd.DataFrame, field_records: pd.DataFrame, start, end) -> pd.DataFrame:
    """Daily rows from start to end: the developer's best score to date, the
    field's best to date, the models holding each, and gap = developer - field.
    `field_records` must contain the developer's records."""
    if not set(org_records["model"]).issubset(set(field_records["model"])):
        raise ValueError("field_records must include every developer record")
    days = pd.date_range(start, end, freq="D")
    org = _daily(step_frontier(org_records, cutoff=end), days, "org")
    fld = _daily(step_frontier(field_records, cutoff=end), days, "field")
    g = org.merge(fld, on="date")
    g["gap"] = g["org_best"] - g["field_best"]
    g["org_leads"] = g["gap"].abs() <= LEAD_TOL
    return g


def lead_spells(gaps: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of days on which the developer holds the field's
    frontier: start, end, days, and the developer's model at the start."""
    lead = gaps["org_leads"].to_numpy()
    if not lead.any():
        return pd.DataFrame(columns=["start", "end", "days", "model"])
    edges = np.diff(np.r_[0, lead.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return pd.DataFrame({"start": gaps["date"].iloc[starts].to_numpy(),
                         "end": gaps["date"].iloc[ends].to_numpy(),
                         "days": ends - starts + 1,
                         "model": gaps["org_model"].iloc[starts].to_numpy()})


def gap_summary(gaps: pd.DataFrame, at, after=None) -> dict:
    """Gap on date `at`; the deepest gap and its date; and the first date on
    or after `after` (default `at`) on which the developer leads again."""
    at = pd.Timestamp(at)
    after = at if after is None else pd.Timestamp(after)
    row = gaps.loc[gaps["date"] == at].iloc[0]
    deepest = gaps.loc[gaps["gap"].idxmin()]
    back = gaps[(gaps["date"] >= after) & gaps["org_leads"]]
    return {"gap_at": float(row["gap"]), "org_model_at": row["org_model"], "field_model_at": row["field_model"],
            "deepest_gap": float(deepest["gap"]), "deepest_gap_date": deepest["date"].date(),
            "lead_regained": back["date"].iloc[0].date() if not back.empty else None,
            "lead_regained_model": back["org_model"].iloc[0] if not back.empty else None}


def cross_ratio(org_post_slope: float, field_pre_slope: float) -> float:
    """The developer's post-break slope over the field's pre-break slope: the
    developer's later pace against the field's earlier pace. NaN if the field's
    pre-break slope is not positive."""
    return org_post_slope / field_pre_slope if field_pre_slope > 0 else float("nan")
