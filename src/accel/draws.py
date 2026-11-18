"""Summaries over score draws that compare two series within the same draw."""
from __future__ import annotations

import pandas as pd


def share_above(draws: pd.DataFrame, a: str, b: str, by: str = "draw_set") -> pd.DataFrame:
    """For each `by` group: the share of draws in which series `a` has a
    higher ratio than series `b` in the same draw. `draws` has one row per
    (by, draw, organization) with a `ratio` that is NaN when unsupported.
    `share_all_draws` counts a draw with either ratio missing as not above;
    `share_both_supported` uses only draws where both ratios exist."""
    if draws.duplicated([by, "draw", "organization"]).any():
        raise ValueError("draws must have one row per (group, draw, organization)")
    rows = []
    for g, d in draws.groupby(by, sort=True):
        wide = d.set_index(["draw", "organization"])["ratio"].unstack()
        if a not in wide or b not in wide:
            raise ValueError(f"{a!r} or {b!r} missing from group {g!r}")
        both = wide[[a, b]].dropna()
        rows.append({by: g, "series": a, "compared_with": b, "n_draws": len(wide),
                     "share_all_draws": float((wide[a] > wide[b]).mean()),
                     "n_both_supported": len(both),
                     "share_both_supported": float((both[a] > both[b]).mean()) if len(both) else float("nan")})
    return pd.DataFrame(rows)


GZIP_FIXED = {"method": "gzip", "mtime": 0}


def write_csv_gz(df: pd.DataFrame, path) -> None:
    """Write a gzip-compressed CSV with the header timestamp fixed at 0, so the
    same table written twice gives the same bytes."""
    df.to_csv(path, index=False, compression=GZIP_FIXED)
