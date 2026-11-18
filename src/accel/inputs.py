"""Frozen-input checks and the model-group table with validated joins.

Each fitted model group gets one release date and one organization; any
conflict is recorded in an audit table, never silently resolved.

Date rule (stated, not inferred): a group's date is the LATEST date among its
metadata rows. That matches how Epoch's loader dates score rows (the latest
version scored) and the date column in Epoch's bundled eci_scores.csv for
98% of models. The earliest date is kept alongside for a sensitivity check.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "b07bd2eb34cc26469ea832daa2b17e6088f127caa3361c0ee78bac1d05671814"  # Wayback capture 20261113150458; see provenance/wayback_comparison/REPORT.md
ECI_COMMIT = "e04929ab751531e558b1a61e86f436487c710ef8"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_input(path: str | Path, expected: str = EXPECTED_SHA256) -> str:
    got = sha256(path)
    if got != expected:
        raise ValueError(f"input hash mismatch for {path}: {got} != {expected}")
    return got


def read_model_metadata(zip_path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        return pd.read_csv(z.open("model_metadata.csv"))


def group_table(meta: pd.DataFrame, groups, org_fallback: dict | None = None,
                allow_unknown_org: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per requested model group: date (latest), date_earliest,
    organization, accessibility. Returns (table, audit).

    Raises if a group is missing from the metadata, has no usable date, or has
    more than one non-null organization. A group with no organization in the
    metadata takes it from `org_fallback` (e.g. Epoch's bundled eci_scores.csv)
    if given, else raises; every fallback is listed in the audit. Date
    disagreements are allowed but listed in the audit.
    """
    m = meta.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m[m["model_group"].isin(set(groups))]
    missing = sorted(set(groups) - set(m["model_group"]))
    if missing:
        raise ValueError(f"groups absent from model metadata: {missing}")
    agg = m.groupby("model_group").agg(
        date=("date", "max"),
        date_earliest=("date", "min"),
        n_dates=("date", "nunique"),
        n_rows=("model_version", "size"),
        n_org=("organization", "nunique"),
        organization=("organization", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA),
        accessibility=("accessibility", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
    )
    if agg["date"].isna().any():
        raise ValueError(f"groups with no date: {sorted(agg.index[agg['date'].isna()])}")
    if (agg["n_org"] > 1).any():
        raise ValueError(f"groups with conflicting organizations: {sorted(agg.index[agg['n_org'] > 1])}")
    no_org = agg["organization"].isna()
    agg["org_source"] = "model_metadata"
    if no_org.any() and org_fallback is not None:
        vals = [org_fallback.get(g) for g in agg.index[no_org]]
        agg.loc[no_org, "organization"] = [v if isinstance(v, str) and v else None for v in vals]
        agg.loc[no_org, "org_source"] = "fallback"
    still = agg["organization"].isna()
    if still.any() and allow_unknown_org:
        agg.loc[still, "organization"] = "unknown"
        agg.loc[still, "org_source"] = "unknown"
    if agg["organization"].isna().any():
        raise ValueError(f"groups with no organization: {sorted(agg.index[agg['organization'].isna()])}")
    issues = []
    for g, r in agg.iterrows():
        if r["n_dates"] > 1:
            issues.append((g, "multiple_dates"))
        elif r["n_rows"] > 1:
            issues.append((g, "multiple_versions_same_date"))
        if r["org_source"] == "fallback":
            issues.append((g, "organization_from_fallback"))
        if r["org_source"] == "unknown":
            issues.append((g, "organization_unknown"))
    audit = pd.DataFrame(issues, columns=["model_group", "issue"]).merge(
        agg.reset_index(), on="model_group", how="left")
    return agg.reset_index().rename(columns={"model_group": "model"}), audit
