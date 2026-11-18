"""Continuous-kink trend diagnostics on frontier points.

Model: y(t) = a + b*t + c*max(0, t - tau), t in years. Pre-slope b, post-slope
b + c. This is our public-data diagnostic; it is not claimed to be METR's
implementation.

Everything here is descriptive. The F-type statistic compares the kink fit
with a single line and has no calibrated null distribution (searching over
tau changes its distribution), so it is reported as a fit comparison, never
as a significance test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .frontier import frontier_in_window

MIN_DATES_PER_SIDE = 4
DAYS_PER_YEAR = 365.25
BASELINE_TOL = 1e-8


def years_since(dates, origin) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates))
    return ((d - pd.Timestamp(origin)).dt.days / DAYS_PER_YEAR).to_numpy(dtype=float)


@dataclass
class KinkFit:
    status: str              # ok | insufficient_dates | nonpositive_baseline
    tau: float
    n_pre: int               # distinct dates strictly before tau
    n_post: int              # distinct dates at or after tau
    pre_slope: float
    post_slope: float
    ratio: float             # post/pre, NaN unless status == ok
    ssr: float
    ssr_line: float
    f_type: float            # descriptive only

    def as_dict(self) -> dict:
        return asdict(self)


def fit_kink(t: np.ndarray, y: np.ndarray, tau: float,
             min_per_side: int = MIN_DATES_PER_SIDE) -> KinkFit:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n_pre = int(np.unique(t[t < tau]).size)
    n_post = int(np.unique(t[t >= tau]).size)
    X = np.column_stack([np.ones_like(t), t, np.clip(t - tau, 0.0, None)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ssr = float(((y - X @ beta) ** 2).sum())
    X1 = X[:, :2]
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    ssr1 = float(((y - X1 @ b1) ** 2).sum())
    dof = max(len(t) - 3, 1)
    f_type = (ssr1 - ssr) / (ssr / dof) if ssr > 0 else float("inf")
    pre, post = float(beta[1]), float(beta[1] + beta[2])
    if n_pre < min_per_side or n_post < min_per_side:
        status, ratio = "insufficient_dates", float("nan")
    elif pre <= BASELINE_TOL * max(1.0, abs(post)):
        # numerically zero or negative pre-slope: a ratio would be meaningless
        status, ratio = "nonpositive_baseline", float("nan")
    else:
        status, ratio = "ok", post / pre
    return KinkFit(status, float(tau), n_pre, n_post, pre, post, ratio, ssr, ssr1, f_type)


def fit_level_shift(t: np.ndarray, y: np.ndarray, tau: float,
                    min_per_side: int = MIN_DATES_PER_SIDE) -> dict:
    """One slope plus a jump at tau: y = a + b*t + d*1[t >= tau]. Same number of
    parameters as the kink, so SSRs compare directly (descriptively)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n_pre = int(np.unique(t[t < tau]).size)
    n_post = int(np.unique(t[t >= tau]).size)
    X = np.column_stack([np.ones_like(t), t, (t >= tau).astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ssr = float(((y - X @ beta) ** 2).sum())
    status = "ok" if (n_pre >= min_per_side and n_post >= min_per_side) else "insufficient_dates"
    return {"status": status, "tau": float(tau), "n_pre": n_pre, "n_post": n_post,
            "slope": float(beta[1]), "jump": float(beta[2]), "ssr": ssr}


def ssr_profile(t: np.ndarray, y: np.ndarray, taus: np.ndarray, kind: str = "kink",
                min_per_side: int = MIN_DATES_PER_SIDE) -> pd.DataFrame:
    """Vectorized break profile for many candidate taus at once, for use inside
    draw loops. kind="kink" fits a + b t + c max(0, t - tau) (same as fit_kink);
    kind="level_shift" fits a + b t + d 1[t >= tau] (same as fit_level_shift).
    Columns: tau, n_pre, n_post, supported (>= min_per_side distinct dates each
    side), slope (b), third (c for a kink, d for a jump), ssr."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    taus = np.asarray(taus, float)
    ut = np.unique(t)
    n_pre = np.searchsorted(ut, taus, side="left")          # distinct dates strictly before tau
    n_post = ut.size - n_pre
    T = np.broadcast_to(t, (taus.size, t.size))
    if kind == "kink":
        third = np.clip(T - taus[:, None], 0.0, None)
    elif kind == "level_shift":
        third = (T >= taus[:, None]).astype(float)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    X = np.stack([np.ones_like(T), T, third], axis=2)         # (k, n, 3)
    beta = np.linalg.pinv(X) @ y                              # (k, 3); pinv handles all-zero third columns
    ssr = ((y[None, :] - np.einsum("knp,kp->kn", X, beta)) ** 2).sum(axis=1)
    return pd.DataFrame({"tau": taus, "n_pre": n_pre, "n_post": n_post,
                         "supported": (n_pre >= min_per_side) & (n_post >= min_per_side),
                         "slope": beta[:, 1], "third": beta[:, 2], "ssr": ssr})


def best_of_profile(profile: pd.DataFrame) -> pd.Series | None:
    """Lowest-SSR supported row of an ssr_profile, or None."""
    ok = profile[profile["supported"]]
    if ok.empty:
        return None
    return ok.loc[ok["ssr"].idxmin()]


def break_profile(t: np.ndarray, y: np.ndarray, taus: np.ndarray,
                  min_per_side: int = MIN_DATES_PER_SIDE) -> pd.DataFrame:
    """Fit the kink at every candidate tau. Rows keep their status, so
    unsupported candidates are visible rather than dropped."""
    return pd.DataFrame([fit_kink(t, y, tau, min_per_side).as_dict() for tau in taus])


def best_supported_break(profile: pd.DataFrame) -> pd.Series | None:
    """Lowest-SSR candidate among those with enough dates on both sides
    (status ok or nonpositive_baseline). None if no candidate qualifies."""
    ok = profile[profile["status"] != "insufficient_dates"]
    if ok.empty:
        return None
    return ok.loc[ok["ssr"].idxmin()]


def series_from_records(records: pd.DataFrame, start, end, origin) -> pd.DataFrame:
    f = frontier_in_window(records, start, end)
    return f.assign(t=years_since(f["date"], origin))


def leave_one_model_out(records: pd.DataFrame, start, end, origin, tau: float,
                        models: list[str] | None = None) -> pd.DataFrame:
    """Influence diagnostics: drop one model from the *records*, rebuild the
    frontier, refit at a fixed tau. A previously dominated model can enter the
    frontier when the model above it is removed."""
    base = series_from_records(records, start, end, origin)
    candidates = models if models is not None else list(base["model"])
    rows = []
    for m in candidates:
        s = series_from_records(records[records["model"] != m], start, end, origin)
        fit = fit_kink(s["t"].to_numpy(), s["score"].to_numpy(), tau)
        entered = sorted(set(s["model"]) - set(base["model"]))
        rows.append({"dropped_model": m, "entered_frontier": ";".join(entered),
                     "n_points": len(s), **fit.as_dict()})
    return pd.DataFrame(rows)


def kink_coefficients(t: np.ndarray, y: np.ndarray, tau: float) -> tuple[np.ndarray, float]:
    """Least-squares (a, b, c) of y = a + b t + c max(0, t - tau), and the SSR.
    Same fit as fit_kink; returns the intercept too, for drawing the line."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(t), t, np.clip(t - tau, 0.0, None)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, float(((y - X @ beta) ** 2).sum())


def level_shift_coefficients(t: np.ndarray, y: np.ndarray, tau: float) -> tuple[np.ndarray, float]:
    """Least-squares (a, b, d) of y = a + b t + d 1[t >= tau], and the SSR.
    Same fit as fit_level_shift."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(t), t, (t >= tau).astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, float(((y - X @ beta) ** 2).sum())
