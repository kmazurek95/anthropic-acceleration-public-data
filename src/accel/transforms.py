"""Scale transformations and the stylized-path illustration.

g_k(x) = (exp(k x) - 1) / k, g_0 = identity. Increasing for every k. k > 0
weights higher points more (convex), k < 0 less (concave). exp(k * 25) is the
marginal weight of a point 25 higher relative to a point lower down.

The stylized path is an ILLUSTRATION on a path built from METR's reported
slopes and assumed dates. It is not a re-estimate of METR's model and does not
show that any particular transformation is plausible.
"""
from __future__ import annotations

from datetime import date

import numpy as np


def g(x, k: float):
    x = np.asarray(x, float)
    if abs(k) < 1e-12:
        return x.copy()
    return np.expm1(k * x) / k


def affine(x, a: float, b: float):
    if b <= 0:
        raise ValueError("positive affine maps only")
    return a + b * np.asarray(x, float)


# Stylized path: METR's reported slopes (15 and 29 points/yr) on assumed dates.
PRE_SLOPE, POST_SLOPE = 15.0, 29.0
T0, TB, T1 = date(2024, 3, 1), date(2025, 10, 7), date(2026, 10, 15)
PRE_YEARS = (TB - T0).days / 365.25
POST_YEARS = (T1 - TB).days / 365.25


def stylized_ratio(k: float, pre_slope=PRE_SLOPE, post_slope=POST_SLOPE,
                   pre_years=PRE_YEARS, post_years=POST_YEARS) -> float:
    """Ratio of average progress rates (post/pre) along the two-segment path
    after rescaling by g_k, with the origin at the break."""
    a = pre_slope * pre_years
    b = post_slope * post_years
    if abs(k) < 1e-12:
        return (b / post_years) / (a / pre_years)
    post = np.expm1(k * b) / post_years
    pre = -np.expm1(-k * a) / pre_years
    return float(post / pre)


def milestone_crossing_time(t, y, threshold: float) -> float:
    """First time a (nondecreasing, piecewise-linear) path reaches `threshold`;
    NaN if never. Used to show that crossing times are invariant when the path
    and the threshold are transformed together."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    hit = np.flatnonzero(y >= threshold)
    if hit.size == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(t[0])
    t0, t1, y0, y1 = t[i - 1], t[i], y[i - 1], y[i]
    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))
