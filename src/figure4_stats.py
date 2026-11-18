"""Fit statistics behind note Figure 4.

Recomputes both panels exactly as src/figures.py figure4() does: Anthropic's fourteen frontier points on the
published scores, the best freely dated change of slope and the best freely dated one-off step (dates from
results/developer_comparison/slope_vs_step_point.csv), with each fit's coefficients and sum of squared residuals. A
third row fits one straight line with no break to the same points, for reference; it is not drawn.
Output: results/developer_comparison/figure4_fit_stats.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from accel import trend
from figures import CC, ORIGIN, anthropic_points


def main():
    p = anthropic_points("epoch_published")
    t, y = p["t"].to_numpy(), p["score"].to_numpy()
    shape = pd.read_csv(CC / "slope_vs_step_point.csv").set_index("source").loc["epoch_published"]
    kink_day, step_day = pd.Timestamp(shape.kink_best_date), pd.Timestamp(shape.step_best_first_day)
    bk, ssr_k = trend.kink_coefficients(t, y, float(trend.years_since([kink_day], ORIGIN)[0]))
    bs, ssr_s = trend.level_shift_coefficients(t, y, float(trend.years_since([step_day], ORIGIN)[0]))
    assert np.isclose(ssr_k, shape.kink_best_ssr) and np.isclose(ssr_s, shape.step_best_ssr)
    rows = [
        {"panel": "left: best change of slope", "date_first": kink_day.date(), "date_last": kink_day.date(),
         "n_points": len(p), "slope_before": bk[1], "slope_after": bk[1] + bk[2], "ratio": (bk[1] + bk[2]) / bk[1],
         "step_points": np.nan, "ssr": ssr_k},
        {"panel": "right: best one-off step", "date_first": step_day.date(),
         "date_last": pd.Timestamp(shape.step_best_last_day).date(), "n_points": len(p),
         "slope_before": bs[1], "slope_after": bs[1], "ratio": 1.0, "step_points": bs[2], "ssr": ssr_s},
    ]
    # Reference (not drawn in Figure 4): one straight line, no break, through the same fourteen points.
    X1 = np.column_stack([np.ones_like(t), t])
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    ssr_1 = float(((y - X1 @ b1) ** 2).sum())
    assert np.isclose(ssr_1, trend.fit_kink(t, y, float(trend.years_since([kink_day], ORIGIN)[0])).ssr_line)
    rows.append({"panel": "reference: one straight line, no break", "date_first": np.nan, "date_last": np.nan,
                 "n_points": len(p), "slope_before": b1[1], "slope_after": b1[1], "ratio": 1.0,
                 "step_points": np.nan, "ssr": ssr_1})
    out = pd.DataFrame(rows)
    out.to_csv(CC / "figure4_fit_stats.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
