"""Slope change against step with the window end moved to 2026-11-13.

The free-date slope change against the free-date step for Anthropic, as in developer_comparison.py, with the trend
window's end moved from 2026-10-31 to the snapshot date so Claude Fable 5.2 (dated 2026-11-09) enters. Same frontier
rule, fixed 2025-10-07 break for the fixed-break comparison, same fit functions (free_shapes, fit_kink,
fit_level_shift); the daily grid of candidate dates runs to the new window end. Point scores under all three sources,
then the refit's 4,000 posterior draws and Epoch's 500 least-squares bootstrap draws, frontier rebuilt in each draw.
With the window end at 2026-10-31 the point rows reproduce slope_vs_step_point.csv exactly.

Outputs (results/developer_comparison/): slope_vs_step_window_end_point.csv,
slope_vs_step_window_end_draw_summary.csv, slope_vs_step_window_end_per_draw.csv.gz.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from accel import draws as draws_mod  # noqa: E402
from accel import trend  # noqa: E402
from developer_comparison import free_shapes, ls_bootstrap  # noqa: E402
from analysis import ORIGIN, TAU_DATE, WINDOW, load_sources  # noqa: E402

OUT = ROOT / "results" / "developer_comparison"
ORG = "Anthropic"
END = "2026-11-13"
STEM = "slope_vs_step_window_end"


def shapes(rec: pd.DataFrame, tau: float, taus: np.ndarray, days: pd.DatetimeIndex) -> dict:
    s = trend.series_from_records(rec[rec.organization == ORG], WINDOW[0], END, ORIGIN)
    t, y = s["t"].to_numpy(), s["score"].to_numpy()
    k = trend.fit_kink(t, y, tau)
    ls_fixed = trend.fit_level_shift(t, y, tau)
    return {"window_end": END, "n_points": len(s), **(free_shapes(t, y, taus, days) or {}),
            "kink_ssr_fixed": k.ssr, "step_ssr_fixed": ls_fixed["ssr"],
            "kink_beats_step_fixed": bool(k.ssr < ls_fixed["ssr"])}


def main():
    sources, _, _ = load_sources()
    released = {k: v[~v["unreleased"]].reset_index(drop=True) for k, v in sources.items()}
    tau = float(trend.years_since([TAU_DATE], ORIGIN)[0])
    days = pd.date_range(WINDOW[0], END, freq="D")
    taus = trend.years_since(days, ORIGIN)

    point = pd.DataFrame([{"source": n, **shapes(r, tau, taus, days)} for n, r in released.items()])
    point.to_csv(OUT / f"{STEM}_point.csv", index=False)

    bayes = np.load(ROOT / "results" / "bayes_refit" / "eci_draws.npz", allow_pickle=False)
    sets = {"bayes_refit": ({"eci": bayes["eci"], "model_names": list(bayes["model_names"])}, released["bayes_refit"]),
            "least_squares": (ls_bootstrap(), released["least_squares"])}
    per = []
    for name, (dr, base) in sets.items():
        names = list(dr["model_names"])
        b = base.set_index("model")
        E = dr["eci"][:, [names.index(m) for m in b.index]]
        t0 = time.time()
        for i in range(E.shape[0]):
            per.append({"draw_set": name, "draw": i, **shapes(b.assign(score=E[i]).reset_index(), tau, taus, days)})
            if i % 500 == 0:
                print(f"  {name}: draw {i}/{E.shape[0]} ({time.time() - t0:.0f}s)", flush=True)
    per = pd.DataFrame(per)
    draws_mod.write_csv_gz(per, OUT / f"{STEM}_per_draw.csv.gz")

    rows = []
    for ds, x in per.groupby("draw_set"):
        sd = x.dropna(subset=["step_best_ssr"])
        rows.append({"draw_set": ds, "window_end": END, "n_draws": len(x),
                     "share_kink_beats_step_fixed": float(x.kink_beats_step_fixed.mean()),
                     "n_free_comparisons": len(sd),
                     "share_step_beats_kink_free": float(sd.step_beats_kink_free.mean()),
                     "median_ssr_difference_step_minus_kink_free": float((sd.step_best_ssr - sd.kink_best_ssr).median()),
                     "kink_best_date_top": sd.kink_best_date.astype(str).value_counts(normalize=True).head(3).round(3).to_dict(),
                     "step_best_first_day_top": sd.step_best_first_day.astype(str).value_counts(normalize=True).head(3).round(3).to_dict(),
                     "n_points_range": f"{x.n_points.min()}-{x.n_points.max()}"})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / f"{STEM}_draw_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80):
        print(point.to_string(index=False))
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
