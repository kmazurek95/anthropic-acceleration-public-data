"""Developer ratios, slope vs step with free dates, and Anthropic's gap to the
frontier, each over the Bayesian posterior and least-squares bootstrap draws.
Writes results/developer_comparison/."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from accel import catchup, inputs, trend  # noqa: E402
from accel import draws as draws_mod  # noqa: E402
from analysis import ORIGIN, TAU_DATE, WINDOW, ZIP, load_sources  # noqa: E402

OUT = ROOT / "results" / "developer_comparison"
FIG = ROOT / "figures"
LSB = ROOT / "results" / "ls_bootstrap"
ORG = "Anthropic"
WIDE_START = "2024-02-29"          # one day earlier: admits Claude 3 Opus (the fifteenth point)


def ls_bootstrap() -> dict:
    """Epoch's least-squares fit with its default bootstrap; cached, and reused only for the same input hash."""
    f = LSB / "eci_draws.npz"
    meta = LSB / "meta.json"
    if f.exists() and meta.exists() and json.loads(meta.read_text()).get("input_sha256") == inputs.EXPECTED_SHA256:
        d = np.load(f, allow_pickle=False)
        return {"eci": d["eci"], "model_names": list(d["model_names"])}
    from eci import fit_eci_model, prepare_benchmark_data
    digest = inputs.verify_input(ZIP)
    t0 = time.time()
    eci_df, _, draws = fit_eci_model(prepare_benchmark_data(ZIP))       # defaults: 500 resamples, seed 12345
    LSB.mkdir(parents=True, exist_ok=True)
    names = np.array(draws["model_names"], dtype=str)
    np.savez_compressed(f, eci=draws["eci"], model_names=names)
    eci_df.to_csv(LSB / "eci_scores_with_bootstrap_ci.csv", index=False)
    (LSB / "meta.json").write_text(json.dumps({
        "input_sha256": digest, "eci_commit": inputs.ECI_COMMIT, "function": "eci.fit_eci_model",
        "settings": "defaults: bootstrap_samples=500, bootstrap_seed=12345, regularization_strength=0.1, "
                    "performance_clip_eps=1e-3; resamples each model's benchmark results with replacement",
        "n_draws": int(draws["eci"].shape[0]), "elapsed_seconds": round(time.time() - t0, 1)}, indent=2))
    return {"eci": draws["eci"], "model_names": list(names)}


def candidate_orgs(released: pd.DataFrame, tau_date) -> list[str]:
    """Developers with >= 4 distinct in-window release dates on each side of the
    break. Necessary (not sufficient) for a supported frontier fit, so no draw
    can make any other developer eligible."""
    w = released[(released.date >= WINDOW[0]) & (released.date <= WINDOW[1])]
    tau = pd.Timestamp(tau_date)
    pre = w[w.date < tau].groupby("organization").date.nunique()
    post = w[w.date >= tau].groupby("organization").date.nunique()
    both = pd.concat([pre, post], axis=1).fillna(0)
    return sorted(both[(both.iloc[:, 0] >= 4) & (both.iloc[:, 1] >= 4)].index)


def series(rec, org=None, start=WINDOW[0]):
    r = rec if org is None else rec[rec.organization == org]
    s = trend.series_from_records(r, start, WINDOW[1], ORIGIN)
    return s["t"].to_numpy(), s["score"].to_numpy(), s


def free_shapes(t, y, taus, days):
    """Best supported kink and best supported level shift over the daily grid.
    The level shift's SSR is flat between release dates, so the best date is
    a span; its first and last days are reported."""
    kp = trend.ssr_profile(t, y, taus, "kink")
    lp = trend.ssr_profile(t, y, taus, "level_shift")
    kb, lb = trend.best_of_profile(kp), trend.best_of_profile(lp)
    if kb is None or lb is None:
        return None
    tied = lp.supported & np.isclose(lp.ssr, lb.ssr, rtol=1e-10, atol=1e-10)
    return {"kink_best_date": days[int(kb.name)].date(), "kink_best_ssr": kb.ssr,
            "kink_best_ratio": (kb.slope + kb.third) / kb.slope if kb.slope > 0 else np.nan,
            "step_best_first_day": days[int(np.flatnonzero(tied)[0])].date(),
            "step_best_last_day": days[int(np.flatnonzero(tied)[-1])].date(),
            "step_best_ssr": lb.ssr, "step_best_jump": lb.third,
            "step_beats_kink_free": bool(lb.ssr < kb.ssr)}


def one_draw(rec, orgs, tau, taus, days):
    dev = []
    for org in orgs + [None]:
        t, y, s = series(rec, org)
        k = trend.fit_kink(t, y, tau)
        dev.append({"organization": org or "All developers", "n_points": len(s), **k.as_dict()})
    dev = pd.DataFrame(dev)
    t, y, _ = series(rec, ORG)
    shapes = free_shapes(t, y, taus, days) or {}
    ls_fixed = trend.fit_level_shift(t, y, tau)
    a = dev.set_index("organization").loc[ORG]
    f = dev.set_index("organization").loc["All developers"]
    g = catchup.gap_series(rec[rec.organization == ORG], rec, *WINDOW)
    gs = catchup.gap_summary(g, at=TAU_DATE)
    other = dev[(dev.organization != ORG) & (dev.organization != "All developers") & (dev.status == "ok")]
    anth_ok = a.status == "ok"
    extra = {"kink_ssr_fixed": a.ssr, "step_ssr_fixed": ls_fixed["ssr"],
             "kink_beats_step_fixed": bool(a.ssr < ls_fixed["ssr"]), **shapes,
             "cross_ratio": catchup.cross_ratio(a.post_slope, f.pre_slope),
             "anthropic_post_slope": a.post_slope, "field_pre_slope": f.pre_slope,
             "gap_at_break": gs["gap_at"], "deepest_gap": gs["deepest_gap"],
             "deepest_gap_date": gs["deepest_gap_date"], "lead_regained": gs["lead_regained"],
             "anthropic_highest": bool(anth_ok and (other.empty or a.ratio > other.ratio.max())),
             "anthropic_only_above_1": bool(anth_ok and a.ratio > 1 and (other.ratio <= 1).all()),
             "n_other_supported": len(other)}
    return dev, extra


def run_draws(name, draws, base, orgs, tau, taus, days):
    names = list(draws["model_names"])
    base = base.set_index("model")
    cols = [names.index(m) for m in base.index]
    E = draws["eci"][:, cols]
    devs, extras = [], []
    t0 = time.time()
    for i in range(E.shape[0]):
        rec = base.assign(score=E[i]).reset_index()
        d, x = one_draw(rec, orgs, tau, taus, days)
        devs.append(d.assign(draw=i))
        extras.append({"draw": i, **x})
        if i % 500 == 0:
            print(f"  {name}: draw {i}/{E.shape[0]} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(devs).assign(draw_set=name), pd.DataFrame(extras).assign(draw_set=name)


def q(s, p):
    return float(s.quantile(p)) if len(s) else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources, info, _ = load_sources()
    released = {k: v[~v["unreleased"]].reset_index(drop=True) for k, v in sources.items()}
    tau = float(trend.years_since([TAU_DATE], ORIGIN)[0])
    days = pd.date_range(*WINDOW, freq="D")
    taus = trend.years_since(days, ORIGIN)
    orgs = candidate_orgs(released["epoch_published"], TAU_DATE)
    print("candidate developers:", orgs)

    # ---- point estimates under each source (no draws) ----
    pts, shapes_pts, catch_pts, gaps_all, spells_all = [], [], [], [], []
    for name, rec in released.items():
        d, x = one_draw(rec, orgs, tau, taus, days)
        pts.append(d.assign(source=name))
        shapes_pts.append({"source": name, **{k: v for k, v in x.items() if k.startswith(("kink", "step"))}})
        g = catchup.gap_series(rec[rec.organization == ORG], rec, *WINDOW)
        gaps_all.append(g.assign(source=name))
        spells_all.append(catchup.lead_spells(g).assign(source=name))
        gs = catchup.gap_summary(g, at=TAU_DATE)
        june = g[g.date == "2024-06-20"].iloc[0]
        tw, yw, sw = series(rec, ORG, start=WIDE_START)
        k15 = trend.fit_kink(tw, yw, tau)
        a = d.set_index("organization").loc[ORG]
        f = d.set_index("organization").loc["All developers"]
        catch_pts.append({"source": name,
                          "anthropic_pre_slope": a.pre_slope, "anthropic_post_slope": a.post_slope,
                          "field_pre_slope": f.pre_slope, "field_post_slope": f.post_slope,
                          "cross_ratio": catchup.cross_ratio(a.post_slope, f.pre_slope),
                          "gap_on_2024-06-20": june.gap, "leader_on_2024-06-20": june.field_model,
                          "gap_at_break": gs["gap_at"], "anthropic_model_at_break": gs["org_model_at"],
                          "field_model_at_break": gs["field_model_at"], "deepest_gap": gs["deepest_gap"],
                          "deepest_gap_date": gs["deepest_gap_date"], "lead_regained": gs["lead_regained"],
                          "lead_regained_model": gs["lead_regained_model"],
                          "ratio_14_points": a.ratio, "n_points_wide_window": len(sw),
                          "first_point_wide_window": f"{sw.model.iloc[0]} ({sw.date.iloc[0].date()})",
                          "ratio_wide_window": k15.ratio, "n_pre_wide": k15.n_pre, "n_post_wide": k15.n_post,
                          "change_from_adding_point": k15.ratio - a.ratio})
    pd.concat(pts).to_csv(OUT / "developer_ratios_point.csv", index=False)
    pd.DataFrame(shapes_pts).to_csv(OUT / "slope_vs_step_point.csv", index=False)
    pd.DataFrame(catch_pts).to_csv(OUT / "catchup_point.csv", index=False)
    pd.concat(gaps_all).to_csv(OUT / "gap_series_daily.csv", index=False)
    pd.concat(spells_all).to_csv(OUT / "lead_spells.csv", index=False)

    # ---- draws ----
    bayes = np.load(ROOT / "results" / "bayes_refit" / "eci_draws.npz", allow_pickle=False)
    draw_sets = {"bayes_refit": ({"eci": bayes["eci"], "model_names": list(bayes["model_names"])},
                                 released["bayes_refit"]),
                 "least_squares": (ls_bootstrap(), released["least_squares"])}
    devs, extras = [], []
    for name, (dr, base) in draw_sets.items():
        d, x = run_draws(name, dr, base, orgs, tau, taus, days)
        devs.append(d)
        extras.append(x)
    devs, extras = pd.concat(devs), pd.concat(extras)
    draws_mod.write_csv_gz(devs, OUT / "developer_ratios_draws.csv.gz")
    draws_mod.write_csv_gz(extras, OUT / "step_and_catchup_per_draw.csv.gz")

    point = pd.concat(pts)
    rows = []
    for (ds, org), g in devs.groupby(["draw_set", "organization"]):
        ok = g[g.status == "ok"]
        pt = point[(point.source == ds) & (point.organization == org)]
        pub = point[(point.source == "epoch_published") & (point.organization == org)]
        rows.append({"draw_set": ds, "organization": org, "n_draws": len(g),
                     "share_supported": len(ok) / len(g), "point_ratio_same_fit": pt.ratio.iloc[0],
                     "point_ratio_published": pub.ratio.iloc[0],
                     "median": q(ok.ratio, .5), "p05": q(ok.ratio, .05), "p95": q(ok.ratio, .95),
                     "share_above_1": float((ok.ratio > 1).mean()) if len(ok) else np.nan,
                     "n_pre_range": f"{g.n_pre.min()}-{g.n_pre.max()}", "n_post_range": f"{g.n_post.min()}-{g.n_post.max()}"})
    dev_summary = pd.DataFrame(rows)
    dev_summary.to_csv(OUT / "developer_ratios_summary.csv", index=False)

    rows = []
    for ds, x in extras.groupby("draw_set"):
        sd = x.dropna(subset=["step_best_ssr"])
        rows.append({"draw_set": ds, "n_draws": len(x),
                     "share_kink_beats_step_fixed": float(x.kink_beats_step_fixed.mean()),
                     "n_free_comparisons": len(sd),
                     "share_step_beats_kink_free": float(sd.step_beats_kink_free.mean()),
                     "median_ssr_difference_step_minus_kink_free": float((sd.step_best_ssr - sd.kink_best_ssr).median()),
                     "kink_best_date_top": sd.kink_best_date.astype(str).value_counts(normalize=True).head(3).round(3).to_dict(),
                     "step_best_first_day_top": sd.step_best_first_day.astype(str).value_counts(normalize=True).head(3).round(3).to_dict(),
                     "cross_ratio_median": q(x.cross_ratio, .5), "cross_ratio_p05": q(x.cross_ratio, .05),
                     "cross_ratio_p95": q(x.cross_ratio, .95),
                     "gap_at_break_median": q(x.gap_at_break, .5), "gap_at_break_p05": q(x.gap_at_break, .05),
                     "gap_at_break_p95": q(x.gap_at_break, .95),
                     "lead_regained_top": x.lead_regained.astype(str).value_counts(normalize=True).head(3).round(3).to_dict(),
                     "share_anthropic_highest": float(x.anthropic_highest.mean()),
                     "share_anthropic_only_above_1": float(x.anthropic_only_above_1.mean())})
    step_catchup = pd.DataFrame(rows)
    step_catchup.to_csv(OUT / "step_and_catchup_draw_summary.csv", index=False)
    (OUT / "run_info.json").write_text(json.dumps({**{k: info[k] for k in ("input_sha256", "n_models")},
                                                   "candidate_developers": orgs, "window": WINDOW,
                                                   "wide_window_start": WIDE_START, "tau_date": TAU_DATE,
                                                   "draw_sets": {k: int(v[0]["eci"].shape[0]) for k, v in draw_sets.items()}},
                                                  indent=2, default=str))
    figure(dev_summary, extras, pd.concat(gaps_all))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(dev_summary.round(3).to_string(index=False))
        print(pd.DataFrame(shapes_pts).to_string(index=False))
        print(pd.DataFrame(catch_pts).T.to_string())
        print(step_catchup.T.to_string())


def figure(dev_summary, extras, gaps):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    order = (dev_summary[dev_summary.draw_set == "bayes_refit"].sort_values("median").organization.tolist())
    for j, (ds, col, off) in enumerate([("bayes_refit", "#2a6f97", -0.15), ("least_squares", "#b5651d", 0.15)]):
        c = dev_summary[dev_summary.draw_set == ds].set_index("organization").reindex(order)
        y = np.arange(len(order)) + off
        ax.errorbar(c["median"], y, xerr=[c["median"] - c.p05, c.p95 - c["median"]], fmt="o", color=col,
                    ms=4, capsize=2, label=f"{ds} draws (median, 90%)")
    sup = dev_summary.pivot_table(index="organization", columns="draw_set", values="share_supported").reindex(order)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"{o}\n(supported {sup.loc[o, 'bayes_refit']:.0%} / {sup.loc[o, 'least_squares']:.0%})"
                        for o in order], fontsize=7)
    ax.axvline(1, color="#555", ls=":", lw=1)
    ax.set_xlabel("post/pre slope ratio, break fixed at 2025-10-07")
    ax.set_title("Developer ratios over score draws (supported draws only)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for src, col in [("epoch_published", "#2a6f97"), ("least_squares", "#b5651d")]:
        g = gaps[gaps.source == src]
        ax.step(g.date, g.gap, where="post", color=col, lw=1.2, label=src)
    ax.axvline(pd.Timestamp(TAU_DATE), color="#555", ls=":", lw=1)
    ax.set_ylabel("Anthropic best minus field best (ECI points)")
    ax.set_title("Anthropic's gap to the all-developer frontier", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)
    ax = axes[2]
    for ds, col in [("bayes_refit", "#2a6f97"), ("least_squares", "#b5651d")]:
        x = extras[extras.draw_set == ds].dropna(subset=["step_best_ssr"])
        ax.hist(x.step_best_ssr - x.kink_best_ssr, bins=60, alpha=0.6, color=col, density=True,
                label=f"{ds}: step wins in {x.step_beats_kink_free.mean():.0%}")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("best step SSR minus best kink SSR, dates free\n(< 0: the step fits better)")
    ax.set_title("Slope or step with free break dates, per draw", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Developers, catch-up and slope vs step (descriptive; score uncertainty only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "developers_step_catchup.png", dpi=150)
    plt.close(fig)


def figure_from_saved():
    figure(pd.read_csv(OUT / "developer_ratios_summary.csv"), pd.read_csv(OUT / "step_and_catchup_per_draw.csv.gz"),
           pd.read_csv(OUT / "gap_series_daily.csv", parse_dates=["date"]))


if __name__ == "__main__":
    figure_from_saved() if "--figure-only" in sys.argv else main()
