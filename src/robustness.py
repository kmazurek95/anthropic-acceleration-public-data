"""Developer ratios, monthly breaks, posterior-draw ratios, single-model swaps,
Gemini 4 Carbon dates, kink vs level shift, and the window end
extended to 2026-11-13. Writes results/robustness/."""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from accel import fri, inputs, trend  # noqa: E402
from analysis import (FRI_CUTOFF, ORIGIN, TAU_DATE, WINDOW, ZIP,  # noqa: E402
                                 load_sources)
from developer_comparison import ls_bootstrap  # noqa: E402

OUT = ROOT / "results" / "robustness"
FIG = ROOT / "figures"
ORG = "Anthropic"
WINDOW_END_EXTENDED = "2026-11-13"     # snapshot date; admits Claude Fable 5.2 (2026-11-09)


def kink_on(records, tau, org=None):
    r = records if org is None else records[records["organization"] == org]
    s = trend.series_from_records(r, *WINDOW, ORIGIN)
    return s, trend.fit_kink(s["t"].to_numpy(), s["score"].to_numpy(), tau)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources, info, _ = load_sources()
    released = {k: v[~v["unreleased"]] for k, v in sources.items()}
    tau = float(trend.years_since([TAU_DATE], ORIGIN)[0])
    days = pd.date_range(*WINDOW, freq="D")
    taus = trend.years_since(days, ORIGIN)

    # Fixed-break kink on each developer's own frontier
    rows = []
    for name, rec in released.items():
        for org in sorted(rec["organization"].unique()):
            s, fit = kink_on(rec, tau, org)
            rows.append({"source": name, "organization": org, "n_points": len(s), **fit.as_dict()})
    orgs = pd.DataFrame(rows)
    orgs.to_csv(OUT / "developer_ratios.csv", index=False)

    # Anthropic with the break on the first of each month
    months = pd.date_range(WINDOW[0], WINDOW[1], freq="MS")
    rows = []
    for name, rec in released.items():
        s = trend.series_from_records(rec[rec.organization == ORG], *WINDOW, ORIGIN)
        for m in months:
            tm = float(trend.years_since([m], ORIGIN)[0])
            rows.append({"source": name, "break": m.date(),
                         **trend.fit_kink(s["t"].to_numpy(), s["score"].to_numpy(), tm).as_dict()})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "anthropic_monthly_breaks.csv", index=False)

    # Anthropic ratio and kink-vs-level-shift SSR in each posterior draw
    d = np.load(ROOT / "results" / "bayes_refit" / "eci_draws.npz", allow_pickle=False)
    names = list(d["model_names"])
    base = released["bayes_refit"][released["bayes_refit"].organization == ORG].set_index("model")
    cols = [names.index(m) for m in base.index]
    draws = d["eci"][:, cols]
    rows = []
    for i in range(draws.shape[0]):
        rec = base.assign(score=draws[i]).reset_index()
        s = trend.series_from_records(rec, *WINDOW, ORIGIN)
        t, y = s["t"].to_numpy(), s["score"].to_numpy()
        k = trend.fit_kink(t, y, tau)
        ls = trend.fit_level_shift(t, y, tau)
        rows.append({"draw": i, "status": k.status, "ratio": k.ratio, "pre_slope": k.pre_slope,
                     "post_slope": k.post_slope, "n_points": len(s), "frontier": ";".join(s["model"]),
                     "kink_ssr": k.ssr, "level_shift_ssr": ls["ssr"]})
    post = pd.DataFrame(rows)
    post.to_csv(OUT / "posterior_draw_ratios.csv", index=False)
    ok = post[post.status == "ok"]
    post_summary = {
        "n_draws": len(post), "n_ok": len(ok), "status_counts": post.status.value_counts().to_dict(),
        "median": ok.ratio.median(), "p05": ok.ratio.quantile(0.05), "p95": ok.ratio.quantile(0.95),
        "n_distinct_frontiers": post.frontier.nunique(),
        "share_kink_beats_level_shift": float((post.kink_ssr < post.level_shift_ssr).mean()),
    }
    pd.Series(post_summary).to_csv(OUT / "posterior_summary.csv", header=["value"])

    # Replace one Anthropic model's score with its value under the other fit
    pub = released["epoch_published"].set_index("model")
    lsq = released["least_squares"].set_index("model")
    anth = pub[pub.organization == ORG].index
    rows = []
    for label, a, b in [("published", pub, lsq), ("least_squares", lsq, pub)]:
        _, f0 = kink_on(a.reset_index(), tau, ORG)
        rows.append({"base": label, "swapped_model": "(none)", "base_score": np.nan, "swapped_score": np.nan,
                     "ratio": f0.ratio, "change": 0.0})
        for m in anth:
            sw = a.copy()
            sw.loc[m, "score"] = b.loc[m, "score"]
            _, f = kink_on(sw.reset_index(), tau, ORG)
            rows.append({"base": label, "swapped_model": m, "base_score": a.loc[m, "score"],
                         "swapped_score": b.loc[m, "score"], "ratio": f.ratio, "change": f.ratio - f0.ratio})
    swaps = pd.DataFrame(rows)
    swaps.to_csv(OUT / "single_model_swaps.csv", index=False)

    # Gemini 4 Carbon under its two recorded dates
    with zipfile.ZipFile(ZIP) as z:
        bundled = pd.read_csv(z.open("epoch_capabilities_index/eci_scores.csv")).set_index("Model")
    gc = "Gemini 4 Carbon"
    rows = []
    for name, rec in released.items():
        for label, date in [("metadata date", rec.set_index("model").loc[gc, "date"]),
                            ("published-file date", pd.Timestamp(bundled.loc[gc, "date"]))]:
            r = rec.copy()
            r.loc[r.model == gc, "date"] = pd.Timestamp(date)
            s, fk = kink_on(r, tau)
            prof = trend.break_profile(s["t"].to_numpy(), s["score"].to_numpy(), taus)
            prof.insert(0, "candidate_date", days.date)
            b = trend.best_supported_break(prof)
            fr = fri.summarise_rises(fri.window_rises(r, FRI_CUTOFF))
            rows.append({"source": name, "gemini_4_carbon_date": f"{label} ({pd.Timestamp(date).date()})",
                         "carbon_on_frontier": gc in set(s.model), "n_pre": fk.n_pre, "n_post": fk.n_post,
                         "fixed_ratio": fk.ratio, "est_break": b["candidate_date"], "est_ratio": b["ratio"],
                         "fri_max_rise": fr["max_rise"], "fri_window": f"{fr['first_max_start']}..{fr['first_max_end']}"})
    gem = pd.DataFrame(rows)
    gem.to_csv(OUT / "gemini_4_carbon_dates.csv", index=False)

    # Kink vs level shift, fixed and estimated break, per source
    rows = []
    for name, rec in released.items():
        s = trend.series_from_records(rec[rec.organization == ORG], *WINDOW, ORIGIN)
        t, y = s["t"].to_numpy(), s["score"].to_numpy()
        k = trend.fit_kink(t, y, tau)
        ls = trend.fit_level_shift(t, y, tau)
        kp = pd.DataFrame([trend.fit_kink(t, y, x).as_dict() for x in taus]).assign(date=days.date)
        lp = pd.DataFrame([trend.fit_level_shift(t, y, x) for x in taus]).assign(date=days.date)
        kb = kp[kp.status != "insufficient_dates"].sort_values("ssr").iloc[0]
        lb = lp[lp.status != "insufficient_dates"].sort_values("ssr").iloc[0]
        rows.append({"source": name, "n_pre": k.n_pre, "n_post": k.n_post,
                     "kink_ssr_fixed": k.ssr, "level_shift_ssr_fixed": ls["ssr"],
                     "level_shift_slope_fixed": ls["slope"], "level_shift_jump_fixed": ls["jump"],
                     "kink_best_date": kb.date, "kink_ssr_best": kb.ssr,
                     "level_shift_best_date": lb.date, "level_shift_ssr_best": lb.ssr,
                     "level_shift_jump_best": lb.jump})
    kvl = pd.DataFrame(rows)
    kvl.to_csv(OUT / "kink_vs_level_shift.csv", index=False)

    # Window end extended to 2026-11-13
    we_point, we_draws = window_end_sensitivity(released, tau)
    we_point.to_csv(OUT / "window_end_point.csv", index=False)
    we_draws.to_csv(OUT / "window_end_draws_summary.csv", index=False)

    figures(orgs, sweep, post, post_summary, swaps)
    with pd.option_context("display.width", 220):
        print(orgs[(orgs.status == "ok")].round(2).to_string(index=False))
        print(sweep[sweep.source == "epoch_published"][["break", "status", "ratio"]].round(2).to_string(index=False))
        print(pd.Series(post_summary).to_string())
        print(swaps[swaps.swapped_model.isin(["(none)", "Claude Opus 4.5"])].round(3).to_string(index=False))
        print(swaps.loc[swaps.change.abs().sort_values(ascending=False).index].head(6).round(3).to_string(index=False))
        print(gem.round(2).to_string(index=False))
        print(kvl.round(2).to_string(index=False))
        print(we_point.round(3).to_string(index=False))
        print(we_draws.round(3).to_string(index=False))


def _window_fits(rec, tau):
    rows = []
    for org in (ORG, None):
        r = rec if org is None else rec[rec["organization"] == org]
        for end in (WINDOW[1], WINDOW_END_EXTENDED):
            s = trend.series_from_records(r, WINDOW[0], end, ORIGIN)
            k = trend.fit_kink(s["t"].to_numpy(), s["score"].to_numpy(), tau)
            rows.append({"series": org or "All developers", "window_end": end, "n_points": len(s),
                         "last_point": f"{s.model.iloc[-1]} ({s.date.iloc[-1].date()})", **k.as_dict()})
    return rows


def window_end_sensitivity(released, tau):
    """Fixed-break ratios with the window ending 2026-10-31 (primary) and
    2026-11-13, for Anthropic and all developers; point estimates per source
    and summaries over both draw sets."""
    point = pd.DataFrame([{"source": name, **row} for name, rec in released.items()
                          for row in _window_fits(rec, tau)])
    bayes = np.load(ROOT / "results" / "bayes_refit" / "eci_draws.npz", allow_pickle=False)
    draw_sets = {"bayes_refit": {"eci": bayes["eci"], "model_names": list(bayes["model_names"])},
                 "least_squares": ls_bootstrap()}
    per = []
    for name, d in draw_sets.items():
        base = released[name].set_index("model")
        names = list(d["model_names"])
        E = d["eci"][:, [names.index(m) for m in base.index]]
        for i in range(E.shape[0]):
            for row in _window_fits(base.assign(score=E[i]).reset_index(), tau):
                per.append({"draw_set": name, "draw": i, **row})
    per = pd.DataFrame(per)
    rows = []
    for (ds, series, end), g in per.groupby(["draw_set", "series", "window_end"], sort=False):
        ok = g[g.status == "ok"]
        rows.append({"draw_set": ds, "series": series, "window_end": end, "n_draws": len(g),
                     "share_supported": len(ok) / len(g),
                     "median": ok.ratio.median(), "p05": ok.ratio.quantile(0.05), "p95": ok.ratio.quantile(0.95),
                     "share_above_1": float((ok.ratio > 1).mean()),
                     "n_pre_range": f"{g.n_pre.min()}-{g.n_pre.max()}",
                     "n_post_range": f"{g.n_post.min()}-{g.n_post.max()}"})
    return point, pd.DataFrame(rows)



def figures(orgs, sweep, post, ps, swaps):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    o = orgs[(orgs.source == "epoch_published") & (orgs.status == "ok")].sort_values("ratio")
    ax = axes[0]
    ax.barh(o.organization, o.ratio, color=["#2a6f97" if x == ORG else "#999" for x in o.organization])
    ax.axvline(1, color="#555", ls=":", lw=1)
    ax.set_xlabel("post/pre slope ratio, break fixed at 2025-10-07")
    ax.set_title("Each developer's own frontier (published scores)", fontsize=9)
    ax = axes[1]
    s = sweep[(sweep.source == "epoch_published") & (sweep.status == "ok")]
    ax.plot(pd.to_datetime(s["break"]), s.ratio, "o-", color="#2a6f97", ms=3)
    a145 = orgs[(orgs.source == "epoch_published") & (orgs.organization == ORG)].ratio.iloc[0]
    ax.axhline(a145, color="#b5651d", ls="--", lw=1, label=f"fixed 2025-10-07: {a145:.2f}")
    ax.axhline(1, color="#555", ls=":", lw=1)
    ax.set_title("Anthropic, break moved to the 1st of each month", fontsize=9); ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)
    ax = axes[2]
    ok = post[post.status == "ok"]
    ax.hist(ok.ratio, bins=50, color="#3a7d44", alpha=0.8)
    ax.axvline(ps["median"], color="k", lw=1)
    ax.axvline(ps["p05"], color="k", ls="--", lw=1); ax.axvline(ps["p95"], color="k", ls="--", lw=1)
    ax.set_title(f"Anthropic ratio per posterior draw: median {ps['median']:.2f}, 90% [{ps['p05']:.2f}, {ps['p95']:.2f}]", fontsize=9)
    ax.set_xlabel("post/pre slope ratio")
    fig.suptitle("Robustness: placebo breaks and score uncertainty (descriptive)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "placebo_and_posterior.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
