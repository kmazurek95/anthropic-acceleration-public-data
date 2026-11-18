"""Write RESULTS.md: every table read from results/, in the order of the note.
Run after analysis.py, robustness.py, developer_comparison.py,
slope_vs_step_window_end.py, bradshaw_specs.py, figures.py and figure4_stats.py."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from accel import draws as draws_mod

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
PD, RB, DC = R / "public_data", R / "robustness", R / "developer_comparison"
BR, FIGS = R / "bradshaw", ROOT / "figures"
FIXED = "fixed_break_2025-10-07"
DRAW_SETS = ("bayes_refit", "least_squares")


def md(df: pd.DataFrame, floatfmt=2) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def iv(m, lo, hi):
    return "" if pd.isna(m) else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def pct(v):
    return "" if pd.isna(v) else f"{float(v):.0%}"


def release_frequency(w, fpts, info, tau="2025-10-07"):
    """Record-setting Anthropic releases per month before and after the break."""
    start, end, brk = (pd.Timestamp(x) for x in (info["window"][0], info["window"][1], tau))
    pre_days, post_days = (brk - start).days, (end - brk).days + 1  # window end inclusive
    rows = []
    for s, g in fpts[fpts.series == "anthropic"].groupby("source", sort=False):
        d = pd.to_datetime(g.date)
        n_pre, n_post = int((d < brk).sum()), int((d >= brk).sum())
        pre_m, post_m = pre_days / (365.25 / 12), post_days / (365.25 / 12)
        rows.append({"source": s, "records pre / post": f"{n_pre} / {n_post}",
                     "days pre / post": f"{pre_days} / {post_days}",
                     "months pre / post": f"{pre_m:.1f} / {post_m:.1f}",
                     "per month pre / post": f"{n_pre / pre_m:.2f} / {n_post / post_m:.2f}",
                     "per year pre / post": f"{12 * n_pre / pre_m:.2f} / {12 * n_post / post_m:.2f}",
                     "rate ratio post / pre": f"{(n_post / post_m) / (n_pre / pre_m):.2f}"})
    w(f"Release frequency of the frontier points: record-setting Anthropic releases (distinct release dates) before "
      f"the break ({start:%Y-%m-%d} to {brk:%Y-%m-%d}) and on or after it ({brk:%Y-%m-%d} to {end:%Y-%m-%d}, end "
      "inclusive). A month is 365.25 / 12 days.\n")
    w(md(pd.DataFrame(rows)) + "\n")


def meta_ai_counts(w, orgs):
    """Meta AI's frontier release dates either side of the fixed break, per score source."""
    m = orgs[orgs.organization == "Meta AI"]
    w("Meta AI's frontier release dates before / on or after the fixed break, by score source: "
      + "; ".join(f"{r.source} {int(r.n_pre)} / {int(r.n_post)} ({r.status})" for r in m.itertuples())
      + ". With fewer than four dates before the break it has no point ratio, so it enters the comparison only in "
      "draws where it has at least four on each side.\n")


LOADER_NOTE = ("Epoch's loader drops models released before 2023-01-01, so the frontier in early 2023 omits "
               "late-2022 models and early-2023 windows can overstate the rise.")


def additional_sections(w):
    w("## Additional checks\n")
    if (RB / "window_end_point.csv").exists():
        pt = pd.read_csv(RB / "window_end_point.csv")
        dr = pd.read_csv(RB / "window_end_draws_summary.csv")
        w("**Window end extended to 2026-11-13.** Break fixed at 2025-10-07; the window's end moves from 2026-10-31 to "
          "the snapshot date so Claude Fable 5.2 (dated 2026-11-09) can enter. n_pre / n_post are distinct release "
          "dates before / on or after the break.\n")
        pt = pt.assign(**{"n_pre / n_post": pt.n_pre.astype(str) + " / " + pt.n_post.astype(str)})
        w(md(pt[["source", "series", "window_end", "n_points", "n_pre / n_post", "pre_slope", "post_slope", "ratio",
                 "last_point"]]) + "\n")
        dr = dr.assign(draws=[iv(m, a, b) for m, a, b in zip(dr["median"], dr.p05, dr.p95)],
                       supported=dr.share_supported.map(lambda v: f"{v:.1%}"),
                       above_1=dr.share_above_1.map(lambda v: f"{v:.1%}"))
        w("Over the draws (frontier rebuilt per draw; ranges of n_pre / n_post across draws):\n")
        w(md(dr[["draw_set", "series", "window_end", "n_draws", "draws", "supported", "above_1", "n_pre_range",
                 "n_post_range"]]) + "\n")
    if (DC / "slope_vs_step_window_end_point.csv").exists():
        sp = pd.read_csv(DC / "slope_vs_step_window_end_point.csv")
        sd = pd.read_csv(DC / "slope_vs_step_window_end_draw_summary.csv")
        w("**Slope change against step with the window end at 2026-11-13** (`src/slope_vs_step_window_end.py`). As "
          "the kink-against-step comparison above, with the window's end moved from 2026-10-31 to the snapshot date "
          "so Claude Fable 5.2 (dated 2026-11-09) enters; candidate dates run to the new end.\n")
        w(md(sp[["source", "window_end", "n_points", "kink_ssr_fixed", "step_ssr_fixed", "kink_best_date",
                 "kink_best_ratio", "kink_best_ssr", "step_best_first_day", "step_best_last_day", "step_best_jump",
                 "step_best_ssr", "step_beats_kink_free"]]) + "\n")
        w(md(pd.DataFrame([{"draw_set": r.draw_set, "draws": int(r.n_draws),
                            "kink beats step, break fixed": f"{r.share_kink_beats_step_fixed:.1%}",
                            "step beats kink, dates free": f"{r.share_step_beats_kink_free:.1%}",
                            "median SSR step minus kink, free": f"{r.median_ssr_difference_step_minus_kink_free:.2f}",
                            "most common best step dates": r.step_best_first_day_top,
                            "n_points across draws": r.n_points_range} for r in sd.itertuples()])) + "\n")
    if (BR / "spec_grid.csv").exists():
        g = pd.read_csv(BR / "spec_grid.csv").fillna({"value": ""})
        run = json.loads((BR / "run_info.json").read_text())
        w("**Specifications behind Julian Bradshaw's estimates** (`src/bradshaw_specs.py`). His figures, by the date "
          "he posted them (US Central): "
          + "; ".join(f"{k}: " + ", ".join(f"{a} {v}" for a, v in d.items()) for k, d in run["targets"].items())
          + ". His chart notes give the method as 'Epoch eci_scores.csv (5 Nov 2026), models from Mar 2024. "
          "Running-max frontiers; OLS, slope change at 7 Oct 2025', with Fable 5.2 included on 10 November. Each "
          "`one_change` row changes one thing from the main specification (the baseline row); `stated_method` rows "
          "combine the changes his notes name.\n")
        w(md(g[["group", "changed", "value", "anthropic_ratio", "anthropic_pre_slope", "anthropic_post_slope",
                "anthropic_n_pre", "anthropic_n_post", "anthropic_first_point", "all_dev_ratio", "all_dev_pre_slope",
                "all_dev_post_slope", "all_dev_n_pre", "all_dev_n_post"]]) + "\n")
    figs = sorted(FIGS.glob("note_fig*.png"))
    if figs:
        w("**Figures for the note** (`src/figures.py`, drawn from the files above): "
          + ", ".join(f"`figures/{f.name}`" for f in figs) + ".\n")
    f4 = R / "developer_comparison" / "figure4_fit_stats.csv"
    if f4.exists():
        w("**Figure 4 fit statistics** (`src/figure4_stats.py`, same fits as `src/figures.py`). Published scores, "
          "Anthropic's fourteen frontier points, each shape at its own best date. The step's sum of squared residuals "
          "is the same on every day from `date_first` to `date_last`. The third row is one straight line with no "
          "break through the same points; it is not drawn in the figure.\n")
        w(md(pd.read_csv(f4).fillna({"date_first": "", "date_last": ""}), 3) + "\n")


def main():
    info = json.loads((PD / "run_info.json").read_text())
    kinks = pd.read_csv(PD / "kink_fits.csv")
    kinks = kinks[~kinks.series.str.contains("incl")]
    infl = pd.read_csv(PD / "influence_checks.csv")
    fri_s = pd.read_csv(PD / "fri_windows_summary.csv")
    fri_w = pd.read_csv(PD / "fri_windows_calendar12.csv", parse_dates=["start", "end"])
    comp = pd.read_csv(PD / "method_comparison.csv")
    audit = pd.read_csv(PD / "metadata_audit.csv")
    scores = pd.read_csv(PD / "public_scores.csv")
    fpts = pd.read_csv(PD / "frontier_points.csv")
    fit_meta = json.loads((R / "bayes_refit" / "fit_meta.json").read_text())
    post = pd.read_csv(RB / "posterior_summary.csv", index_col=0)["value"]
    sweep = pd.read_csv(RB / "anthropic_monthly_breaks.csv")
    swaps = pd.read_csv(RB / "single_model_swaps.csv")
    gem = pd.read_csv(RB / "gemini_4_carbon_dates.csv")
    run = json.loads((DC / "run_info.json").read_text())
    dev_summary = pd.read_csv(DC / "developer_ratios_summary.csv")
    c2p = pd.read_csv(DC / "slope_vs_step_point.csv")
    c3p = pd.read_csv(DC / "catchup_point.csv").set_index("source")
    dsum = pd.read_csv(DC / "step_and_catchup_draw_summary.csv").set_index("draw_set")
    spells = pd.read_csv(DC / "lead_spells.csv")
    sources = list(dict.fromkeys(scores["source"]))
    dev_by = dev_summary.set_index(["draw_set", "organization"])

    out = []
    w = out.append
    w("# Results\n")
    w("Generated by `src/build_results.py`; every number is read from a file in `results/`. Method and data: "
      "`README.md`. Slopes are ECI points per year; a ratio is the post-break slope over the pre-break slope. "
      "Score sources: `epoch_published` (Epoch's published scores), `least_squares` (Epoch's prior fitting method, "
      "refit here), `bayes_refit` (Epoch's current Bayesian method, refit here). Intervals are 90%.\n")

    # 1. Estimates
    rows = []
    for s in sources:
        a = kinks[(kinks.source == s) & (kinks.series == "anthropic") & (kinks.spec == FIXED)].iloc[0]
        d = kinks[(kinks.source == s) & (kinks.series == "all_developers") & (kinks.spec == FIXED)].iloc[0]
        row = {"source": s, "Anthropic pre slope": a.pre_slope, "Anthropic post slope": a.post_slope,
               "Anthropic ratio": a.ratio, "points pre / post": f"{int(a.n_pre)} / {int(a.n_post)}",
               "all-developer ratio": d.ratio}
        if s in DRAW_SETS:
            an, al = dev_by.loc[(s, "Anthropic")], dev_by.loc[(s, "All developers")]
            row["Anthropic over draws"] = iv(an["median"], an.p05, an.p95)
            row["all developers over draws"] = iv(al["median"], al.p05, al.p95)
        rows.append(row)
    w("## Estimates of the acceleration\n")
    w("Break fixed at 2025-10-07. Draws: 4,000 posterior draws (`bayes_refit`) or 500 bootstrap resamples "
      "(`least_squares`), frontier rebuilt in each.\n")
    w(md(pd.DataFrame(rows).fillna("")) + "\n")

    # 2. Fourteen points
    w("## Anthropic's frontier\n")
    a = fpts[(fpts.series == "anthropic")].pivot_table(index=["model", "date"], columns="source", values="score")
    w("Frontier points (Anthropic releases that beat every earlier Anthropic release), window "
      f"{info['window'][0]} to {info['window'][1]}:\n")
    w(md(a.reset_index().sort_values("date"), 1) + "\n")
    w("Leave one model out (break fixed; frontier rebuilt after each deletion):\n")
    li = infl[(infl.series == "anthropic") & (infl.status == "ok")]
    g = li.groupby("source")
    lsum = g.agg(n_drops=("ratio", "size"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max")).reset_index()
    lsum["drop_giving_min"] = g.apply(lambda x: x.loc[x.ratio.idxmin(), "dropped_model"]).values
    lsum["drop_giving_max"] = g.apply(lambda x: x.loc[x.ratio.idxmax(), "dropped_model"]).values
    w(md(lsum) + "\n")
    fp = c3p[["ratio_14_points", "first_point_wide_window", "n_points_wide_window", "ratio_wide_window",
              "change_from_adding_point"]].reset_index()
    w(f"Window opened on {run['wide_window_start']} (adds Claude 3 Opus):\n")
    w(md(fp, 3) + "\n")
    release_frequency(w, fpts, info)

    # 3. Confidence
    w("## Score uncertainty and fitting method\n")
    w(f"Anthropic ratio over {int(float(post['n_draws']))} posterior draws: median {float(post['median']):.2f}, "
      f"90% [{float(post['p05']):.2f}, {float(post['p95']):.2f}]; {int(float(post['n_distinct_frontiers']))} distinct "
      "frontier sets occur.\n")
    w("Score differences between sources (first minus second; frontier models = any model on the Anthropic or "
      "all-developer frontier in the window):\n")
    cshow = comp.set_index("pair")[["spearman_all", "max_abs_diff_all", "model_max_all", "mean_diff_all",
                                    "n_frontier_models_in_window", "max_abs_diff_frontier_window",
                                    "model_max_frontier_window", "mean_abs_diff_frontier_window"]]
    cshow = cshow.apply(lambda col: col.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v))
    w(md(cshow.T.reset_index().rename(columns={"index": "quantity"})) + "\n")
    fr = fpts[fpts.series.isin(["anthropic", "all_developers"])].groupby("source").score.agg(["min", "max"])
    w("Score range of the frontier points in the window (Anthropic and all-developer frontiers, released "
      f"models, {info['window'][0]} to {info['window'][1]}): "
      + "; ".join(f"{s} {r['min']:.1f} to {r['max']:.1f}" for s, r in fr.iterrows()) + ".\n")
    w("Largest single-model effects: one Anthropic model's score replaced by its value under the other fit:\n")
    top = swaps[swaps.swapped_model != "(none)"].assign(abs_change=lambda x: x.change.abs())
    w(md(top.sort_values("abs_change", ascending=False).head(8)[
        ["base", "swapped_model", "base_score", "swapped_score", "ratio", "change"]], 3) + "\n")
    d = fit_meta["diagnostics"]
    w(f"Bayesian refit: max R-hat {d['max_rhat']:.4f}, min bulk ESS {d['min_bulk_ess']:.0f}, "
      f"{d['divergences']} divergences, {d['elapsed_seconds'] / 60:.1f} min. {fit_meta['backend']}.\n")

    # 4. Relative to what
    w("## Other developers and catch-up\n")
    w(f"Developers with at least four distinct release dates on each side of the break: "
      f"{', '.join(run['candidate_developers'])}. `supported` is the share of draws with enough frontier dates on "
      "each side; `> 1` is the share of supported draws above 1.\n")
    t = dev_summary.assign(interval=[iv(m, a, b) for m, a, b in zip(dev_summary["median"], dev_summary.p05, dev_summary.p95)])
    t = t.pivot_table(index="organization", columns="draw_set", aggfunc="first",
                      values=["point_ratio_published", "point_ratio_same_fit", "interval", "share_supported",
                              "share_above_1"])
    cols = {"published point": ("point_ratio_published", "bayes_refit")}
    for ds in DRAW_SETS:
        cols.update({f"{ds} point": ("point_ratio_same_fit", ds), f"{ds} draws": ("interval", ds),
                     f"{ds} supported": ("share_supported", ds), f"{ds} > 1": ("share_above_1", ds)})
    t = pd.DataFrame({k: t[v] for k, v in cols.items()})
    dr = pd.read_csv(DC / "developer_ratios_draws.csv.gz", usecols=["draw_set", "draw", "organization", "ratio"])
    above = pd.concat([draws_mod.share_above(dr, o, "All developers") for o in t.index if o != "All developers"])
    for ds in DRAW_SETS:
        t[f"{ds} > all developers"] = above[above.draw_set == ds].set_index("series").share_all_draws.reindex(t.index)
    t = t[[c for c in ["published point"] + [f"{ds} {k}" for ds in DRAW_SETS
                                              for k in ("point", "draws", "supported", "> 1", "> all developers")]]]
    for c in t.columns:
        if c.endswith("> all developers"):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.1%}")
        else:
            t[c] = t[c].map(pct) if c.endswith(("supported", "> 1")) else t[c].map(
                lambda v: "" if pd.isna(v) else (v if isinstance(v, str) else f"{float(v):.2f}"))
    w(md(t.reset_index()) + "\n")
    an = above[above.series == "Anthropic"].set_index("draw_set")
    w("`> all developers`: share of all draws in which the developer's ratio is above the all-developer "
      "ratio in the same draw; a draw where either is unsupported counts as not above. For Anthropic, among draws "
      "where both are supported: " + "; ".join(f"{ds} {r.share_both_supported:.1%} of {int(r.n_both_supported)}"
                                               for ds, r in an.iterrows()) + ".\n")
    meta_ai_counts(w, pd.read_csv(RB / "developer_ratios.csv"))
    meta = draws_mod.share_above(dr, "Meta AI", "Anthropic").set_index("draw_set")
    w("Meta AI against Anthropic, draw by draw: draws in which Meta AI is supported and its ratio exceeds "
      "Anthropic's in the same draw (a draw where Anthropic is unsupported counts as not above): "
      + "; ".join(f"{ds} {round(r.share_all_draws * r.n_draws)} of {int(r.n_draws)} draws ({r.share_all_draws:.1%})"
                  for ds, r in meta.iterrows()) + ".\n")
    w(md(pd.DataFrame([{"draw_set": ds,
                        "Anthropic highest supported ratio": f"{dsum.loc[ds].share_anthropic_highest:.1%}",
                        "Anthropic only supported developer above 1": f"{dsum.loc[ds].share_anthropic_only_above_1:.1%}"}
                       for ds in DRAW_SETS])) + "\n")
    w("Catch-up: gap = Anthropic's best score to date minus the best score to date across all developers "
      "(0 = Anthropic holds the frontier). Cross ratio = Anthropic post-break slope / all-developer pre-break slope.\n")
    cols = ["anthropic_pre_slope", "anthropic_post_slope", "field_pre_slope", "field_post_slope", "cross_ratio",
            "gap_at_break", "deepest_gap", "deepest_gap_date", "lead_regained", "lead_regained_model"]
    c3t = c3p[cols].apply(lambda s: s.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v))
    w(md(c3t.T.reset_index().rename(columns={"index": "quantity"})) + "\n")
    w("Spells in which Anthropic holds the all-developer frontier (published scores):\n")
    w(md(spells[spells.source == "epoch_published"][["start", "end", "days", "model"]]) + "\n")
    w(md(pd.DataFrame([{"draw_set": ds,
                        "cross ratio": iv(r.cross_ratio_median, r.cross_ratio_p05, r.cross_ratio_p95),
                        "gap at break": iv(r.gap_at_break_median, r.gap_at_break_p05, r.gap_at_break_p95),
                        "date lead regained (most common)": r.lead_regained_top}
                       for ds, r in ((ds, dsum.loc[ds]) for ds in DRAW_SETS)])) + "\n")

    # 5. Breakpoint vs step
    w("## Breakpoint and step\n")
    w("Anthropic with the break on the first of each month (months with fewer than four dates on a side omitted):\n")
    sok = sweep[sweep.status == "ok"].pivot_table(index="break", columns="source", values="ratio").reset_index()
    w(md(sok) + "\n")
    w("Break date estimated (lowest SSR over every day in the window):\n")
    est = kinks[kinks.spec != FIXED][["source", "series", "break_date", "n_pre", "n_post", "pre_slope",
                                      "post_slope", "ratio"]]
    w(md(est.sort_values(["series", "source"])) + "\n")
    w("Two-segment (kink) vs one slope with a step (level shift); same parameter count, so SSRs compare directly. "
      "A step's best date is a span between releases.\n")
    w(md(c2p[["source", "kink_ssr_fixed", "step_ssr_fixed", "kink_best_date", "kink_best_ratio", "kink_best_ssr",
              "step_best_first_day", "step_best_last_day", "step_best_jump", "step_best_ssr",
              "step_beats_kink_free"]]) + "\n")
    w(md(pd.DataFrame([{"draw_set": ds, "draws": int(r.n_draws),
                        "kink beats step, break fixed": f"{r.share_kink_beats_step_fixed:.1%}",
                        "step beats kink, dates free": f"{r.share_step_beats_kink_free:.1%}",
                        "median SSR step minus kink, free": f"{r.median_ssr_difference_step_minus_kink_free:.2f}",
                        "most common best step dates": r.step_best_first_day_top}
                       for ds, r in ((ds, dsum.loc[ds]) for ds in DRAW_SETS)])) + "\n")

    # 6. FRI
    w("## FRI fast year\n")
    w(f"All-developer frontier, complete 12-calendar-month windows ending by {info['fri_cutoff']}; threshold 30 points.\n")
    fs = fri_s[~fri_s.unreleased_included][["source", "window", "max_rise", "first_max_start", "first_max_end",
                                            "start_model", "end_model", "any_window_ge_threshold"]]
    w(md(fs) + "\n")
    rows = []
    for s in sources:
        x = fri_w[(fri_w.source == s) & (fri_w.start >= "2024-01-01")]
        r = x.loc[x.rise.idxmax()]
        rows.append({"source": s, "max_rise, windows from 2024": r.rise, "start": r.start.date(),
                     "end": r.end.date(), "start_model": r.start_model, "end_model": r.end_model})
    w(LOADER_NOTE + " Restricted to windows starting in 2024 or later:\n")
    w(md(pd.DataFrame(rows)) + "\n")

    # 7. Additional checks
    additional_sections(w)

    # 8. Data notes
    w("## Data notes\n")
    with zipfile.ZipFile(ROOT / "data" / "benchmark_data_2026-11-13_wayback-20261113150458.zip") as z:
        bundled = pd.read_csv(z.open("epoch_capabilities_index/eci_scores.csv")).set_index("Model")
    pub = scores[scores.source == "epoch_published"].drop_duplicates("model").set_index("model")
    same = (pd.to_datetime(bundled["date"]) == pd.to_datetime(pub["date"]).reindex(bundled.index)).mean()
    w(f"- Fit input: {info['n_models']} models, {info['n_benchmarks']} benchmarks, {info['n_rows']} model-benchmark rows.")
    w(f"- Excluded as unreleased: {', '.join(info['unreleased_models'])}.")
    w(f"- Metadata audit (`results/public_data/metadata_audit.csv`): {audit.issue.value_counts().to_dict()}.")
    w(f"- Release dates used here match the dates in Epoch's `eci_scores.csv` for {same:.1%} of models.")
    w("- Gemini 4 Carbon under its two recorded dates (all-developer frontier):\n")
    w(md(gem) + "\n")
    (ROOT / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
