"""Frontier fits under three score sources: two-segment fits at the fixed and
estimated breaks, leave-one-out, break profiles, FRI windows, method comparison.
Specification in README.md. Writes results/public_data/ and figures/."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from accel import frontier as fr, fri, inputs, trend  # noqa: E402

ZIP = ROOT / "data" / "benchmark_data_2026-11-13_wayback-20261113150458.zip"   # see data/README.md
OUT = ROOT / "results" / "public_data"
FIG = ROOT / "figures"
WINDOW = ("2024-03-01", "2026-10-31")
ORIGIN = "2024-03-01"
TAU_DATE = "2025-10-07"
FRI_CUTOFF = "2026-11-03"
ORG = "Anthropic"


def load_sources() -> tuple[dict[str, pd.DataFrame], dict]:
    from eci import fit_eci_model, prepare_benchmark_data
    digest = inputs.verify_input(ZIP)
    df = prepare_benchmark_data(ZIP)
    ls, _, _ = fit_eci_model(df, bootstrap_samples=0)
    with zipfile.ZipFile(ZIP) as z:
        pub = pd.read_csv(z.open("epoch_capabilities_index/eci_scores.csv"))
    src = {"least_squares": ls[["Model", "eci"]], "epoch_published": pub[["Model", "eci"]]}
    refit = ROOT / "results" / "bayes_refit" / "eci_scores_refit.csv"
    if refit.exists():
        src["bayes_refit"] = pd.read_csv(refit)[["Model", "eci"]]
    models = sorted(df["model"].unique())
    for k, v in src.items():
        if sorted(v["Model"]) != models:
            raise ValueError(f"{k}: model set differs from the eligible fit input")
    org_fallback = dict(zip(pub["Model"], pub["Organization"]))   # only for groups with no organization in metadata
    table, audit = inputs.group_table(inputs.read_model_metadata(ZIP), models, org_fallback=org_fallback,
                                       allow_unknown_org=True)   # unknown orgs can never enter the Anthropic series
    table["unreleased"] = table["accessibility"].str.contains("Unreleased", na=False)
    info = {"input_sha256": digest, "n_models": len(models), "n_benchmarks": int(df.benchmark.nunique()),
            "n_rows": len(df), "sources": list(src),
            "unreleased_models": sorted(table.loc[table.unreleased, "model"])}
    return {k: table.merge(v.rename(columns={"Model": "model", "eci": "score"}), on="model")
            for k, v in src.items()}, info, audit


def series_specs():
    # (series label, organization filter or None, include unreleased?)
    return [("anthropic", ORG, False), ("all_developers", None, False),
            ("all_developers_incl_unreleased", None, True)]


def select(records, org, incl_unreleased):
    r = records if incl_unreleased else records[~records["unreleased"]]
    return r if org is None else r[r["organization"] == org]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sources, info, audit = load_sources()
    audit.to_csv(OUT / "metadata_audit.csv", index=False)
    tau = float(trend.years_since([TAU_DATE], ORIGIN)[0])
    days = pd.date_range(*WINDOW, freq="D")
    taus = trend.years_since(days, ORIGIN)
    label = {"input_sha256": info["input_sha256"][:12], "window": f"{WINDOW[0]}..{WINDOW[1]}"}

    scores, fpts, kinks, profiles, best, infl, fri_rows, fri_all = [], [], [], [], [], [], [], []
    for name, recs in sources.items():
        scores.append(recs.assign(source=name))
        for series, org, incl in series_specs():
            r = select(recs, org, incl)
            s = trend.series_from_records(r, *WINDOW, ORIGIN)
            fpts.append(s.assign(source=name, series=series))
            fit = trend.fit_kink(s["t"].to_numpy(), s["score"].to_numpy(), tau)
            kinks.append({"source": name, "series": series, "spec": "fixed_break_2025-10-07",
                          "n_points": len(s), **fit.as_dict(), **label})
            prof = trend.break_profile(s["t"].to_numpy(), s["score"].to_numpy(), taus)
            prof.insert(0, "candidate_date", days.date)
            profiles.append(prof.assign(source=name, series=series))
            b = trend.best_supported_break(prof)
            if b is not None:
                best.append({"source": name, "series": series, "spec": "estimated_break_min_ssr",
                             "break_date": b["candidate_date"], "n_points": len(s),
                             **{k: b[k] for k in ["status", "n_pre", "n_post", "pre_slope", "post_slope",
                                                  "ratio", "ssr", "f_type"]}, **label})
            infl.append(trend.leave_one_model_out(r, *WINDOW, ORIGIN, tau).assign(source=name, series=series))
        for incl in (False, True):
            r = select(recs, None, incl)
            for kind in ("calendar12", "days365"):
                w = fri.window_rises(r, FRI_CUTOFF, kind=kind)
                s = fri.summarise_rises(w)
                fri_rows.append({"source": name, "unreleased_included": incl, "cutoff": FRI_CUTOFF, **s,
                                 "input_sha256": label["input_sha256"]})
                if kind == "calendar12" and not incl:
                    fri_all.append(w.assign(source=name))

    scores = pd.concat(scores)
    scores.to_csv(OUT / "public_scores.csv", index=False)
    pd.concat(fpts).to_csv(OUT / "frontier_points.csv", index=False)
    kinks = pd.DataFrame(kinks)
    best = pd.DataFrame(best)
    pd.concat([kinks, best]).to_csv(OUT / "kink_fits.csv", index=False)
    pd.concat(profiles).to_csv(OUT / "break_profiles.csv", index=False)
    infl = pd.concat(infl)
    infl.to_csv(OUT / "influence_checks.csv", index=False)
    fri_tab = pd.DataFrame(fri_rows)
    fri_tab.to_csv(OUT / "fri_windows_summary.csv", index=False)
    pd.concat(fri_all).to_csv(OUT / "fri_windows_calendar12.csv", index=False)

    # Method comparison: every pair of sources, all models and frontier models in the window.
    wide = scores.pivot_table(index="model", columns="source", values="score")
    fp = pd.concat(fpts)
    in_window_frontier = set(fp.loc[~fp.series.str.contains("incl"), "model"])
    comp = []
    names = list(sources)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = wide[a] - wide[b]
            dfr = d[d.index.isin(in_window_frontier)]
            imax = dfr.abs().idxmax()
            comp.append({"pair": f"{a} - {b}", "spearman_all": wide[a].corr(wide[b], method="spearman"),
                         "max_abs_diff_all": d.abs().max(), "model_max_all": d.abs().idxmax(),
                         "mean_diff_all": d.mean(),
                         "n_frontier_models_in_window": len(dfr),
                         "max_abs_diff_frontier_window": abs(dfr[imax]), "model_max_frontier_window": imax,
                         "signed_diff_at_that_model": dfr[imax],
                         "mean_abs_diff_frontier_window": dfr.abs().mean(),
                         "spearman_frontier_window": wide.loc[dfr.index, a].corr(wide.loc[dfr.index, b], method="spearman")})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "method_comparison.csv", index=False)
    wide.assign(in_window_frontier=wide.index.isin(in_window_frontier)).to_csv(OUT / "scores_wide.csv")
    (OUT / "run_info.json").write_text(json.dumps({**info, "tau_date": TAU_DATE, "origin": ORIGIN,
                                                    "fri_cutoff": FRI_CUTOFF, "window": WINDOW}, indent=2))
    figures(sources, fp, kinks, pd.concat(profiles), pd.concat(fri_all), wide, in_window_frontier, tau)
    print(kinks[["source", "series", "status", "n_pre", "n_post", "pre_slope", "post_slope", "ratio"]].round(3).to_string(index=False))
    print(best[["source", "series", "break_date", "status", "pre_slope", "post_slope", "ratio"]].round(3).to_string(index=False))
    print(fri_tab[["source", "unreleased_included", "window", "max_rise", "first_max_start", "first_max_end",
                   "start_model", "end_model", "any_window_ge_threshold"]].round(2).to_string(index=False))
    print(comp.round(3).T.to_string())


COLORS = {"least_squares": "#b5651d", "epoch_published": "#2a6f97", "bayes_refit": "#3a7d44"}


def figures(sources, fp, kinks, profiles, fri_w, wide, frontier_models, tau):
    tb = pd.Timestamp(TAU_DATE)
    # Trends
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, series in zip(axes, ["anthropic", "all_developers"]):
        for name in sources:
            s = fp[(fp.source == name) & (fp.series == series)]
            k = kinks[(kinks.source == name) & (kinks.series == series)].iloc[0]
            t = np.linspace(s.t.min(), s.t.max(), 100)
            X = np.column_stack([np.ones_like(t), t, np.clip(t - tau, 0, None)])
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(s)), s.t, np.clip(s.t - tau, 0, None)]), s.score, rcond=None)[0]
            dates = pd.Timestamp(ORIGIN) + pd.to_timedelta(t * trend.DAYS_PER_YEAR, "D")
            ax.plot(s.date, s.score, "o", ms=4, color=COLORS[name], alpha=0.8)
            lab = f"{name}: {k.pre_slope:.1f} -> {k.post_slope:.1f}/yr, ratio {k.ratio:.2f}" if k.status == "ok" else f"{name}: {k.status}"
            ax.plot(dates, X @ beta, "-", color=COLORS[name], lw=1.5, label=lab)
        ax.axvline(tb, color="#999", ls=":", lw=1)
        ax.set_title(f"{series.replace('_', ' ')} frontier, kink fixed at {TAU_DATE}")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("ECI")
    fig.suptitle("Public frontier trends, March 2024 to October 2026 (diagnostic, not a METR reproduction)", fontsize=10)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "frontier_trends.png", dpi=150); plt.close(fig)
    # Break profiles
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, series in zip(axes, ["anthropic", "all_developers"]):
        for name in sources:
            p = profiles[(profiles.source == name) & (profiles.series == series)]
            okp = p[p.status != "insufficient_dates"]
            ax.plot(pd.to_datetime(okp.candidate_date), okp.ssr, color=COLORS[name], label=name)
        ax.axvline(tb, color="#999", ls=":", lw=1)
        ax.set_title(f"{series.replace('_', ' ')}: SSR by candidate break"); ax.set_ylabel("SSR (ECI points²)")
        ax.legend(fontsize=7)
    fig.suptitle("Break profiles (descriptive; only candidates with >=4 release dates each side)", fontsize=10)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "break_profiles.png", dpi=150); plt.close(fig)
    # FRI
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for name in sources:
        w = fri_w[fri_w.source == name]
        ax.plot(w.start, w.rise, color=COLORS[name], label=f"{name} (max {w.rise.max():.1f})")
    ax.axhline(30, color="#c00", ls="--", lw=1, label="FRI fast-year line (30)")
    ax.set_ylabel("12-month frontier rise (ECI points)"); ax.set_xlabel("window start")
    ax.set_title(f"All-developer frontier, 12-calendar-month windows ending by {FRI_CUTOFF}", fontsize=10)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "fri_windows.png", dpi=150); plt.close(fig)
    # Method comparison
    names = list(sources)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ref = "epoch_published"
    for name in names:
        if name == ref:
            continue
        ax.scatter(wide[ref], wide[name], s=10, color=COLORS[name], alpha=0.5, label=f"{name} (all models)")
        fm = wide[wide.index.isin(frontier_models)]
        ax.scatter(fm[ref], fm[name], s=28, facecolors="none", edgecolors=COLORS[name], lw=1.2,
                   label=f"{name} (window frontier models)")
    lims = [wide.min().min() - 2, wide.max().max() + 2]
    ax.plot(lims, lims, color="#999", lw=1)
    ax.set_xlabel("Epoch published ECI (bundled)"); ax.set_ylabel("refit ECI")
    ax.set_title("Same data, different fitting method", fontsize=10); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "method_comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
