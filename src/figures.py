"""The note's four figures, drawn only from
existing result files; nothing is re-estimated except the fitted lines, which
are checked against the stored fits before drawing.

  Figure 1  Anthropic's fourteen frontier points, two-segment fits on published
            scores and on the prior (least-squares) method.
  Figure 2  Anthropic ratio over the 4,000 posterior draws of the Bayesian refit.
  Figure 3  Each developer's ratio over both draw sets, with 90% intervals.
  Figure 4  The best freely dated change of slope against the best freely dated
            step, published scores, with each fit's SSR.

Palette: navy, rust and grey from the author's Substack style. PNG, 1440 px wide
(twice a 720-pixel column). Writes figures/note_fig*.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from accel import trend  # noqa: E402

R = ROOT / "results"
PE, RB, CC = R / "public_data", R / "robustness", R / "developer_comparison"
FIG = ROOT / "figures"
ORIGIN, TAU_DATE = "2024-03-01", "2025-10-07"
NAVY, RUST, GREY, INK = "#0B2545", "#C44536", "#6B7280", "#1F2937"
WIDTH_IN, DPI = 7.2, 200                     # 1440 px
NAMED = ["Claude 3.5 Sonnet", "Claude Sonnet 4.5", "Claude Opus 4.5", "Claude Fable 5", "Claude Opus 5.5"]
SOURCE_LABEL = {"epoch_published": "Epoch's published scores", "least_squares": "Epoch's prior fitting method",
                "bayes_refit": "Epoch's current method, refit"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GREY, "grid.linewidth": 0.4, "grid.alpha": 0.25,
        "xtick.major.size": 3, "ytick.major.size": 3, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.titlelocation": "left", "axes.titleweight": "bold"})


def note(fig, text):
    """Footnote under the axes; the layout leaves room for it."""
    h = 0.045 * (text.count("\n") + 1)
    fig.get_layout_engine().set(rect=(0, h, 1, 1 - h))
    fig.text(0.01, 0.008, text, fontsize=7, color=GREY)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


def anthropic_points(source: str) -> pd.DataFrame:
    f = pd.read_csv(PE / "frontier_points.csv", parse_dates=["date"])
    return f[(f.source == source) & (f.series == "anthropic")].sort_values("date").reset_index(drop=True)


def stored_kink(source: str) -> pd.Series:
    k = pd.read_csv(PE / "kink_fits.csv")
    return k[(k.source == source) & (k.series == "anthropic") & (k.spec == "fixed_break_2025-10-07")].iloc[0]


def kink_line(p: pd.DataFrame, tau: float, check: pd.Series | None = None):
    """Fitted two-segment line over the span of the points; asserts the slopes
    and SSR equal the stored fit when one is given."""
    beta, ssr = trend.kink_coefficients(p["t"].to_numpy(), p["score"].to_numpy(), tau)
    if check is not None:
        assert np.isclose(beta[1], check.pre_slope) and np.isclose(beta[1] + beta[2], check.post_slope)
        assert np.isclose(ssr, check.ssr)
    tt = np.linspace(p["t"].min(), p["t"].max(), 200)
    return tt, beta[0] + beta[1] * tt + beta[2] * np.clip(tt - tau, 0, None), beta, ssr


def to_date(t):
    return pd.Timestamp(ORIGIN) + pd.to_timedelta(np.asarray(t) * 365.25, unit="D")


def break_line(ax, label=True):
    ax.axvline(pd.Timestamp(TAU_DATE), color=GREY, ls=":", lw=1)
    if label:
        ax.text(pd.Timestamp(TAU_DATE), ax.get_ylim()[0], " 7 Oct 2025", color=GREY, fontsize=7.5,
                va="bottom", ha="left")


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def figure1():
    tau = float(trend.years_since([TAU_DATE], ORIGIN)[0])
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.2), layout="constrained")
    labels = {}
    for src, col, filled in [("least_squares", RUST, False), ("epoch_published", NAVY, True)]:
        p = anthropic_points(src)
        k = stored_kink(src)
        tt, yy, beta, _ = kink_line(p, tau, k)
        ax.plot(to_date(tt), yy, color=col, lw=2, ls="-" if filled else "--", zorder=2)
        ax.scatter(p.date, p.score, s=34, zorder=3, color=col if filled else "white", edgecolor=col, lw=1.4)
        labels[src] = (to_date(tt[-1:])[0], yy[-1], k)
    ax.set_ylim(126, 174)
    break_line(ax)
    pub = anthropic_points("epoch_published").set_index("model")
    offsets = {"Claude 3.5 Sonnet": (6, -12), "Claude Sonnet 4.5": (-6, 9), "Claude Opus 4.5": (6, -12),
               "Claude Fable 5": (-6, 9), "Claude Opus 5.5": (-6, 9)}
    for m in NAMED:
        dx, dy = offsets[m]
        ax.annotate(m.replace("Claude ", ""), (pub.loc[m, "date"], pub.loc[m, "score"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=INK, ha="left" if dx > 0 else "right")
    for src, yfrac, mark in [("epoch_published", 0.95, "solid line, filled dots"),
                             ("least_squares", 0.83, "dashed line, open dots")]:
        k = labels[src][2]
        ax.text(0.02, yfrac, f"{SOURCE_LABEL[src]} ({mark})\n{k.pre_slope:.1f} → {k.post_slope:.1f} points/yr "
                f"= {k.ratio:.2f}x", transform=ax.transAxes, fontsize=8, va="top",
                color=NAVY if src == "epoch_published" else RUST)
    ax.set_ylabel("Epoch Capabilities Index")
    date_axis(ax)
    ax.set_title("Anthropic's frontier releases, March 2024 to October 2026", fontsize=10)
    note(fig, "Each point is an Anthropic release that beat every earlier Anthropic release. Lines: two "
             "segments joined at 7 Oct 2025.")
    save(fig, "note_fig1_anthropic_frontier.png")


def figure2():
    d = pd.read_csv(RB / "posterior_draw_ratios.csv")
    ok = d[d.status == "ok"].ratio
    ps = pd.read_csv(RB / "posterior_summary.csv", index_col=0)["value"]
    med, lo, hi = float(ps["median"]), float(ps["p05"]), float(ps["p95"])
    assert np.isclose(ok.median(), med) and np.isclose(ok.quantile(0.05), lo)
    pub, ls = stored_kink("epoch_published").ratio, stored_kink("least_squares").ratio
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.6), layout="constrained")
    bins = np.arange(1.0, 2.31, 0.02)
    ax.axvspan(lo, hi, color=NAVY, alpha=0.08, lw=0)
    ax.hist(ok, bins=bins, color=NAVY, edgecolor="white", linewidth=0.6)
    peak = np.histogram(ok, bins=bins)[0].max()
    ax.set_ylim(0, peak * 1.45)
    ax.plot([med, med], [0, peak * 1.38], color=INK, lw=1.2)
    ax.text(med, peak * 1.38, f" median {med:.2f}", fontsize=8, color=INK, va="top", ha="left")
    ax.text(lo - 0.01, peak * 0.9, f"Shaded: 90% of draws,\n{lo:.2f} to {hi:.2f}", fontsize=8, color=INK,
            va="top", ha="right")
    for v, txt, ha, h in [(pub, f"Epoch's published scores {pub:.2f} ", "right", 1.38),
                          (ls, f" prior fitting method {ls:.2f}", "left", 1.13)]:
        ax.plot([v, v], [0, peak * h], color=RUST, lw=1.2, ls="--")
        ax.text(v, peak * h, txt, fontsize=8, color=RUST, ha=ha, va="top" if h > 1.3 else "bottom")
    ax.set_xlim(1.0, 2.3)
    ax.set_xlabel("Ratio of Anthropic's slope after 7 Oct 2025 to its slope before")
    ax.set_ylabel("Number of draws")
    ax.set_title(f"Anthropic's ratio across {len(ok):,} draws of the model scores", fontsize=10)
    note(fig, "Draws from Epoch's current (Bayesian) model, refit; the frontier is rebuilt and refit in each "
             "draw. Covers score uncertainty only.")
    save(fig, "note_fig2_ratio_draws.png")


def figure3():
    s = pd.read_csv(CC / "developer_ratios_summary.csv")
    pt = pd.read_csv(CC / "developer_ratios_point.csv")
    pt = pt[pt.source == "epoch_published"].set_index("organization")
    order = s[s.draw_set == "bayes_refit"].sort_values("median").organization.tolist()
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.6), layout="constrained")
    for ds, col, off, filled in [("bayes_refit", NAVY, 0.16, True), ("least_squares", RUST, -0.16, False)]:
        c = s[s.draw_set == ds].set_index("organization").reindex(order)
        y = np.arange(len(order)) + off
        ax.hlines(y, c.p05, c.p95, color=col, lw=2)
        ax.scatter(c["median"], y, s=36, zorder=3, color=col if filled else "white", edgecolor=col, lw=1.4)
    ax.axvline(1.0, color=GREY, lw=1, ls=":")
    sup = s.pivot_table(index="organization", columns="draw_set", values="share_supported")
    labels = []
    for o in order:
        n = f"{int(pt.loc[o, 'n_pre'])} | {int(pt.loc[o, 'n_post'])}"
        extra = "" if sup.loc[o].min() >= 0.9 else f"  (in {sup.loc[o, 'bayes_refit']:.0%} / {sup.loc[o, 'least_squares']:.0%} of draws)"
        labels.append(f"{o}   {n}{extra}")
    ax.set_yticks(np.arange(len(order)), labels)
    ax.grid(False)
    ax.set_xlabel("Ratio of slope after 7 Oct 2025 to slope before (median and 90% of draws)")
    top = len(order) - 1
    c_top = s[(s.draw_set == "bayes_refit") & (s.organization == order[-1])].iloc[0]
    ax.text(c_top.p95 + 0.05, top + 0.16, "current method (4,000 draws)", color=NAVY, fontsize=8, va="center")
    l_top = s[(s.draw_set == "least_squares") & (s.organization == order[-1])].iloc[0]
    ax.text(l_top.p95 + 0.05, top - 0.16, "prior method (500 draws)", color=RUST, fontsize=8, va="center")
    ax.text(1.0, -0.9, " no change of pace", color=GREY, fontsize=7.5, va="center")
    ax.set_ylim(-1.2, len(order) - 0.4)
    ax.set_title("Each developer's change of pace at 7 October 2025", fontsize=10)
    note(fig, "Numbers after each name: frontier releases before | after the break (published scores). "
             "A developer enters a draw only with 4+ release dates each side.")
    save(fig, "note_fig3_developers.png")


def figure4():
    p = anthropic_points("epoch_published")
    t, y = p["t"].to_numpy(), p["score"].to_numpy()
    shape = pd.read_csv(CC / "slope_vs_step_point.csv").set_index("source").loc["epoch_published"]
    kink_day = pd.Timestamp(shape.kink_best_date)
    step_day = pd.Timestamp(shape.step_best_first_day)
    tau_k = float(trend.years_since([kink_day], ORIGIN)[0])
    tau_s = float(trend.years_since([step_day], ORIGIN)[0])
    bk, ssr_k = trend.kink_coefficients(t, y, tau_k)
    bs, ssr_s = trend.level_shift_coefficients(t, y, tau_s)
    assert np.isclose(ssr_k, shape.kink_best_ssr) and np.isclose(ssr_s, shape.step_best_ssr)
    tt = np.linspace(t.min(), t.max(), 400)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.6), sharex=True, sharey=True, layout="constrained")
    fits = [(axes[0], bk[0] + bk[1] * tt + bk[2] * np.clip(tt - tau_k, 0, None), ssr_k, kink_day,
             f"Best change of slope: {kink_day.day} {kink_day:%b %Y}",
             f"{bk[1]:.1f} → {bk[1] + bk[2]:.1f} points/yr ({(bk[1] + bk[2]) / bk[1]:.2f}x)"),
            (axes[1], bs[0] + bs[1] * tt + bs[2] * (tt >= tau_s), ssr_s, step_day,
             "Best step: end of May 2026", f"{bs[1]:.1f} points/yr, step of {bs[2]:.1f} points")]
    for ax, yy, ssr, when, title, desc in fits:
        ax.scatter(p.date, p.score, s=26, color=NAVY, zorder=3)
        d = to_date(tt)
        if ax is axes[1]:
            cut = tt < tau_s
            ax.plot(d[cut], yy[cut], color=RUST, lw=2)
            ax.plot(d[~cut], yy[~cut], color=RUST, lw=2)
        else:
            ax.plot(d, yy, color=RUST, lw=2)
        ax.axvline(when, color=GREY, ls=":", lw=1)
        ax.set_title(title, fontsize=9)
        ax.text(0.03, 0.95, f"{desc}\nsum of squared residuals {ssr:.1f}", transform=ax.transAxes, fontsize=8,
                color=INK, va="top")
        date_axis(ax)
        ax.tick_params(axis="x", labelrotation=0)
    axes[0].set_ylabel("Epoch Capabilities Index")
    last = pd.Timestamp(shape.step_best_last_day)
    note(fig, "Each shape chose its own date: the day in the window that gives it the lowest sum of squared "
              "residuals. Published scores, the same fourteen points.\nThe step fits equally well on any day from "
              f"{step_day.day} {step_day:%b} to {last.day} {last:%b %Y}.")
    save(fig, "note_fig4_slope_vs_step.png")


def main():
    style()
    figure1()
    figure2()
    figure3()
    figure4()
    print("wrote", ", ".join(sorted(x.name for x in FIG.glob("note_fig*.png"))))


if __name__ == "__main__":
    main()
