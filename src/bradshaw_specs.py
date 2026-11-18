"""Search for a public-data specification that gives Julian Bradshaw's figures.

He reported Epoch-ECI acceleration of about 1.45x for Anthropic on 5 November,
then 1.24x for Anthropic and 0.90x for all frontier models on 10 November
(dates US Central; the comments are timestamped 2026-11-06 and 2026-11-11 UTC;
LessWrong comments on the linkpost of METR's report and on the AI Futures
Project's Q3.5 post).
His chart notes state the method: "Epoch eci_scores.csv (5 Nov 2026), models
from Mar 2024. Running-max frontiers; OLS, slope change at 7 Oct 2025", and the
10 November chart says "incl. Fable 5.2". His 5 November chart also gives 1.14x
(data from Mar 2024) and 1.46x (data from Dec 2023) for Anthropic.

Every row changes one thing from the main specification (frontier built on full
history, window 2024-03-01 to 2026-10-31, break 2025-10-07, latest metadata
release date, published scores). Rows in group "stated_method" combine the
changes his chart notes name. Nothing is tuned to his numbers.

Output: results/bradshaw/spec_grid.csv and run_info.json.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from accel import frontier, trend  # noqa: E402
from analysis import ORIGIN, TAU_DATE, WINDOW, ZIP, load_sources  # noqa: E402

OUT = ROOT / "results" / "bradshaw"
ORG = "Anthropic"
BASE = {"series": "frontier points", "construction": "full history", "start": WINDOW[0], "end": WINDOW[1],
        "break": TAU_DATE, "date_rule": "latest metadata date", "source": "epoch_published"}
TARGETS = {"2026-11-10, incl. Fable 5.2": {"Anthropic": 1.24, "All developers": 0.90,
                                            "slopes": "Anthropic 17.0 -> 20.9; all labs 18.8 -> 16.8"},
           "2026-11-05, data from Mar 2024": {"Anthropic": 1.14, "slopes": "Anthropic 17.2 -> 19.6"},
           "2026-11-05, data from Dec 2023": {"Anthropic": 1.46},
           "2026-11-05, all labs running max": {"All developers": 0.82}}   # dates US Central


def points(rec: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """The (date, score) series one specification fits."""
    start, end = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
    if spec["series"] == "all models":
        r = rec[(rec.date >= start) & (rec.date <= end)]
        return r[["date", "score", "model"]]
    c = spec["construction"]
    if c == "full history":
        return frontier.frontier_in_window(rec, start, end)[["date", "score", "model"]]
    if c == "window only":
        return frontier.frontier_window_only(rec, start, end)[["date", "score", "model"]]
    if c == "running max at each release":
        dates = sorted(rec[(rec.date >= start) & (rec.date <= end)].date.unique())
    elif c == "running max daily":
        dates = pd.date_range(start, end, freq="D")
    else:
        raise ValueError(c)
    return frontier.running_max_at(rec, dates, cutoff=end)[["date", "score", "model"]].dropna()


def fit(rec: pd.DataFrame, spec: dict) -> dict:
    p = points(rec, spec)
    tau = float(trend.years_since([spec["break"]], ORIGIN)[0])
    k = trend.fit_kink(trend.years_since(p["date"], ORIGIN), p["score"].to_numpy(), tau)
    first = p.sort_values(["date", "score"], ascending=[True, False]).iloc[0]
    return {"first_point": f"{first.model} ({first.date.date()}, {first.score:.1f})", "n_points": len(p), "n_pre": k.n_pre, "n_post": k.n_post, "status": k.status,
            "pre_slope": k.pre_slope, "post_slope": k.post_slope, "ratio": k.ratio}


def specs():
    yield "baseline", "none", "", {}
    yield "one_change", "series", "all models, not frontier only", {"series": "all models"}
    for v in ["2023-12-01", "2024-01-01", "2024-02-01", "2024-04-01", "2024-06-01"]:
        yield "one_change", "window start", v, {"start": v}
    for v in ["2026-11-11", "2026-11-13"]:
        yield "one_change", "window end", v, {"end": v}
    for v in ["2025-09-29", "2025-10-15", "2025-11-01", "2025-11-24"]:
        yield "one_change", "break date", v, {"break": v}
    for v in ["earliest metadata date", "eci_scores.csv date"]:
        yield "one_change", "release-date rule", v, {"date_rule": v}
    for v in ["least_squares", "bayes_refit"]:
        yield "one_change", "score source", v, {"source": v}
    for v in ["window only", "running max at each release", "running max daily"]:
        yield "one_change", "frontier construction", v, {"construction": v}
    for src in ["epoch_published", "least_squares", "bayes_refit"]:
        yield ("stated_method", "window only + end 2026-11-13 (10 Nov chart)", src,
               {"construction": "window only", "end": "2026-11-13", "source": src})
        yield ("stated_method", "window only + start 2023-12-01 (5 Nov 'from Dec 2023')", src,
               {"construction": "window only", "start": "2023-12-01", "source": src})
        yield ("stated_method", "window only + start 2023-12-01 + end 2026-11-13", src,
               {"construction": "window only", "start": "2023-12-01", "end": "2026-11-13", "source": src})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources, info, _ = load_sources()
    released = {k: v[~v["unreleased"]].reset_index(drop=True) for k, v in sources.items()}
    with zipfile.ZipFile(ZIP) as z:
        pub_dates = pd.read_csv(z.open("epoch_capabilities_index/eci_scores.csv")).set_index("Model")["date"]
    rows = []
    for group, dim, value, change in specs():
        spec = {**BASE, **change}
        rec = released[spec["source"]].copy()
        if spec["date_rule"] == "earliest metadata date":
            rec["date"] = pd.to_datetime(rec["date_earliest"])
        elif spec["date_rule"] == "eci_scores.csv date":
            rec["date"] = pd.to_datetime(rec["model"].map(pub_dates))
        row = {"group": group, "changed": dim, "value": value, **spec}
        for name, r in [("anthropic", rec[rec.organization == ORG]), ("all_dev", rec)]:
            row.update({f"{name}_{k}": v for k, v in fit(r, spec).items()})
        rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "spec_grid.csv", index=False)
    (OUT / "run_info.json").write_text(json.dumps({"input_sha256": info["input_sha256"], "baseline": BASE,
                                                   "targets": TARGETS}, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(grid[["group", "changed", "value", "anthropic_ratio", "anthropic_pre_slope", "anthropic_post_slope",
                    "anthropic_n_pre", "anthropic_n_post", "all_dev_ratio", "all_dev_pre_slope",
                    "all_dev_post_slope"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
