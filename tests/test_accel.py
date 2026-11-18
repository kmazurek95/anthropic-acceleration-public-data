"""Edge-case tests for the pipeline modules."""
import numpy as np
import pandas as pd
import pytest

from accel import catchup, fri, frontier, inputs, transforms, trend


def recs(rows):
    return pd.DataFrame(rows, columns=["model", "date", "organization", "score"])


# --- scale ------------------------------------------------------------------

def _two_segment(tau=1.5, pre=15.0, post=29.0, n=40, noise=0.0, seed=0):
    t = np.linspace(0, 2.6, n)
    y = 100 + pre * t + (post - pre) * np.clip(t - tau, 0, None)
    y = y + np.random.default_rng(seed).normal(0, noise, n)
    return t, y


def test_affine_invariance_of_slope_ratio():
    t, y = _two_segment(noise=0.5)
    base = trend.fit_kink(t, y, 1.5).ratio
    for a, b in [(0, 2.0), (-50, 0.3), (12.5, 7.0)]:
        assert trend.fit_kink(t, transforms.affine(y, a, b), 1.5).ratio == pytest.approx(base, rel=1e-9)


def test_nonlinear_monotone_rescaling_changes_ratio():
    t, y = _two_segment()
    base = trend.fit_kink(t, y - y.min(), 1.5).ratio
    concave = trend.fit_kink(t, transforms.g(y - y.min(), -0.03), 1.5).ratio
    convex = trend.fit_kink(t, transforms.g(y - y.min(), 0.03), 1.5).ratio
    assert concave < base < convex


def test_stylized_path_known_values():
    # Identity ratio 1.93; ratio 1.0 near weight 0.54 for a point 25 higher.
    assert transforms.stylized_ratio(0.0) == pytest.approx(1.93, abs=0.005)
    from scipy.optimize import brentq
    k = brentq(lambda k: transforms.stylized_ratio(k) - 1.0, -0.2, 0.2)
    assert np.exp(25 * k) == pytest.approx(0.54, abs=0.005)


def test_milestone_crossing_time_invariant_when_threshold_transformed_too():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 35.0, 60.0])
    thr = 30.0
    base = transforms.milestone_crossing_time(t, y, thr)
    for k in (-0.05, 0.04):
        # path transformed at the observed points; crossing interpolated within the
        # segment, so compare the bracketing segment (invariant) and the exact time
        # under a transform applied to both path and threshold at the knots
        yk, tk = transforms.g(y, k), transforms.g(thr, k)
        seg_base = np.searchsorted(y, thr)
        seg_k = np.searchsorted(yk, tk)
        assert seg_base == seg_k
    # exact invariance for a strictly monotone step path (no interpolation)
    steps = transforms.milestone_crossing_time(t, np.array([10, 20, 30, 60.0]), 30.0)
    steps_k = transforms.milestone_crossing_time(t, transforms.g([10, 20, 30, 60.0], -0.05),
                                                 float(transforms.g(30.0, -0.05)))
    assert steps == pytest.approx(steps_k)
    assert base == pytest.approx(1 + 10 / 15)   # linear interpolation inside the bracketing segment


# --- kink fits --------------------------------------------------------------

def test_known_kink_recovered():
    t, y = _two_segment(tau=1.5, pre=15, post=29)
    fit = trend.fit_kink(t, y, 1.5)
    assert fit.status == "ok"
    assert fit.pre_slope == pytest.approx(15)
    assert fit.post_slope == pytest.approx(29)
    prof = trend.break_profile(t, y, np.linspace(0.6, 2.0, 141))
    best = trend.best_supported_break(prof)
    assert best["tau"] == pytest.approx(1.5, abs=0.02)


def test_no_break_gives_ratio_near_one():
    t = np.linspace(0, 2.6, 40)
    y = 100 + 20 * t + np.random.default_rng(1).normal(0, 0.3, 40)
    fit = trend.fit_kink(t, y, 1.3)
    assert fit.ratio == pytest.approx(1.0, abs=0.1)


def test_sparse_dates_flagged_not_forced():
    t = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3, 1.4])
    y = 100 + 10 * t
    fit = trend.fit_kink(t, y, 0.5)       # 3 dates before, 5 after
    assert fit.status == "insufficient_dates"
    assert np.isnan(fit.ratio)
    prof = trend.break_profile(t, y, np.array([0.5, 0.9]))
    assert set(prof["status"]) == {"insufficient_dates"}
    assert trend.best_supported_break(prof) is None


def test_repeated_dates_count_once_per_side():
    t = np.array([0, 0, 0, 0, 0.5, 1.0, 1.5, 2.0, 2.5])
    fit = trend.fit_kink(t, 100 + 10 * t, 0.75)
    assert fit.n_pre == 2 and fit.status == "insufficient_dates"


@pytest.mark.parametrize("pre", [0.0, -5.0])
def test_nonpositive_baseline_reported(pre):
    t = np.linspace(0, 2, 20)
    y = 100 + pre * t + 20 * np.clip(t - 1.0, 0, None)
    fit = trend.fit_kink(t, y, 1.0)
    assert fit.status == "nonpositive_baseline"
    assert np.isnan(fit.ratio)
    assert fit.pre_slope == pytest.approx(pre, abs=1e-9)


def test_level_shift_vs_kink_discriminate():
    t = np.linspace(0, 2.6, 40)
    jump = 100 + 15 * t + 8 * (t >= 1.5)
    ls = trend.fit_level_shift(t, jump, 1.5)
    assert ls["slope"] == pytest.approx(15) and ls["jump"] == pytest.approx(8)
    assert ls["ssr"] < trend.fit_kink(t, jump, 1.5).ssr
    kinked = 100 + 15 * t + 14 * np.clip(t - 1.5, 0, None)
    assert trend.fit_kink(t, kinked, 1.5).ssr < trend.fit_level_shift(t, kinked, 1.5)["ssr"]


# --- frontier ---------------------------------------------------------------

def test_same_day_permutation_does_not_change_frontier():
    rows = [("A", "2025-01-01", "X", 100), ("B", "2025-02-01", "X", 105),
            ("C", "2025-02-01", "Y", 105), ("D", "2025-02-01", "Y", 103),
            ("E", "2025-03-01", "Y", 110)]
    r = recs(rows)
    f1 = frontier.step_frontier(r)
    f2 = frontier.step_frontier(r.iloc[::-1])
    f3 = frontier.step_frontier(r.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_frame_equal(f1, f3)
    feb = f1[f1["date"] == "2025-02-01"].iloc[0]
    assert feb["model"] == "B"                       # tie broken by name
    assert feb["same_day_models"] == "B;C;D"
    assert list(f1["model"]) == ["A", "B", "E"]


def test_frontier_rebuilt_after_deletion():
    r = recs([("A", "2025-01-01", "X", 100), ("B", "2025-02-01", "X", 110),
              ("C", "2025-03-01", "X", 105), ("D", "2025-04-01", "X", 120)])
    assert list(frontier.step_frontier(r)["model"]) == ["A", "B", "D"]
    # Dropping B lets C (previously dominated) become a frontier point.
    assert list(frontier.step_frontier(r[r.model != "B"])["model"]) == ["A", "C", "D"]


def test_window_preserves_prior_frontier():
    r = recs([("A", "2024-01-01", "X", 120), ("B", "2024-06-01", "X", 110),
              ("C", "2024-09-01", "X", 125)])
    w = frontier.frontier_in_window(r, "2024-03-01", "2024-12-31")
    # B is dominated by A (before the window); trimming must not promote it.
    assert list(w["model"]) == ["C"]


def test_cutoff_enforced():
    r = recs([("A", "2025-01-01", "X", 100), ("B", "2025-06-01", "X", 130)])
    f = frontier.step_frontier(r, cutoff="2025-05-31")
    assert list(f["model"]) == ["A"]
    rises = fri.window_rises(r, cutoff="2025-05-31")
    assert rises.empty or rises["end"].max() <= pd.Timestamp("2025-05-31")


def test_duplicate_models_rejected():
    r = recs([("A", "2025-01-01", "X", 100), ("A", "2025-02-01", "X", 101)])
    with pytest.raises(ValueError):
        frontier.step_frontier(r)


# --- metadata ---------------------------------------------------------------

def test_inconsistent_metadata():
    meta = pd.DataFrame({
        "model_version": ["a1", "a2", "b1", "c1", "c2"],
        "model_group": ["A", "A", "B", "C", "C"],
        "date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-04-01"],
        "organization": ["X", "X", "Y", "Z", "W"],
        "accessibility": ["API access"] * 5,
    })
    with pytest.raises(ValueError, match="conflicting organizations"):
        inputs.group_table(meta, ["A", "B", "C"])
    table, audit = inputs.group_table(meta, ["A", "B"])
    a = table.set_index("model").loc["A"]
    assert a["date"] == pd.Timestamp("2025-02-01")
    assert a["date_earliest"] == pd.Timestamp("2025-01-01")
    assert list(audit["model_group"]) == ["A"]
    with pytest.raises(ValueError, match="absent"):
        inputs.group_table(meta, ["A", "Q"])
    meta.loc[meta.model_group == "B", "organization"] = None
    with pytest.raises(ValueError, match="no organization"):
        inputs.group_table(meta, ["A", "B"])
    table, audit = inputs.group_table(meta, ["A", "B"], org_fallback={"B": "Y2"})
    assert table.set_index("model").loc["B", "organization"] == "Y2"
    assert "organization_from_fallback" in set(audit["issue"])
    table, audit = inputs.group_table(meta, ["A", "B"], org_fallback={}, allow_unknown_org=True)
    assert table.set_index("model").loc["B", "organization"] == "unknown"
    assert "organization_unknown" in set(audit["issue"])


# --- FRI windows ------------------------------------------------------------

def test_calendar_month_windows_and_leap_year():
    r = recs([("A", "2024-02-29", "X", 100), ("B", "2025-02-28", "X", 140),
              ("C", "2025-06-01", "X", 150)])
    rises = fri.window_rises(r, cutoff="2025-12-31", kind="calendar12")
    first = rises.iloc[0]
    assert first["start"] == pd.Timestamp("2024-02-29")
    assert first["end"] == pd.Timestamp("2025-02-28")     # leap day + 12 months
    assert first["rise"] == pytest.approx(40)
    d365 = fri.window_rises(r, cutoff="2025-12-31", kind="days365")
    assert d365.iloc[0]["end"] == pd.Timestamp("2025-02-28")  # 2024 is leap: 365 days lands on Feb 28
    assert (rises["end"] <= pd.Timestamp("2025-12-31")).all()
    s = fri.summarise_rises(rises, threshold=30)
    assert s["any_window_ge_threshold"] and s["max_rise"] == pytest.approx(50)


def test_no_window_starts_before_first_frontier_point():
    r = recs([("A", "2025-01-10", "X", 100), ("B", "2025-09-01", "X", 110)])
    rises = fri.window_rises(r, cutoff="2026-03-01")
    assert rises["start"].min() == pd.Timestamp("2025-01-10")
    assert rises["start_score"].notna().all()


def test_hash_check(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    with pytest.raises(ValueError):
        inputs.verify_input(p)
    assert inputs.verify_input(p, inputs.sha256(p)) == inputs.sha256(p)


# --- vectorized profiles (used inside draw loops) ---------------------------

@pytest.mark.parametrize("kind", ["kink", "level_shift"])
def test_ssr_profile_matches_single_fits(kind):
    rng = np.random.default_rng(7)
    t = np.sort(np.r_[rng.uniform(0, 2.6, 14), 0.5, 0.5])          # includes a repeated date
    y = 120 + 15 * t + 10 * np.clip(t - 1.6, 0, None) + rng.normal(0, 1, t.size)
    taus = np.r_[-0.5, np.linspace(0, 2.6, 60), 3.0]                  # includes taus outside the data
    prof = trend.ssr_profile(t, y, taus, kind=kind)
    for _, r in prof.iterrows():
        if kind == "kink":
            f = trend.fit_kink(t, y, r.tau)
            ssr, slope, n_pre, n_post = f.ssr, f.pre_slope, f.n_pre, f.n_post
        else:
            f = trend.fit_level_shift(t, y, r.tau)
            ssr, slope, n_pre, n_post = f["ssr"], f["slope"], f["n_pre"], f["n_post"]
        assert r.ssr == pytest.approx(ssr, rel=1e-8, abs=1e-8)
        assert r.slope == pytest.approx(slope, rel=1e-8, abs=1e-8)
        assert (r.n_pre, r.n_post) == (n_pre, n_post)
        assert r.supported == (n_pre >= 4 and n_post >= 4)
    best = trend.best_of_profile(prof)
    assert best is not None and best.supported
    assert best.ssr == prof[prof.supported].ssr.min()


def test_ssr_profile_unsupported_everywhere_gives_none():
    t = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    prof = trend.ssr_profile(t, 100 + t, np.array([0.5, 0.9]))
    assert not prof.supported.any()
    assert trend.best_of_profile(prof) is None


# --- catch-up ---------------------------------------------------------------

def _field():
    return recs([("A1", "2024-06-01", "A", 130), ("B1", "2024-09-01", "B", 128),
                 ("B2", "2025-01-01", "B", 135), ("A2", "2025-03-01", "A", 133),
                 ("B3", "2025-06-01", "B", 140), ("A3", "2025-09-01", "A", 142)])


def test_gap_series_and_lead_spells():
    f = _field()
    g = catchup.gap_series(f[f.organization == "A"], f, "2024-06-01", "2025-12-31").set_index("date")
    assert g.loc["2024-06-01", "gap"] == 0 and g.loc["2024-06-01", "org_leads"]
    assert g.loc["2024-09-15", "gap"] == 0                  # B1 (128) does not pass A1 (130)
    assert g.loc["2025-01-01", "gap"] == pytest.approx(-5)   # B2 135 vs A1 130
    assert g.loc["2025-03-01", "gap"] == pytest.approx(-2)   # A2 133 vs B2 135
    assert g.loc["2025-06-01", "gap"] == pytest.approx(-7)   # B3 140 vs A2 133
    assert g.loc["2025-09-01", "gap"] == 0 and g.loc["2025-09-01", "org_model"] == "A3"
    assert (g["gap"] <= 0).all()
    sp = catchup.lead_spells(g.reset_index())
    assert list(sp.start.dt.date.astype(str)) == ["2024-06-01", "2025-09-01"]
    assert list(sp.end.dt.date.astype(str)) == ["2024-12-31", "2025-12-31"]
    s = catchup.gap_summary(g.reset_index(), at="2025-07-01")
    assert s["gap_at"] == pytest.approx(-7) and s["field_model_at"] == "B3"
    assert s["deepest_gap"] == pytest.approx(-7) and str(s["deepest_gap_date"]) == "2025-06-01"
    assert str(s["lead_regained"]) == "2025-09-01" and s["lead_regained_model"] == "A3"


def test_gap_series_keeps_history_before_window():
    f = _field()
    g = catchup.gap_series(f[f.organization == "A"], f, "2025-02-01", "2025-04-01").set_index("date")
    assert g.loc["2025-02-01", "gap"] == pytest.approx(-5)   # carries A1 and B2 from before the window


def test_gap_series_requires_field_to_contain_developer():
    f = _field()
    with pytest.raises(ValueError):
        catchup.gap_series(f[f.organization == "A"], f[f.organization == "B"], "2024-06-01", "2025-12-31")


def test_cross_ratio():
    assert catchup.cross_ratio(20.78, 17.42) == pytest.approx(20.78 / 17.42)
    assert np.isnan(catchup.cross_ratio(20.0, 0.0))


def test_window_start_one_day_earlier_adds_exactly_that_point():
    r = recs([("P", "2024-02-29", "X", 125), ("Q", "2024-06-20", "X", 130), ("R", "2024-10-22", "X", 133)])
    base = trend.series_from_records(r, "2024-03-01", "2026-10-31", "2024-03-01")
    wider = trend.series_from_records(r, "2024-02-29", "2026-10-31", "2024-03-01")
    assert list(base.model) == ["Q", "R"] and list(wider.model) == ["P", "Q", "R"]
    assert wider.t.iloc[0] == pytest.approx(-1 / 365.25)


# --- additions: window-only frontier, running max, coefficients, paired shares ---

from accel import draws as draws_mod  # noqa: E402


def test_window_only_frontier_promotes_a_dominated_model():
    r = recs([("Opus", "2024-02-29", "A", 125.5), ("Haiku", "2024-03-07", "A", 117.7),
              ("Sonnet", "2024-06-20", "A", 130.0)])
    full = frontier.frontier_in_window(r, "2024-03-01", "2026-10-31")
    trimmed = frontier.frontier_window_only(r, "2024-03-01", "2026-10-31")
    assert list(full.model) == ["Sonnet"]
    assert list(trimmed.model) == ["Haiku", "Sonnet"]


def test_running_max_at_carries_forward_and_is_nan_before_first_point():
    r = recs([("a", "2024-01-10", "X", 1.0), ("b", "2024-02-01", "X", 0.5), ("c", "2024-03-01", "X", 2.0)])
    out = frontier.running_max_at(r, ["2024-03-05", "2024-01-01", "2024-02-15"])
    assert list(out.date.dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-02-15", "2024-03-05"]
    assert np.isnan(out.score.iloc[0])
    assert list(out.score.iloc[1:]) == [1.0, 2.0] and list(out.model.iloc[1:]) == ["a", "c"]


def test_coefficients_match_the_fit_functions():
    t, y = _two_segment(noise=0.7, seed=3)
    beta, ssr = trend.kink_coefficients(t, y, 1.5)
    k = trend.fit_kink(t, y, 1.5)
    assert beta[1] == pytest.approx(k.pre_slope) and beta[1] + beta[2] == pytest.approx(k.post_slope)
    assert ssr == pytest.approx(k.ssr)
    beta, ssr = trend.level_shift_coefficients(t, y, 1.5)
    ls = trend.fit_level_shift(t, y, 1.5)
    assert beta[1] == pytest.approx(ls["slope"]) and beta[2] == pytest.approx(ls["jump"])
    assert ssr == pytest.approx(ls["ssr"])


def test_share_above_counts_unsupported_draws_as_not_above():
    d = pd.DataFrame({"draw_set": ["s"] * 6, "draw": [0, 0, 1, 1, 2, 2],
                      "organization": ["A", "All", "A", "All", "A", "All"],
                      "ratio": [1.5, 1.0, 0.9, 1.0, np.nan, 1.0]})
    out = draws_mod.share_above(d, "A", "All").iloc[0]
    assert out.n_draws == 3 and out.share_all_draws == pytest.approx(1 / 3)
    assert out.n_both_supported == 2 and out.share_both_supported == pytest.approx(0.5)
    with pytest.raises(ValueError):
        draws_mod.share_above(pd.concat([d, d.iloc[:1]]), "A", "All")


def test_extending_the_window_end_adds_only_later_points():
    r = recs([("p", "2025-06-01", "X", 1.0), ("q", "2026-10-20", "X", 2.0), ("r", "2026-11-09", "X", 3.0)])
    base = trend.series_from_records(r, "2024-03-01", "2026-10-31", "2024-03-01")
    ext = trend.series_from_records(r, "2024-03-01", "2026-11-13", "2024-03-01")
    assert list(base.model) == ["p", "q"] and list(ext.model) == ["p", "q", "r"]


def test_write_csv_gz_is_byte_stable(tmp_path):
    import time
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "t.csv.gz"
    draws_mod.write_csv_gz(df, p)
    first = p.read_bytes()
    time.sleep(1.1)
    draws_mod.write_csv_gz(df, p)
    assert p.read_bytes() == first and first[4:8] == bytes(4)
    pd.testing.assert_frame_equal(pd.read_csv(p), df)
