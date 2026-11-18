# [Anthropic's Acceleration: What the Public Data Show](https://kmazurek95.github.io/writing/essays/anthropic-acceleration-public-data/)

All code and numbers to reproduce results can be found here for the note linked above. In its report of 5 November
2026, METR estimated that, as of October 2026, the capabilities of Anthropic models were increasing at a rate 1.9
times the rate before October 2025. The 1.9 is METR's fit to Anthropic's internal capabilities index; its overall judgment is about 2x.
Here I look at the corresponding result implied by the Epoch Capabilities Index (ECI) published by Epoch AI.

These are simply curves fitted to 14 points corresponding to Anthropic frontier models. There is no statistical
testing of whether there is a "kink" at any particular point. Intervals reflect uncertainty around each model's
score, but assume the exact dates of model release and of the kink, and that there are two linear segments. So this
is not any kind of test or estimate of METR's figure of 1.9, which is based on Anthropic's internal data. Nor is there
any suggestion as to what may have led to this.

## Method

- **Scores.** Epoch's benchmark snapshot (`data/README.md`) is fitted with Epoch's `eci-public` code at commit
  `e04929ab`, giving three score sources. `epoch_published` is Epoch's published scores. `least_squares` is Epoch's
  prior method, refit here. `bayes_refit` is Epoch's current Bayesian method, refit here.
- **Release dates.** The latest date among a model group's rows in `model_metadata.csv`. Models flagged as unreleased
  are excluded.
- **Frontier.** Release days whose best score beats every earlier day, built on full history. Window: 2024-03-01 to
  2026-10-31.
- **Fit.** y = a + b·t + c·max(0, t − τ), with t in years and τ = 2025-10-07 unless stated. The ratio is (b + c)/b,
  reported only with at least four distinct release dates on each side of the break.
- **Uncertainty.** The frontier is rebuilt and refit in each of 4,000 posterior draws (`bayes_refit`) and each of
  Epoch's 500 bootstrap resamples (`least_squares`).
- **FRI fast year.** The all-developer frontier over complete 12-calendar-month windows ending by 2026-11-03.

## Reproduce

Python 3.12, from the repository root. Create and activate a virtual environment, then download Epoch's data
snapshot into `data/` (the Wayback Machine capture described in `data/README.md`; every script checks its SHA-256):

```
python -m venv .venv
.venv\Scripts\activate        # Windows; on macOS/Linux: source .venv/bin/activate
python -c "import urllib.request as u; u.urlretrieve('https://web.archive.org/web/20261113150458id_/https://epoch.ai/data/benchmark_data.zip', 'data/benchmark_data_2026-11-13_wayback-20261113150458.zip')"
pip install -r requirements.lock.txt
python -m pytest -q
python src/analysis.py
python src/robustness.py
python src/developer_comparison.py
python src/slope_vs_step_window_end.py
python src/bradshaw_specs.py
python src/figures.py
python src/figure4_stats.py
python src/build_results.py
```

This writes `results/`, `figures/` and `RESULTS.md` (every table, in the note's order). The score draws behind the
intervals ship in `results/bayes_refit/` and `results/ls_bootstrap/`. `python src/fit_bayesian.py --zip
data/benchmark_data_2026-11-13_wayback-20261113150458.zip --out results/bayes_refit` refits the Bayesian model (about 5 minutes on 4
cores). MCMC on another machine will not reproduce the draws exactly, so the shipped draws are the ones behind the
reported numbers.

Code was written with AI coding assistants under my direction and review.

Code: MIT license. Data: Epoch AI, CC BY 4.0 (see `data/README.md`).
