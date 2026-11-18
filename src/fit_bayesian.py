"""Fit Epoch's Bayesian ECI on the frozen input with fit_bayesian_eci defaults.
Only the backend differs: PyTensor in numba mode. --cores 1 runs chains sequentially.
Usage: python src/fit_bayesian.py --zip data/benchmark_data_2026-11-13_wayback-20261113150458.zip --out results/bayes_refit"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "mode=NUMBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor
    import arviz as az
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from accel.inputs import verify_input, ECI_COMMIT
    import eci
    from eci import bayesian, prepare_benchmark_data, fit_bayesian_eci

    digest = verify_input(args.zip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = prepare_benchmark_data(args.zip)
    print(f"mode={pytensor.config.mode} models={df.model.nunique()} benchmarks={df.benchmark.nunique()} rows={len(df)}", flush=True)

    if args.smoke:
        model = bayesian.build_bayesian_model(df)
        t = time.time()
        with model:
            pm.sample(draws=20, tune=20, chains=2, cores=2, progressbar=False,
                      compute_convergence_checks=False, random_seed=1)
        print(f"SMOKE OK parallel 2 chains in {time.time() - t:.1f}s", flush=True)
        return

    if args.cores is not None:
        # Epoch's _sample passes cores=chains; override it without touching the model or sampler settings.
        orig = pm.sample
        def sample_with_cores(*a, **k):
            k["cores"] = args.cores
            return orig(*a, **k)
        bayesian.pm.sample = sample_with_cores

    t = time.time()
    eci_df, edi_df, draws = fit_bayesian_eci(df, progressbar=False)
    elapsed = time.time() - t
    trace = draws["trace"]
    rhat = az.rhat(trace, var_names=list(bayesian._DIAGNOSTIC_VARS))
    ess = az.ess(trace, var_names=list(bayesian._DIAGNOSTIC_VARS))
    diag = {
        "max_rhat": max(float(np.nanmax(v.to_numpy())) for v in rhat.data_vars.values()),
        "min_bulk_ess": min(float(np.nanmin(v.to_numpy())) for v in ess.data_vars.values()),
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "started_from": draws["started_from"],
        "attempts": draws["attempts"],
        "elapsed_seconds": round(elapsed, 1),
    }
    eci_df.to_csv(out / "eci_scores_refit.csv", index=False)
    edi_df.to_csv(out / "edi_scores_refit.csv", index=False)
    np.savez_compressed(out / "eci_draws.npz", eci=draws["eci"], model_names=np.array(draws["model_names"]))
    meta = {
        "input_sha256": digest, "eci_commit": ECI_COMMIT, "eci_version": eci.__version__,
        "backend": f"PyTensor mode={pytensor.config.mode} (numba), no C++ compiler",
        "cores": args.cores if args.cores is not None else "chains (Epoch default)",
        "settings": "fit_bayesian_eci defaults: draws=None (ladder), tune=None, chains=4, target_accept=0.95, "
                    "dense_mass_matrix=True, random_seed=12345, performance_clip_eps=1e-3, ci_level=0.90, "
                    "anchors Claude 3.5 Sonnet=130, GPT-5=150",
        "diagnostics": diag,
    }
    (out / "fit_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("FIT DONE", json.dumps(diag, default=str), flush=True)


if __name__ == "__main__":   # required on Windows: worker processes re-import this module
    main()
