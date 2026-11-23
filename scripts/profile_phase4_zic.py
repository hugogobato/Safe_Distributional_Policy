"""Profile a nonconfirmatory ZIC-BCF fit at the production tree count."""

import argparse
from pathlib import Path
from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage
from time import perf_counter

import numpy as np

from safe_dist_policy.learners import ZICBCFBridge
from safe_dist_policy.phase4_design import DESIGNS, simulated_data
from safe_dist_policy.phase4_run import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path to preserve existing timing results.")
    burn, draws = (500, 1000) if args.full else (50, 100)
    design = DESIGNS["d05-main"]
    sample, _ = simulated_data(design, 999, n_eval=1000)
    estimator = ZICBCFBridge(design.beta, 999, burn=burn, draws=draws, trees=200)
    start = perf_counter()
    estimator.fit(sample.train)
    fit_time = perf_counter() - start
    start = perf_counter()
    scores = estimator.predict(sample.evaluation.x)
    predict_time = perf_counter() - start
    result = {
        "study": "nonconfirmatory production-tree timing", "design": design.name, "seed": 999,
        "training_n": sample.train.n, "test_n": len(scores.score), "burn": burn, "draws": draws,
        "prognostic_trees": 200, "moderating_trees": 50, "fit_seconds": fit_time,
        "projection_prediction_seconds": predict_time,
        "peak_python_rss_mib": getrusage(RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_r_rss_mib": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024.0,
        "finite_heldout_scores": bool(np.isfinite(scores.score).all()),
    }
    atomic_json(args.output, result)
    print(result, flush=True)
    return 0 if result["finite_heldout_scores"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
