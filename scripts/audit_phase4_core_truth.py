"""Verify which legacy truth shortcuts affect the frozen core checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from safe_dist_policy.phase4_design import (
    CORE,
    baseline_quantile,
    covariates,
    population,
)
from safe_dist_policy.phase4_run import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4_sim"))
    args = parser.parse_args()
    checked = 0
    empty = []
    for d in CORE:
        cutoff = baseline_quantile(d, d.beta)
        is_equal = d.dgp == 3 and d.label == "equality"
        is_harm = d.dgp == 3 and d.label == "strict"
        expected_pos = np.array([is_equal or is_harm, is_equal or is_harm or d.dgp == 4])
        expected_neg = np.array([is_equal, is_equal or d.dgp == 4])
        for seed in d.seeds:
            path = args.output_dir / "checkpoints" / "core" / d.name / f"seed_{seed}.json"
            saved = json.loads(path.read_text())
            if saved["status"] != "ok":
                raise ValueError(f"Core output failed for {d.name} {seed}.")
            rng = np.random.default_rng(np.random.SeedSequence([d.dgp, seed]).spawn(3)[2])
            x_eval = covariates(d, 100_000, rng)
            arms = population(d, x_eval)
            tail = arms.survival(cutoff)
            score = np.column_stack((tail[:, 1] - tail[:, 0], arms.positive[:, 1] - arms.positive[:, 0]))
            assert np.array_equal(np.all(score >= 0, axis=0), expected_pos), (d.name, seed, "positive")
            assert np.array_equal(np.all(score <= 0, axis=0), expected_neg), (d.name, seed, "negative")
            for row in saved["rows"]:
                if row["policy_selected"] and row["deployment"] == 0:
                    empty.append({
                        "design": d.name, "seed": seed, "method": row["method"], "guarantee": row["guarantee"],
                    })
            checked += 1
    certified_empty = [record for record in empty if record["guarantee"] in {"joint", "indicator"}]
    result = {
        "core_population_draws_checked": checked,
        "global_sign_and_equality_knowledge_agreed_for_every_draw": True,
        "selected_zero_switch_nonguaranteed_rows": [record for record in empty if record not in certified_empty],
        "selected_zero_switch_certificates": certified_empty,
        "primary_certificate_truth_unaffected": not certified_empty,
    }
    atomic_json(args.output_dir / "core_truth_revision_audit.json", result)
    print(f"{checked} original evaluation populations checked; {len(certified_empty)} certified zero-observed switches.")
    return 1 if certified_empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
