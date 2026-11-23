"""Report the prespecified focal Gate G3 comparisons using paired seeds."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from merge_phase4 import mean_ci

from safe_dist_policy.phase4_run import atomic_json

METHODS = ("LCT-LTT", "B1-survival", "B1-opra", "B2", "B3", "B4", "B5", "B5+ext")
FOCAL = ("d05-main", "d06-n5000-s025-q90", "d09-meps-proxy", "d09-ohie-proxy")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/phase4_sim"))
    args = parser.parse_args()
    focal = args.results / "focal"
    paths = args.results / "core_dgp1_5_r50.csv", focal / "stress_dgp6_10.csv"
    by_design = defaultdict(dict)
    for path in paths:
        for row in read_rows(path):
            d, m, seed = row["design"], row["method"], int(row["seed"])
            if d in FOCAL and m in METHODS:
                key = d, m
                if seed in by_design[key]:
                    raise ValueError(f"Duplicate focal result {(d, m, seed)}.")
                by_design[key][seed] = row

    results = {}
    for design in FOCAL:
        methods = {m: by_design[(design, m)] for m in METHODS}
        assert all(set(records) == set(range(1000, 1050)) for records in methods.values()), design
        contender = methods["LCT-LTT"]

        def compare(baseline, field, factor=1.0):
            return mean_ci([float(contender[s][field]) - factor * float(baseline[s][field]) for s in sorted(contender)])

        def metric(r, field):
            return sum(float(x[field]) for x in r.values()) / len(r)

        b1_names = ("B1-survival", "B1-opra")
        best_dep = max(b1_names, key=lambda m: metric(methods[m], "deployment"))
        best_welfare = max(b1_names, key=lambda m: metric(methods[m], "safe_welfare"))
        summary = {}
        for method, rows in methods.items():
            summary[method] = {
                "deployment": metric(rows, "deployment"),
                "safe_welfare": metric(rows, "safe_welfare"),
                "safe_welfare_lower": metric(rows, "safe_welfare_lower"),
                "unsafe_definite_count": sum(r["prefix_definite"] == "True" for r in rows.values()),
                "unsafe_central_count": sum(r["prefix_unsafe"] == "True" for r in rows.values()),
                "unsafe_possible_count": sum(r["prefix_possible"] == "True" for r in rows.values()),
                "selected_quantile_unsafe_count": sum(r["q_unsafe"] == "True" for r in rows.values()),
                "selected_extensive_unsafe_count": sum(r["ext_unsafe"] == "True" for r in rows.values()),
                "selected_policy_count": sum(float(r["deployment"]) > 0 for r in rows.values()),
            }
        results[design] = {
            "summary": summary,
            "best_b1_for_deployment": best_dep,
            "best_b1_for_welfare": best_welfare,
            "deployment_difference_from_best_b1_with_95ci": compare(methods[best_dep], "deployment"),
            "safe_welfare_above_25pct_gain_over_b1_with_95ci": (
                compare(methods[best_welfare], "safe_welfare", 1.25)
                if metric(methods[best_welfare], "safe_welfare") > 0 else None
            ),
            "safe_welfare_difference_from_best_b1_with_95ci": compare(methods[best_welfare], "safe_welfare"),
            "safe_welfare_above_15pct_gain_over_b2_with_95ci": compare(methods["B2"], "safe_welfare", 1.15),
            "deployment_difference_from_matched_b5_with_95ci": compare(methods["B5+ext"], "deployment"),
            "safe_welfare_difference_from_matched_b5_with_95ci": compare(methods["B5+ext"], "safe_welfare"),
            "safe_welfare_above_15pct_gain_over_matched_b5_with_95ci": compare(methods["B5+ext"], "safe_welfare", 1.15),
            "paired_deployment_counts_lct_higher_equal_lower": [
                sum(op(float(contender[s]["deployment"]), float(methods["B5+ext"][s]["deployment"])) for s in contender)
                for op in (lambda a, b: a > b, lambda a, b: a == b, lambda a, b: a < b)
            ],
        }

    by_profile = defaultdict(list)
    for row in read_rows(focal / "resource_profiles.csv"):
        by_profile[row["design"]].append(row)
    for design, rows in by_profile.items():
        results[design]["baseline_cutoff_inclusion"] = {
            "inside": sum(r["b5_cutoff_in_lct_interval"] == "True" for r in rows),
            "inside_and_matched_b5_prefix_covers_lct": sum(
                r["b5_cutoff_in_lct_interval"] == "True" and r["b5_passes_all_localized_prefix"] == "True" for r in rows
            ),
            "matched_b5_prefix_covers_lct": sum(r["b5_passes_all_localized_prefix"] == "True" for r in rows),
        }

    atomic_json(focal / "gate_contrasts.json", results)
    print(f"Wrote {len(results)} focal design summaries and paired intervals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
