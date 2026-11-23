#!/usr/bin/env python3
"""Run the five-seed smoke suite and keep atomic per-seed checkpoints."""

import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from safe_dist_policy.certification import RiskSpec, certify_localized
from safe_dist_policy.dgps import null_recovery, smoke_split
from safe_dist_policy.eval import run_benchmark
from safe_dist_policy.learners import ZICBCFBridge


def publish_json(path: Path, payload):
    with NamedTemporaryFile(mode="w", prefix=".phase3-", dir=path.parent, delete=False) as stream:
        scratch = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.link(scratch, path)  # create atomically without replacing prior results
    finally:
        scratch.unlink()


def capture_seed(seed: int, destination: Path):
    start = perf_counter()
    split = smoke_split(seed)
    results = run_benchmark(split, seed)
    zic = None
    if seed == 0:
        start_r = perf_counter()
        bridge = ZICBCFBridge(.90, seed, burn=10, draws=12, trees=12).fit(split.train)
        pred = bridge.predict(split.evaluation.x[:25])
        zic = {
            "evaluation_rows": len(pred.score),
            "finite_predictions": bool(np.isfinite(pred.score).all()),
            "train_rows": split.train.n,
            "seconds": perf_counter() - start_r,
            "burn": 10, "draws": 12, "trees": 12,
        }
    rows = []
    for entry in results:
        values = asdict(entry)
        values["threshold"] = float(values["threshold"]) if np.isfinite(values["threshold"]) else (
            "baseline" if values["threshold"] > 0 else "all_candidates"
        )
        evaluation = values.pop("evaluation")
        rows.append({"seed": seed, **values, **evaluation})
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    publish_json(destination, {
        "seed": seed, "rows": rows, "resources": {
            "wall_seconds": perf_counter() - start,
            "peak_self_mib": own / 1024,
            "peak_r_child_mib": child / 1024,
            "upper_bound_process_peak_mib": (own + child) / 1024,
        },
        "zicbcf": zic,
    })


def null_panel():
    rows = []
    for strict in (False, True):
        for seed in range(500):
            cal, family = null_recovery(seed, unsafe=strict)
            cert = certify_localized(cal, family, RiskSpec(), 0.5)
            rows.append({
                "seed": seed, "strict_unsafe": strict, "positive_candidate_certified": bool(cert.passed),
                "unsafe_prefix": bool(strict and cert.passed), "tested": cert.tested,
            })
    return rows


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_rows(path: Path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/phase3_smoke"))
    parser.add_argument("--seed", type=int, choices=range(5))
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(exist_ok=True, parents=True)
    checkpoint = out / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    if args.seed is not None:
        capture_seed(args.seed, checkpoint / f"seed_{args.seed}.json")
        return

    for name in ("smoke_summary.csv", "null_checks.csv", "resource_profile.json"):
        if (out / name).exists():
            parser.error(f"Output already exists: {out / name}")
    records = []
    for seed in range(5):
        path = checkpoint / f"seed_{seed}.json"
        if not path.exists():
            subprocess.run(
                [sys.executable, __file__, "--out-dir", str(out), "--seed", str(seed)], check=True,
            )
        records.append(json.loads(path.read_text()))
        print(f"Smoke seed {seed} complete", flush=True)
    nulls = null_panel()
    write_rows(out / "null_checks.csv", nulls)
    all_rows = [row for record in records for row in record["rows"]]
    write_rows(out / "smoke_summary.csv", all_rows)
    peak = None
    scope_line = Path("/proc/self/cgroup").read_text().splitlines()[0]
    scope = Path("/sys/fs/cgroup") / scope_line.split(":", 2)[2].lstrip("/")
    if (scope / "memory.peak").exists():
        peak = int((scope / "memory.peak").read_text())
    publish_json(out / "resource_profile.json", {
        "seeds": [{"seed": record["seed"], **record["resources"]} for record in records],
        "zicbcf_projection_smoke": records[0]["zicbcf"],
        "cgroup_memory_peak_bytes": peak,
        "strict_near_null_fwer": sum(x["unsafe_prefix"] for x in nulls) / 500,
        "equality_positive_certifications": sum(
            x["positive_candidate_certified"] for x in nulls if not x["strict_unsafe"]
        ),
        "sha256": {
            str(path.relative_to(out)): checksum(path)
            for path in sorted((*checkpoint.glob("seed_*.json"), out / "smoke_summary.csv", out / "null_checks.csv"))
        },
    })


if __name__ == "__main__":
    main()
