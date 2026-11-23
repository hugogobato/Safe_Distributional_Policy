"""Parallel, checkpointed confirmatory runner for Phase 4 Protocol v2 (Gate G3 v2)."""

import argparse
import csv
import json
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from safe_dist_policy.phase4_run import THREADS, atomic_json
from safe_dist_policy.phase4_v2 import V2_DESIGNS, run_v2_replication


def _worker(args: tuple[str, int, str]) -> tuple[str, int, str]:
    for name in THREADS:
        os.environ[name] = "1"
    design_name, seed, out_dir = args
    cp = Path(out_dir) / "checkpoints" / design_name / f"seed_{seed}.json"
    if cp.exists():
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if rec.get("status") == "ok":
            return design_name, seed, "cached"
    rows = run_v2_replication(design_name, seed)
    atomic_json(cp, {"status": "ok", "design": design_name, "seed": seed, "rows": rows})
    return design_name, seed, "ok"


def summarize(output_dir: Path, designs: list[str], seeds: range) -> dict[str, object]:
    all_rows: list[dict[str, object]] = []
    for d in designs:
        for s in seeds:
            cp = output_dir / "checkpoints" / d / f"seed_{s}.json"
            rec = json.loads(cp.read_text(encoding="utf-8"))
            all_rows.extend(rec["rows"])

    csv_path = output_dir / "v2_all_rows.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in all_rows:
        grouped[(str(r["design"]), str(r["target"]), str(r["method"]))].append(r)

    summary_rows = []
    summary_json: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (d, t, m), recs in sorted(grouped.items()):
        n = len(recs)
        dep = float(np.mean([float(r["deployment"]) for r in recs]))
        sw = float(np.mean([float(r["safe_welfare"]) for r in recs]))
        wg = float(np.mean([float(r["welfare_gain"]) for r in recs]))
        fwer = float(np.mean([bool(r["prefix_unsafe"]) for r in recs]))
        sel_u = float(np.mean([bool(r["selected_unsafe"]) for r in recs]))
        pass_rate = float(np.mean([bool(r["deployed"]) for r in recs]))
        entry = {
            "design": d, "target": t, "method": m, "n_seeds": n,
            "deployment": dep, "safe_welfare": sw, "welfare_gain": wg,
            "prefix_fwer": fwer, "selected_unsafe_rate": sel_u, "pass_rate": pass_rate,
        }
        summary_rows.append(entry)
        summary_json.setdefault(d, {}).setdefault(t, {})[m] = {
            "n_seeds": n, "deployment": dep, "safe_welfare": sw,
            "welfare_gain": wg, "prefix_fwer": fwer, "selected_unsafe_rate": sel_u, "pass_rate": pass_rate,
        }

    sum_csv = output_dir / "v2_summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    atomic_json(output_dir / "v2_summary.json", summary_json)
    return summary_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4_v2"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--design", action="append", choices=sorted(V2_DESIGNS))
    args = parser.parse_args()

    for name in THREADS:
        os.environ[name] = "1"
    designs = args.design or list(V2_DESIGNS.keys())
    seeds = range(1000, 1000 + args.n_seeds)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    tasks = [(d, s, str(out)) for d in designs for s in seeds]
    started = perf_counter()
    print(f"Running {len(tasks)} Phase 4 v2 tasks across {len(designs)} designs with {args.workers} workers...", flush=True)

    done = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            d, s, st = fut.result()
            done += 1
            if done == 1 or done % 25 == 0 or done == len(tasks):
                print(f"Completed {done}/{len(tasks)} ({d} seed={s} {st}) in {perf_counter() - started:.1f}s", flush=True)

    summary = summarize(out, designs, seeds)
    print("\n=== Phase 4 Protocol v2 Confirmatory Summary ===")
    for d in designs:
        for t, methods in summary[d].items():
            print(f"\n[{d} | {t}]")
            print(f"  {'Method':18s} {'Deploy':>8s} {'SafeWelf':>9s} {'FWER':>7s} {'Pass':>7s}")
            for m, vals in methods.items():
                print(
                    f"  {m:18s} {100*vals['deployment']:7.2f}% {vals['safe_welfare']:9.4f} "
                    f"{100*vals['prefix_fwer']:6.1f}% {100*vals['pass_rate']:6.1f}%"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
