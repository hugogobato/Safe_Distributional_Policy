"""Checkpointed spawn-parallel runner for Phase 4 Protocol v2 stress blocks (Blocks A-F)."""

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
from safe_dist_policy.phase4_v2_stress import run_stress_task, stress_block_tasks


def _worker(args: tuple[str, str, int, str]) -> tuple[str, str, int, str]:
    for name in THREADS:
        os.environ[name] = "1"
    block, design_name, seed, out_dir = args
    cp = Path(out_dir) / "checkpoints" / block / design_name / f"seed_{seed}.json"
    if cp.exists():
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if rec.get("status") == "ok":
            return block, design_name, seed, "cached"
    rows, profile = run_stress_task(block, design_name, seed)
    atomic_json(cp, {"status": "ok", "block": block, "design": design_name, "seed": seed, "rows": rows, "profile": profile})
    return block, design_name, seed, "ok"


def summarize(output_dir: Path, tasks: list[tuple[str, str, int]], tag: str) -> None:
    all_rows: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for block, d, s in tasks:
        cp = output_dir / "checkpoints" / block / d / f"seed_{s}.json"
        if not cp.exists():
            continue
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if rec.get("status") == "ok":
            all_rows.extend(rec["rows"])
            profiles.append(rec["profile"])

    if not all_rows:
        return
    with (output_dir / f"{tag}_all_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    if profiles:
        with (output_dir / f"{tag}_profiles.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(profiles[0].keys()))
            w.writeheader()
            w.writerows(profiles)

    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in all_rows:
        grouped[(str(r["block"]), str(r["design"]), str(r["target"]), str(r["method"]))].append(r)

    summary = []
    for (b, d, t, m), recs in sorted(grouped.items()):
        summary.append({
            "block": b,
            "design": d,
            "target": t,
            "method": m,
            "n_seeds": len(recs),
            "deployment": float(np.mean([float(r["deployment"]) for r in recs])),
            "safe_welfare": float(np.mean([float(r["safe_welfare"]) for r in recs])),
            "welfare_gain": float(np.mean([float(r["welfare_gain"]) for r in recs])),
            "prefix_fwer": float(np.mean([bool(r["prefix_unsafe"]) for r in recs])),
            "pass_rate": float(np.mean([bool(r["deployed"]) for r in recs])),
            "mean_cert_seconds": float(np.mean([float(r["cert_seconds"]) for r in recs])),
        })
    with (output_dir / f"{tag}_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    atomic_json(output_dir / f"{tag}_summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        choices=("core_1_4", "dgp6_crossover", "dgp7_misspec", "dgp8_pareto", "dgp10_scaling", "zic_sensitivity"),
    )
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4_v2_stress"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    for name in THREADS:
        os.environ[name] = "1"
    if args.tasks_file:
        payload = json.loads(args.tasks_file.read_text(encoding="utf-8"))
        tasks = [(str(t["block"]), str(t["design"]), int(t["seed"])) for t in payload["tasks"]]
        tag = args.tag or args.tasks_file.stem
    elif args.block:
        tasks = stress_block_tasks(args.block)
        tag = args.tag or args.block
    else:
        parser.error("Provide --block or --tasks-file.")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    print(f"[{tag}] Running {len(tasks)} tasks with {args.workers} spawn workers -> {out}", flush=True)

    done = 0
    if args.workers == 1:
        for b, d, s in tasks:
            _, _, _, st = _worker((b, d, s, str(out)))
            done += 1
            if done == 1 or done % 10 == 0 or done == len(tasks):
                print(f"[{tag}] {done}/{len(tasks)} ({d} seed={s} {st}) in {perf_counter() - started:.1f}s", flush=True)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futs = [pool.submit(_worker, (b, d, s, str(out))) for b, d, s in tasks]
            for fut in as_completed(futs):
                _, d, s, st = fut.result()
                done += 1
                if done == 1 or done % 25 == 0 or done == len(tasks):
                    print(f"[{tag}] {done}/{len(tasks)} ({d} seed={s} {st}) in {perf_counter() - started:.1f}s", flush=True)

    summarize(out, tasks, tag)
    print(f"[{tag}] Finished {len(tasks)} tasks in {perf_counter() - started:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
