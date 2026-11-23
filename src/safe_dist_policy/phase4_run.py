"""Checkpointed local and notebook execution for the frozen Phase 4 suite."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from .phase4_design import CORE, DESIGNS, STRESS, ZIC_DESIGNS

PROTOCOL_SHA256 = "ae7775408c386b0d125215202811a1b019d55571ae18549b02adb925d501874b"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class Task:
    study: str
    design: str
    seed: int

    @property
    def key(self):
        return f"{self.study}/{self.design}/seed_{self.seed}"


def task_grid(study):
    core = [Task("core", d.name, s) for d in CORE for s in d.seeds] if study in ("core", "all") else []
    stress = (
        [Task("stress", d.name, s) for d in STRESS for s in d.seeds]
        if study in ("stress", "remote", "all") else []
    )
    zic = (
        [Task("zic", name, s) for name in ZIC_DESIGNS for s in range(1000, 1020)]
        if study in ("zic", "remote", "all") else []
    )
    return core + stress + zic


def read_checkpoint(path: Path, task: Task, *, strict: bool = True):
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("task") != task.__dict__ or record.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError(f"Incompatible checkpoint {path}.")
    expected_source = os.environ.get("SAFE_DIST_POLICY_SOURCE_SHA256")
    if expected_source is not None and record.get("source_sha256") != expected_source:
        raise ValueError(f"Unexpected source release in {path}.")
    if strict and record["status"] != "ok":
        raise ValueError(f"Failed checkpoint {path}.")
    return record


def atomic_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(content, handle, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def checkpoint(output: Path, task: Task):
    return output / "checkpoints" / f"{task.key}.json"


def _one(task: Task, output: Path, reduced_zic: bool):
    from .phase4_eval import run_replication

    path = checkpoint(output, task)
    start = perf_counter()
    record: dict[str, object] = {
        "task": task.__dict__, "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": os.environ.get("SAFE_DIST_POLICY_SOURCE_SHA256"),
    }
    try:
        rows, profile = run_replication(DESIGNS[task.design], task.seed, task.study, reduced_zic=reduced_zic)
        record.update(status="ok", rows=rows, profile=profile, reduced_zic=reduced_zic)
        atomic_json(path, record)
        return 0
    except Exception:
        record.update(status="failed", error=traceback.format_exc(), seconds=perf_counter() - start)
        atomic_json(path, record)
        return 1


def merge_checkpoint_digest(output: Path, tasks, path: Path):
    hashes = {}
    for task in tasks:
        cp = checkpoint(output, task)
        if cp.exists():
            hashes[cp.relative_to(output).as_posix()] = hashlib.sha256(cp.read_bytes()).hexdigest()
    atomic_json(path, {"protocol_sha256": PROTOCOL_SHA256, "sha256": hashes})


def run_tasks(tasks, output: Path, workers: int, *, retry_failed: bool = False, reduced_zic: bool = False, digest_name="batch_checksums.json"):
    remaining = []
    failed = []
    for task in tasks:
        cp = checkpoint(output, task)
        record = read_checkpoint(cp, task, strict=False)
        if record and record["status"] == "ok":
            if record.get("reduced_zic", False) != reduced_zic:
                raise ValueError(f"Refusing to replace a checkpoint with different fit settings: {cp}")
            continue
        if record:
            if not retry_failed:
                failed.append(task.key)
                continue
            backup = cp.with_name(f"{cp.stem}.failed-{hashlib.sha256(cp.read_bytes()).hexdigest()[:16]}.json")
            if not backup.exists():
                cp.rename(backup)
        remaining.append(task)

    environment = dict(os.environ)
    environment.update({name: "1" for name in THREADS})
    started = perf_counter()

    def launch(task):
        args = [sys.executable, "-m", "safe_dist_policy.phase4_run", "--output-dir", str(output), "--one", json.dumps(task.__dict__)]
        if reduced_zic:
            args.append("--reduced-zic")
        result = subprocess.run(args, env=environment, check=False)
        cp = checkpoint(output, task)
        if not cp.exists():
            atomic_json(cp, {
                "task": task.__dict__, "protocol_sha256": PROTOCOL_SHA256,
                "source_sha256": environment.get("SAFE_DIST_POLICY_SOURCE_SHA256"),
                "status": "failed", "error": f"Isolated worker exited {result.returncode} without a checkpoint.",
            })
        return task, result.returncode

    print(f"Starting {len(remaining)} of {len(tasks)} tasks with {workers} isolated workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(launch, task) for task in remaining]
        for i, future in enumerate(as_completed(futures), 1):
            task, code = future.result()
            if code:
                failed.append(task.key)
                print(f"FAILED {task.key} (status {code})", flush=True)
            if i == 1 or i % 10 == 0 or i == len(futures):
                print(f"Completed {i} / {len(futures)} in {perf_counter() - started:.1f}s", flush=True)

    merge_checkpoint_digest(output, tasks, output / digest_name)
    print(f"{len(failed)} failed tasks", flush=True)
    return 1 if failed else 0


def main(default_study=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("core", "stress", "zic", "remote", "all"), default=default_study or "core")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4_sim"))
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--design", action="append", choices=sorted(DESIGNS))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reduced-zic", action="store_true")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.one:
        return _one(Task(**json.loads(args.one)), output, args.reduced_zic)
    if args.tasks_file:
        manifest = json.loads(args.tasks_file.read_text(encoding="utf-8"))
        if manifest["protocol_sha256"] != PROTOCOL_SHA256:
            parser.error("The assigned shard uses a different protocol.")
        tasks = [Task(**record) for record in manifest["tasks"]]
    else:
        tasks = task_grid(args.study)
        if args.design:
            tasks = [t for t in tasks if t.design in args.design]
        if args.pilot:
            tasks = list({(t.study, t.design): Task(t.study, t.design, 999) for t in tasks}.values())
    if not tasks or len(tasks) != len(set(tasks)):
        parser.error("Provide at least one task and no duplicate tasks.")
    if args.reduced_zic and not args.pilot and not args.tasks_file:
        parser.error("The reduced ZIC fit is only for bridge verification.")
    if args.pilot and args.output_dir == Path("results/phase4_sim"):
        parser.error("Keep pilot checkpoints under a separate output directory.")
    if any(DESIGNS[t.design].n_cal >= 100_000 for t in tasks) and args.workers > 1:
        parser.error("Use a single isolated worker for the 100,000-row scaling cell.")
    return run_tasks(
        tasks, output, args.workers, retry_failed=args.retry_failed, reduced_zic=args.reduced_zic,
        digest_name=f"{args.tasks_file.stem}_checksums.json" if args.tasks_file else f"{args.study}_checksums.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
