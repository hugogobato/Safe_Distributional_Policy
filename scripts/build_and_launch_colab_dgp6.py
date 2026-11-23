"""Build 12 ultra-lightweight (~2.5 KB) GitHub-backed Colab notebooks for DGP 6 crossover
(2,400 tasks) and launch them strictly on Google Chrome authuser=6..9 (3 slots each).
"""

import argparse
import asyncio
import importlib.util
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat as nb

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLAB_FLEET_PATH = pathlib.Path(
    "/home/hugo/.local/share/agy-profiles/gemini/home/.gemini/config/skills/colab-fleet-runner/scripts/colab_fleet.py"
)
ALLOWED_ACCOUNTS = (6, 7, 8, 9)
REPO_URL = "https://github.com/hugogobato/Safe_Distributional_Policy.git"


def _load_colab_fleet():
    spec = importlib.util.spec_from_file_location("colab_fleet", COLAB_FLEET_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _balanced_dgp6_shards(n_shards: int = 12) -> list[list[dict[str, object]]]:
    sys.path.insert(0, str(ROOT / "src"))
    from safe_dist_policy.phase4_design import DESIGNS
    from safe_dist_policy.phase4_v2_stress import stress_block_tasks

    tasks = stress_block_tasks("dgp6_crossover")
    bins: list[list[dict[str, object]]] = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards

    def weight(item: tuple[str, str, int]) -> float:
        d = DESIGNS[item[1]]
        return 0.25 + (d.n_cal / 5000.0) ** 0.85

    for b, d, s in sorted(tasks, key=lambda x: (-weight(x), x[1], x[2])):
        idx = min(range(n_shards), key=lambda i: (loads[i], i))
        bins[idx].append({"block": b, "design": d, "seed": s})
        loads[idx] += weight((b, d, s))
    return bins


def write_shard_configs(config_dir: pathlib.Path, n_shards: int = 12) -> list[str]:
    config_dir.mkdir(parents=True, exist_ok=True)
    shards = _balanced_dgp6_shards(n_shards)
    names: list[str] = []
    for idx, tasks in enumerate(shards, 1):
        shard_name = f"dgp6_shard_{idx:02d}"
        names.append(shard_name)
        (config_dir / f"{shard_name}.json").write_text(
            json.dumps({"shard": shard_name, "tasks": tasks}, indent=2) + "\n",
            encoding="utf-8",
        )
    return names


RUN_CELL = '''import os, shutil, subprocess, sys, traceback, zipfile
from pathlib import Path

workspace = Path("phase4_v2_runs").resolve() / shard_name
source_dir = workspace / "repo"
run_dir = workspace / "results"
workspace.mkdir(parents=True, exist_ok=True)
run_dir.mkdir(parents=True, exist_ok=True)
setup_log = run_dir / "setup.log"
run_log = run_dir / "worker.log"
run_error = None

def run_quiet(cmd, *, cwd=workspace, env=None, log_path=setup_log):
    """Redirect verbose setup output to disk to keep Colab browser DOM & RAM tiny."""
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\\n$ " + " ".join(map(str, cmd)) + "\\n")
        res = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        print(f"Command failed ({res.returncode}): {cmd}\\n--- log tail ---\\n{tail}", flush=True)
        raise subprocess.CalledProcessError(res.returncode, cmd)

def run_milestone(cmd, *, cwd=workspace, env=None):
    """Stream only concise milestone lines to stdout; full log goes to disk."""
    with run_log.open("a", encoding="utf-8") as fh:
        with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                if line.startswith("[") or "Error" in line or "Traceback" in line:
                    print(line, end="", flush=True)
            code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)

R_BOOTSTRAP = r"""
options(timeout = 600L)
if (!requireNamespace("grf", quietly = TRUE)) {
  codename <- tryCatch(trimws(system("lsb_release -cs", intern = TRUE)), error = function(e) "")
  repos <- c()
  if (nzchar(codename)) {
    repos <- c(repos, sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", codename))
  }
  repos <- c(repos, "https://cloud.r-project.org")
  ok <- FALSE
  for (r in repos) {
    tryCatch({
      install.packages("grf", repos = r, Ncpus = 2L, quiet = TRUE)
      if (requireNamespace("grf", quietly = TRUE)) { ok <- TRUE; break }
    }, error = function(e) NULL)
  }
  if (!ok) stop("Failed to install R package grf.")
}
cat("grf", as.character(packageVersion("grf")), "\\n")
"""

try:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"

    print(f"[{shard_name}] Cloning repo @ {commit_sha[:10]}...", flush=True)
    if source_dir.exists():
        shutil.rmtree(source_dir)
    run_quiet(["git", "clone", "--depth", "1", repo_url, str(source_dir)], env=env)
    run_quiet(["git", "fetch", "--depth", "1", "origin", commit_sha], cwd=source_dir, env=env)
    run_quiet(["git", "checkout", commit_sha], cwd=source_dir, env=env)

    if not shutil.which("Rscript"):
        print(f"[{shard_name}] Installing r-base...", flush=True)
        run_quiet(["apt-get", "update", "-qq"], env=env)
        run_quiet(["apt-get", "install", "-y", "-qq", "r-base", "r-base-dev", "libopenblas-dev"], env=env)

    print(f"[{shard_name}] Ensuring R grf...", flush=True)
    r_boot = workspace / "install_grf.R"
    r_boot.write_text(R_BOOTSTRAP, encoding="utf-8")
    run_quiet(["Rscript", "--vanilla", str(r_boot)], env=env)

    env["PYTHONPATH"] = str(source_dir / "src")
    tasks_file = source_dir / "configs" / "phase4_v2_dgp6" / f"{shard_name}.json"
    print(f"[{shard_name}] Starting DGP 6 tasks with 2 spawn workers...", flush=True)
    run_milestone(
        [
            sys.executable, "-u", str(source_dir / "scripts" / "run_phase4_v2_stress.py"),
            "--tasks-file", str(tasks_file),
            "--output-dir", str(run_dir),
            "--workers", "2",
            "--tag", shard_name,
        ],
        cwd=source_dir,
        env=env,
    )
except Exception:
    run_error = traceback.format_exc()
    (run_dir / "notebook_error.txt").write_text(run_error, encoding="utf-8")
    print(run_error[-3000:], flush=True)
'''

PACK_CELL = '''output_file = str(workspace / f"{shard_name}_results.zip")
pack_error = None
try:
    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(run_dir).as_posix())
    print(f"[{shard_name}] Packed {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)", flush=True)
except Exception:
    pack_error = traceback.format_exc()
    print("PACKAGING ERROR:", pack_error, flush=True)

if pack_error is None:
    try:
        from google.colab import files
        files.download(output_file)
        print("Downloaded:", output_file, flush=True)
    except Exception as e:
        print("(Not on Colab / download skipped):", e, flush=True)

if run_error is not None:
    raise RuntimeError(f"Shard {shard_name} failed during execution. See notebook_error.txt.")
if pack_error is not None:
    raise RuntimeError(f"Shard {shard_name} failed during ZIP packaging.")
'''


def _current_commit_sha() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def build_notebooks(out_dir: pathlib.Path, n_shards: int = 12) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.ipynb"):
        old.unlink()
    commit_sha = _current_commit_sha()
    config_dir = ROOT / "configs" / "phase4_v2_dgp6"
    shard_names = [f"dgp6_shard_{i:02d}" for i in range(1, n_shards + 1)]
    for name in shard_names:
        assert (config_dir / f"{name}.json").exists(), f"Missing {config_dir / f'{name}.json'}"

    nb_paths: list[pathlib.Path] = []
    for shard_name in shard_names:
        cfg = (
            f"shard_name = {shard_name!r}\n"
            f"repo_url = {REPO_URL!r}\n"
            f"commit_sha = {commit_sha!r}\n"
        )
        book = nb.v4.new_notebook(
            cells=[
                nb.v4.new_markdown_cell(
                    f"# Phase 4 v2 DGP 6 Crossover — `{shard_name}`\n\n"
                    f"Lightweight GitHub-backed notebook (`{REPO_URL}` @ `{commit_sha[:10]}`). "
                    "Redirects setup output to disk to keep browser tab memory minimal and downloads one ZIP."
                ),
                nb.v4.new_code_cell(cfg),
                nb.v4.new_code_cell(RUN_CELL),
                nb.v4.new_code_cell(PACK_CELL),
            ],
            metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        )
        nb.validate(book)
        path = out_dir / f"{shard_name}.ipynb"
        nb.write(book, path)
        nb_paths.append(path)

    # Pre-1980 ZIP timestamp check
    chk = out_dir / "_pkg_check"
    chk.mkdir(exist_ok=True)
    f = chk / "epoch.txt"
    f.write_bytes(b"ok")
    os.utime(f, (0, 0))
    z = chk / "t.zip"
    with ZipFile(z, "w", ZIP_DEFLATED, strict_timestamps=False) as zf:
        zf.write(f, "epoch.txt")
    with ZipFile(z, "r") as zf:
        assert zf.read("epoch.txt") == b"ok"
    shutil.rmtree(chk)
    sizes = [p.stat().st_size for p in nb_paths]
    print(
        f"Built {len(nb_paths)} lightweight notebooks in {out_dir} "
        f"(each ~{sum(sizes)/len(sizes)/1024:.1f} KB, pinned to {commit_sha[:10]})",
        flush=True,
    )
    return nb_paths


async def launch_u6_to_u9(
    nb_paths: list[pathlib.Path],
    report_path: pathlib.Path,
    *,
    check_delay_sec: int = 300,
    terminate_lingering: bool = False,
) -> None:
    cf = _load_colab_fleet()
    from playwright.async_api import async_playwright

    targets = [(u, slot) for slot in (1, 2, 3) for u in ALLOWED_ACCOUNTS]
    if len(nb_paths) > len(targets):
        raise ValueError(f"Cannot launch {len(nb_paths)} notebooks across {len(targets)} slots (u/6..u/9).")

    cdp_url = cf.ensure_browser_running("chrome", resync_auth=True)
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        ctx = browser.contexts[0]
        schedule = [(u, slot, nb_path) for (u, slot), nb_path in zip(targets, nb_paths)]
        print(
            f"[COLAB u/6..u/9] Launching {len(schedule)} lightweight notebooks strictly on Chrome authuser={ALLOWED_ACCOUNTS}...",
            flush=True,
        )
        sem = asyncio.Semaphore(2)

        async def worker(idx: int, u: int, slot: int, nb_path: pathlib.Path):
            assert u in ALLOWED_ACCOUNTS, f"Safety guard: unauthorized account u/{u}"
            async with sem:
                try:
                    res = await cf.upload_and_run_single(
                        ctx, "chrome", u, slot, nb_path, terminate_lingering=terminate_lingering,
                    )
                    print(
                        f"  [{idx}/{len(schedule)}] chrome u/{u} slot {slot} ({nb_path.name}): "
                        f"{res['status']} in {res.get('elapsed_sec')}s -> {res.get('drive_url', '')}",
                        flush=True,
                    )
                    return res
                except Exception as exc:
                    print(f"  [{idx}/{len(schedule)}] chrome u/{u} slot {slot} ({nb_path.name}): LAUNCH_ERROR -> {exc}", flush=True)
                    return {
                        "browser": "chrome", "account_idx": u, "slot_idx": slot,
                        "notebook": nb_path.name, "status": "LAUNCH_ERROR", "error": str(exc), "page": None,
                    }

        tasks = []
        for idx, (u, slot, nb_path) in enumerate(schedule, 1):
            tasks.append(asyncio.create_task(worker(idx, u, slot, nb_path)))
            if idx < len(schedule):
                await asyncio.sleep(random.uniform(1.8, 2.8))
        records = list(await asyncio.gather(*tasks))

        for pg in list(ctx.pages):
            if pg.url in ("about:blank", "chrome://new-tab-page/"):
                try:
                    await pg.close()
                except Exception:
                    pass

        if check_delay_sec > 0:
            print(f"\n[WAIT] Waiting {check_delay_sec}s ({check_delay_sec/60:.1f} min) before health check...", flush=True)
            t0 = time.time()
            while True:
                rem = check_delay_sec - (time.time() - t0)
                if rem <= 0:
                    break
                await asyncio.sleep(min(30.0, rem))
                for rec in records:
                    pg = rec.get("page")
                    if pg:
                        try:
                            await cf.handle_colab_dialogs(pg, allow_terminate_others=False)
                        except Exception:
                            pass
                print(f"  [WAIT] {time.time() - t0:.0f}s / {check_delay_sec}s elapsed...", flush=True)

        print("\n[HEALTH CHECK] Inspecting all 12 tabs (chrome u/6..u/9)...", flush=True)
        report = []
        healthy = 0
        for rec in records:
            pg = rec.get("page")
            clean = {k: v for k, v in rec.items() if k != "page"}
            if rec["status"] != "LAUNCHED" or pg is None:
                entry = {**clean, "health": "ERROR", "reason": rec.get("error", "Launch failed")}
            else:
                await cf.handle_colab_dialogs(pg, allow_terminate_others=False)
                info = await cf.inspect_tab_status(pg)
                verdict = cf.classify_health(info)
                entry = {
                    **clean,
                    "health": verdict["health"],
                    "reason": verdict["reason"],
                    "last_log": info.get("lastLog", ""),
                    "run_states": info.get("runStates", []),
                }
                if verdict["health"] in ("RUNNING_OK", "COMPLETED_OK"):
                    healthy += 1
            report.append(entry)
            print(
                f"  - chrome u/{entry.get('account_idx')} slot {entry.get('slot_idx')} "
                f"({entry.get('notebook')}): [{entry['health']}] {entry['reason']}",
                flush=True,
            )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n[SUMMARY] Healthy/Completed: {healthy}/{len(records)} | Report: {report_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks-dir", type=pathlib.Path, default=ROOT / "colab_notebooks" / "phase4_v2_dgp6")
    parser.add_argument("--n-shards", type=int, default=12)
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--check-delay-sec", type=int, default=300)
    parser.add_argument("--terminate-lingering-sessions", action="store_true")
    args = parser.parse_args()

    if args.write_configs:
        names = write_shard_configs(ROOT / "configs" / "phase4_v2_dgp6", n_shards=args.n_shards)
        print(f"Wrote {len(names)} shard config files to {ROOT / 'configs' / 'phase4_v2_dgp6'}", flush=True)
        return 0

    nb_paths = build_notebooks(args.notebooks_dir, n_shards=args.n_shards)
    if args.build_only:
        return 0
    asyncio.run(
        launch_u6_to_u9(
            nb_paths,
            args.notebooks_dir / "health_report.json",
            check_delay_sec=args.check_delay_sec,
            terminate_lingering=args.terminate_lingering_sessions,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
