"""Build independently runnable Phase 4 notebooks and their portable source repo."""

import argparse
import base64
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import nbformat as nb

from safe_dist_policy.phase4_design import DESIGNS
from safe_dist_policy.phase4_run import (
    PROTOCOL_SHA256,
    Task,
    checkpoint,
    read_checkpoint,
    task_grid,
)

VENDOR_COMMIT = "b410fcc30cf127343c56614e74f2ad7369daa394"
ROOT = Path(__file__).resolve().parents[1]
PYTHON_UV = "0.11.26"
README = """# Safe Distributional Policy Phase 4

This source folder is ready to upload to a GitHub repository. It contains the simulator, command-line runners, frozen preregistration, pinned Python environment, and the tracked source of the GPL zicbcf dependency at commit `b410fcc30cf127343c56614e74f2ad7369daa394`. The simulator uses fully synthetic data. The completed local core results are kept in the research project.

## Google Colab

Open one notebook per listed shard and run all cells on a CPU runtime. Each notebook embeds this source release and its own task list, so it can run without a GitHub upload, a Drive mount or an earlier notebook. It builds an isolated Python 3.12 environment from `uv.lock` and installs the required R packages. Most shards use two isolated workers. Scaling and ZIC-BCF shards use one.

Each notebook produces one uniquely named result ZIP. Keep partial ZIPs produced when a task or setup fails. Returning to the same runtime and rerunning preserves successful local checkpoints. On a new runtime, the optional resume ZIP configuration restores the previous checkpoints for the same shard before running it again. The final ZIP includes setup or worker failures when known and uses a local-save fallback outside Colab.

## Run from source

Install Python with uv and use a recent R with recommended packages. The shell commands below preserve successful checkpoints and verify hashes.

```sh
uv sync --locked --no-dev --python 3.12
Rscript --vanilla scripts/colab_r_bootstrap.R .
Rscript --vanilla scripts/colab_r_bootstrap.R . zic
.venv/bin/python scripts/run_phase4_core.py --workers 2
.venv/bin/python scripts/run_phase4_stress.py --tasks-file configs/phase4_stress_01.json --workers 2
```

For the full study, merge the local core directory and the notebook ZIPs from the research project root. Set the archive directory to contain these Phase 4 ZIPs only.

```sh
uv pip install --python .venv/bin/python -r requirements-phase4-analysis.txt
.venv/bin/python scripts/merge_phase4.py \
  --output-dir results/phase4_sim --archives /path/to/phase4-zips --source-index /path/to/shard_index.json
```

The merger checks the task grid, schema, hashes, and protocol and fails if any required checkpoint is missing or failed. It refuses inconsistent duplicate successful results, preserves retries, and writes the merged tables and the crossover figure. The Phase 4 decision is based on the inspected merged evidence, with thresholds documented in [the frozen protocol](docs/phase4_preregistration.md). Core-only summaries can be reconstructed with `--study core`.

## ZIC-BCF

The vendored R model fits on training observations alone. The forecast on new covariates is a secondary projection from its fitted posterior summaries. See the vendor license and [the R bridge](src/safe_dist_policy/forest_bridge.R).
"""


def balanced(tasks, n_bins, weight):
    bins = [[] for _ in range(n_bins)]
    loads = [0.0] * n_bins
    for task in sorted(tasks, key=lambda t: (-weight(t), t.design, t.seed)):
        index = min(range(n_bins), key=lambda i: (loads[i], i))
        bins[index].append(task)
        loads[index] += weight(task)
    return bins


def shard_specs(completed=()):
    tasks = [task for task in task_grid("stress") if task not in completed]
    ordinary = [t for t in tasks if DESIGNS[t.design].n_cal < 20_000]
    mid = [t for t in tasks if DESIGNS[t.design].n_cal == 20_000]
    big = [t for t in tasks if DESIGNS[t.design].n_cal == 100_000]

    def normalized_cost(task):
        d = DESIGNS[task.design]
        return (0.20 + (d.n_cal / 5000) ** 0.75) * (1.18 if d.dgp in (8, 9, 10) else 1)

    shards = {
        f"phase4_stress_{i:02d}": (group, 2, False)
        for i, group in enumerate(balanced(ordinary, 12, normalized_cost), 1)
    }
    shards["phase4_scaling_20000"] = (mid, 1, False)
    for p in (10, 50):
        shards[f"phase4_scaling_100000_p{p}"] = ([t for t in big if DESIGNS[t.design].p == p], 1, False)
    for i, group in enumerate(balanced(task_grid("zic"), 6, lambda _: 1), 1):
        name = "phase4_zicbcf_colab" if i == 1 else f"phase4_zicbcf_colab_{i:02d}"
        shards[name] = (group, 1, True)
    flattened = [t for group, _, _ in shards.values() for t in group]
    assert len(flattened) == len(set(flattened)) and set(flattened) == set(task_grid("remote")) - set(completed)
    return shards


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_sha(obj):
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_vendor(source_dir, destination):
    destination.mkdir(parents=True)
    raw = subprocess.check_output([
        "git", "-C", str(source_dir), "archive", "--format=tar", VENDOR_COMMIT, "--",
        "DESCRIPTION", "LICENSE", "NAMESPACE", "R", "src", "man",
    ])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter="data")
    (destination / "SOURCE_REVISION").write_text(VENDOR_COMMIT + "\n")


def create_source(destination, vendor, shards):
    destination.mkdir(parents=True)
    files = (
        "pyproject.toml", "uv.lock", "requirements-phase4-analysis.txt",
        "docs/phase4_preregistration.md",
        "scripts/run_phase4_core.py", "scripts/run_phase4_stress.py", "scripts/merge_phase4.py",
        "scripts/colab_r_bootstrap.R",
    )
    source = ROOT / "src" / "safe_dist_policy"
    files = [*files, *(str(p.relative_to(ROOT)) for p in source.iterdir() if p.suffix in (".py", ".R"))]
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copytree(ROOT / "tests", destination / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    copy_vendor(vendor, destination / "vendor" / "zicbcf")
    (destination / "README.md").write_text(README)
    (destination / ".gitignore").write_text(".venv/\n__pycache__/\n*.py[cod]\n*.egg-info/\n.pytest_cache/\n.ruff_cache/\n.mypy_cache/\n")
    configs = destination / "configs"
    configs.mkdir()
    for name, (group, workers, zic) in shards.items():
        write_json(configs / f"{name}.json", {
            "protocol_sha256": PROTOCOL_SHA256, "tasks": [t.__dict__ for t in group],
            "workers": workers, "zic": zic,
        })
    entries = {str(p.relative_to(destination)): sha(p.read_bytes()) for p in sorted(destination.rglob("*")) if p.is_file()}
    write_json(destination / "source_manifest.json", {
        "protocol_sha256": PROTOCOL_SHA256, "zicbcf_commit": VENDOR_COMMIT, "sha256": entries,
    })


def source_archive(source, path):
    with ZipFile(path, "w", ZIP_DEFLATED, compresslevel=9, strict_timestamps=False) as archive:
        for entry in sorted(source.rglob("*")):
            if entry.is_file():
                name = entry.relative_to(source).as_posix()
                member = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                archive.writestr(member, entry.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)


START = '''import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

resume_archive = ""  # Optional path to a previously downloaded ZIP for this shard.
workspace = Path(os.environ.get("SAFE_DIST_PHASE4_ROOT", "phase4_runs")).resolve() / shard_name
workspace.mkdir(parents=True, exist_ok=True)
source_dir = workspace / "source"
run_dir = workspace / "results"
run_dir.mkdir(exist_ok=True)
log_file = run_dir / "notebook.log"
assigned = run_dir / f"{shard_name}_assignment.json"
assigned.write_text(json.dumps(assignment, indent=2) + "\\n")
run_error = None

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

def checked_output(command, cwd=workspace, env=None):
    rendered = " ".join(map(str, command))
    print(rendered, flush=True)
    with log_file.open("a") as log:
        log.write(f"\\n$ {rendered}\\n")
        with subprocess.Popen(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1) as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)

source = base64.b64decode(embedded_source)
if digest(source) != source_sha256:
    raise ValueError("Embedded source SHA-256 mismatch.")
with ZipFile(io.BytesIO(source)) as archive:
    archive.extractall(source_dir)
source_manifest = json.loads((source_dir / "source_manifest.json").read_text())
for name, expected in source_manifest["sha256"].items():
    if digest((source_dir / name).read_bytes()) != expected:
        raise ValueError(f"Source SHA-256 mismatch for {name}.")
'''

RUN = '''def run_shard():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["SAFE_DIST_POLICY_SOURCE_SHA256"] = source_sha256
    env["UV_PROJECT_ENVIRONMENT"] = str(source_dir / ".venv")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if "R_LIBS_USER" not in env:
        r_library = workspace / "R-library"
        r_library.mkdir(exist_ok=True)
        env["R_LIBS_USER"] = str(r_library)
    old_error = run_dir / "notebook_error.txt"
    if old_error.exists():
        old_error.replace(run_dir / f"notebook_error-{digest(old_error.read_bytes())[:16]}.txt")
    if resume_archive:
        with ZipFile(resume_archive) as archive:
            record = json.loads(archive.read("archive_manifest.json"))
            if (record["protocol_sha256"], record["source_sha256"], record["shard"]) != (
                    assignment["protocol_sha256"], source_sha256, shard_name):
                raise ValueError("Resume archive has a different source, shard or protocol.")
            known = {f"checkpoints/{t['study']}/{t['design']}/seed_{t['seed']}.json"
                     for t in assignment["tasks"]}
            for name in known.intersection(archive.namelist()):
                contents = archive.read(name)
                if digest(contents) != record["sha256"][name]:
                    raise ValueError(f"Resume checkpoint SHA-256 mismatch for {name}.")
                destination = run_dir / name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(contents)

    uv = shutil.which("uv")
    uv_version = subprocess.run([uv, "--version"], text=True, capture_output=True, check=False).stdout if uv else ""
    if uv_version.startswith("uv UV_PIN ") or uv_version.strip() == "uv UV_PIN":
        uv_command = [uv]
    else:
        checked_output([sys.executable, "-m", "pip", "install", "uv==UV_PIN"], env=env)
        uv_command = [sys.executable, "-m", "uv"]
    checked_output([*uv_command, "sync", "--locked", "--no-dev", "--python", "3.12"],
                   cwd=source_dir, env=env)
    python = source_dir / ".venv" / "bin" / "python"

    if any(not shutil.which(tool) for tool in ("Rscript", "gfortran", "g++")):
        checked_output(["apt-get", "update"], env=env)
        checked_output(["apt-get", "install", "-y", "r-base", "r-base-dev",
                        "libopenblas-dev", "gfortran", "g++"], env=env)
    if not os.environ.get("SAFE_DIST_COLAB_SKIP_R_SETUP"):
        cmd = ["Rscript", "--vanilla", str(source_dir / "scripts" / "colab_r_bootstrap.R"), str(source_dir)]
        if assignment["zic"]:
            cmd.append("zic")
        checked_output(cmd, env=env)
    cmd = [str(python), "-u", str(source_dir / "scripts" / "run_phase4_stress.py"),
           "--tasks-file", str(assigned), "--output-dir", str(run_dir), "--workers", str(assignment["workers"]),
           "--retry-failed"]
    if local_bridge_verification:
        cmd.append("--reduced-zic")
    checked_output(cmd, cwd=source_dir, env=env)

try:
    run_shard()
except Exception:
    run_error = traceback.format_exc()
    (run_dir / "notebook_error.txt").write_text(run_error)
    print(run_error, flush=True)
'''

FINISH = '''output_file = str(workspace / f"{shard_name}_results.zip")
outputs = {p.relative_to(run_dir).as_posix(): p for p in sorted(run_dir.rglob("*")) if p.is_file()}
manifest = {
    "protocol_sha256": assignment["protocol_sha256"], "source_sha256": source_sha256,
    "shard": shard_name,
    "sha256": {name: digest(path.read_bytes()) for name, path in outputs.items()},
    "complete": run_error is None,
}
with ZipFile(output_file, "w", ZIP_DEFLATED, strict_timestamps=False) as archive:
    for name, path in outputs.items():
        archive.write(path, name)
    archive.writestr("archive_manifest.json", json.dumps(manifest, indent=2) + "\\n")

try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file, flush=True)
except Exception as e:
    print("(Not on Colab / download skipped):", e, flush=True)
    print("Saved:", output_file, flush=True)

if run_error:
    raise RuntimeError("Shard execution failed. The downloaded archive includes notebook_error.txt.")
'''


def notebook(shard, assignment, payload, *, verification=False):
    source_sha = sha(payload)
    start_config = "\n".join((
        f"shard_name = {shard!r}", f"assignment = {assignment!r}",
        f"source_sha256 = {source_sha!r}", f"local_bridge_verification = {verification!r}",
    ))
    text = (
        f"# {shard}\n\n"
        f"Run all cells on a CPU runtime. This notebook contains its frozen source and {len(assignment['tasks'])} "
        "independent tasks. It stores checkpoints as it runs and triggers one ZIP download at the end."
    )
    book = nb.v4.new_notebook(cells=[
        nb.v4.new_markdown_cell(text),
        nb.v4.new_code_cell(start_config),
        nb.v4.new_code_cell("embedded_source = " + repr(base64.b64encode(payload).decode())),
        nb.v4.new_code_cell(START),
        nb.v4.new_code_cell(RUN.replace("UV_PIN", PYTHON_UV)),
        nb.v4.new_code_cell(FINISH),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nb.validate(book)
    return book


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--with-local-verification", action="store_true")
    parser.add_argument("--completed-stress", type=Path, help="Results directory of successful local focal runs.")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "phase4_source"
    zip_path = output / "phase4_source.zip"
    notebook_dir = output / "notebooks"
    if any(p.exists() for p in (destination, zip_path, notebook_dir)):
        parser.error("Use a fresh output directory to preserve existing source and notebook releases.")
    notebook_dir.mkdir()
    completed = set()
    if args.completed_stress:
        for task in task_grid("stress"):
            record = read_checkpoint(checkpoint(args.completed_stress, task), task)
            if record is not None:
                completed.add(task)
    shards = shard_specs(completed)
    create_source(destination, args.vendor, shards)
    source_archive(destination, zip_path)
    payload = zip_path.read_bytes()
    summaries = {}
    for name, (group, workers, zic) in shards.items():
        assignment = json.loads((destination / "configs" / f"{name}.json").read_text())
        nb.write(notebook(name, assignment, payload), notebook_dir / f"{name}.ipynb")
        summaries[name] = {
            "tasks": len(group), "workers": workers, "zic": zic,
            "designs": sorted({t.design for t in group}),
            "assignment_sha256": json_sha(assignment),
        }
    write_json(output / "shard_index.json", {
        "protocol_sha256": PROTOCOL_SHA256, "source_sha256": sha(payload), "shards": summaries,
        "completed_local_stress_tasks": [t.__dict__ for t in sorted(completed, key=lambda t: t.key)],
    })
    if args.with_local_verification:
        group = [
            Task("stress", "d06-n5000-s025-q90", 999), Task("stress", "d07-k05", 999),
            Task("stress", "d08-cap2", 999), Task("stress", "d09-meps-proxy", 999),
            Task("stress", "d09-ohie-proxy", 999), Task("stress", "d10-n1000-p10", 999),
            Task("zic", "d05-main", 999),
        ]
        assignment = {"protocol_sha256": PROTOCOL_SHA256, "tasks": [t.__dict__ for t in group], "workers": 2, "zic": True}
        write_json(output / "local_verification_tasks.json", assignment)
        nb.write(notebook("phase4_local_verification", assignment, payload, verification=True), output / "local_verification.ipynb")


if __name__ == "__main__":
    main()
