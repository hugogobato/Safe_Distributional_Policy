"""Verify, merge and summarize the Phase 4 task archives."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import beta, t

from safe_dist_policy.phase4_design import DESIGNS, STRESS
from safe_dist_policy.phase4_run import (
    PROTOCOL_SHA256,
    atomic_json,
    checkpoint,
    task_grid,
)

BASE_METHODS = {
    "LCT-LTT", "Oracle", "B1-survival", "B1-opra", "B2", "B3", "B4", "B5", "B5+ext",
    "LCT-LTT-eb", "LCT-LTT-5pct", "B2-5pct", "B5+ext-5pct",
}


def expected_methods(task):
    if task.study == "zic":
        return {"LCT-LTT", "LCT-ZIC-proj"}
    dgp = DESIGNS[task.design].dgp
    methods = set(BASE_METHODS)
    if dgp == 4:
        methods.update(("LCT-LTT-Qonly", "B2-Qonly", "B1-survival-Qonly", "B1-opra-Qonly"))
    if dgp in (3, 7):
        methods.add("LCT-LTT-fixed-ladder")
    if dgp == 7:
        methods.update(("LCT-LTT-misspec", "LCT-LTT-gamma", "LCT-LTT-misspec-fixed-ladder", "LCT-LTT-gamma-fixed-ladder"))
    if dgp == 8:
        methods.update(("LCT-LTT-Qonly", "B2-Qonly", "LCT-LTT-CVaR", "B1-opra-CVaR"))
    return methods


def verify_record(raw: bytes, task, *, smoke=False):
    record = json.loads(raw)
    if record.get("task") != task.__dict__ or record.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError(f"Task or frozen protocol mismatch for {task.key}.")
    if record["status"] != "ok":
        return record
    if record.get("reduced_zic", False) and not smoke:
        raise ValueError(f"Reduced ZIC verification cannot be used as confirmatory evidence: {task.key}.")
    rows = record["rows"]
    methods = [row["method"] for row in rows]
    if set(methods) != expected_methods(task) or len(methods) != len(set(methods)):
        raise ValueError(f"Wrong method rows for {task.key}: {methods}.")
    profile = record["profile"]
    d = DESIGNS[task.design]
    if profile["n_cal"] != d.n_cal or profile["n_eval"] != 100_000 or profile["n_train"] != math.ceil(2 * d.n_cal / 3):
        raise ValueError(f"Wrong shared split size for {task.key}.")
    for row in rows:
        if not 0 <= row["deployment"] <= 1 or not row["tested"] >= row["prefix_size"]:
            raise ValueError(f"Invalid evaluation or testing counts in {task.key} {row['method']}.")
        if not bool(row["prefix_definite"]) <= bool(row["prefix_unsafe"]) <= bool(row["prefix_possible"]):
            raise ValueError(f"Truth bounds out of order in {task.key} {row['method']}.")
        if any(isinstance(value, float) and not np.isfinite(value) for value in row.values()):
            raise ValueError(f"Nonfinite metric in {task.key} {row['method']}.")
    return record


def import_archive(path: Path, output: Path, lookup, *, smoke=False, index=None):
    with ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate archive members in {path}.")
        manifest = json.loads(archive.read("archive_manifest.json"))
        if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"Incompatible preregistration in {path}.")
        source_sha = manifest.get("source_sha256")
        shard = manifest.get("shard")
        if index is not None and (source_sha != index["source_sha256"] or shard not in index["shards"]):
            raise ValueError(f"Incompatible source release or shard in {path}.")
        assignment = json.loads(archive.read(f"{shard}_assignment.json"))
        encoded = json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
        if index is not None and hashlib.sha256(encoded).hexdigest() != index["shards"][shard]["assignment_sha256"]:
            raise ValueError(f"Unexpected assignment for shard {shard}.")
        assigned = {
            f"checkpoints/{t['study']}/{t['design']}/seed_{t['seed']}.json" for t in assignment["tasks"]
        }
        files = manifest["sha256"]
        if set(members) != {*files, "archive_manifest.json"}:
            raise ValueError(f"Archive inventory mismatch in {path}.")
        for name, digest in files.items():
            source = archive.read(name)
            if hashlib.sha256(source).hexdigest() != digest:
                raise ValueError(f"SHA-256 mismatch for {name} in {path}.")
            if name in lookup:
                task = lookup[name]
                if name not in assigned:
                    raise ValueError(f"Out-of-assignment checkpoint {name} in {path}.")
                record = verify_record(source, task, smoke=smoke)
                if record.get("source_sha256") != source_sha:
                    raise ValueError(f"Incompatible checkpoint source metadata for {task.key} in {path}.")
                target = checkpoint(output, task)
                if target.exists() and target.read_bytes() != source:
                    previous = verify_record(target.read_bytes(), task, smoke=smoke)
                    if previous["status"] == "ok" and record["status"] == "ok":
                        raise ValueError(f"Conflicting successful checkpoints for {task.key}.")
                    if previous["status"] == "ok":
                        target = target.with_name(f"{target.stem}.failed-{digest[:16]}.json")
                    else:
                        old_digest = hashlib.sha256(target.read_bytes()).hexdigest()
                        backup = target.with_name(f"{target.stem}.failed-{old_digest[:16]}.json")
                        target.replace(backup)
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_bytes(target, source)
            elif re.fullmatch(r"checkpoints/[a-z]+/[-a-z0-9]+/seed_\d+\.failed-[a-f0-9]+\.json", name):
                # Retry histories are kept without treating them as another replication.
                expected = name.split(".failed-", 1)[0] + ".json"
                if expected not in lookup or expected not in assigned:
                    raise ValueError(f"Unexpected failed copy in {path}: {name}.")
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                _atomic_bytes(destination, source)
            elif name.startswith("checkpoints/"):
                raise ValueError(f"Unexpected task in {path}: {name}.")


def _atomic_bytes(path: Path, raw: bytes):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(raw)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    half = float(t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.
    return mean, mean - half, mean + half


def binomial_ci(events):
    n = len(events)
    k = sum(events)
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return k / n, lo, hi


def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[row["study"], row["design"], row["method"]].append(row)

    estimates = []
    for (study, design, method), group in sorted(cells.items()):
        summary = {"study": study, "design": design, "method": method, "n": len(group)}
        for field in (
            "deployment", "welfare_gain", "safe_welfare", "safe_welfare_lower", "delta_quantile",
            "delta_extensive", "delta_cvar", "trap_switch_fraction", "cert_seconds", "cvar_conditional_delta",
        ):
            values = np.array([r[field] for r in group if r.get(field) is not None])
            if len(values) == 0:
                continue
            avg, lo, hi = mean_ci(values)
            q1, med, q3 = np.quantile(values, (0.25, 0.50, 0.75))
            summary.update({
                f"{field}_mean": avg, f"{field}_lo": lo, f"{field}_hi": hi,
                f"{field}_median": float(med), f"{field}_iqr": float(q3 - q1),
            })
        for field in (
            "prefix_unsafe", "prefix_definite", "prefix_possible", "q_unsafe", "ext_unsafe", "policy_selected",
            "heldout_switch_observed", "cvar_conditional_prefix_unsafe",
        ):
            events = [r[field] for r in group if r.get(field) is not None]
            if not events:
                continue
            rate, lo, hi = binomial_ci(events)
            summary.update({field: rate, f"{field}_lo": lo, f"{field}_hi": hi})
        estimates.append(summary)

    pairs = []
    for (study, design, method), primary in sorted(cells.items()):
        if method not in ("LCT-LTT", "LCT-ZIC-proj"):
            continue
        primary_by_seed = {r["seed"]: r for r in primary}
        for comparator in (
            "B1-survival", "B1-opra", "B1-best", "B1-per-seed-envelope", "B2", "B5", "B5+ext",
            "LCT-LTT-eb", "LCT-LTT-5pct", "LCT-LTT",
        ):
            if comparator == method:
                continue
            key = study, design, comparator
            fields = ("deployment", "safe_welfare", "safe_welfare_lower")
            winners = {field: comparator for field in fields}
            if comparator in ("B1-best", "B1-per-seed-envelope"):
                survival = {r["seed"]: r for r in cells.get((study, design, "B1-survival"), [])}
                opra = {r["seed"]: r for r in cells.get((study, design, "B1-opra"), [])}
                b1_seeds = sorted(primary_by_seed.keys() & survival.keys() & opra.keys())
                if not b1_seeds:
                    continue
                if comparator == "B1-best":
                    winners = {
                        f: "B1-survival" if sum(survival[s][f] for s in b1_seeds) >= sum(opra[s][f] for s in b1_seeds)
                        else "B1-opra" for f in fields
                    }
                compared = {
                    s: {
                        f: max(survival[s][f], opra[s][f]) if comparator == "B1-per-seed-envelope"
                        else (survival[s][f] if winners[f] == "B1-survival" else opra[s][f])
                        for f in fields
                    }
                    for s in b1_seeds
                }
            else:
                compared = {r["seed"]: r for r in cells.get(key, [])}
            common = sorted(primary_by_seed.keys() & compared.keys())
            if not common:
                continue
            pair = {"study": study, "design": design, "method": method, "comparator": comparator, "n": len(common)}
            for field in fields:
                values = [primary_by_seed[s][field] - compared[s][field] for s in common]
                avg, lo, hi = mean_ci(values)
                base = float(np.mean([compared[s][field] for s in common]))
                pair.update({
                    f"{field}_diff": avg, f"{field}_lo": lo, f"{field}_hi": hi,
                    f"{field}_relative": avg / base if base > 0 else None,
                    f"{field}_comparator": winners[field],
                })
            pairs.append(pair)
    return estimates, pairs


def save_table(output, prefix, rows):
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    rows = [{field: row.get(field) for field in fields} for row in rows]
    destination = output / f"{prefix}.csv"
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output, delete=False) as stream:
        tmp = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, destination)
    destination = output / f"{prefix}.parquet"
    with NamedTemporaryFile(dir=output, delete=False) as stream:
        tmp = Path(stream.name)
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    os.replace(tmp, destination)


def make_crossover(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {(row["design"], row["method"]): row for row in rows if row["design"].startswith("d06")}
    if {d.name for d in STRESS if d.dgp == 6} - {design for design, _ in cells}:
        return []
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.1), sharex=True, sharey=True, layout="constrained")
    welfare_fig, welfare_axes = plt.subplots(3, 4, figsize=(13, 8.1), sharex=True, sharey=True, layout="constrained")
    methods = ("LCT-LTT", "B1-survival", "B1-opra", "B2", "B5+ext")
    for beta_value, row_axes, welfare_row in zip((80, 90, 95), axes, welfare_axes):
        for safe, ax, welfare_ax in zip((10, 25, 50, 80), row_axes, welfare_row):
            for method in methods:
                sizes, y, lower, upper, w, wlo, whi = [], [], [], [], [], [], []
                for n in (1000, 2500, 5000, 10000):
                    cell = cells.get((f"d06-n{n}-s{safe:03d}-q{beta_value}", method))
                    if cell:
                        sizes.append(n)
                        y.append(cell["deployment_mean"])
                        lower.append(cell["deployment_lo"])
                        upper.append(cell["deployment_hi"])
                        w.append(cell["safe_welfare_mean"])
                        wlo.append(cell["safe_welfare_lo"])
                        whi.append(cell["safe_welfare_hi"])
                if sizes:
                    line, = ax.plot(sizes, y, "o-", label=method, markersize=3)
                    ax.fill_between(sizes, lower, upper, color=line.get_color(), alpha=0.10)
                    welfare_ax.plot(sizes, w, "o-", label=method, markersize=3)
                    welfare_ax.fill_between(sizes, wlo, whi, color=line.get_color(), alpha=0.10)
            for panel in (ax, welfare_ax):
                panel.set_title(f"β .{beta_value} · safe region {safe}%")
                panel.set_xscale("log")
                panel.set_xticks((1000, 2500, 5000, 10000), ("1k", "2.5k", "5k", "10k"))
                panel.grid(alpha=0.15)
    for figure, all_axes, ylabel in (
        (fig, axes, "Certified deployment"), (welfare_fig, welfare_axes, "Safety-adjusted welfare"),
    ):
        handles, labels = all_axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="outside upper center", ncols=len(methods))
        figure.supxlabel("Calibration sample size")
        figure.supylabel(ylabel)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    path = figures / "crossover_deployment_welfare.pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pages:
        pages.savefig(fig)
        pages.savefig(welfare_fig)
    plt.close(fig)
    plt.close(welfare_fig)
    return [path]


def make_core_figure(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core = {(row["design"], row["method"]): row for row in rows if row["study"] == "core"}
    if not core:
        return None
    designs = ("d01-main", "d02-main", "d03-equality", "d03-strict", "d04-main", "d05-main")
    descriptions = ("DGP 1  Recovery", "DGP 2  Smooth hurdle", "DGP 3  Equality", "DGP 3  Strict near-null",
                    "DGP 4  Location shift", "DGP 5  Mean/tail conflict")
    colors = {"LCT-LTT": "#007d79", "B1-survival": "#58429d", "B1-opra": "#9570bc", "B2": "#bb7510", "B5+ext": "#525969"}
    offsets = np.linspace(-.30, .30, len(colors))
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.3), sharey=True, layout="constrained")
    for ax, endpoint, scale, title in zip(
        axes, ("deployment", "safe_welfare"), (100, 1), ("Mean switch rate (%)", "Mean safety-adjusted welfare"),
    ):
        for offset, (method, color) in zip(offsets, colors.items()):
            selected = [core[(d, method)] for d in designs]
            mean, lo, hi = [np.array([row[f"{endpoint}_{end}"] for row in selected]) * scale for end in ("mean", "lo", "hi")]
            ax.errorbar(mean, np.arange(len(designs)) + offset, xerr=[mean - lo, hi - mean],
                        fmt="o", markersize=3.7, color=color, capsize=2, linewidth=1.1, label=method)
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_yticks(np.arange(len(designs)), descriptions)
        ax.axvline(0, linewidth=.8, color=".55")
        ax.grid(axis="x", color=".90", linewidth=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False)
    fig.suptitle("Quantile and extensive-margin certification", x=.01, ha="left", fontsize=14, fontweight="semibold")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    path = figures / "core_method_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase4_sim"))
    p.add_argument("--archives", nargs="*", type=Path, default=[])
    p.add_argument("--study", default="all", choices=("all", "core", "stress", "zic", "remote"))
    p.add_argument("--tasks-file", type=Path)
    p.add_argument("--source-index", type=Path)
    p.add_argument("--verify-smoke", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.tasks_file:
        from safe_dist_policy.phase4_run import Task
        spec = json.loads(args.tasks_file.read_text(encoding="utf-8"))
        if spec["protocol_sha256"] != PROTOCOL_SHA256:
            p.error("Unexpected task manifest.")
        tasks = [Task(**record) for record in spec["tasks"]]
    else:
        tasks = task_grid(args.study)
    lookup = {checkpoint(output, task).relative_to(output).as_posix(): task for task in tasks}
    source_index = None
    if args.source_index:
        source_index = json.loads(args.source_index.read_text(encoding="utf-8"))
        if source_index.get("protocol_sha256") != PROTOCOL_SHA256:
            p.error("Unexpected source release protocol.")
    elif args.archives and not args.verify_smoke:
        p.error("Provide --source-index with confirmatory notebook result ZIPs.")
    for archive in args.archives:
        if archive.is_dir():
            for path in sorted(archive.glob("*.zip")):
                import_archive(path, output, lookup, smoke=args.verify_smoke, index=source_index)
        else:
            import_archive(archive, output, lookup, smoke=args.verify_smoke, index=source_index)

    local_manifest = output / "core_checksums.json"
    local_core = {}
    if local_manifest.exists():
        payload = json.loads(local_manifest.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("Local core checkpoint manifest has the wrong protocol.")
        local_core = payload["sha256"]

    rows, missing, failures, profiles = [], [], [], []
    hashes = {}
    for name, task in lookup.items():
        path = output / name
        if not path.is_file():
            missing.append(task.key)
            continue
        raw = path.read_bytes()
        record = verify_record(raw, task, smoke=args.verify_smoke)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if task.study == "core" and local_core and hashes[name] != local_core.get(name):
            raise ValueError(f"Local core checkpoint manifest failed for {task.key}.")
        if record["status"] != "ok":
            failures.append(task.key)
            continue
        design = DESIGNS[task.design]
        shared = {
            **task.__dict__, "dgp": design.dgp, "beta": design.beta, "p": design.p,
            "safe_mass": design.safe_mass, "kappa": design.kappa,
            "design_overlap": design.kappa + .14 * (1 - 2 * design.kappa) if design.dgp == 7 else design.kappa,
            "n_cal": design.n_cal, "cap": record["profile"]["cap"],
        }
        rows.extend({**shared, **row} for row in record["rows"])
        profiles.append({**shared, **record["profile"]})

    if args.verify_smoke:
        print(f"Verified {len(profiles)} smoke checkpoints, {len(rows)} rows", flush=True)
        return 1 if missing or failures else 0
    core = [row for row in rows if row["study"] == "core"]
    stress = [row for row in rows if row["study"] in ("stress", "zic")]
    save_table(output, "core_dgp1_5_r50", core)
    save_table(output, "stress_dgp6_10", stress)
    save_table(output, "resource_profiles", profiles)
    summaries, paired = summarize(rows)
    save_table(output, "method_summary", summaries)
    save_table(output, "paired_comparisons", paired)
    if not missing and not failures:
        make_core_figure(summaries, output)
        make_crossover(summaries, output)
    published = {path for path in output.rglob("*") if path.suffix in {".csv", ".parquet", ".pdf"} and (
        path.parent == output or path.parent == output / "figures"
    )}
    hashes.update({path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in published})
    label = args.tasks_file.stem if args.tasks_file else args.study
    status = {
        "protocol_sha256": PROTOCOL_SHA256, "study": label, "complete": not missing and not failures,
        "selected_tasks": len(tasks), "successful_tasks": len(profiles), "rows": len(rows),
        "missing": missing, "failed": failures, "sha256": hashes,
    }
    atomic_json(output / f"{label}_merge_status.json", status)
    print(f"{len(profiles)} successful tasks, {len(rows)} method rows, {len(missing)} missing, {len(failures)} failed")
    return 1 if (missing or failures) and not args.allow_partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
