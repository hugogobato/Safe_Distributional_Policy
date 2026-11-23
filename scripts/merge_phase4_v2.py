"""Merge all 12 Colab DGP 6 shard ZIPs with local Phase 4 v2 & stress results."""

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", type=Path, default=Path("/home/hugo/Downloads"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4_v2_stress/dgp6_merged"))
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    zips = sorted(args.downloads_dir.glob("dgp6_shard_*_results.zip"))
    print(f"Found {len(zips)} DGP 6 shard ZIPs in {args.downloads_dir}", flush=True)
    if len(zips) < 12:
        missing = sorted({f"dgp6_shard_{i:02d}_results.zip" for i in range(1, 13)} - {z.name for z in zips})
        print("Missing shards:", missing, flush=True)

    checkpoints: dict[tuple[str, int], dict] = {}
    errors: list[str] = []
    for zpath in zips:
        with zipfile.ZipFile(zpath, "r") as zf:
            if "notebook_error.txt" in zf.namelist():
                errors.append(f"{zpath.name}: {zf.read('notebook_error.txt').decode('utf-8', errors='replace')[:300]}")
            for name in zf.namelist():
                if name.startswith("checkpoints/dgp6_crossover/") and name.endswith(".json"):
                    rec = json.loads(zf.read(name).decode("utf-8"))
                    if rec.get("status") == "ok":
                        key = (str(rec["design"]), int(rec["seed"]))
                        checkpoints[key] = rec

    print(f"Loaded {len(checkpoints)} / 2400 unique ok DGP 6 checkpoints.", flush=True)
    if errors:
        print("Shard errors:", errors, flush=True)

    all_rows = [row for rec in checkpoints.values() for row in rec["rows"]]
    if not all_rows:
        return 1

    with (out / "dgp6_all_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in all_rows:
        grouped[(str(r["design"]), str(r["target"]), str(r["method"]))].append(r)

    summary = []
    for (d, t, m), recs in sorted(grouped.items()):
        summary.append({
            "design": d,
            "target": t,
            "method": m,
            "n_seeds": len(recs),
            "deployment": float(np.mean([float(r["deployment"]) for r in recs])),
            "safe_welfare": float(np.mean([float(r["safe_welfare"]) for r in recs])),
            "welfare_gain": float(np.mean([float(r["welfare_gain"]) for r in recs])),
            "prefix_fwer": float(np.mean([bool(r["prefix_unsafe"]) for r in recs])),
            "pass_rate": float(np.mean([bool(r["deployed"]) for r in recs])),
        })
    with (out / "dgp6_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    (out / "dgp6_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} cell-method summary rows to {out / 'dgp6_summary.csv'}", flush=True)
    return 0 if len(checkpoints) == 2400 and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
