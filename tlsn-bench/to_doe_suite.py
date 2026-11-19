#!/usr/bin/env python3
"""Import local TLSNotary benchmark results as a suite result folder.

  to_doe_suite.py [results/results.csv] [--suite tlsn-local] [--id ID]

Writes `$DOES_PROJECT_DIR/doe-suite-results/<suite>_<id>/` in the layout that
the experiment framework produces for a remote run, so that a super-ETL config can combine the
TLSNotary results with the zkML suites (see
`doe-suite-config/super_etl/combined_plots.yml`):

  <suite>_<id>/suite_design.yml
  <suite>_<id>/exp_<backend>/run_<i>/rep_0/config.json      {workload, backend, row, bytes}
  <suite>_<id>/exp_<backend>/run_<i>/rep_0/local/host_0/results.csv

One experiment per backend (provekit, bb, external), one run per (workload,
row, bytes) point; results.csv holds the repetitions of that point in the
format of `bench.py`. A future suite design for the TLSNotary benchmark
should write the same results.csv per run (one row per repetition) so the
super-ETL does not change: only the suite name and id in `$SUITE_ID$` do.

The suite id defaults to the modification time of the CSV, so re-importing
unchanged results gives the same folder.
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(here / "results" / "results.csv"))
    ap.add_argument("--suite", default="tlsn-local")
    ap.add_argument("--id", default=None, help="suite id (default: CSV mtime)")
    ap.add_argument("--results-dir", default=None,
                    help="results dir (default: $DOES_PROJECT_DIR/doe-suite-results)")
    args = ap.parse_args()

    project = os.environ.get("DOES_PROJECT_DIR", str(here.parent))
    results_dir = Path(args.results_dir or Path(project) / "doe-suite-results")
    suite_id = args.id or str(int(os.path.getmtime(args.csv)))
    suite_dir = results_dir / f"{args.suite}_{suite_id}"

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    # backend -> (workload, row, bytes) -> [csv rows]
    points = defaultdict(lambda: defaultdict(list))
    for r in rows:
        points[r["backend"]][(r["workload"], r["row"], int(r["bytes"]))].append(r)

    if suite_dir.exists():
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True)

    design = {}
    for backend, pts in sorted(points.items()):
        exp = f"exp_{backend}"
        keys = sorted(pts)
        design[exp] = {
            "n_repetitions": 1,
            "common_roles": [],
            "host_types": {"local": {"n": 1, "check_status": True, "init_roles": [], "$CMD$": [{"main": "[% my_run.cmd %]"}]}},
            "base_experiment": {
                "backend": backend,
                "workload": "$FACTOR$",
                "row": "$FACTOR$",
                "bytes": "$FACTOR$",
                "cmd": "imported from tlsn-bench/results/results.csv (local run)",
            },
            "factor_levels": [{"workload": w, "row": row, "bytes": n} for (w, row, n) in keys],
            "except_filters": [],
        }
        for i, (w, row, n) in enumerate(keys):
            rep_dir = suite_dir / exp / f"run_{i}" / "rep_0"
            host_dir = rep_dir / "local" / "host_0"
            host_dir.mkdir(parents=True)
            config = {"backend": backend, "workload": w, "row": row, "bytes": n,
                      "cmd": design[exp]["base_experiment"]["cmd"]}
            (rep_dir / "config.json").write_text(json.dumps(config, indent=4))
            with open(host_dir / "results.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=header)
                wr.writeheader()
                wr.writerows(pts[(w, row, n)])

    design["$ETL$"] = {}
    # suite_design.yml is read with a YAML loader; JSON is valid YAML.
    (suite_dir / "suite_design.yml").write_text(json.dumps(design, indent=2))

    n_runs = sum(len(p) for p in points.values())
    print(f"wrote {suite_dir} ({len(points)} experiments, {n_runs} runs, {len(rows)} rows)")
    print(f"super-ETL: $SUITE_ID$: {{{args.suite}: \"{suite_id}\"}}")


if __name__ == "__main__":
    main()
