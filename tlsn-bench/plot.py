#!/usr/bin/env python3
"""Plots and tables for the TLSNotary / Artemis benchmark.

  plot.py [results/results.csv] [--out results]

Writes sweep_prover.{pdf,png} (prover time against payload size, one panel per
backend, log-log, median over runs) and summary.md (tables for the sweep and
the age check).
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# label, colour, marker, linestyle. Colours validated for colour-vision
# deficiency (adjacent pairs); "No commitment" is a neutral reference line.
STYLE = {
    "sha256": ("SHA-256 (as shipped)", "#0072B2", "o", "-"),
    "keccak": ("Keccak-256", "#CC79A7", "^", "-"),
    "blake3": ("BLAKE3", "#E69F00", "v", "-"),
    "poseidon2": ("Poseidon2 (faster of t16 / native)", "#D55E00", "s", "-"),
    "artemis": ("Artemis", "#009E73", "D", "-"),
    "artemis_estimate": ("Artemis (cost estimate, not sound)", "#009E73", "D", ":"),
    "nocom": ("No commitment", "#666666", "x", "--"),
}
PANELS = [("provekit", "ProveKit (WHIR, pairing-free)"), ("bb", "Barretenberg (UltraHonk, KZG)")]


def load(path):
    """(workload, backend, row, bytes) -> {metric: median over runs}."""
    groups = defaultdict(list)
    for r in csv.DictReader(open(path)):
        groups[(r["workload"], r["backend"], r["row"], int(r["bytes"]))].append(r)
    out = {}
    for key, rows in groups.items():
        med = {}
        for metric in ("prover_s", "verifier_s", "proof_bytes", "max_rss_bytes", "constraints",
                       "opening_s", "commit_s", "witgen_s"):
            vals = [float(r[metric]) for r in rows if r[metric] not in ("", None)]
            med[metric] = statistics.median(vals) if vals else None
        med["runs"] = len(rows)
        out[key] = med
    return out


def series(data, backend):
    """row -> {bytes: metrics}, with the bb Artemis estimate completed by the
    external KZG opening and Poseidon2 reduced to its faster variant."""
    rows = defaultdict(dict)
    for (workload, b, row, n), m in data.items():
        if workload == "sweep" and b == backend:
            rows[row][n] = dict(m)
    if backend == "bb":
        for n, m in rows.get("artemis_estimate", {}).items():
            ext = data.get(("sweep", "external", "kzg", n))
            if ext:
                m["prover_s"] += ext["prover_s"]
                m["verifier_s"] += ext["verifier_s"]
                m["proof_bytes"] += ext["proof_bytes"]
    # Rows with several circuit variants: keep the faster one per size.
    for name, variants in (("poseidon2", ("poseidon2_t16", "poseidon2_native")),
                           ("keccak", ("keccak256", "keccak256_bb")),
                           ("blake3", ("blake3", "blake3_bb"))):
        variants = [v for v in variants if v in rows]
        best = {}
        for n in sorted({n for v in variants for n in rows[v]}):
            cands = [(v, rows[v][n]) for v in variants if n in rows[v]]
            variant, m = min(cands, key=lambda c: c[1]["prover_s"])
            best[n] = dict(m, variant=variant)
        if best:
            rows[name] = best
    return rows


def fmt_bytes(n):
    return f"{n} B" if n < 1024 else (f"{n / 1024:.1f} KiB" if n < 2**20 else f"{n / 2**20:.0f} MiB")


def plot(data, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (backend, title) in zip(axes, PANELS):
        rows = series(data, backend)
        for row, (label, colour, marker, ls) in STYLE.items():
            pts = sorted(rows.get(row, {}).items())
            if not pts:
                continue
            xs = [n for n, _ in pts]
            ys = [max(m["prover_s"], 1e-3) for _, m in pts]
            ax.plot(xs, ys, color=colour, marker=marker, linestyle=ls, linewidth=2, markersize=5,
                    label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Committed payload size [bytes]")
        ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Prover time [s]")
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"sweep_prover.{ext}", dpi=200)


def tables(data, out):
    lines = ["# TLSNotary / Artemis benchmark: summary", "",
             "Medians over the runs in the CSV. Prover time excludes ACVM witness generation, "
             "key generation and setup; the Artemis rows include the IPA opening "
             "(ProveKit) or the KZG opening (Barretenberg estimate).", ""]
    for backend, title in PANELS:
        rows = series(data, backend)
        order = [r for r in ("nocom", "sha256", "keccak", "blake3", "poseidon2", "artemis", "artemis_estimate") if r in rows]
        sizes = sorted({n for r in order for n in rows[r]})
        if not sizes:
            continue
        for metric, name, f in (
            ("prover_s", "Prover time [s]", lambda v: f"{v:.3f}"),
            ("constraints", "Constraints (ProveKit R1CS) / gates (bb circuit size)", lambda v: f"{int(v)}"),
            ("verifier_s", "Verifier time [ms]", lambda v: f"{v * 1e3:.1f}"),
            ("proof_bytes", "Proof size [kB]", lambda v: f"{v / 1e3:.1f}"),
            ("max_rss_bytes", "Prover peak memory [GiB]", lambda v: f"{v / 2**30:.2f}"),
        ):
            lines += [f"## {title}: {name}", "",
                      "| Payload | " + " | ".join(STYLE[r][0] for r in order) + " |",
                      "|---|" + "---|" * len(order)]
            for n in sizes:
                cells = []
                for r in order:
                    v = rows[r].get(n, {}).get(metric)
                    cells.append(f(v) if v is not None else "n/a")
                lines.append(f"| {fmt_bytes(n)} | " + " | ".join(cells) + " |")
            lines.append("")
        if "artemis" in rows and "sha256" in rows:
            lines += [f"## {title}: prover-time ratio to Artemis", "",
                      "| Payload | SHA-256 / Artemis | Keccak-256 / Artemis | BLAKE3 / Artemis | "
                      "Poseidon2 / Artemis | Artemis / No commitment |",
                      "|---|---|---|---|---|---|"]
            for n in sizes:
                a = rows["artemis"].get(n, {}).get("prover_s")
                def ratio(r, inv=False):
                    v = rows.get(r, {}).get(n, {}).get("prover_s")
                    if not a or not v:
                        return "n/a"
                    return f"{(a / v if inv else v / a):.1f}x"
                lines.append(f"| {fmt_bytes(n)} | {ratio('sha256')} | {ratio('keccak')} | {ratio('blake3')} | {ratio('poseidon2')} | {ratio('nocom', True)} |")
            lines.append("")
    age = {(b, row): m for (w, b, row, _), m in data.items() if w == "age"}
    if age:
        lines += ["## Age check (10 committed bytes)", "",
                  "| Backend | Row | Constraints / gates | Prover [ms] | Verifier [ms] | Proof [kB] |",
                  "|---|---|---|---|---|---|"]
        for (b, row), m in sorted(age.items()):
            lines.append(f"| {b} | {row} | {int(m['constraints'])} | {m['prover_s'] * 1e3:.1f} | "
                         f"{m['verifier_s'] * 1e3:.1f} | {m['proof_bytes'] / 1e3:.1f} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(here / "results/results.csv"))
    ap.add_argument("--out", default=str(here / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.csv)
    plot(data, out)
    tables(data, out)
    print(f"wrote {out}/sweep_prover.pdf, sweep_prover.png, summary.md")


if __name__ == "__main__":
    main()
