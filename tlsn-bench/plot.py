#!/usr/bin/env python3
"""Plots and tables for the TLSNotary / Artemis benchmark.

  plot.py [results/results.csv | <dir> ...] [--out results] [--backends bb] [--machine NAME]

Writes sweep_prover.{pdf,png} (prover time against payload size, one panel per
backend, log-log, median over runs) and summary.md (tables for the sweep and
the age check).

Several sources can be given; a directory stands for every results.csv below it
(a doe-suite result folder) together with its lego_layout.json files. A point
(workload, backend, row, bytes) is taken from the first source that has it, so
runs of different sessions or machines are never mixed in one median. AWS:

  plot.py ../doe-suite-results/tlsn_<id> ../doe-suite-results/tlsn-lego_<id> \
      --out results/aws --backends bb --machine "AWS r6i.8xlarge"
"""

import argparse
import csv
import json
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
    "lego_estimate": ("LegoSNARK CP_link (cost estimate)", "#56B4E9", "P", (0, (1, 1.5))),
    "nocom": ("No commitment", "#666666", "x", "--"),
}
PANELS = [("provekit", "ProveKit (WHIR, pairing-free)"), ("bb", "Barretenberg (UltraHonk, KZG)")]


def source_files(source, name):
    """A file, or every file called `name` below a directory."""
    source = Path(source)
    return sorted(source.rglob(name)) if source.is_dir() else [source]


def load(sources):
    """(workload, backend, row, bytes) -> {metric: median over runs}; first source wins per point."""
    groups = {}
    for source in sources:
        own = defaultdict(list)
        for path in source_files(source, "results.csv"):
            for r in csv.DictReader(open(path)):
                own[(r["workload"], r["backend"], r["row"], int(r["bytes"]))].append(r)
        for key, rows in own.items():
            groups.setdefault(key, rows)
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
    external KZG opening, the bb LegoSNARK estimate rebased on the nocom row,
    and Poseidon2 reduced to its faster variant."""
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
        # LegoSNARK proves the no-commitment circuit unchanged, and its rows were measured in a later
        # session than the nocom rows (the same `bb prove` drifts by several percent between sessions).
        # Use the nocom row's bb time plus the link prover (opening_s); the CSV keeps the measured sum.
        for n, m in rows.get("lego_estimate", {}).items():
            base = rows.get("nocom", {}).get(n)
            if base and m.get("opening_s") is not None:
                m["prover_s"] = base["prover_s"] + m["opening_s"]
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


def plot(data, out, panels, machine=None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(panels), figsize=(10 if len(panels) > 1 else 6.4, 4.4), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (backend, title) in zip(axes, panels):
        rows = series(data, backend)
        for row, (label, colour, marker, ls) in STYLE.items():
            pts = sorted(rows.get(row, {}).items())
            if not pts:
                continue
            xs = [n for n, _ in pts]
            ys = [max(m["prover_s"], 1e-3) for _, m in pts]
            # The LegoSNARK estimate lies within 3% of "No commitment": large markers drawn below
            # that line keep both visible.
            extra = {"markersize": 10, "zorder": 1.9} if row == "lego_estimate" else {"markersize": 5}
            ax.plot(xs, ys, color=colour, marker=marker, linestyle=ls, linewidth=2, label=label, **extra)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title + (f", {machine}" if machine else ""), fontsize=10)
        ax.set_xlabel("Committed payload size [bytes]")
        ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Prover time [s]")
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8,
               ncol=4 if len(panels) > 1 else 2, loc="lower center")
    fig.tight_layout(rect=(0, 0.11 if len(panels) > 1 else 0.17, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"sweep_prover.{ext}", dpi=200)


def tables(data, out, panels, layout, machine=None):
    lines = ["# TLSNotary / Artemis benchmark: summary" + (f" ({machine})" if machine else ""), "",
             "Medians over the runs in the CSV. Prover time excludes ACVM witness generation, "
             "key generation and setup; the Artemis rows include the IPA opening "
             "(ProveKit) or the KZG opening (Barretenberg estimate). The LegoSNARK row is the no-commitment "
             "row plus the CP_link prover on the wire columns of a real Barretenberg proof (same circuit; "
             "the CSV rows hold the sum with a `bb prove` run of their own session).", ""]
    for backend, title in panels:
        rows = series(data, backend)
        order = [r for r in ("nocom", "sha256", "keccak", "blake3", "poseidon2", "artemis", "artemis_estimate",
                             "lego_estimate") if r in rows]
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
    lines += lego_table(data, layout)
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


def load_layout(sources, out):
    """lego_layout.json (written by bench.py) next to a CSV source or below a directory source."""
    layout = {}
    for source in sources:
        source = Path(source)
        files = source_files(source, "lego_layout.json") if source.is_dir() else [source.parent / "lego_layout.json"]
        for path in files:
            if path.exists():
                for key, val in json.loads(path.read_text()).items():
                    layout.setdefault(key, val)
    return layout


def lego_table(data, layout):
    """Layout facts and link cost of the LegoSNARK rows."""
    lines = []
    for row, title in (("lego_estimate", "no-commitment circuit, byte cells linked (the plotted row)"),
                       ("lego_packed_estimate", "variant that packs in the circuit, packed cells linked")):
        sizes = sorted(int(k.split("/")[1]) for k in layout if k.split("/")[0] == row)
        if not sizes:
            continue
        lines += [f"## Barretenberg: LegoSNARK CP_link, {title}", "",
                  "Link prover, Pedersen commitment and link verifier are in-process medians. "
                  "Overhead columns: prover time over no commitment. LegoSNARK: the link prover (the circuit "
                  "is the no-commitment circuit) or, for the packed variant, the difference of the `bb prove` "
                  "medians of the two LegoSNARK circuits (measured back to back) plus the link prover. "
                  "Artemis: `artemis_estimate` + KZG opening - `nocom` (measured back to back).", "",
                  "| Payload | Gates | Linked wires | Non-zero cells in the MSM | of at most 16 bits | "
                  "Link prover [ms] | LegoSNARK overhead [ms] | Artemis estimate overhead [ms] | "
                  "Pedersen commit [ms] (not counted) | Link verifier [ms] | Setup [s] | Keys [MB] |",
                  "|---|" + "---|" * 11]
        for n in sizes:
            m = layout[f"{row}/{n}"]
            get = lambda r, b="bb": data.get(("sweep", b, r, n), {}).get("prover_s")  # noqa: E731
            nocom, art, ext = get("nocom"), get("artemis_estimate"), get("kzg", "external")
            link = data.get(("sweep", "bb", row, n), {}).get("opening_s")
            ref = data.get(("sweep", "bb", "lego_estimate", n))
            over = link
            if row != "lego_estimate":  # (bb + link) - (bb_ref + link_ref) + link_ref
                over = get(row) - ref["prover_s"] + ref["opening_s"] if ref else None
            art_over = art + ext - nocom if art and ext and nocom else None
            ms = lambda v: "n/a" if v is None else f"{v * 1e3:.1f}"  # noqa: E731
            lines.append(
                f"| {fmt_bytes(n)} | {m['gates']} | {', '.join(m['linked_wires'])} | {m['msm_cells']} | "
                f"{100 * m['msm_small_share']:.2f}% | {ms(link)} | {ms(over)} | {ms(art_over)} | "
                f"{ms(m['commit_s'])} | {ms(m['verify_s'])} | {m['setup_s']:.2f} | "
                f"{(m['key_p_bytes'] + m['key_pedersen_bytes']) / 1e6:.1f} |")
        lines.append("")
    return lines


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", default=[str(here / "results/results.csv")],
                    help="results.csv files or directories holding them; first source wins per point")
    ap.add_argument("--out", default=str(here / "results"))
    ap.add_argument("--backends", default="provekit,bb", help="panels to draw")
    ap.add_argument("--machine", default=None, help="machine name for the panel titles and the summary")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = [p for p in PANELS if p[0] in args.backends.split(",")]
    data = load(args.sources)
    plot(data, out, panels, args.machine)
    tables(data, out, panels, load_layout(args.sources, out), args.machine)
    print(f"wrote {out}/sweep_prover.pdf, sweep_prover.png, summary.md")


if __name__ == "__main__":
    main()
