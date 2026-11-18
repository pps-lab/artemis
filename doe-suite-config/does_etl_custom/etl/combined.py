"""Combined zkML + TLSNotary prover-time table (super-ETL `combined_plots.yml`).

The pipeline reads two kinds of result files:

- zkML suites (`output.csv`, one row per run, columns `model`, `cpsnark`, `pc_type`, `Prover time`),
- TLSNotary suites (`results.csv` in the format of `tlsn-bench/bench.py`, one row per repetition, columns
  `workload`, `backend`, `row`, `bytes`, `prover_s`, ...). Until a doe-suite design exists, the local results are
  imported with `tlsn-bench/to_doe_suite.py`.

`CombinedProverTimeTransformer` brings both into one long table (app, pc_type, cpsnark, x, prover_time_sec) and
`CombinedProverTableLoader` writes it as a LaTeX table (rows: CP-SNARK schemes, grouped by proof-system family;
column groups: one per application) in which every missing cell is explained: not supported by the proof system,
not implemented, or not run.
"""

import math
import re
import os
import statistics
from typing import Any, Dict, List, Optional

import pandas as pd

from doespy.etl.steps.loaders import Loader
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.extractors import Extractor

from .custom import parse_rust_duration


class TlsnCsvExtractor(Extractor):
    """`results.csv` of the TLSNotary benchmark. Its own `run` column (repetition index) is renamed to
    `tlsn_rep` so that it does not overwrite doe-suite's `run`."""

    file_regex: List[str] = [r"^results\.csv$"]

    def extract(self, path: str, options: Dict) -> List[Dict]:
        if os.path.getsize(path) == 0:   # a run that died before its first measurement (e.g. out of memory)
            return []
        df = pd.read_csv(path)
        df = df.rename(columns={"run": "tlsn_rep"})
        df["app"] = "tlsn"
        return df.to_dict(orient="records")


class RuntimeInfoExtractor(Extractor):
    """`runtime_info.yaml` written by `/usr/bin/time -f '{wall_time: %E, max_rss: %M}'`. A run killed by a signal
    prepends "Command terminated by signal 9", which is not valid YAML, so `max_rss` (KiB) is read with a regex."""

    file_regex: List[str] = [r"^runtime_info\.yaml$"]

    def extract(self, path: str, options: Dict) -> List[Dict]:
        text = open(path).read()
        m = re.search(r"max_rss:\s*(\d+)", text)
        if m is None:
            return []
        return [{"max_rss": int(m.group(1)), "killed": "non-zero status" in text or "terminated by signal" in text}]


class CombinedProverTimeTransformer(Transformer):
    """zkML and TLSNotary rows -> columns `app, pc_type, cpsnark, x, prover_time_sec, estimate, n, variant`.

    - zkML: `x` is the model; median over runs if a configuration appears more than once.
    - TLSNotary: `x` is the payload size in bytes (as a string), only rows of `tlsn_workload`. The backend
      is mapped to the proof-system family (`tlsn_backend_pc`), circuit rows to schemes (`tlsn_rows`, a list
      of rows or a dict backend -> rows; Poseidon2 is fixed per backend: TACEO's width-16 sponge on ProveKit,
      the native width-4 permutation on Barretenberg; with several rows the faster median wins), and the Barretenberg Artemis
      estimate is completed with the external KZG opening (`tlsn_add_external`).
    - LegoSNARK (`tlsn_rebase`): the circuit is the no-commitment circuit, so the value is the `nocom` median of
      the table's TLSNotary suite plus the median link prover time (`opening_s`) of the LegoSNARK rows, as in
      `tlsn-bench/plot.py`. `tlsn_suite_rows` restricts a suite to the listed circuit rows, so that the
      reference rows a LegoSNARK suite measures for itself (`nocom`, ...) are not pooled with the main suite.
    """

    zkml_pc_map: Dict[str, str] = {"kzg_bls": "kzg"}

    tlsn_workload: str = "sweep"
    tlsn_backend_pc: Dict[str, str] = {"bb": "kzg", "provekit": "ipa"}
    tlsn_rows: Dict[str, Any] = {
        "no_com": ["nocom"],
        "poly": ["artemis", "artemis_estimate"],
        "poseidon": {"provekit": ["poseidon2_t16"], "bb": ["poseidon2_native"]},
        "sha256": ["sha256"],
        "keccak": ["keccak256", "keccak256_bb"],
        "blake3": ["blake3", "blake3_bb"],
        "cp_link_lego": ["lego_estimate"],
    }
    tlsn_estimate_rows: List[str] = ["artemis_estimate", "lego_estimate"]
    # circuit row -> external (pcs_bench) row whose prover time is added
    tlsn_add_external: Dict[str, str] = {"artemis_estimate": "kzg"}
    # circuit row -> row whose prover time replaces the row's own proof time; the row's `opening_s` is added
    tlsn_rebase: Dict[str, str] = {"lego_estimate": "nocom"}
    # suite name -> circuit rows taken from that suite (suites not listed contribute all their rows)
    tlsn_suite_rows: Dict[str, List[str]] = {}
    # TLSNotary payload sizes (bytes) to keep; empty keeps all
    tlsn_sizes: List[int] = []

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        out = []
        is_tlsn = df["source_file"] == "results.csv"
        out.append(self._zkml(df[~is_tlsn & (df["source_file"] == "output.csv")],
                              df[df["source_file"] == "runtime_info.yaml"]))
        tl = df[is_tlsn]
        if self.tlsn_sizes and len(tl) > 0:
            tl = tl[pd.to_numeric(tl["bytes"], errors="coerce").isin(self.tlsn_sizes)]
        out.append(self._tlsn(tl))
        res = pd.concat([o for o in out if o is not None and len(o) > 0], ignore_index=True)
        return res

    def _zkml(self, df: pd.DataFrame, rt: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        if len(df) == 0:
            return None
        df = df[df["Prover time"].notna()].copy()
        # peak memory of the run (`/usr/bin/time` max_rss in KiB, runtime_info.yaml of the same run)
        df["max_rss_bytes"] = float("nan")
        if rt is not None and len(rt) > 0 and "max_rss" in rt:
            keys = [k for k in ("suite_name", "suite_id", "exp_name", "run", "rep", "host_type", "host_idx")
                    if k in df and k in rt]
            rss = rt.dropna(subset=["max_rss"]).drop_duplicates(subset=keys).set_index(keys)["max_rss"]
            idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else df[keys[0]]
            df["max_rss_bytes"] = pd.to_numeric(rss.reindex(idx).values, errors="coerce") * 1024
        df["pc_type"] = df["pc_type"].replace(self.zkml_pc_map)
        df["prover_time_sec"] = df["Prover time"].apply(parse_rust_duration)
        df["verifier_time_sec"] = df["Verifier time"].apply(_mean_duration) if "Verifier time" in df else float("nan")
        df["proof_size_bytes"] = pd.to_numeric(df["Proof size"], errors="coerce") if "Proof size" in df else float("nan")
        g = df.groupby(["pc_type", "cpsnark", "model"])
        res = g[METRICS].median()
        res["n"] = g.size()
        res = res.reset_index().rename(columns={"model": "x"})
        res["app"] = "zkml"
        res["estimate"] = False
        res["variant"] = res["cpsnark"]
        return res

    def _tlsn(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if len(df) == 0:
            return None
        df = df[df["workload"] == self.tlsn_workload].copy()
        df["bytes"] = df["bytes"].astype(int)
        cols = ["prover_s", "verifier_s", "proof_bytes", "max_rss_bytes", "opening_s"]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else float("nan")
        for suite, keep in self.tlsn_suite_rows.items():
            df = df[(df["suite_name"] != suite) | df["row"].isin(keep)]
        g = df.groupby(["backend", "row", "bytes"])
        med = g[cols].median()
        med["n"] = g.size()
        med = med.to_dict(orient="index")   # (backend, row, bytes) -> {metric: median}

        def ok(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v))

        res = []
        for backend, pc in self.tlsn_backend_pc.items():
            for scheme, variants in self.tlsn_rows.items():
                if isinstance(variants, dict):
                    variants = variants.get(backend, [])
                sizes = sorted({b for (be, r, b), m in med.items()
                                if be == backend and r in variants and ok(m["prover_s"])})
                for b in sizes:
                    cands = []
                    for v in variants:
                        m = med.get((backend, v, b))
                        if m is None or not ok(m["prover_s"]):
                            continue
                        m = dict(m)
                        ext = self.tlsn_add_external.get(v)
                        if ext is not None:
                            e = med.get(("external", ext, b))
                            if e is None:
                                continue  # an incomplete estimate is not shown
                            m["prover_s"] += e["prover_s"]
                            m["verifier_s"] += e["verifier_s"]
                            m["proof_bytes"] += e["proof_bytes"]
                            if ok(e["max_rss_bytes"]):
                                m["max_rss_bytes"] = max(m["max_rss_bytes"], e["max_rss_bytes"])
                        base = self.tlsn_rebase.get(v)
                        if base is not None:
                            # prover: the base row's proof plus the link prover; the other metrics are the
                            # row's own (proof + link proof, verify + link verify, max of both processes)
                            mb = med.get((backend, base, b))
                            if mb is None or not ok(m["opening_s"]):
                                continue
                            m["prover_s"] = mb["prover_s"] + m["opening_s"]
                        cands.append((m["prover_s"], v, m))
                    if not cands:
                        continue
                    _, v, m = min(cands, key=lambda c: c[0])
                    res.append({
                        "app": "tlsn", "pc_type": pc, "cpsnark": scheme, "x": str(b),
                        "prover_time_sec": m["prover_s"], "verifier_time_sec": m["verifier_s"],
                        "proof_size_bytes": m["proof_bytes"], "max_rss_bytes": m["max_rss_bytes"],
                        "n": m["n"], "estimate": v in self.tlsn_estimate_rows, "variant": f"{backend}:{v}",
                    })
        return pd.DataFrame(res)


def _mean_duration(value) -> float:
    """Mean of a list of Rust durations as printed by time_circuit ("[6.4ms, 6.2ms]"), or a single duration."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = [p.strip() for p in str(value).strip().strip("[]").split(",") if p.strip()]
    if not parts:
        return float("nan")
    return statistics.mean(parse_rust_duration(p) for p in parts)


METRICS = ["prover_time_sec", "verifier_time_sec", "proof_size_bytes", "max_rss_bytes"]


def _fmt_time(x: float) -> str:
    """Seconds with about three significant digits, whole seconds from 100 s (thin-space grouped from 10^4)."""
    if x >= 10000:
        return f"{x:,.0f}".replace(",", "\\,")
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 0.1:
        return f"{x:.2f}"
    if x >= 0.01:
        return f"{x:.3f}"
    return "$<$0.01"


def _printed(x: float) -> float:
    """The value as printed by `_fmt_time`."""
    s = _fmt_time(x)
    return 0.0 if s.startswith("$<$") else float(s.replace("\\,", ""))


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _rule_note(rule):
    """Footnote of a missing-cell rule: `note`, else `text` with its line breaks removed."""
    note = rule.get("note") or rule.get("text") or ""
    return " ".join(note.replace("\\\\", " ").split())


class CombinedProverTableLoader(Loader):
    """LaTeX prover-time table over applications (`apps`, each with its columns `x`), proof-system families
    (`blocks`) and schemes (`rows`).

    - `transpose: false`: row groups = families, rows = schemes; column groups = applications, columns = x.
    - `transpose: true`: row groups = applications, rows = x; column groups = families, columns = schemes.

    Each row group starts with a line naming the proof system of every (application, family) pair
    (`apps[].systems`). Cells without a measurement are resolved with the first matching rule of `missing`
    (keys `cpsnark`, `pc_type`, `app`, `x`: lists, all optional; `kind`: `unsupported` | `not_implemented` |
    `not_run`; `note` (or `text`): footnote below the table; `merge: false`: marker in every cell). Otherwise
    adjacent cells of the same rule within one group are merged into one rectangle with a single marker. A cell with no rule is `--` (not run). Values of estimate rows get a
    dagger. With `bold_best`, the fastest scheme other than `no_com` is bold per (application, family, x).

    `metrics` ([{key, label, scale}], optional): stack one block of row groups per metric (column `key` of the
    transformer output divided by `scale`), each headed by its `label`; rules of `missing` may then also match on
    `metric`. Without it, the table shows `prover_time_sec`. Bold marks the smallest value per metric (ties at the
    printed precision are all bold); a metric with `bold: false` has none.

    Writes `<file_name>.tex`, `<file_name>.csv` (cell matrix), `<file_name>_long.csv` and a standalone
    `<file_name>_preview.tex`.
    """

    file_name: str = "prover_time_table"
    caption: str = ""
    label: str = "tab:eval:prover"
    transpose: bool = False
    placement: str = "t"
    bold_best: bool = True
    mark_estimates: bool = False   # dagger on estimate rows (e.g. the bb Artemis estimate)
    tabcolsep_pt: float = 3.0
    font_size: str = r"\small"
    group_gap_pt: float = 8.0

    # [{key, label, columns: [{key, label, sublabel?}], systems: {pc_type: label}}]
    apps: List[Dict] = []
    # [{key, label}]
    blocks: List[Dict] = [{"key": "kzg", "label": "Pairing-based"}, {"key": "ipa", "label": "Transparent"}]
    # [{key, label}]
    rows: List[Dict] = []
    missing: List[Dict] = []
    # [{key, label, scale?, bold?}]
    metrics: List[Dict] = []
    bold_legend: str = r"\textbf{bold}: fastest scheme with a commitment"

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        output_dir = self.get_output_dir(etl_info)
        os.makedirs(output_dir, exist_ok=True)

        metrics = self.metrics or [{"key": "prover_time_sec", "label": None}]
        vals = {}
        for r in df.to_dict(orient="records"):
            for m in metrics:
                v = r.get(m["key"])
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                vals[(m["key"], r["app"], r["pc_type"], r["cpsnark"], str(r["x"]))] = (
                    v / m.get("scale", 1), bool(r["estimate"]))

        # --- grid definition: groups of lines, each line/column carries a partial cell key
        apps = {a["key"]: a for a in self.apps}
        if self.transpose:
            row_groups = [{"key": {"app": a["key"]}, "label": a["label"],
                           "lines": [{"part": {"x": str(c["key"])}, "label": c["label"], "sublabel": c.get("sublabel")}
                                     for c in a["columns"]]} for a in self.apps]
            col_groups = [{"key": {"pc": b["key"]}, "label": b["label"],
                           "cols": [{"part": {"cpsnark": r["key"]}, "label": r["label"]} for r in self.rows]}
                          for b in self.blocks]
        else:
            row_groups = [{"key": {"pc": b["key"]}, "label": b["label"],
                           "lines": [{"part": {"cpsnark": r["key"]}, "label": r["label"]} for r in self.rows]}
                          for b in self.blocks]
            col_groups = [{"key": {"app": a["key"]}, "label": a["label"],
                           "cols": [{"part": {"x": str(c["key"])}, "label": c["label"], "sublabel": c.get("sublabel")}
                                    for c in a["columns"]]} for a in self.apps]

        # one block of row groups per metric; the first group of a block carries the metric heading
        row_groups = [{**rg, "key": {**rg["key"], "metric": m["key"]}, "metric": m,
                       "heading": m.get("label") if i == 0 else None}
                      for m in metrics for i, rg in enumerate(row_groups)]

        def cell_key(rg, ln, cg, col):
            d = {**rg["key"], **cg["key"], **ln["part"], **col["part"]}
            return d["metric"], d["app"], d["pc"], d["cpsnark"], d["x"]

        def system_label(rg, cg):
            d = {**rg["key"], **cg["key"]}
            return apps[d["app"]]["systems"].get(d["pc"], "")

        notes = []

        def note_mark(note):
            if not note:
                return ""
            if note not in notes:
                notes.append(note)
            return f"$^{{{chr(ord('a') + notes.index(note))}}}$"

        def rule_for(metric, app, pc, scheme, x):
            for i, rule in enumerate(self.missing):
                if all(key not in rule or val in [str(v) for v in rule[key]]
                       for key, val in (("metric", metric), ("app", app), ("pc_type", pc), ("cpsnark", scheme),
                                        ("x", x))):
                    return i, rule
            return None, None

        best = {}
        if self.bold_best:
            shown = {r["key"] for r in self.rows}
            cnt = {}
            no_bold = {m["key"] for m in metrics if m.get("bold") is False}
            for (metric, app, pc, scheme, x), (t, _) in vals.items():
                if scheme == "no_com" or scheme not in shown or metric in no_bold:
                    continue
                t = _printed(t)   # ties at the printed precision are all bold
                k = (metric, app, pc, x)
                cnt.setdefault(k, []).append(t)
                if k not in best or t < best[k]:
                    best[k] = t
            # no bold when every scheme with a commitment ties (e.g. a constant proof size)
            best = {k: t for k, t in best.items() if len(cnt[k]) == 1 or any(v != t for v in cnt[k])}

        # --- header
        n_cols = sum(len(cg["cols"]) for cg in col_groups)
        has_row_sub = any(ln.get("sublabel") for rg in row_groups for ln in rg["lines"])
        has_col_sub = any(c.get("sublabel") for cg in col_groups for c in cg["cols"])
        n_lab = 2 if has_row_sub else 1
        gap = f"@{{\\hspace{{{self.group_gap_pt}pt}}}}"
        col_spec = "l" * n_lab + "".join(gap + "r" * len(cg["cols"]) for cg in col_groups)

        lines = []
        hdr, cmid, c = [""] * n_lab, [], n_lab + 1
        for cg in col_groups:
            k = len(cg["cols"])
            hdr.append(f"\\multicolumn{{{k}}}{{c}}{{{cg['label']}}}")
            cmid.append(f"\\cmidrule(lr){{{c}-{c + k - 1}}}")
            c += k
        lines.append(" & ".join(hdr) + r" \\")
        lines.append(" ".join(cmid))
        lines.append(" & ".join([""] * n_lab + [col["label"] for cg in col_groups for col in cg["cols"]]) + r" \\")
        if has_col_sub:
            lines.append(" & ".join([""] * n_lab + [f"{{\\scriptsize {col['sublabel']}}}" if col.get("sublabel") else ""
                                                   for cg in col_groups for col in cg["cols"]]) + r" \\")

        used = set()
        csv_rows = []
        marked = set()
        for rg in row_groups:
            lines.append(r"\midrule")
            if rg.get("heading"):
                lines.append(f"\\multicolumn{{{n_lab + n_cols}}}{{l}}{{\\textbf{{{rg['heading']}}}}} \\\\")
            sys_cells = [f"\\multicolumn{{{n_lab}}}{{l}}{{\\textit{{{rg['label']}}}}}"]
            for cg in col_groups:
                sys_cells.append(f"\\multicolumn{{{len(cg['cols'])}}}{{c}}{{\\textit{{{system_label(rg, cg)}}}}}")
            lines.append(" & ".join(sys_cells) + r" \\")
            lines.append(f"\\cmidrule(lr){{{n_lab + 1}-{n_lab + n_cols}}}")

            # resolve the cells of this row group: grid[i][j] for line i, flat column j
            flat = [(gi, cg, col) for gi, cg in enumerate(col_groups) for col in cg["cols"]]
            grid = []
            for ln in rg["lines"]:
                row_cells = []
                csv_row = {"metric": rg["metric"]["key"], "group": rg["label"], "line": ln["label"]}
                for gi, cg, col in flat:
                    metric, app, pc, scheme, x = cell_key(rg, ln, cg, col)
                    v = vals.get((metric, app, pc, scheme, x))
                    name = f"{cg['label']}:{col['label']}"
                    if v is not None and not math.isnan(v[0]):
                        t, est = v
                        s = _fmt_time(t)
                        if self.bold_best and scheme != "no_com" and best.get((metric, app, pc, x)) == _printed(t):
                            s = f"\\textbf{{{s}}}"
                        if est and self.mark_estimates:
                            s += r"$^\dagger$"
                            used.add("estimate")
                        row_cells.append({"kind": "val", "text": s})
                        csv_row[name] = f"{t:.6g}" + (" (estimate)" if est else "")
                    else:
                        ri, rule = rule_for(metric, app, pc, scheme, x)
                        kind = rule.get("kind", "not_run") if rule else "not_run"
                        row_cells.append({"kind": kind, "ri": ri, "rule": rule, "cg": gi})
                        csv_row[name] = kind
                grid.append(row_cells)
                csv_rows.append(csv_row)

            # rectangles of the same text rule within one column group: extend right, then down
            def mergeable(cell, ri, gi):
                return (cell["kind"] not in ("val", "not_run") and cell.get("ri") == ri and cell.get("cg") == gi
                        and "rect" not in cell)

            for i, row_cells in enumerate(grid):
                for j, cell in enumerate(row_cells):
                    if (cell["kind"] in ("val", "not_run") or "rect" in cell or not cell["rule"]
                            or cell["rule"].get("merge", True) is False):
                        continue
                    ri, gi = cell["ri"], cell["cg"]
                    w = 1
                    while j + w < len(row_cells) and mergeable(row_cells[j + w], ri, gi):
                        w += 1
                    h = 1
                    while i + h < len(grid) and all(mergeable(grid[i + h][j + d], ri, gi) for d in range(w)):
                        h += 1
                    for di in range(h):
                        for dj in range(w):
                            grid[i + di][j + dj]["rect"] = (i, j, w, h)

            # render
            for i, (ln, row_cells) in enumerate(zip(rg["lines"], grid)):
                cells = [ln["label"]] + ([f"{{\\scriptsize {ln.get('sublabel') or ''}}}"] if has_row_sub else [])
                j = 0
                while j < len(row_cells):
                    cell = row_cells[j]
                    kind = cell["kind"]
                    if kind == "val":
                        cells.append(cell["text"])
                        j += 1
                        continue
                    used.add(kind)
                    rule = cell["rule"]
                    mark = note_mark(_rule_note(rule)) if rule else ""
                    if rule and rule.get("merge", True) is False:
                        # unmerged rule: marker in every cell, footnote mark only on its first cell in the group
                        mark = "" if (id(rg), cell["ri"]) in marked else mark
                        marked.add((id(rg), cell["ri"]))
                    fill = r"\cpFill" if kind == "unsupported" else ""
                    if "rect" not in cell:
                        short = {"unsupported": r"\cpUnsupportedShort",
                                 "not_implemented": r"\cpNotImplShort"}.get(kind, r"\cpNotRun")
                        cells.append(f"\\multicolumn{{1}}{{c}}{{{fill}{short}{mark}}}")
                        j += 1
                        continue
                    i0, j0, w, h = cell["rect"]
                    short = r"\cpUnsupportedShort" if kind == "unsupported" else r"\cpNotImplShort"
                    inner = ""
                    if i == i0 + h - 1:
                        body = f"{short}{mark}"
                        # multirow sits in the last row, so the cell colour of later rows does not hide it
                        inner = body if h == 1 else f"\\multirow{{-{h}}}{{*}}{{{body}}}"
                    cells.append(f"\\multicolumn{{{w}}}{{c}}{{{fill}{inner}}}")
                    j += w
                lines.append(" & ".join(cells) + r" \\")

        legend = []
        if "unsupported" in used or "not_implemented" in used:
            legend.append(r"$\bot$: the scheme does not support the proof system")
        if "not_run" in used:
            legend.append(r"\cpNotRun{}: not run")
        if self.bold_best:
            legend.append(self.bold_legend)
        if "estimate" in used:
            legend.append(r"$^\dagger$: cost estimate")
        note_lines = [f"$^{{{chr(ord('a') + i)}}}${n.rstrip('.')}" for i, n in enumerate(notes)]

        macros = r"""% Missing-cell markers (\providecommand: the paper may redefine them). Shading needs colortbl
% (\usepackage{colortbl} or \usepackage[table]{xcolor}); without it, cells are not shaded. Needs multirow.
\providecommand{\cpFill}{\ifdefined\cellcolor\cellcolor{gray!15}\fi}
\providecommand{\cpFillLegend}{\ifdefined\cellcolor\colorbox{gray!15}{\strut\itshape shaded}\else\textit{grey text}\fi}
\providecommand{\cpUnsupported}[1]{{\color{black!65}\footnotesize\itshape #1}}
\providecommand{\cpUnsupportedShort}{{\color{black!65}$\bot$}}
\providecommand{\cpNotImpl}[1]{{\color{gray}\footnotesize\itshape #1}}
\providecommand{\cpNotImplShort}{{\color{black!65}$\bot$}}
\providecommand{\cpNotRun}{{\color{gray}--}}
"""
        body = "\n".join(lines)
        foot = "; ".join(legend) + "."
        if note_lines:
            foot += r" \\ " + ". ".join(note_lines) + "."
        table = f"""% generated by CombinedProverTableLoader (doe-suite-config/does_etl_custom/etl/combined.py)
{macros}\\begin{{table*}}[{self.placement}]
\\centering
{self.font_size}
\\setlength{{\\tabcolsep}}{{{self.tabcolsep_pt}pt}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.3em}}
\\parbox{{\\textwidth}}{{\\footnotesize {foot}}}
\\caption{{{self.caption}}}
\\label{{{self.label}}}
\\end{{table*}}
"""
        with open(os.path.join(output_dir, f"{self.file_name}.tex"), "w") as f:
            f.write(table)

        preview = (
            "\\documentclass{article}\n\\usepackage[textwidth=7in,textheight=9in]{geometry}\n"
            "\\usepackage{booktabs,multirow,colortbl,xcolor}\n\\begin{document}\n"
            f"\\input{{{self.file_name}.tex}}\n\\end{{document}}\n"
        )
        with open(os.path.join(output_dir, f"{self.file_name}_preview.tex"), "w") as f:
            f.write(preview)

        pd.DataFrame(csv_rows).to_csv(os.path.join(output_dir, f"{self.file_name}.csv"), index=False)
        df.to_csv(os.path.join(output_dir, f"{self.file_name}_long.csv"), index=False)


def _fmt_matrix(x: float, style: str) -> str:
    """Number formats of the original hand-written results table (tab:results); `sig` for small values."""
    if style == "int":
        return f"{x:.0f}"
    if style == "k":   # verifier ms: integers, thousands as 3.5k
        return f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"
    if style == "sig":   # about two significant digits, thousands as 3.8k
        if x >= 1000:
            return f"{x / 1000:.1f}k"
        if x >= 10:
            return f"{x:.0f}"
        if x >= 1:
            return f"{x:.1f}"
        if x >= 0.01:
            return f"{x:.2f}"
        return "$<$0.01"
    # "mem": two decimals below 100, integers above
    return f"{x:.0f}" if x >= 100 else f"{x:.2f}"


class ResultsMatrixLoader(Loader):
    """The original layout of the results table (tab:results): column groups = metrics, columns = the x values of a
    section (models for zkML, use cases for TLSNotary); per section, row groups = proof-system families (rotated
    label), rows = schemes. Each section starts with its name and its own column labels; a section with fewer
    columns leaves the remaining ones empty. Missing cells are `-` and are explained in the caption."""

    file_name: str = "results_table"
    caption: str = ""
    label: str = "tab:results"
    placement: str = "t"
    tabcolsep_pt: float = 2.0
    # [{app, label, columns: [{key, label}], blocks: [{key, label}], fmt?: {metric: style}}]
    sections: List[Dict] = []
    # [{key, label}]
    rows: List[Dict] = []
    # [{key, label, scale, fmt: int | k | mem | sig}]
    metrics: List[Dict] = []

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        output_dir = self.get_output_dir(etl_info)
        os.makedirs(output_dir, exist_ok=True)
        vals = {(r["app"], r["pc_type"], r["cpsnark"], str(r["x"])): r for r in df.to_dict(orient="records")}

        n = max(len(sec["columns"]) for sec in self.sections)
        spec = "ll" + "".join(f"*{{{n}}}{{c}}" for _ in self.metrics)
        hdr = ["", ""] + [f"\\multicolumn{{{n}}}{{c}}{{{m['label']}}}" for m in self.metrics]
        cmid = " ".join(f"\\cmidrule(lr){{{3 + i * n}-{2 + (i + 1) * n}}}" for i in range(len(self.metrics)))
        lines = [" & ".join(hdr) + r" \\", cmid]
        csv_rows = []
        for si, sec in enumerate(self.sections):
            cols = sec["columns"] + [None] * (n - len(sec["columns"]))
            if si > 0:
                lines.append(r"\midrule")
            sub = [f"\\multicolumn{{2}}{{l}}{{\\textit{{{sec['label']}}}}}"] + \
                  [c["label"] if c else "" for _ in self.metrics for c in cols]
            lines.append(" & ".join(sub) + r" \\")
            for b in sec["blocks"]:
                lines.append(r"\midrule")
                for i, r in enumerate(self.rows):
                    first = f"\\multirow{{{len(self.rows)}}}{{*}}{{\\rotatebox{{90}}{{{b['label']}}}}}" if i == 0 else ""
                    cells = [first, r["label"]]
                    csv_row = {"app": sec["app"], "pc_type": b["key"], "cpsnark": r["key"]}
                    for m in self.metrics:
                        style = sec.get("fmt", {}).get(m["key"], m.get("fmt", "mem"))
                        for c in cols:
                            if c is None:
                                cells.append("")
                                continue
                            v = vals.get((sec["app"], b["key"], r["key"], str(c["key"])), {}).get(m["key"])
                            if v is None or (isinstance(v, float) and math.isnan(v)):
                                cells.append("-")
                            else:
                                cells.append(_fmt_matrix(v / m.get("scale", 1), style))
                                csv_row[f"{m['key']}:{c['key']}"] = v
                    lines.append(" & ".join(cells) + r" \\")
                    csv_rows.append(csv_row)
        body = "\n".join(lines)
        table = f"""% generated by ResultsMatrixLoader (doe-suite-config/does_etl_custom/etl/combined.py)
\\begin{{table*}}[{self.placement}]
\\centering
\\setlength{{\\tabcolsep}}{{{self.tabcolsep_pt}pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{spec}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\caption{{{self.caption}}}
\\label{{{self.label}}}
\\end{{table*}}
"""
        with open(os.path.join(output_dir, f"{self.file_name}.tex"), "w") as f:
            f.write(table)
        pd.DataFrame(csv_rows).to_csv(os.path.join(output_dir, f"{self.file_name}.csv"), index=False)
