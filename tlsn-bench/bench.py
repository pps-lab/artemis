#!/usr/bin/env python3
"""Sweep driver for the TLSNotary / Artemis benchmark (see README.md).

Each measurement is one CSV row. ProveKit rows are timed in-process by
`artemis-bench`; Barretenberg rows are the wall-clock time of the `bb` CLI,
with the median wall-clock time of the same command on an empty circuit
subtracted (both values are kept in the CSV).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CIRCUITS = ROOT / "circuits"
BUILD = ROOT / ".build"
ARTEMIS_BENCH = ROOT / "provekit/target/release/artemis-bench"
PCS_BENCH = ROOT / "pcs/target/release/pcs_bench"
DIGEST = ROOT / "pcs/target/release/digest"
NARGO = os.environ.get("NARGO", "nargo")
BB = os.environ.get("BB", "bb")

# BN254 scalar field modulus.
R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
PACK = 31

# Payload sizes in bytes: tlsn-extension plugins (maxRecvData).
SIZES = {
    520: "swissbank",
    2400: "duolingo/spotify",
    4000: "twitter",
    10000: "discord_dm",
    16384: "uber",
    65536: "idme",
}

# row -> (circuit template, artemis link on ProveKit)
SWEEP_ROWS = {
    "nocom": ("nocom", False),
    "sha256": ("sha256", False),
    # Keccak-256 / BLAKE3: explicit Noir code (ProveKit has neither black box)
    # and, for Barretenberg, also the black-box variants (`_bb`).
    "keccak256": ("keccak256", False),
    "keccak256_bb": ("keccak256_bb", False),
    "blake3": ("blake3", False),
    "blake3_bb": ("blake3_bb", False),
    "poseidon2_t16": ("poseidon2_t16", False),
    "poseidon2_native": ("poseidon2_native", False),
    "artemis": ("artemis", True),
    "artemis_estimate": ("artemis_estimate", False),
}
BACKEND_ROWS = {
    "provekit": ["nocom", "sha256", "keccak256", "blake3", "poseidon2_t16", "poseidon2_native", "artemis"],
    # Barretenberg cannot expose a challenge, so the Artemis row is the
    # (unsound) in-circuit Horner cost estimate; add an `external` opening.
    "bb": ["nocom", "sha256", "keccak256", "keccak256_bb", "blake3", "blake3_bb", "poseidon2_t16", "poseidon2_native", "artemis_estimate"],
}
# Largest payload that fits a 48 GB machine (prover RAM extrapolated from the
# smaller sizes: ~50 GiB and ~42 GiB at 64 KiB). `--all-sizes` lifts the caps.
LOCAL_CAP = {("provekit", "keccak256"): 16384, ("bb", "keccak256"): 16384}
AGE_ROWS = {
    "provekit": [("age_nocom", False), ("age_sha256", False), ("age_keccak256", False),
                 ("age_blake3", False), ("age_poseidon2", False), ("age_artemis", True)],
    "bb": [("age_nocom", False), ("age_sha256", False), ("age_keccak256", False), ("age_keccak256_bb", False),
           ("age_blake3", False), ("age_blake3_bb", False), ("age_poseidon2", False)],
}

FIELDS = [
    "workload", "backend", "row", "bytes", "run", "constraints", "prover_s", "verifier_s",
    "proof_bytes", "max_rss_bytes", "witgen_s", "commit_s", "opening_s", "prover_raw_s",
    "verifier_raw_s", "baseline_prove_s", "baseline_verify_s", "version", "note",
]


def sh(cmd, cwd=None):
    """Run a command and return its stdout."""
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{proc.stderr[-2000:]}")
    return proc.stdout


def timed(cmd):
    """Run a command; return (wall seconds, the child's peak RSS in bytes)."""
    start = time.perf_counter()
    pid = os.posix_spawn(
        str(cmd[0]), [str(c) for c in cmd], os.environ,
        file_actions=[(os.POSIX_SPAWN_OPEN, 1, os.devnull, os.O_WRONLY, 0),
                      (os.POSIX_SPAWN_OPEN, 2, os.devnull, os.O_WRONLY, 0)],
    )
    _, status, usage = os.wait4(pid, 0)
    wall = time.perf_counter() - start
    if status != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} exited with status {status}")
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return wall, rss


def payload_bytes(n):
    rng = random.Random(n)
    return [rng.randrange(256) for _ in range(n)]


def digest(alg, data):
    """Keccak-256 / BLAKE3 of `data` as a list of bytes (pcs `digest` helper)."""
    out = subprocess.run([str(DIGEST), alg], input=bytes(data), capture_output=True, check=True)
    return list(bytes.fromhex(out.stdout.decode().strip()))


def hash_alg(template):
    for alg in ("keccak256", "blake3"):
        if alg in template:
            return alg
    return None


def horner(payload, mask, z):
    """z * (sum_i u_i z^i + mask z^k), u = payload packed 31 bytes LE."""
    chunks = [int.from_bytes(bytes(payload[i:i + PACK]), "little") for i in range(0, len(payload), PACK)]
    acc = mask
    for u in reversed(chunks):
        acc = (acc * z + u) % R
    return acc * z % R


def prover_toml(template, n):
    payload = payload_bytes(n)
    mask = 0x1234567890ABCDEF
    lines = []
    if template == "sha256":
        lines.append(f"committed_hash = {list(hashlib.sha256(bytes(payload)).digest())}")
    if hash_alg(template):
        lines.append(f"committed_hash = {digest(hash_alg(template), payload)}")
    if template == "artemis_estimate":
        z = 0x2A2A2A2A2A2A2A2A2A2A2A2A2A2A2A2A
        lines.append(f'eval_point = "{z}"')
        lines.append(f'claimed_eval = "{horner(payload, mask, z)}"')
    lines.append(f"payload = {payload}")
    if template in ("artemis", "artemis_estimate"):
        lines.append(f'mask = "{mask}"')
    return "\n".join(lines) + "\n"


def age_toml(template):
    dob = list(b"1985-03-12")
    blinder = [120, 80, 62, 10, 76, 60, 130, 98, 147, 161, 139, 126, 27, 236, 36, 56]
    lines = [f"payload = {dob}"]
    if template == "age_sha256":
        lines.append(f"blinder = {blinder}")
        lines.append(f"committed_hash = {list(hashlib.sha256(bytes(dob + blinder)).digest())}")
    if hash_alg(template):
        lines.append(f"blinder = {blinder}")
        lines.append(f"committed_hash = {digest(hash_alg(template), dob + blinder)}")
    if template == "age_poseidon2":
        lines.append('blinder = "0x0f1e2d3c4b5a6978"')
    if template == "age_artemis":
        lines.append('mask = "0x0f1e2d3c4b5a6978"')
    lines.append("\n[proof_date]\nyear = 2025\nmonth = 9\nday = 2")
    return "\n".join(lines) + "\n"


def instantiate(template, n, toml):
    """Copy a circuit template to .build/, set N, write Prover.toml."""
    name = template if n is None else f"{template}_{n}"
    dst = BUILD / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "src").mkdir(parents=True)
    nargo = (CIRCUITS / template / "Nargo.toml").read_text()
    nargo = nargo.replace('"../vendor/', f'"{CIRCUITS}/vendor/')
    (dst / "Nargo.toml").write_text(nargo)
    src = (CIRCUITS / template / "src/main.nr").read_text()
    if n is not None:
        assert "global N: u32 = 1024;" in src, template
        src = src.replace("global N: u32 = 1024;", f"global N: u32 = {n};")
    (dst / "src/main.nr").write_text(src)
    (dst / "Prover.toml").write_text(toml)
    return dst


def versions():
    nargo = subprocess.run([NARGO, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    bb = subprocess.run([BB, "--version"], capture_output=True, text=True).stdout.strip()
    pk = subprocess.run(["git", "-C", str(ROOT / "provekit"), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
    return {"bb": f"bb {bb}; {nargo}", "provekit": f"provekit {pk} + artemis patch; noir v1.0.0-beta.26"}


def run_provekit(dst, link, runs):
    keys = dst / "keys"
    args = [ARTEMIS_BENCH, "prepare", "--circuit", dst, "--out", keys]
    out = sh(args + (["--artemis"] if link else []))
    info = json.loads(out.strip().splitlines()[-1])
    rows = []
    for run in range(runs):
        args = [ARTEMIS_BENCH, "prove", "--keys", keys, "--input", dst / "Prover.toml"]
        out = sh(args + (["--artemis"] if link else []))
        m = json.loads(out.strip().splitlines()[-1])
        rows.append({
            "run": run, "constraints": info["constraints"], "prover_s": m["prover_s"],
            "verifier_s": m["verifier_s"], "proof_bytes": m["proof_bytes"],
            "max_rss_bytes": m["max_rss_bytes"], "witgen_s": m["witgen_s"],
            "commit_s": m.get("commit_s", ""), "opening_s": m.get("opening_s", ""),
        })
    return rows


def bb_once(dst, name):
    target = dst / "target"
    prove, rss = timed([shutil.which(BB), "prove", "-b", target / f"{name}.json", "-w",
                        target / f"{name}.gz", "-k", target / "vk", "-o", target])
    verify, _ = timed([shutil.which(BB), "verify", "-k", target / "vk", "-p", target / "proof",
                       "-i", target / "public_inputs"])
    return prove, verify, rss


def bb_setup(dst, name):
    sh([NARGO, "execute", "--silence-warnings"], cwd=dst)
    target = dst / "target"
    out = sh([BB, "gates", "-b", target / f"{name}.json"])
    gates = json.loads(out[out.index("{"):])["functions"][0]["circuit_size"]
    sh([BB, "write_vk", "-b", target / f"{name}.json", "-o", target])
    return gates


_baseline = None


def bb_baseline(runs):
    global _baseline
    if _baseline is None:
        dst = instantiate("empty", None, 'x = "1"\ny = "1"\n')
        bb_setup(dst, "empty")
        samples = [bb_once(dst, "empty") for _ in range(max(runs, 5))]
        _baseline = (statistics.median(s[0] for s in samples), statistics.median(s[1] for s in samples))
    return _baseline


def run_bb(dst, template, runs):
    base_p, base_v = bb_baseline(runs)
    gates = bb_setup(dst, template)
    rows = []
    for run in range(runs):
        prove, verify, rss = bb_once(dst, template)
        rows.append({
            "run": run, "constraints": gates, "prover_s": max(prove - base_p, 0.0),
            "verifier_s": max(verify - base_v, 0.0),
            "proof_bytes": (dst / "target/proof").stat().st_size, "max_rss_bytes": rss,
            "prover_raw_s": prove, "verifier_raw_s": verify, "baseline_prove_s": base_p,
            "baseline_verify_s": base_v,
        })
    return rows


def run_external(n, runs):
    """Opening of the external commitment on its own (for the bb estimate)."""
    elements = -(-n // PACK) + 1
    rows = []
    for scheme in ("kzg", "ipa"):
        out = sh([PCS_BENCH, scheme, str(elements), str(runs)])
        m = json.loads(out.strip().splitlines()[-1])
        rows.append((scheme, {
            "run": "median", "constraints": elements, "prover_s": m["open_s"],
            "verifier_s": m["verify_s"], "proof_bytes": m["proof_bytes"], "commit_s": m["commit_s"],
        }))
    return rows


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.csv = csv.DictWriter(self.file, fieldnames=FIELDS)
        if new:
            self.csv.writeheader()

    def write(self, base, rows):
        for row in rows:
            self.csv.writerow({**base, **row})
        self.file.flush()
        med = statistics.median(r["prover_s"] for r in rows)
        rss = max((r.get("max_rss_bytes") or 0) for r in rows)
        print(f"  {base['backend']:9} {base['row']:17} {str(base['bytes']):>8} B  "
              f"constraints={rows[0]['constraints']:<9} prover={med:8.3f}s  rss={rss / 2**30:5.2f} GiB",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workload", choices=["sweep", "age"])
    ap.add_argument("--sizes", default="520,2400,4000,10000,16384,65536",
                    help="payload sizes in bytes (sweep)")
    ap.add_argument("--backends", default="provekit,bb")
    ap.add_argument("--rows", default="", help="restrict to these rows")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/results.csv"))
    ap.add_argument("--note", default="")
    ap.add_argument("--all-sizes", action="store_true", help="ignore the local memory caps")
    ap.add_argument("--clean-build", action="store_true",
                    help="delete each instantiated circuit (keys, target) after measuring it (bounds disk use)")
    args = ap.parse_args()

    vers = versions()
    writer = Writer(args.out)
    backends = args.backends.split(",")
    only = set(filter(None, args.rows.split(",")))

    if args.workload == "sweep":
        for n in map(int, args.sizes.split(",")):
            print(f"{n} bytes ({SIZES.get(n, 'custom')})", flush=True)
            for backend in backends:
                for row in BACKEND_ROWS[backend]:
                    if only and row not in only:
                        continue
                    if row == "blake3_bb" and n > 1024:
                        continue  # Barretenberg's BLAKE3 black box takes at most 1024 bytes
                    if not args.all_sizes and n > LOCAL_CAP.get((backend, row), n):
                        print(f"  {backend:9s} {row:20s} skipped at {n} B (memory; --all-sizes to run)")
                        continue
                    template, link = SWEEP_ROWS[row]
                    dst = instantiate(template, n, prover_toml(template, n))
                    base = {"workload": "sweep", "backend": backend, "row": row, "bytes": n,
                            "version": vers[backend], "note": args.note}
                    rows = run_provekit(dst, link, args.runs) if backend == "provekit" \
                        else run_bb(dst, template, args.runs)
                    writer.write(base, rows)
                    if args.clean_build:
                        shutil.rmtree(dst, ignore_errors=True)
            if "bb" in backends and (not only or "external" in only):
                for scheme, row in run_external(n, args.runs):
                    writer.write({"workload": "sweep", "backend": "external", "row": scheme,
                                  "bytes": n, "version": "tlsn-bench-pcs", "note": args.note}, [row])
    else:
        for backend in backends:
            for template, link in AGE_ROWS[backend]:
                if only and template not in only:
                    continue
                dst = instantiate(template, None, age_toml(template))
                base = {"workload": "age", "backend": backend, "row": template, "bytes": 10,
                        "version": vers[backend], "note": args.note}
                rows = run_provekit(dst, link, args.runs) if backend == "provekit" \
                    else run_bb(dst, template, args.runs)
                writer.write(base, rows)
                if args.clean_build:
                    shutil.rmtree(dst, ignore_errors=True)


if __name__ == "__main__":
    main()
