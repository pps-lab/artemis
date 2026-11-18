#!/usr/bin/env bash
# Build everything and run the benchmark. See README.md.
#
#   ./run.sh setup            clone + patch + build ProveKit, build the pcs crate
#   ./run.sh test             soundness tests of the Artemis link and the pcs crate
#   ./run.sh small            age check + sweep up to 64 KiB, all rows (peak ~20 GiB RAM)
#   ./run.sh plot             plots and tables from results/results.csv
set -euo pipefail
cd "$(dirname "$0")"

PROVEKIT_REV=893f6d58
NOTE=${NOTE:-}
RUNS=${RUNS:-5}
SIZES=520,2400,4000,10000,16384,65536

case "${1:-}" in
  setup)
    if [ ! -d provekit/.git ]; then
      git clone https://github.com/worldfnd/provekit provekit
      git -C provekit checkout "$PROVEKIT_REV"
      git -C provekit apply ../provekit-artemis.patch
    fi
    (cd provekit && cargo build --release -p artemis-bench)
    (cd pcs && cargo build --release)
    ;;
  test)
    (cd pcs && cargo test --release)
    (cd provekit && cargo test --release -p artemis-bench)
    ;;
  small)
    python3 bench.py age --runs "$RUNS" --note "$NOTE"
    python3 bench.py sweep --sizes "$SIZES" --runs "$RUNS" --note "$NOTE"
    ;;
  plot)
    python3 plot.py
    ;;
  *)
    sed -n '2,7p' "$0"; exit 2 ;;
esac
