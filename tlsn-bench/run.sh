#!/usr/bin/env bash
# Build everything and run the benchmark. See README.md.
#
#   ./run.sh setup            clone + patch + build ProveKit, build the pcs crate
#   ./run.sh test             soundness tests of the Artemis link and the pcs crate
#   ./run.sh small            age check + sweep up to 64 KiB, all rows (peak ~20 GiB RAM)
#   ./run.sh setup-bb-dump    clone + patch + build Barretenberg with the wire dump (LegoSNARK rows)
#   ./run.sh lego             LegoSNARK CP_link rows on Barretenberg, all sizes
#   ./run.sh plot             plots and tables from results/results.csv
set -euo pipefail
cd "$(dirname "$0")"

PROVEKIT_REV=893f6d58
BB_TAG=v5.0.0-nightly.20260324   # the release of the `bb` binary used for timing
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
  setup-bb-dump)
    # Needs cmake and ninja; builds with the system clang (Apple clang 21 works), no AVM, no node module.
    if [ ! -d aztec-packages/.git ]; then
      git clone --depth 1 --branch "$BB_TAG" --filter=blob:none --sparse \
        https://github.com/AztecProtocol/aztec-packages aztec-packages
      git -C aztec-packages sparse-checkout set barretenberg/cpp
      git -C aztec-packages apply ../barretenberg-dump.patch
    fi
    # BB_CC / BB_CXX pick the compiler (Ubuntu: clang-20 / clang++-20 from apt.llvm.org); BB_JOBS limits
    # the parallel compile jobs.
    (cd aztec-packages/barretenberg/cpp && cmake --preset default -DAVM=OFF -DBB_LITE=ON \
        ${BB_CC:+-DCMAKE_C_COMPILER=$BB_CC} ${BB_CXX:+-DCMAKE_CXX_COMPILER=$BB_CXX} \
      && cmake --build build --target bb ${BB_JOBS:+-j $BB_JOBS})
    (cd pcs && cargo build --release)
    ;;
  lego)
    python3 bench.py sweep --sizes "$SIZES" --runs "$RUNS" --note "$NOTE" \
      --backends bb --rows lego_estimate,lego_packed_estimate
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
    sed -n '2,9p' "$0"; exit 2 ;;
esac
