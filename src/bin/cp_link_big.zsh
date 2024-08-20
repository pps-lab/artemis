#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release

cargo run --package zkml --bin cp_link --release 20 1000000 1  > 1000000_1.log &
wait
cargo run --package zkml --bin cp_link --release 20 700000 6  > 700000_6.log &
wait
# cargo run --package zkml --bin cp_link --release 24 10000000 1  > 10000000_1.log &
# wait
cargo run --package zkml --bin cp_link --release 23 7500000 10  > 7500000_10.log &
wait
cargo run --package zkml --bin cp_link --release 26 40000000 10  > 40000000_10.log &
wait