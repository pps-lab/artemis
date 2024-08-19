#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release

cargo run --package zkml --bin cp_link --release 10 1000 1 > 1000_1.log &
wait
cargo run --package zkml --bin cp_link --release 10 1000 10  > 1000_10.log &
wait
cargo run --package zkml --bin cp_link --release 14 10000 1  > 10000_1.log &
wait
cargo run --package zkml --bin cp_link --release 14 10000 10  > 10000_10.log &
wait
cargo run --package zkml --bin cp_link --release 17 100000 1  > 100000_1.log &
wait
cargo run --package zkml --bin cp_link --release 17 100000 10  > 100000_10.log &
wait
cargo run --package zkml --bin cp_link --release 20 1000000 1  > 1000000_1.log &
wait
cargo run --package zkml --bin cp_link --release 20 1000000 10  > 1000000_10.log &
wait
cargo run --package zkml --bin cp_link --release 24 10000000 1  > 10000000_1.log &
wait
cargo run --package zkml --bin cp_link --release 24 10000000 10  > 10000000_10.log &
wait
cargo run --package zkml --bin cp_link --release 26 50000000 1  > 100000000_1.log &
wait
cargo run --package zkml --bin cp_link --release 26 50000000 10  > 100000000_10.log &
wait