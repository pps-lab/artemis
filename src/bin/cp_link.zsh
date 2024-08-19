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
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 1 &> 100000_1.log &
# wait
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 10 &> 100000_10.log &
# wait
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 1 &> 100000_1.log &
# wait
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 10 &> 100000_10.log &
# wait