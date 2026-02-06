#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release

# cargo run --package zkml --bin cp_link --release 10 1000 1 > 1000_1.log &
# wait
# cargo run --package zkml --bin cp_link --release 10 1000 10  > 1000_10.log &
# wait
cargo run --package zkml --bin cp_link --release 15 10000 1  &> logs/mnist_1_lite.log &
wait
cargo run --package zkml --bin cp_link --release 17 100000 1  &> logs/snet_1_lite.log &
wait
cargo run --package zkml --bin cp_link --release 19 280000 1  &> logs/cifar10_1_lite.log &
wait
cargo run --package zkml --bin cp_link --release 20 750000 1  &> logs/dlrm_1_lite.log &
wait
cargo run --package zkml --bin cp_link --release 23 3500000 1  &> logs/mobilenet_1_lite.log &
wait
cargo run --package zkml --bin cp_link --release 23 7500000 1  &> logs/vgg_2_lite.log &
wait
cargo run --package zkml --bin cp_link --release 25 19500000 1  &> logs/diffusion_1_lite.log &
wait
cargo run --package zkml --bin cp_link --release 25 27000000 1  &> logs/gpt2_3_lite.log &
wait
# cargo run --package zkml --bin cp_link --release 14 10000 10  > 10000_10.log &
# wait
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 1 &> 100000_1.log &
# wait
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 10 &> 100000_10.log &
# wait
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 1 &> 100000_1.log &
# wait
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --nocapture  17 100000 10 &> 100000_10.log &
# wait