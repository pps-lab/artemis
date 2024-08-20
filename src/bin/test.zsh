#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa true 0 17 40 halo2pngs/snetpc_ipa.png > halo2logs/snet_ipa.txt &
# wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 0 17 5 halo2pngs/snet_kzg.png > halo2logs/snet_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 0 15 10 halo2pngs/mnist_kzg.png > halo2logs/mnist_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 0 20 6 halo2pngs/dlrm_kzg.png > halo2logs/dlrm_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 0 23 10 halo2pngs/vgg_kzg.png > halo2logs/vgg_kzg.txt &
wait

./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 2 17 5 halo2pngs/snet_poly_kzg.png > halo2logs/snet_poly_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 2 15 10 halo2pngs/mnist_poly_kzg.png > halo2logs/mnist_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg false 2 20 6 halo2pngs/dlrm_poly_kzg.png > halo2logs/dlrm_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg false 2 23 10 halo2pngs/vgg_poly_kzg.png > halo2logs/vgg_poly_kzg.txt &
wait

# ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 2 26 10 halo2pngs/gpt2_poly_kzg.png > halo2logs/gpt2_poly_kzg.txt &
wait

# ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 10 halo2pngs/gpt2_kzg.png > halo2logs/gpt2_kzg.txt &
wait

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
cargo run --package zkml --bin cp_link --release 17 100000 5  > 100000_5.log &
wait
cargo run --package zkml --bin cp_link --release 20 1000000 1  > 1000000_1.log &
wait
cargo run --package zkml --bin cp_link --release 20 700000 6  > 700000_6.log &
wait
# cargo run --package zkml --bin cp_link --release 24 10000000 1  > 10000000_1.log &
# wait
cargo run --package zkml --bin cp_link --release 23 7500000 10  > 7500000_10.log &
wait
cargo run --package zkml --bin cp_link --release 26 40000000 10  > 40000000_5.log &
wait
# cargo run --package zkml --bin cp_link --release 26 50000000 10  > 100000000_10.log &
# wait

#sudo shutdown now -h
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3