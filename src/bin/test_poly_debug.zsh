#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 1 17 5 halo2pngs/snet_poly_kzg.png > halo2logs/snet_debug.txt &
# wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 1 15 10 halo2pngs/mnist_poly_kzg.png
wait