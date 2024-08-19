#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa true 0 17 40 halo2pngs/snetpc_ipa.png > halo2logs/snet_ipa.txt &
# wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 0 16 10 halo2pngs/snet_kzg.png > halo2logs/snet_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 0 15 10 halo2pngs/mnist_kzg.png > halo2logs/mnist_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 0 21 10 halo2pngs/dlrm_kzg.png > halo2logs/dlrm_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 0 22 10 halo2pngs/vgg_kzg.png > halo2logs/vgg_kzg.txt &
wait


#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3