#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa true 0 17 40 pngs/snetpc_ipa.png > logs/snet_ipa.txt &
# wait
# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 1 17 5  > logs/snet_poly_kzg.txt &
# wait
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 1 15 10  > logs/mnist_poly_kzg.txt &
# wait

# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 1 15 10  > logs/mnist_poly_kzg.txt &
# wait

./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg false 0 20 6  > logs/dlrm_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg false 0 23 10  > logs/vgg_nocom_kzg.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 0 25 13  > logs/gpt2_nocom_kzg.txt &
wait

./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 1 20 6  > logs/dlrm_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 2 23 10  > logs/vgg_poly_kzg.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 3 25 13  > logs/gpt2_poly_kzg.txt &
wait

#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3