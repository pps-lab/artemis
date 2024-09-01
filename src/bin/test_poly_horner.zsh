#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa true 0 17 40 pngs/snetpc_ipa.png > logs/snet_ipa.txt &
# wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 1 17 5 pngs/snet_poly_kzg.png > logs/snet_polyh_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 1 15 10 pngs/mnist_poly_kzg.png > logs/mnist_polyh_kzg.txt &
wait
./target/release/time_circuit examples/cifar/cifar10.msgpack examples/cifar/cifar10_input.msgpack kzg true 1 19 15 pngs/cifar10_poly_kzg.png > logs/cifar10_polyh_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 1 20 6 pngs/dlrm_poly_kzg.png > logs/dlrm_polyh_kzg.txt &
wait
./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack kzg true 1 23 20 pngs/mobilenet_poly_kzg.png &> logs/mobilenet_polyh_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 2 23 10 pngs/vgg_poly_kzg.png > logs/vgg_polyh_kzg.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 3 25 13 pngs/gpt2_poly_kzg.png > logs/gpt2_polyh_kzg.txt &
wait
./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg true 1 25 13 pngs/diffusion_poly_kzg.png > logs/diffusion_polyh_kzg.txt &
wait
sudo shutdown -h
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3