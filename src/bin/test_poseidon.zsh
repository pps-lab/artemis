#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release
#mkdir params_kzg
#mkdir params_ipa

./target/release/time_circuit examples/cifar/shallownet_p.msgpack examples/mnist/inp.msgpack kzg true 0 18 10 halo2pngs/snet_pos_kzg_18_10.png > halo2logs/snet_pos_kzg_18_10.txt &
wait
./target/release/time_circuit examples/cifar/mnist_p.msgpack examples/mnist/inp.msgpack kzg true 0 17 10 halo2pngs/mnist_pos_kzg_17_10.png > halo2logs/mnist_pos_kzg_17_10.txt &
wait
./target/release/time_circuit examples/cifar/dlrm_p.msgpack examples/cifar/dlrm_input.msgpack kzg true 0 21 10 halo2pngs/dlrm_pos_kzg_21_10.png > halo2logs/dlrm_pos_kzg_21_10.txt &
wait
./target/release/time_circuit examples/cifar/vgg_p.msgpack examples/cifar/vgg_input.msgpack kzg true 0 25 10 halo2pngs/vgg_pos_kzg_25_10.png > halo2logs/vgg_pos_kzg_25_10.txt &
wait


./target/release/time_circuit examples/mnist/shallownet_p.msgpack examples/mnist/inp.msgpack kzg true 0 18 20 halo2pngs/snet_pos_kzg_18_20.png > halo2logs/snet_pos_kzg_18_20.txt &
wait
./target/release/time_circuit examples/cifar/mnist_p.msgpack examples/mnist/inp.msgpack kzg true 0 17 20 halo2pngs/mnist_pos_kzg_17_20.png > halo2logs/mnist_pos_kzg_17_20.txt &
wait
./target/release/time_circuit examples/cifar/dlrm_p.msgpack examples/cifar/dlrm_input.msgpack kzg true 0 21 20 halo2pngs/dlrm_pos_kzg_21_20.png > halo2logs/dlrm_pos_kzg_21_20.txt &
wait
./target/release/time_circuit examples/cifar/vgg_p.msgpack examples/cifar/vgg_input.msgpack kzg true 0 25 20 halo2pngs/vgg_pos_kzg_25_20.png > halo2logs/vgg_pos_kzg_25_20.txt &
wait
# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# # wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 2 26 10 halo2pngs/gpt2_poly_kzg.png &> halo2logs/gpt2_poly_kzg.txt &
# wait

# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# # wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 10 halo2pngs/gpt2_kzg.png &> halo2logs/gpt2_kzg.txt &
# wait
# sudo shutdown now -h
