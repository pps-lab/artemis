#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release
#mkdir params_kzg
#mkdir params_ipa
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 0 23 10 halo2pngs/vgg_kzg.png &> halo2logs/vgg_kzg.txt &
wait

# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# # wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 2 26 10 halo2pngs/gpt2_poly_kzg.png &> halo2logs/gpt2_poly_kzg.txt &
# wait

# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# # wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 10 halo2pngs/gpt2_kzg.png &> halo2logs/gpt2_kzg.txt &
# wait
sudo shutdown now -h