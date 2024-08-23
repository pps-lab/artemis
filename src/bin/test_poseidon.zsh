#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release
#mkdir params_kzg
#mkdir params_ipa


./target/release/time_circuit examples/mnist/shallownet_p.msgpack examples/mnist/inp.msgpack kzg false 0 19 8 pngs/snet_pos_kzg.png > logs/snet_pos_kzg.txt &
wait
./target/release/time_circuit examples/cifar/mnist_p.msgpack examples/mnist/inp.msgpack kzg false 0 19 10 pngs/mnist_pos_kzg.png > logs/mnist_pos_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm_p.msgpack examples/cifar/dlrm_input.msgpack kzg false 0 21 11 pngs/dlrm_pos_kzg.png > logs/dlrm_pos_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg_p.msgpack examples/cifar/vgg_input.msgpack kzg false 0 25 12 pngs/vgg_pos_kzg.png &> logs/vgg_pos_kzg.txt &
wait

# ./target/release/time_circuit examples/mnist/shallownet_p.msgpack examples/mnist/inp.msgpack kzg true 0 19 13 pngs/snet_pos_kzg_18_12.png > logs/snet_pos_kzg_18_12.txt &
# wait
# ./target/release/time_circuit examples/cifar/mnist_p.msgpack examples/mnist/inp.msgpack kzg true 0 19 13 pngs/mnist_pos_kzg_17_12.png > logs/mnist_pos_kzg_17_12.txt &
# wait
# ./target/release/time_circuit examples/cifar/dlrm_p.msgpack examples/cifar/dlrm_input.msgpack kzg true 0 21 13 pngs/dlrm_pos_kzg_21_12.png > logs/dlrm_pos_kzg_21_12.txt &
# wait
# ./target/release/time_circuit examples/cifar/vgg_p.msgpack examples/cifar/vgg_input.msgpack kzg true 0 25 13 pngs/vgg_pos_kzg_25_12.png > logs/vgg_pos_kzg_25_12.txt &
# wait

# ./target/release/time_circuit examples/mnist/shallownet_p.msgpack examples/mnist/inp.msgpack kzg true 0 19 14 pngs/snet_pos_kzg_18_12.png > logs/snet_pos_kzg_18_12.txt &
# wait
# ./target/release/time_circuit examples/cifar/mnist_p.msgpack examples/mnist/inp.msgpack kzg true 0 19 14 pngs/mnist_pos_kzg_17_12.png > logs/mnist_pos_kzg_17_12.txt &
# wait
# ./target/release/time_circuit examples/cifar/dlrm_p.msgpack examples/cifar/dlrm_input.msgpack kzg true 0 21 14 pngs/dlrm_pos_kzg_21_12.png > logs/dlrm_pos_kzg_21_12.txt &
# wait
# ./target/release/time_circuit examples/cifar/vgg_p.msgpack examples/cifar/vgg_input.msgpack kzg true 0 25 14 pngs/vgg_pos_kzg_25_12.png > logs/vgg_pos_kzg_25_12.txt &
# wait

# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 pngs/diffusion_kzg.png > logs/diffusion_kzg.txt &
# # wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 2 26 10 pngs/gpt2_poly_kzg.png &> logs/gpt2_poly_kzg.txt &
# wait

# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 pngs/diffusion_kzg.png > logs/diffusion_kzg.txt &
# # wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 10 pngs/gpt2_kzg.png &> logs/gpt2_kzg.txt &
# wait
# sudo shutdown now -h
