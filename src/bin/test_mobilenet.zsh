#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa true 0 17 40 pngs/snetpc_ipa.png > logs/snet_ipa.txt &
# wait
# ./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack kzg false 0 23 20 pngs/mobilenet_nocom_kzg_23.png &> logs/mobilenet_nocom_23.txt &
# wait
./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack kzg true 1 23 20 pngs/mobilenet_poly_kzg.png &> logs/mobilenet_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack kzg false 0 23 20 pngs/mobilenet_nocom_kzg.png &> logs/mobilenet_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/mobilenet_p.msgpack examples/cifar/mobilenet_input.msgpack kzg false 0 24 20 pngs/mobilenet_pos_kzg.png &> logs/mobilenet_pos_kzg.txt &
wait
# ./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack kzg false 0 24 10 pngs/mobilenet_nocom_kzg_24.png &> logs/mobilenet_nocom_24.txt &
# wait
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 1 15 10 pngs/mnist_poly_kzg.png > logs/mnist_poly_kzg.txt &
# wait
# ./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 1 20 6 pngs/dlrm_poly_kzg.png > logs/dlrm_poly_kzg.txt &
# wait
# ./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 2 23 10 pngs/vgg_poly_kzg.png > logs/vgg_poly_kzg.txt &
# wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 3 25 13 pngs/gpt2_poly_kzg.png > logs/gpt2_poly_kzg.txt &
# wait

#sudo shutdown -h
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3