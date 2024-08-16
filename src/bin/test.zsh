#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

#./target/release/draw_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 0 diffusionpc.png &> diffusionpc.txt &
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 2 17 pngs/snetpc17_1.png &> logs/snetpc17_1.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 1 17 pngs/snetpc17_2.png &> logs/snetpc17_2.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 0 17 pngs/snet17.png &> logs/snet17.txt &
wait
./target/release/time_circuit examples/mnist/poseidon_model.msgpack examples/mnist/inp.msgpack kzg true 0 17 pngs/snetpos17.png &> logs/snetpos17.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 2 16 pngs/snetpc16_1.png &> logs/snetpc16_1.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 1 16 pngs/snetpc16_2.png &> logs/snetpc16_2.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 0 16 pngs/snet16.png &> logs/snet16.txt &
wait
./target/release/time_circuit examples/mnist/poseidon_model.msgpack examples/mnist/inp.msgpack kzg true 0 16 pngs/snetpos16.png &> logs/snetpos16.txt &
wait
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 0 loc &> logloc.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 0 mpc &> log0.txt &; pid0=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 1 mpc &> log1.txt &; pid1=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 2 mpc &> log2.txt &; pid2=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 3 mpc &> log3.txt &; pid3=$!
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3