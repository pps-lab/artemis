#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release
#mkdir params_kzg
#mkdir params_ipa
./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 2 25 13 halo2pngs/gpt2_kzg.png > halo2logs/gpt2_kzg.txt &
wait
#./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg false gpt20.svg &> logpt20.txt 
#./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg true gpt21.svg &> logpt21.txt &
# ./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg false gpt20.svg &> logpt20.txt &
# ./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg true gpt20.svg &> logpt20.txt &
wait
