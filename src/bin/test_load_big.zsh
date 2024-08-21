#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release
#mkdir params_kzg
#mkdir params_ipa
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 10 halo2pngs/gpt2_poly_kzg.png &> halo2logs/draw_gpt2_poly_kzg_10.txt &
wait
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 15 halo2pngs/gpt2_poly_kzg.png &> halo2logs/draw_gpt2_poly_kzg_15.txt &
wait
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 20 halo2pngs/gpt2_poly_kzg.png &> halo2logs/draw_gpt2_poly_kzg_20.txt &
wait
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 25 halo2pngs/gpt2_poly_kzg.png &> halo2logs/draw_gpt2_poly_kzg_25.txt &
wait
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 30 halo2pngs/gpt2_poly_kzg.png &> halo2logs/draw_gpt2_poly_kzg_30.txt &
wait
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 35 halo2pngs/gpt2_poly_kzg.png &> halo2logs/draw_gpt2_poly_kzg_35.txt &
wait
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 40 halo2pngs/gpt2_poly_kzg.png &> halo2logs/draw_gpt2_poly_kzg_40.txt &
wait
# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# # wait


# # ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 2 25 13 halo2pngs/diffusion_kzg.png > halo2logs/diffusion_kzg.txt &
# # wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 26 10 halo2pngs/gpt2_kzg.png &> halo2logs/gpt2_kzg.txt &
# wait
sudo shutdown now -h