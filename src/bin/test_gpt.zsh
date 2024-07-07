#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release
#mkdir params_kzg
#mkdir params_ipa
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 0 gpt1.png &> gpt1.txt &
./target/release/draw_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 0 gpt0.png &> gpt0.txt &
#./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg false gpt20.svg &> logpt20.txt 
#./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg true gpt21.svg &> logpt21.txt &
# ./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg false gpt20.svg &> logpt20.txt &
# ./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg true gpt20.svg &> logpt20.txt &
wait
