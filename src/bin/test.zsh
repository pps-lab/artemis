#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa

# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa true 0 17 40 pngs/snetpc_ipa.png > logs/snet_ipa.txt &
# wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 0 17 5 pngs/snet_nocom_kzg.png > logs/snet_nocom_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 0 15 10 pngs/mnist_nocom_kzg.png > logs/mnist_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg false 0 20 6 pngs/dlrm_nocom_kzg.png > logs/dlrm_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg false 0 23 10 pngs/vgg_nocom_kzg.png > logs/vgg_nocom_kzg.txt &
wait

./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 1 17 5 pngs/snet_poly_kzg.png > logs/snet_poly_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 1 15 10 pngs/mnist_poly_kzg.png > logs/mnist_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 1 20 6 pngs/dlrm_poly_kzg.png > logs/dlrm_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 2 23 10 pngs/vgg_poly_kzg.png > logs/vgg_poly_kzg.txt &
wait

./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 3 25 13 pngs/gpt2_poly_kzg.png > logs/gpt2_poly_kzg.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 0 25 13 pngs/gpt2_nocom_kzg.png > logs/gpt2_nocom_kzg.txt &
wait

./target/release/time_circuit examples/mnist/shallownet_p.msgpack examples/mnist/inp.msgpack kzg false 0 19 8 pngs/snet_pos_kzg.png > logs/snet_pos_kzg.txt &
wait
./target/release/time_circuit examples/cifar/mnist_p.msgpack examples/mnist/inp.msgpack kzg false 0 19 10 pngs/mnist_pos_kzg.png > logs/mnist_pos_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm_p.msgpack examples/cifar/dlrm_input.msgpack kzg false 0 21 11 pngs/dlrm_pos_kzg.png > logs/dlrm_pos_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg_p.msgpack examples/cifar/vgg_input.msgpack kzg false 0 25 12 pngs/vgg_pos_kzg.png &> logs/vgg_pos_kzg.txt &
wait

# cargo run --package zkml --bin cp_link --release 10 1000 1 > 1000_1.log &
# wait
# cargo run --package zkml --bin cp_link --release 10 1000 10  > 1000_10.log &
# wait
# cargo run --package zkml --bin cp_link --release 14 10000 1  > 10000_1.log &
# wait
# cargo run --package zkml --bin cp_link --release 14 10000 10  > 10000_10.log &
# wait
# cargo run --package zkml --bin cp_link --release 17 100000 1  > 100000_1.log &
# wait
# cargo run --package zkml --bin cp_link --release 17 100000 10  > 100000_10.log &
# wait
# cargo run --package zkml --bin cp_link --release 17 100000 5  > 100000_5.log &
# wait
# cargo run --package zkml --bin cp_link --release 20 1000000 1  > 1000000_1.log &
# wait
# cargo run --package zkml --bin cp_link --release 20 700000 6  > 700000_6.log &
# wait
# # cargo run --package zkml --bin cp_link --release 24 10000000 1  > 10000000_1.log &
# # wait
# cargo run --package zkml --bin cp_link --release 23 7500000 10  > 7500000_10.log &
# wait
# cargo run --package zkml --bin cp_link --release 26 40000000 10  > 40000000_5.log &
# wait
# cargo run --package zkml --bin cp_link --release 26 50000000 10  > 100000000_10.log &
# wait

#sudo shutdown now -h
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3