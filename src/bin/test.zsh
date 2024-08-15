#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo +nightly build --release
#mkdir params_kzg
#mkdir params_ipa
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 0 witnesscol.png &> witnesscol.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 0 polycommit.png &> polycommit.txt &

./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 0 witnesscolsn.png &> witnesscolsn_new.txt &
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 0 polycommitsn.png &> polycommitsn_new.txt &

#./target/release/time_circuit examples/cifar/medium_model_nocom.msgpack examples/cifar/input.msgpack kzg true 0 witnesscolcf.png &> witnesscolcf.txt &
#./target/release/time_circuit examples/cifar/medium_model_nocom.msgpack examples/cifar/input.msgpack kzg false 0 polycommitcf.png &> polycommitcf.txt &

#./target/release/time_circuit examples/cifar/medium_model_com.msgpack examples/cifar/input.msgpack kzg true 0 witnesscolcfposeidon.png &> witnesscolcfposeidon.txt &
#./target/release/time_circuit examples/cifar/medium_model_com.msgpack examples/cifar/input.msgpack kzg false 0 polycommitcfposeidon.png &> polycommitcfposeidon.txt &
#/target/release/time_circuit examples/mnist/poseidon_model.msgpack examples/mnist/inp.msgpack kzg false 0 snposeidon.png &> snposeidon.txt &
# cargo test --package zkml --bin cp_link --release -- test_cplink --exact --show-output  1024 1
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 8 nocom1_8.svg &> log_nocom1_8.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 16 nocom1_16.svg &> log_nocom1_16.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 32 nocom132.png &> log_nocom1_32_n.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 64 nocom164.png &> log_nocom1_64_n.txt &

# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 8 nocom0_8.svg &> log_nocom0_8.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 16 nocom0_16.svg &> log_nocom0_16.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 32 nocom0_32_n.svg &> log_nocom0_32_n.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 64 nocom0_64_n.svg &> log_nocom0_64_n.txt &
# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true nocom_sn1.svg &> log_nocom_sn1.txt &
# ./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false nocom_sn0.svg &> log_nocom_sn0.txt &
# ./target/release/estimate_size examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg &> log_nocom_size.txt &
#./target/release/time_circuit examples/mnist/poseidon_model.msgpack examples/mnist/inp.msgpack kzg &> log_com_poseidon.txt &
# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 0 mpc &> log0snn.txt &; pid0=$!
# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 1 mpc &> log1snn.txt &; pid1=$!
# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 2 mpc &> log2snn.txt &; pid2=$!
# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 3 mpc &> log3snn.txt &; pid3=$!
wait $pid0 $pid1 $pid2 $pid3

# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 0 loc &> logloc.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 0 mpc &> log0.txt &; pid0=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 1 mpc &> log1.txt &; pid1=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 2 mpc &> log2.txt &; pid2=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 3 mpc &> log3.txt &; pid3=$!
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3