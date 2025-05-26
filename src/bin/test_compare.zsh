#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release
#mkdir params_kzg
#mkdir params_ipa

./src/bin/test_bls.zsh mnist kzg no_com 1 . &> mnist_bls.log &
wait
./src/bin/test.zsh mnist kzg no_com 1 . &> mnist.log &
wait
./src/bin/test_bls.zsh resnet18 kzg no_com 1 . &> resnet18_bls.log &
wait
./src/bin/test.zsh resnet18 kzg no_com 1 . &> resnet18.log &
wait
./src/bin/test_bls.zsh dlrm kzg no_com 1 . &> dlrm_bls.log &
wait
./src/bin/test.zsh dlrm kzg no_com 1 . &> dlrm.log &
wait

#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3