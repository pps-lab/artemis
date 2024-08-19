#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

# python ../../python/converter.py --model ../../examples/mnist/model.tflite --model_output ../../examples/mnist/converted_model.msgpack --config_output ../../examples/mnist/config.msgpack --scale_factor 512 --k 17 --num_cols 10 --num_randoms 1024 --commit
#python ../../python/converter.py --model ../../examples/dlrm/dlrm_model2.tflite --model_output ../../examples/mnist/converted_model.msgpack --config_output ../../examples/dlrm/config.msgpack --scale_factor 512 --k 17 --num_cols 10 --num_randoms 1024 --commit
# python python/converter.py --model lenet_medium_cifar10_no_softmax.tflite --model_output examples/cifar/medium_model_com.msgpack --config_output examples/cifar/medium_config_com.msgpack --scale_factor 512 --k 21 --num_cols 10 --num_randoms 1024 --commit
# python python/converter.py --model lenet_medium_cifar10_no_softmax.tflite --model_output examples/cifar/medium_model_nocom.msgpack --config_output examples/cifar/medium_config_nocom.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024
#python python/converter.py --model examples/zkml_tflite/dlrm.tflite --model_output examples/zkml_tflite/dlrm.msgpack --config_output examples/cifar/dlrm_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024
# python python/converter.py --model examples/zkml_tflite/diffusion.tflite --model_output examples/cifar/diffusion.msgpack --config_output examples/cifar/diffusion_config.msgpack --scale_factor 512 --k 22 --num_cols 20 
#python python/converter.py --model examples/zkml_tflite/vgg16.tflite --model_output examples/zkml_tflite/vgg.msgpack --config_output examples/cifar/vgg_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024
# python python/fake_input_converter.py --model_config examples/twitter/config.msgpack --output examples/twitter/input.msgpack 
# python python/converter.py --model examples/zkml_tflite/twitter.tflite --model_output examples/cifar/twitter.msgpack --config_output examples/cifar/twitter_config.msgpack --scale_factor 512 --k 23 --num_cols 10
# python python/converter.py --model examples/zkml_tflite/twitter.tflite --model_output examples/cifar/twitter.msgpack --config_output examples/cifar/twitter_config.msgpack --scale_factor 512 --k 23 --num_cols 10  
python python/fake_input_converter.py --model_config examples/twitter/config.msgpack --output examples/twitter/input.msgpack 

# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 0 mpc &> log0snn.txt &; pid0=$!
# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 1 mpc &> log1snn.txt &; pid1=$!
# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 2 mpc &> log2snn.txt &; pid2=$!
# ./target/release/time_circuit examples/shallownet_mnist/new_converted_model.msgpack examples/mnist/inp.msgpack kzg 3 mpc &> log3snn.txt &; pid3=$!

# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 0 loc &> logloc.txt &
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 0 mpc &> log0.txt &; pid0=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 1 mpc &> log1.txt &; pid1=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 2 mpc &> log2.txt &; pid2=$!
# ./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg 3 mpc &> log3.txt &; pid3=$!
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3