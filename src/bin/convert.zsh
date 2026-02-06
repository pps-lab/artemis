#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

# python ../../python/converter.py --model ../../examples/mnist/model.tflite --model_output ../../examples/mnist/converted_model.msgpack --config_output ../../examples/mnist/config.msgpack --scale_factor 512 --k 17 --num_cols 10 --num_randoms 1024 --commit
#python ../../python/converter.py --model ../../examples/dlrm/dlrm_model2.tflite --model_output ../../examples/mnist/converted_model.msgpack --config_output ../../examples/dlrm/config.msgpack --scale_factor 512 --k 17 --num_cols 10 --num_randoms 1024 --commit
# python python/converter.py --model lenet_medium_cifar10_no_softmax.tflite --model_output examples/cifar/medium_model_com.msgpack --config_output examples/cifar/medium_config_com.msgpack --scale_factor 512 --k 21 --num_cols 10 --num_randoms 1024 --commit
# python python/converter.py --model lenet_medium_cifar10_no_softmax.tflite --model_output examples/cifar/medium_model_nocom.msgpack --config_output examples/cifar/medium_config_nocom.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024
# python python/converter.py --model examples/zkml_tflite/dlrm.tflite --model_output examples/zkml_tflite/dlrm.msgpack --config_output examples/cifar/dlrm_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024
# python python/converter.py --model examples/zkml_tflite/diffusion.tflite --model_output examples/cifar/diffusion.msgpack --config_output examples/cifar/diffusion_config.msgpack --scale_factor 512 --k 22 --num_cols 20 
# python python/converter.py --model examples/zkml_tflite/vgg16.tflite --model_output examples/cifar/vgg.msgpack --config_output examples/cifar/vgg_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024
# python python/converter.py --model examples/zkml_tflite/gpt2.tflite --model_output examples/cifar/gpt2.msgpack --config_output examples/cifar/gpt2_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024

python python/fake_input_converter.py --model_config examples/cifar/cifar10_config.msgpack --output examples/cifar/cifar10_input.msgpack
# python python/converter.py --model examples/zkml_tflite/cifar10.tflite --model_output examples/cifar/cifar10.msgpack --config_output examples/cifar/cifar10_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024 
# #python python/converter.py --model examples/zkml_tflite/mobilebert.tflite --model_output examples/cifar/mobilebert.msgpack --config_output examples/cifar/mobilebert_config.msgpack --scale_factor 512 --k 22 --num_cols 20 
#python python/converter.py --model examples/zkml_tflite/mobilenet.tflite --model_output examples/cifar/mobilenet.msgpack --config_output examples/cifar/mobilenet_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 20000
# python python/converter.py --model examples/zkml_tflite/mnist.tflite --model_output examples/cifar/mnist.msgpack --config_output examples/cifar/mnist_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024 

# python python/converter.py --model examples/zkml_tflite/dlrm.tflite --model_output examples/cifar/dlrm_p.msgpack --config_output examples/cifar/dlrm_config_p.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024 --commit
# python python/converter.py --model examples/zkml_tflite/diffusion.tflite --model_output examples/cifar/diffusion_p.msgpack --config_output examples/cifar/diffusion_config_p.msgpack --scale_factor 512 --k 22 --num_cols 20  --commit
# python python/converter.py --model examples/zkml_tflite/vgg16.tflite --model_output examples/cifar/vgg_p.msgpack --config_output examples/cifar/vgg_config_p.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024 --commit
# python python/converter.py --model examples/zkml_tflite/gpt2.tflite --model_output examples/cifar/gpt2_p.msgpack --config_output examples/cifar/gpt2_config_p.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024 --commit

# python python/converter.py --model examples/zkml_tflite/cifar10.tflite --model_output examples/cifar/cifar10_p.msgpack --config_output examples/cifar/cifar10_p_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024 --commit
# #python python/converter.py --model examples/zkml_tflite/mobilebert.tflite --model_output examples/cifar/mobilebert_p.msgpack --config_output examples/cifar/mobilebert_p_config.msgpack --scale_factor 512 --k 22 --num_cols 20  --commit
# python python/converter.py --model examples/zkml_tflite/mobilenet.tflite --model_output examples/cifar/mobilenet_p.msgpack --config_output examples/cifar/mobilenet_p_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 20000 --commit
# python python/converter.py --model examples/zkml_tflite/mnist.tflite --model_output examples/cifar/mnist_p.msgpack --config_output examples/cifar/mnist_p_config.msgpack --scale_factor 512 --k 19 --num_cols 10 --num_randoms 1024 --commit
