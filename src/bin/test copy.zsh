# #!/usr/bin/env zsh
# set -xe
# trap "exit" INT TERM
# trap "kill 0" EXIT

# cargo +nightly build --release
# #mkdir params_kzg
# #mkdir params_ipa
#./target/release/time_circuit examples/twitter/config.msgpack examples/twitter/input.msgpack kzg false 0 17 5 pngs/twitter_nocom_kzg.png
# shallownet
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg false 0 17 5 false > logs/snet_nocom_kzg.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack kzg true 1 17 5 false > logs/snet_poly_kzg.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa false 0 17 5 false > logs/snet_nocom_ipa.txt &
wait
./target/release/time_circuit examples/mnist/shallownet_model.msgpack examples/mnist/inp.msgpack ipa true 1 17 5 false > logs/snet_poly_ipa.txt &
wait

# mnist
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg false 0 15 10 false > logs/mnist_nocom_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg true 1 15 10 false > logs/mnist_poly_kzg.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack ipa false 0 15 10 false > logs/mnist_nocom_ipa.txt &
wait
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack ipa true 1 15 10 false > logs/mnist_poly_ipa.txt &
wait

# resnet
./target/release/time_circuit examples/cifar/cifar10.msgpack examples/cifar/cifar10_input.msgpack kzg true 1 19 15 false > logs/cifar10_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/cifar10.msgpack examples/cifar/cifar10_input.msgpack kzg false 0 19 15 false > logs/cifar10_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/cifar10.msgpack examples/cifar/cifar10_input.msgpack ipa true 1 19 15 false > logs/cifar10_poly_ipa.txt &
wait
./target/release/time_circuit examples/cifar/cifar10.msgpack examples/cifar/cifar10_input.msgpack ipa false 0 19 15 false > logs/cifar10_nocom_ipa.txt &
wait

# dlrm
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg false 0 20 6 false > logs/dlrm_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 1 20 6 false > logs/dlrm_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack ipa false 0 20 6 false > logs/dlrm_nocom_ipa.txt &
wait
./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack ipa true 1 20 6 false > logs/dlrm_poly_ipa.txt &
wait

# mobilenet
./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack kzg true 1 23 20 false &> logs/mobilenet_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack kzg false 0 23 20 false &> logs/mobilenet_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack ipa true 1 23 20 false &> logs/mobilenet_poly_ipa.txt &
wait
./target/release/time_circuit examples/cifar/mobilenet.msgpack examples/cifar/mobilenet_input.msgpack ipa false 0 23 20 false &> logs/mobilenet_nocom_ipa.txt &
wait

# vgg
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg true 2 23 10 false > logs/vgg_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack kzg false 0 23 10 false > logs/vgg_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack ipa true 2 23 10 false > logs/vgg_poly_ipa.txt &
wait
./target/release/time_circuit examples/cifar/vgg.msgpack examples/cifar/vgg_input.msgpack ipa false 0 23 10 false > logs/vgg_nocom_ipa.txt &
wait

# gpt2
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 3 25 13 false > logs/gpt2_poly_kzg.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 0 25 13 false > logs/gpt2_nocom_kzg.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack ipa true 3 25 13 false > logs/gpt2_poly_ipa.txt &
wait
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack ipa false 0 25 13 false > logs/gpt2_nocom_ipa.txt &
wait

# diffusion
./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg true 1 25 15 false > logs/diffusion_poly_kzg.txt &
wait
./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 0 25 15 false > logs/diffusion_nocom_kzg.txt &
wait
./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack ipa true 1 25 15 false > logs/diffusion_poly_ipa.txt &
wait
./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack ipa false 0 25 15 false > logs/diffusion_nocom_ipa.txt &
wait

# ./target/release/time_circuit examples/cifar/cifar10_p.msgpack examples/cifar/cifar10_input.msgpack kzg false 0 19 15 pngs/cifar10_pos_kzg.png > logs/cifar10_pos_kzg.txt &
# wait

# ./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg false 0 20 6 pngs/dlrm_nocom_kzg.png > logs/dlrm_nocom_kzg.txt &
# wait



# ./target/release/time_circuit examples/cifar/dlrm.msgpack examples/cifar/dlrm_input.msgpack kzg true 1 20 6 pngs/dlrm_poly_kzg.png > logs/dlrm_poly_kzg.txt &
# wait


# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg true 3 25 13 pngs/gpt2_poly_kzg.png > logs/gpt2_poly_kzg.txt &
# wait
# ./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg false 0 25 13 pngs/gpt2_nocom_kzg.png > logs/gpt2_nocom_kzg.txt &
# wait

# ./target/release/time_circuit examples/mnist/shallownet_p.msgpack examples/mnist/inp.msgpack kzg false 0 19 8 pngs/snet_pos_kzg.png > logs/snet_pos_kzg.txt &
# wait
# ./target/release/time_circuit examples/cifar/mnist_p.msgpack examples/mnist/inp.msgpack kzg false 0 19 10 pngs/mnist_pos_kzg.png > logs/mnist_pos_kzg.txt &
# wait
# ./target/release/time_circuit examples/cifar/dlrm_p.msgpack examples/cifar/dlrm_input.msgpack kzg false 0 21 11 pngs/dlrm_pos_kzg.png > logs/dlrm_pos_kzg.txt &
# wait
# ./target/release/time_circuit examples/cifar/vgg_p.msgpack examples/cifar/vgg_input.msgpack kzg false 0 25 12 pngs/vgg_pos_kzg.png &> logs/vgg_pos_kzg.txt &
# wait





# ./target/release/time_circuit examples/cifar/diffusion.msgpack examples/cifar/diffusion_input.msgpack kzg false 0 25 13 pngs/diffusion_pos_kzg.png > logs/diffusion_pos_kzg.txt &
# wait

#sudo shutdown now -h
#~/../../dev/null
# wait $pid0 $pid1 $pid2 $pid3