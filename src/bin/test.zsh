#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

# arg1: name = ['mnist', 'resnet', 'dlrm', 'mobilenet', 'vgg', 'gpt2', 'diffusion']
# arg2: pc_type = ['kzg', 'ipa']
# arg3: cp_snark = ['nocom', 'poly', 'cp_link', 'pos', 'cp_link_plus']
# arg4: num_runs - int
# arg5: code directory
# example: ./test.zsh mnist kzg nocom 5

name="$1"
pc_type="$2"
cp_snark="$3"
num_runs="$4"
dir="$5"

cargo +nightly build --release --manifest-path $dir'/Cargo.toml'
mkdir -p results
cols=0
rows=0
poly_cols=0
cplink=false
poly_com=false

case "$name" in
    mnist)
        cols=10
        rows=15
        poly_cols=1
        # Add MNIST-specific commands here
        # e.g., python mnist_script.py
        ;;
    
    resnet18)
        cols=8
        rows=19
        name='cifar10'
        poly_cols=1
        echo "Running ResNet task..."
        # Add ResNet-specific commands here
        # e.g., python resnet_script.py
        ;;
    
    dlrm)
        cols=33
        k=17
        poly_cols=5
        echo "Running DLRM task..."
        # Add DLRM-specific commands here
        # e.g., python dlrm_script.py
        ;;
    
    mobilenet)
        cols=20
        k=23
        poly_cols=1
        echo "Running MobileNet task..."
        # Add MobileNet-specific commands here
        # e.g., python mobilenet_script.py
        ;;
    
    vgg)
        cols=16
        k=22
        poly_cols=4
        echo "Running VGG task..."
        # Add VGG-specific commands here
        # e.g., python vgg_script.py
        ;;
    
    gpt2)
        cols=13
        k=25
        poly_cols=3
        echo "Running GPT-2 task..."
        # Add GPT-2-specific commands here
        # e.g., python gpt2_script.py
        ;;
    
    diffusion)
        cols=29
        k=24
        poly_cols=2
        echo "Running Diffusion task..."
        # Add Diffusion-specific commands here
        # e.g., python diffusion_script.py
        ;;
    
    *)
        echo "Error: Unknown case '$case'"
        echo "Available cases: mnist, resnet, dlrm, mobilenet, vgg, gpt2, diffusion"
        exit 1
        ;;
esac

case "$cp_snark" in
    nocom)
        cp_link=false
        poly_com=false
        poly_cols=0
        # Add MNIST-specific commands here
        # e.g., python mnist_script.py
        ;;
    
    poly)
        cp_link=false
        poly_com=true
        echo "Running ResNet task..."
        # Add ResNet-specific commands here
        # e.g., python resnet_script.py
        ;;
    
    cp_link)
        cp_link=true
        poly_com=false
        echo "Running DLRM task..."
        # Add DLRM-specific commands here
        # e.g., python dlrm_script.py
        ;;
    
    poseidon)
        cp_link=false
        poly_com=false
        name=$name'_p'
        echo "Running MobileNet task..."
        # Add MobileNet-specific commands here
        # e.g., python mobilenet_script.py
        ;;

    cp_link_slow)
        cp_link=true
        poly_com=false
        echo "Running DLRM task..."
        # Add DLRM-specific commands here
        # e.g., python dlrm_script.py
        ;;

    
    *)
        echo "Error: Unknown case '$case'"
        echo "Available cases: mnist, resnet, dlrm, mobilenet, vgg, gpt2, diffusion"
        exit 1
        ;;
esac

$dir/target/release/time_circuit $dir/examples/cifar/$name.msgpack $dir/examples/cifar/$1_input.msgpack $pc_type false $poly_cols $rows $cols false $num_runs $dir