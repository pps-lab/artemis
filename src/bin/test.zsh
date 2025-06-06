#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
#trap "kill 0" EXIT

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
mkdir -p $dir'/params_ipa'
mkdir -p $dir'/params_kzg_Bls12'
mkdir -p $dir'/params_kzg_Bn256'

cols=0
rows=0
poly_cols=0
cplink=false
poly_com=false
pedersen=false
slow=true

case "$name" in
    mnist)
        if [ "$cp_snark" = "poseidon" ]; then
            cols=10
            rows=19
            poly_cols=0
        # elif [ "$cp_snark" = "pedersen" ]; then
        #     cols=10
        #     rows=19
        else
            cols=10
            rows=15
            poly_cols=3
        fi

        ;;
    
    resnet18)
        if [ "$cp_snark" = "poseidon" ]; then
            cols=13
            rows=20
            poly_cols=0
        else
            cols=15
            rows=19
            poly_cols=1
        fi
        name='cifar10'
        echo "Running ResNet task..."
        ;;
    
    dlrm)
        if [ "$cp_snark" = "poseidon" ]; then
            cols=11
            rows=21
            poly_cols=0
        elif [ "$cp_snark" = "cp_link" ] || [ "$cp_snark" = "apollo_slow" ]; then
            cols=6
            rows=20
        else
            cols=9
            rows=19
            poly_cols=2
        fi
        echo "Running DLRM task..."
        ;;
    
    mobilenet)
        if [ "$cp_snark" = "poseidon" ]; then
            cols=20
            rows=24
            poly_cols=0
        else 
            cols=20
            rows=23
            poly_cols=1
        fi 
        echo "Running MobileNet task..."
        ;;
    
    vgg)
        if [ "$cp_snark" = "poseidon" ]; then
            cols=12
            rows=25
            poly_cols=0
        elif [ "$cp_snark" = "cp_link" ] || [ "$cp_snark" = "apollo_slow" ]; then
            cols=10
            rows=24
        else 
            cols=17
            rows=22
            poly_cols=4
        fi 
        echo "Running VGG task..."
        ;;
    
    gpt2)
        if [ "$cp_snark" = "poseidon" ]; then
            cols=20
            rows=27
            poly_cols=0
        elif [ "$cp_snark" = "cp_link" ] || [ "$cp_snark" = "apollo_slow" ]; then
            cols=10
            rows=27
        else 
            cols=13
            rows=25
            poly_cols=3
        fi 
        echo "Running GPT-2 task..."
        ;;
    
    diffusion)
        if [ "$cp_snark" = "poseidon" ]; then
            cols=16
            rows=26
            poly_cols=0
        elif [ "$cp_snark" = "cp_link" ] || [ "$cp_snark" = "apollo_slow" ]; then
            cols=15
            rows=25
        else 
            cols=29
            rows=24
            poly_cols=2
        fi 
        echo "Running Diffusion task..."
        ;;
    
    *)
        echo "Error: Unknown case '$case'"
        echo "Available cases: mnist, resnet, dlrm, mobilenet, vgg, gpt2, diffusion"
        exit 1
        ;;
esac

name_ipt=$name

case "$cp_snark" in
    no_com)
        cp_link=false
        poly_com=false
        poly_cols=0
        ;;

    cp_link_fast)
        cp_link=true
        poly_com=false
        slow=false
        poly_cols=0
        ;; 

    apollo_slow)
        cp_link=true
        poly_com=false
        slow=false
        poly_cols=1
        ;; 
    poly)
        cp_link=false
        poly_com=true
        ;;
    
    cp_link+)
        cp_link=true
        poly_com=false
        ;;
    
    poseidon)
        cp_link=false
        poly_com=false
        name=$name'_p'
        ;;

    cp_link)
        cp_link=true
        poly_com=false
        poly_cols=0
        ;;

    pedersen)
        cp_link=false
        poly_com=true
        pedersen=true
        # if [ "$poly_cols" -lt 3 ]; then
        #     poly_cols=3
        # fi
        ;;
    *)
        echo "Error: Unknown case '$case'"
        echo "Available cases: mnist, resnet, dlrm, mobilenet, vgg, gpt2, diffusion"
        exit 1
        ;;
esac

$dir/target/release/time_circuit $dir/examples/cifar/$name.msgpack $dir/examples/cifar/$name_ipt\_input.msgpack $pc_type $poly_com $poly_cols $rows $cols $cp_link $pedersen $num_runs $dir $slow
