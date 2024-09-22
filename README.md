# Artemis: Efficient Commit-and-Prove SNARKs for zkML

This repository contains the implementation of Apollo and Artemis, our CP-SNARK constructions from [our paper](https://arxiv.org/abs/2409.12055).
The underlying zkML implementation is based on the [zkML](https://github.com/uiuc-kang-lab/zkml) paper/repo.

## Quickstart

To run with 'model' for polynomial commitment type ('pc_type') and CP-SNARK type 'cp_snark' do the following:

model takes values in ['mnist', 'resnet', 'dlrm', 'mobilenet', 'vgg', 'gpt2', 'diffusion']

polynomial commitment takes values in ['kzg', 'ipa']

CP-SNARK takes values in ['nocom', 'poly', 'cp_link', 'pos', 'cp_link_plus']

```sh
rustup override set nightly
cargo build --release

./src/bin/test.zsh ‘model_name’  ‘pc_type’  ‘cp_snark’  ‘num_runs’ 1 .

```
