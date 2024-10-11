# Artemis: Efficient Commit-and-Prove SNARKs for zkML

This repository contains the implementation of Apollo and Artemis, our CP-SNARK constructions from [our paper](https://arxiv.org/abs/2409.12055).
The underlying zkML implementation is based on the [zkML](https://github.com/uiuc-kang-lab/zkml) paper/repo.

## Requirements
Rust

### Artifacts
This repository requires several artifacts related to the model definitions used in the evaluation.
The model definitions (msgpack files) are stored in `examples/models` (360MB) in an S3 bucket,
and can be downloaded using the following command:
```sh
aws s3 sync s3://pps-artemis-artifacts/models examples/models
```

The setup parameters for the polynomial commitments, `params_ipa` (15GB), `params_kzg` (64GB), take up a significant amount of time
for the larger models to generate (although less than the time of generating the proof).
To optimize this, it might be useful to generate the setup parameters once and store them for later use.
The script looks for the setup parameters in the `params_ipa` and `params_kzg` directories in the root code directory.
In case the setup parameters do not exist in the expected locations, they will be automatically generated when running the benchmarking script.

## Quickstart (local)

To run with `model` for polynomial commitment type (`pc_type`) and CP-SNARK type `cp_snark` do the following:

model takes values in `['mnist', 'resnet', 'dlrm', 'mobilenet', 'vgg', 'gpt2', 'diffusion']`

polynomial commitment takes values in `['kzg', 'ipa']`

CP-SNARK takes values in `['nocom', 'poly', 'cp_link', 'pos', 'cp_link_plus']`

```sh
rustup override set nightly
cargo build --release

./src/bin/test.zsh ‘model_name’  ‘pc_type’  ‘cp_snark’  ‘num_runs’ 1 .

```

## Running with Doe Suite (remote)
The evaluation is built using [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite), which allows straightforward reproducibility of results by defining experiments in a configuration file (suite) and executing them on a set of machines. We provide the necessary DoE-Suite commands to reproduce all results. However, it is also possible to obtain the individual commands used to invoke the framework and run them manually.
For details on running the suites, please refer to the [DoE-Suite documentation](https://nicolas-kuechler.github.io/doe-suite/installation.html#base-installation).

Our artifact contains the following suites:
<details>
    <summary>Available Suites</summary>

| Suite                                                                               | Models                                                           | Est. Duration |
|-------------------------------------------------------------------------------------|------------------------------------------------------------------|---------------|
| [model-small](doe-suite-config/designs/model-small.yml)                             | mnist, resnet18, dlrm                                            | TODO          |
| [model-mobilenet](doe-suite-config/designs/mobilenet.yml)                           | MobileNet                                                        | TODO          |
| [model-vgg](doe-suite-config/designs/vgg.yml)                                       | VGG                                                              | TODO            |
| [model-diffusion](doe-suite-config/designs/diffusion.yml)                           | Diffusion                                                        | TODO           |
| [model-gpt2](doe-suite-config/designs/model-gpt2.yml)                               | GPT-2                                                            | TODO           |

__TODO__: Add CPLink models and poly_ipa?

</details>