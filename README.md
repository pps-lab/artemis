# Artemis: Efficient Commit-and-Prove SNARKs

This repository contains the implementation of Artemis, our commit-and-prove SNARK construction, and the
benchmarks of the evaluation:

- **zkML**: Artemis on halo2 (KZG and IPA) for seven neural networks from MNIST to GPT-2, against the
  baselines Lunar, LegoSNARK, and Poseidon and SHA-256 commitments recomputed in the circuit.
- **TLSNotary**: a predicate proof over a committed TLS response of 520 B to 64 KiB (Noir circuits on
  ProveKit and Barretenberg), comparing Artemis with the SHA-256 commitment TLSNotary uses today, Keccak-256,
  BLAKE3, Poseidon and LegoSNARK ([tlsn-bench/](tlsn-bench/README.md)).

The security proof is in [artemis_security_proof.pdf](artemis_security_proof.pdf).

The zkML circuits (`src/layers`, `src/gadgets`, `python/` and most of `src/model.rs`) come from the
[zkML](https://github.com/uiuc-kang-lab/zkml) implementation. `halo2/` is a fork of halo2 with the extensions
the CP-SNARKs need: extra openings in the multi-open argument, and faster MSM, FFT and lookup code.

## Repository layout

| Path | Content |
|---|---|
| `src/` | zkML circuits, the CP-SNARK gadgets and commitments (`src/gadgets`, `src/commitments`), the prover/verifier driver `src/bin/time_circuit.rs` |
| `src/bin/test.zsh` | entry point of every benchmark run (build, setup, prove, verify; writes `results/output.csv`) |
| `halo2/` | halo2 fork (`halo2_proofs`, `halo2curves`) |
| `doe-suite-config/` | DoE-Suite experiment designs (`designs/`), machine types (`group_vars/`), setup roles and the ETL configs (`super_etl/`) |
| `doe-suite-results.zip` | raw results of all suites used in the paper |
| `doe-suite-results-super/` | tables and plots of the paper, generated from the raw results |
| `tlsn-bench/` | TLSNotary application benchmark (Noir circuits on ProveKit and Barretenberg); see [tlsn-bench/README.md](tlsn-bench/README.md) |

## Requirements

- Rust: the nightly toolchain is pinned in `rust-toolchain.toml` and picked up automatically by `cargo`
  (do not use `cargo +nightly`).
- `zsh`, and the AWS CLI to download the model files.
- For remote runs: Python 3 and the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite) requirements
  (git submodule `doe-suite/`).

### Model files

The model definitions (msgpack files, 360 MB) are in a public S3 bucket and are expected in `examples/cifar`:
```sh
aws s3 sync s3://anonymized-artemis-artifacts/examples/models examples/cifar --no-sign-request
```

### Setup parameters

The KZG and IPA parameters are generated on first use and stored in `params_kzg_Bn256/`, `params_kzg_Bls12/`
and `params_ipa/` in the code directory, then reused. For the largest models they take up to 64 GB (KZG) and
15 GB (IPA), and generating them takes a while (less than proving). Setup time is not included in the measured
prover time.

## Running a single benchmark (local)

```sh
cargo build --release
./src/bin/test.zsh <model> <pc_type> <cp_snark> <num_verifier_runs> <code_dir>
# example: Artemis on MNIST with KZG
./src/bin/test.zsh mnist kzg poly 5 .
```

- `model`: `mnist`, `resnet18`, `dlrm`, `mobilenet`, `vgg`, `diffusion`, `gpt2`
- `pc_type`: `kzg`, `ipa`
- `cp_snark`:

| `cp_snark` | Scheme in the paper | PC |
|---|---|---|
| `no_com` | no commitment (baseline) | KZG, IPA |
| `poly` | Artemis (Horner) | KZG, IPA |
| `poly_merged` | Artemis (Horner over the weights in the general advice columns) | KZG, IPA |
| `pedersen` | Artemis (inner product) | KZG, IPA |
| `cp_link` | Lunar | KZG |
| `cp_link_fast` | Lunar (multi-column) | KZG |
| `cp_link_lego` | LegoSNARK CP_link | KZG |
| `poseidon` | Poseidon commitment in the circuit | KZG, IPA |
| `sha256` | SHA-256 commitment in the circuit | KZG, IPA |

The run writes `results/output.csv` with the prover time, the verifier time (`num_verifier_runs` repetitions)
and the proof size. The prover time is halo2 `create_proof` plus the scheme's own proving work; committing to
the weights and the setup are not counted. `MOCK_VERIFY=1` additionally checks the circuit with halo2's
`MockProver`.

The larger models need a lot of memory: mnist, resnet18 and dlrm were run with 256 GB, mobilenet and vgg with
512 GB, diffusion and gpt2 with 1 TB (Poseidon on diffusion and gpt2 with 2 TB).

## Running the evaluation with DoE-Suite (remote, AWS)

Each experiment of the paper is a DoE-Suite suite in `doe-suite-config/designs/`. A suite starts EC2 instances
(eu-west-1, stock Ubuntu 24.04; `roles/setup-base` installs the dependencies), clones this repository on them,
runs `test.zsh` for every combination of factors and fetches the results. For setup see the
[DoE-Suite documentation](https://nicolas-kuechler.github.io/doe-suite/installation.html#base-installation).

1. Put this repository in a git repository that the instances can clone, and set its URL and branch in
   `doe-suite-config/group_vars/all/main.yml` (`git_remote_repository`).
2. Check out the submodule (`git submodule update --init doe-suite`) and set the DoE-Suite environment
   variables, e.g. in `doe-suite-config/.envrc`:
   ```sh
   export DOES_PROJECT_DIR=<path to this repository>
   export DOES_PROJECT_ID_SUFFIX=<short id, used to tag the AWS resources>
   export DOES_SSH_KEY_NAME=<EC2 key pair name>
   export DOES_AWS_USER=ubuntu
   ```
3. Run a suite and fetch its results into `doe-suite-results/`:
   ```sh
   cd doe-suite
   make run suite=poly-small id=new
   make status suite=poly-small id=last
   ```

Suites used in the paper (machine types in `doe-suite-config/group_vars/`):

| Suites | Models | Machine |
|---|---|---|
| `nocom-small`, `poly-small`, `poly-ip-small`, `cplink-small`, `cplink_fast-small`, `lego-small`, `poseidon-small`, `sha256-small` (and IPA variants) | mnist, resnet18, dlrm | r6i.8xlarge |
| `nocom-med`, `poly-med`, `poly-ip-med`, `cplink-med`, `cplink_fast-med`, `lego-med`, `poseidon-med`, `sha256-med` (and IPA variants) | mobilenet, vgg | r6i.16xlarge |
| `nocom-large`, `poly-large`, `lego-large`, `cplink_fast-large`, `sha256-large` (and IPA variants) | diffusion, gpt2 | r6i.32xlarge |
| `xlarge-poseidon_kzg`, `xlarge-poseidon_ipa` | diffusion, gpt2 | x2idn.metal |
| `tlsn`, `tlsn-lego`, `tlsn-poseidon1` | TLSNotary benchmark ([tlsn-bench](tlsn-bench/README.md)) | r6i.8xlarge |

## TLSNotary benchmark

`tlsn-bench/` is self-contained: it builds a patched ProveKit (with the Artemis hook) and, for the LegoSNARK
rows, a patched Barretenberg, and runs every commitment scheme on the same Noir predicate circuit for the
payload sizes of the tlsn-extension plugins (520 B to 64 KiB). Locally:
```sh
cd tlsn-bench
./run.sh setup && ./run.sh test && ./run.sh small && ./run.sh plot
```
On AWS the suites `tlsn`, `tlsn-lego` and `tlsn-poseidon1` run the same commands. Rows, measurement method,
results and caveats are in [tlsn-bench/README.md](tlsn-bench/README.md); the tables are in
`tlsn-bench/results/summary.md` (local) and `tlsn-bench/results/aws/summary.md` (AWS).

## Results

`doe-suite-results.zip` contains the raw results of the 41 suites behind the paper's tables and plots: the zkML
suites, the SHA-256 and LegoSNARK baselines, and the TLSNotary suites `tlsn`, `tlsn-lego` and `tlsn-poseidon1`.
The generated outputs are in `doe-suite-results-super/`:

- `combined_plots/prover_table/`: prover time of all schemes on the zkML models and on TLSNotary
- `combined_plots/results_table/`: prover time, verifier time and proof size
- `rerun_plots/grid/`: per-model plots of prover time, verifier time and proof size

To regenerate them, unzip the raw results at the repository root and run the super-ETL from `doe-suite/`:
```sh
unzip doe-suite-results.zip
cd doe-suite
make etl-super config=combined_plots out=../doe-suite-results-super pipelines="prover_table results_table"
make etl-super config=rerun_plots out=../doe-suite-results-super
```
