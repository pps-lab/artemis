---
name: doe-suite-experiments
description: Run remote experiments (and process results) using the doe-suite framework. Use when running experiments remotely or when processing experiment results to create plots or other diagrams.
---

This skill orchestrates the complete workflow for running and operating a DoE-Suite experiment:

1. Validates the suite exists and is configured
2. Runs experiment suites on remote infrastructure (e.g., AWS)
3. Monitors progress and collects results
3. Runs and edits super-etl pipelines
4. Edits and creates new suites and configurations.

## Prerequisites
The doe-suite invocation will tell you if things are improperly configured. Do not check this before.

- `.envrc` file must be configured with:
  - `DOES_PROJECT_DIR`: Project root directory
  - `DOES_PROJECT_ID_SUFFIX`: Project identifier suffix
  - `DOES_SSH_KEY_NAME`: AWS SSH key name
  - `DOES_CLOUD`: Cloud provider (aws)
  - `AWS_PROFILE`: AWS credentials profile
  - `AWS_REGION`: AWS region to deploy in
- AWS credentials must be configured
- SSH key must exist in the target AWS region

## Available Suites
Available suites are defined in `doe-suite-config` as yaml files.

### Small Model Suites (r6i.8xlarge - 32 vCPU, 256GB RAM)
- `apollo-small`: mnist, resnet18, dlrm with Apollo (cp_link+) + KZG
- `poly-small`: mnist, resnet18, dlrm with Artemis (poly) + KZG
- `poly-small-ipa`: mnist, resnet18, dlrm with Artemis + IPA
- `poly-ip-small`: mnist, resnet18, dlrm with Artemis Inner Product (pedersen) + KZG
- `poly-ip-small_ipa`: mnist, resnet18, dlrm with Artemis Inner Product + IPA
- `nocom-small`: mnist, resnet18, dlrm with no commitment baseline + KZG
- `nocom-small-ipa`: mnist, resnet18, dlrm with no commitment + IPA
- `cplink-small`: mnist, resnet18, dlrm with Lunar (cp_link) + KZG
- `cplink_fast-small`: mnist, resnet18, dlrm with Lunar Fast + KZG
- `poseidon-small`: mnist, resnet18, dlrm with Poseidon + KZG
- `poseidon-small_ipa`: mnist, resnet18, dlrm with Poseidon + IPA

### Medium Model Suites (r6i.16xlarge - 64 vCPU, 512GB RAM)
- `apollo-med`: mobilenet, vgg with Apollo + KZG
- `poly-med`: mobilenet, vgg with Artemis + KZG
- `poly-med-ipa`: mobilenet, vgg with Artemis + IPA
- `poly-ip-med`: mobilenet, vgg with Artemis Inner Product + KZG
- `poly-ip-med_ipa`: mobilenet, vgg with Artemis Inner Product + IPA
- `nocom-med`: mobilenet, vgg with no commitment + KZG
- `nocom-med-ipa`: mobilenet, vgg with no commitment + IPA
- `cplink-med`: mobilenet, vgg with Lunar + KZG
- `cplink_fast-med`: mobilenet, vgg with Lunar Fast + KZG
- `poseidon-med`: mobilenet, vgg with Poseidon + KZG
- `poseidon-med_ipa`: mobilenet, vgg with Poseidon + IPA

### Large Model Suites (r6i.32xlarge - 128 vCPU, 1TB RAM)
- `apollo-large`: diffusion, gpt2 with Apollo + KZG
- `poly-large`: diffusion, gpt2 with Artemis + KZG
- `poly-large-ipa`: diffusion, gpt2 with Artemis + IPA
- `poly-ip-large`: diffusion, gpt2 with Artemis Inner Product + KZG
- `poly-ip-large_ipa`: diffusion, gpt2 with Artemis Inner Product + IPA
- `nocom-large`: diffusion, gpt2 with no commitment + KZG
- `nocom-large-ipa`: diffusion, gpt2 with no commitment + IPA
- `cplink-large`: diffusion, gpt2 with Lunar + KZG
- `cplink_fast-large`: diffusion, gpt2 with Lunar Fast + KZG
- `xlarge-poseidon_kzg`: diffusion, gpt2 with Poseidon + KZG
- `xlarge-poseidon_ipa`: diffusion, gpt2 with Poseidon + IPA

## Workflow Steps

### 1. Launch Experiment

```bash
source .envrc && make run suite=<suite-name> id=new
```

This will:
- Create a new suite ID (Unix timestamp)
- Set up cloud networking (VPC, subnet, security groups)
- Launch instance(s) with appropriate size
- Install dependencies
- Clone code from GitHub
- Enqueue all experiment jobs
- Start execution

**Expected Duration**:
- Small suites: 15-30 minutes per model
- Medium suites: 1-3 hours per model
- Large suites: 4-12 hours per model

### 2. Monitor Progress
Make sure to keep monitoring runs continuously

```bash
source .envrc && make status suite=<suite-name> id=last
```

This shows:
- Total jobs completed/remaining
- Current job status (queued/running/finished)
- Any errors or warnings

**Check periodically** (every 5-10 minutes for small models, 30-60 minutes for large models)

### 3. Review Results

Once completed, results are in:
```
doe-suite-results/<suite-name>_<suite-id>/exp_<exp-name>/run_<N>/rep_0/small/host_0/
```

Key files:
- `output.csv`: Prover time, verifier time, proof size
- `runtime_info.yaml`: Wall time and max memory (RSS)
- `stderr.log`: Compilation warnings (normal)
- `stdout.log`: Execution output

### 4. Process Results with ETL
Available Super ETL configurations are available in `doe-suite-config/super_etl`.
Each configuration file defines pipelines, and experiments to load at the top:
```
$SUITE_ID$:
  nocom-small: "<run_id1>"
  poly-small: "<run_id2>"
```
To add results from a new run, make sure to update the `run_id` to the new run.

```bash
source .envrc && make etl-super config=poly_plots out=./doe-suite-results-super pipelines="overview"
```

This aggregates results from all experiments and generates:
- Combined CSV files
- Summary statistics
- Plots (if configured)
- 
The output location is `./doe-suite-results-super`


## Common Issues & Solutions

### SSH Key Not Found
**Error**: `InvalidKeyPair.NotFound: The key pair 'xxx' does not exist`

**Solution**:
- Verify `DOES_SSH_KEY_NAME` in `.envrc` matches a key in the target AWS region
- Update `.envrc` to use correct key name
- Ensure `AWS_REGION` matches where your key exists

### Region Mismatch
**Error**: AMI or resources not available in region

**Solution**:
- Check `AWS_REGION` in `.envrc` matches your configuration
- Verify AMI IDs in `doe-suite-config/group_vars/<size>/main.yml` exist in target region

### Code Build Failures
**Error**: Compilation errors during setup

**Solution**:
- Ensure GitHub repository `main` branch is in working state
- Check that all dependencies are specified in the setup role
- Review `stderr.log` from failed experiment for specific errors

### Experiment Panics
**Error**: `thread 'main' panicked at src/bin/time_circuit.rs`

**Solution**:
- Check that `test.zsh` script passes correct number of arguments to binary
- Verify argument order matches what `time_circuit.rs` expects
- Test locally first with same parameters

### ETL Pipeline Errors
**Error**: `file=runtime_info.yaml matches no extractor`

**Solution**:
- Ensure suite design includes `YamlExtractor` in extractors list
- Check that experiment completed successfully (output files exist)
- Verify ETL configuration in suite design matches expected output structure

## Results Interpretation

### Metrics Collected

1. **Prover Time** (`prover_time_sec`)
   - Time to generate the proof
   - Measured in seconds, displayed in minutes
   - Main performance indicator

2. **Verifier Time** (`verifier_time_sec`)
   - Time to verify the proof
   - Averaged over 5 repetitions with stddev
   - Measured in seconds

3. **Proof Size** (`proof_size_bytes`)
   - Size of generated proof
   - Measured in bytes, displayed in kB
   - Communication cost indicator

4. **Memory Usage** (`max_rss`)
   - Maximum resident set size
   - Measured in kibibytes, displayed in GB
   - Peak memory consumption

### CP-SNARK Schemes Comparison

- **No Commitment (no_com)**: Baseline without commitment (fastest, but not CP-SNARK)
- **Artemis Horner (poly)**: Polynomial commitment using Horner evaluation
- **Artemis Inner Product (pedersen)**: Inner product argument approach
- **Apollo (cp_link+)**: Enhanced CP-Link construction
- **Lunar (cp_link)**: Basic CP-Link construction
- **Lunar Fast (cp_link_fast)**: Optimized multi-column CP-Link
- **Poseidon (poseidon)**: Poseidon hash-based commitment

## Important Notes

⚠️ **Cost Warning**: Running experiments on AWS incurs costs. Instances are charged per hour:
- r6i.8xlarge: ~$2.00/hour
- r6i.16xlarge: ~$4.00/hour
- r6i.32xlarge: ~$8.00/hour

💡 **Infrastructure Cleanup**: The system automatically terminates instances after completion. To manually clean up:
```bash
source .envrc && make clean-cloud
```

🔄 **Retrying Failed Runs**: If an experiment fails, fix the code and re-run with `id=new` to create a fresh run.

📊 **Comparing Results**: Use `make etl-super` to aggregate results from multiple suite IDs for comparison.

## Example Workflow

```bash
# 1. Launch experiment
cd doe-suite
source .envrc
make run suite=apollo-small id=new

# 2. Wait ~45 minutes, then check status
make status suite=apollo-small id=last

# 3. Once complete, review results
ls -lh ../doe-suite-results/apollo-small_<id>/exp_apollo_kzg_small/

# 4. Process results with ETL
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview"

# 5. View aggregated results
cat ../doe-suite-results-super/poly_plots/overview/overview.csv
```

## Working Directory

All doe-suite `make` commands must be run from the `doe-suite` directory:
```bash
cd /Users/hidde/PhD/zkml/artemis/doe-suite
```
