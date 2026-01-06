# Artemis Evaluation Framework Documentation

## Project Overview

Artemis is an efficient Commit-and-Prove SNARK implementation for zkML (Zero-Knowledge Machine Learning). The repository contains implementations of Apollo and Artemis CP-SNARK constructions, built on top of the zkML framework.

## Evaluation Infrastructure: DoE-Suite

The evaluation framework uses **DoE-Suite** (Design of Experiments Suite), a tool for remote experiment management that orchestrates benchmarking experiments on:
- AWS EC2 instances
- ETHZ Euler cluster
- Existing remote machines

### Key DoE-Suite Concepts

- **Suite**: A collection of experiments defined in YAML configuration files
- **Factor**: A parameter that varies between different runs (following Design of Experiments methodology)
- **Level**: A specific value a factor takes in a run
- **ETL Pipeline**: Extract-Transform-Load pipeline for processing experiment results

### Machine Sizes

The evaluation uses three tiers of AWS EC2 instances:

| Size   | Instance Type   | vCPUs | RAM   | Volume | Use Case                          |
|--------|-----------------|-------|-------|--------|-----------------------------------|
| Small  | r6i.8xlarge     | 32    | 256GB | 250GB | mnist, resnet18, dlrm             |
| Medium | r6i.16xlarge    | 64    | 512GB | 250GB | mobilenet, vgg                    |
| Large  | r6i.32xlarge    | 128   | 1TB   | 250GB | diffusion, gpt2                   |

All machines use custom AMIs with pre-installed dependencies (ami-0714b2f0040bb3a42 in eu-north-1 Stockholm region).

## Experiment Structure

### Command Template

All experiments execute the following command:
```bash
/usr/bin/time -f '{wall_time: %E, max_rss: %M}' -o results/runtime_info.yaml \
  <code_dir>/src/bin/test.zsh <model> <pc_type> <cpsnark> <num_verifier_reps> <code_dir>
```

Where:
- `num_verifier_reps`: 5 (number of times to repeat verification for statistical significance)

### Experimental Factors

#### 1. Models (`model`)

| Model      | Size    | Description                      |
|------------|---------|----------------------------------|
| mnist      | ~8k     | MNIST digit classification       |
| resnet18   | ~280k   | ResNet-18 image classification   |
| dlrm       | ~764k   | Deep Learning Recommendation Model |
| mobilenet  | ~3.5m   | MobileNet image classification   |
| vgg        | ~15m    | VGG-16 image classification      |
| diffusion  | ~19.5m  | Diffusion model                  |
| gpt2       | ~81m    | GPT-2 language model             |

#### 2. Polynomial Commitment Schemes (`pc_type`)

- `kzg`: KZG polynomial commitment
- `ipa`: IPA (Inner Product Argument) polynomial commitment

#### 3. CP-SNARK Schemes (`cpsnark`)

| Scheme          | Label                    | Description                              |
|-----------------|--------------------------|------------------------------------------|
| `no_com`        | No Commitment            | Baseline without commitment              |
| `poly`          | Artemis (Horner)         | Polynomial evaluation using Horner's method |
| `pedersen`      | Artemis (Inner Product)  | Inner product based approach             |
| `cp_link`       | Lunar                    | CP-Link construction                     |
| `cp_link+`      | Apollo                   | Enhanced CP-Link construction            |
| `cp_link_fast`  | Lunar (Multi-column)     | Fast CP-Link with multi-column optimization |
| `poseidon`      | Poseidon                 | Poseidon hash-based commitment           |

## Complete Experiment Inventory

The evaluation comprises **36 unique experiment configurations** organized by CP-SNARK scheme, model size, and polynomial commitment type:

### 1. No Commitment Baseline (`nocom`)

- **nocom-small** (Suite ID: 1746596630)
  - Models: mnist, resnet18, dlrm
  - PC: KZG
  - Machine: r6i.8xlarge

- **nocom-small-ipa** (Suite ID: 1748257682)
  - Models: mnist, resnet18, dlrm
  - PC: IPA
  - Machine: r6i.8xlarge

- **nocom-med** (Suite ID: 1746992843)
  - Models: mobilenet, vgg
  - PC: KZG
  - Machine: r6i.16xlarge

- **nocom-med-ipa** (Suite ID: 1748257665)
  - Models: mobilenet, vgg
  - PC: IPA
  - Machine: r6i.16xlarge

- **nocom-large** (Suite ID: 1747133727)
  - Models: diffusion, gpt2
  - PC: KZG
  - Machine: r6i.32xlarge

- **nocom-large-ipa** (Suite ID: 1748257674)
  - Models: diffusion, gpt2
  - PC: IPA
  - Machine: r6i.32xlarge

### 2. Artemis Horner (`poly`)

- **poly-small** (Suite ID: 1748250734)
  - Models: mnist, resnet18, dlrm
  - PC: KZG
  - Machine: r6i.8xlarge

- **poly-small-ipa** (Suite ID: 1748252802)
  - Models: mnist, resnet18, dlrm
  - PC: IPA
  - Machine: r6i.8xlarge

- **poly-med** (Suite ID: 1746987414)
  - Models: mobilenet, vgg
  - PC: KZG
  - Machine: r6i.16xlarge

- **poly-med-ipa** (Suite ID: 1748252748)
  - Models: mobilenet, vgg
  - PC: IPA
  - Machine: r6i.16xlarge

- **poly-large** (Suite ID: 1747133715)
  - Models: diffusion, gpt2
  - PC: KZG
  - Machine: r6i.32xlarge

- **poly-large-ipa** (Suite ID: 1748252816)
  - Models: diffusion, gpt2
  - PC: IPA
  - Machine: r6i.32xlarge

### 3. Artemis Inner Product (`poly-ip` / `pedersen`)

- **poly-ip-small** (Suite ID: 1748596610)
  - Models: mnist, resnet18, dlrm
  - PC: KZG
  - Machine: r6i.8xlarge

- **poly-ip-small_ipa** (Suite ID: 1748517607)
  - Models: mnist, resnet18, dlrm
  - PC: IPA
  - Machine: r6i.8xlarge

- **poly-ip-med** (Suite ID: 1748596620)
  - Models: mobilenet, vgg
  - PC: KZG
  - Machine: r6i.16xlarge

- **poly-ip-med_ipa** (Suite ID: 1748517602)
  - Models: mobilenet, vgg
  - PC: IPA
  - Machine: r6i.16xlarge

- **poly-ip-large** (Suite ID: 1748596626)
  - Models: diffusion, gpt2
  - PC: KZG
  - Machine: r6i.32xlarge

- **poly-ip-large_ipa** (Suite ID: 1748517593)
  - Models: diffusion, gpt2
  - PC: IPA
  - Machine: r6i.32xlarge

### 4. Apollo (`apollo` / `cp_link+`)

- **apollo-small** (Suite ID: 1748545862)
  - Models: mnist, resnet18, dlrm
  - PC: KZG
  - Machine: r6i.8xlarge

- **apollo-med** (Suite ID: 1748545870)
  - Models: mobilenet, vgg
  - PC: KZG
  - Machine: r6i.16xlarge

- **apollo-large** (Suite ID: 1748545880)
  - Models: diffusion, gpt2
  - PC: KZG
  - Machine: r6i.32xlarge

### 5. Lunar (`cplink` / `cp_link`)

- **cplink-small** (Suite ID: 1748545816)
  - Models: mnist, resnet18, dlrm
  - PC: KZG
  - Machine: r6i.8xlarge

- **cplink-med** (Suite ID: 1748545807)
  - Models: mobilenet, vgg
  - PC: KZG
  - Machine: r6i.16xlarge

- **cplink-large** (Suite ID: 1748546060)
  - Models: diffusion, gpt2
  - PC: KZG
  - Machine: r6i.32xlarge

### 6. Lunar Fast Multi-column (`cplink_fast`)

- **cplink_fast-small** (Suite ID: 1748779025)
  - Models: mnist, resnet18, dlrm
  - PC: KZG
  - Machine: r6i.8xlarge

- **cplink_fast-med** (Suite ID: 1748784094)
  - Models: mobilenet, vgg
  - PC: KZG
  - Machine: r6i.16xlarge

- **cplink_fast-large** (Suite ID: 1748779019)
  - Models: diffusion, gpt2
  - PC: KZG
  - Machine: r6i.32xlarge

### 7. Poseidon Hash-based Commitment (`poseidon`)

- **poseidon-small** (Suite ID: 1748504139)
  - Models: mnist, resnet18, dlrm
  - PC: KZG
  - Machine: r6i.8xlarge

- **poseidon-small_ipa** (Suite ID: 1748507383)
  - Models: mnist, resnet18, dlrm
  - PC: IPA
  - Machine: r6i.8xlarge

- **poseidon-med** (Suite ID: 1748505016)
  - Models: mobilenet, vgg
  - PC: KZG
  - Machine: r6i.16xlarge

- **poseidon-med_ipa** (Suite ID: 1748505978)
  - Models: mobilenet, vgg
  - PC: IPA
  - Machine: r6i.16xlarge

- **xlarge-poseidon_kzg** (Suite ID: 1748511438)
  - Models: diffusion, gpt2
  - PC: KZG
  - Machine: r6i.32xlarge

- **xlarge-poseidon_ipa** (Suite ID: 1748504953)
  - Models: diffusion, gpt2
  - PC: IPA
  - Machine: r6i.32xlarge

## ETL Pipeline Configuration

The `poly_plots.yml` configuration defines three ETL pipelines for processing experiment results:

### 1. Overview Pipeline

**Purpose**: Quick summary of all experiments with error detection

**Configuration**:
- Experiments: All 36 experiments (*)
- Extractors:
  - `ErrorExpectedFileExtractor`: Checks for expected `output.csv` file
  - `IgnoreExtractor`: Ignores `stdout.log` and `runtime_info.yaml`
  - `CsvExtractor`: Extracts data from `output.csv` files
- Transformers: None
- Loaders:
  - `CsvSummaryLoader`: Creates summary CSV with empty rows skipped

### 2. Grid KZG Pipeline

**Purpose**: Generate grid plots comparing different CP-SNARK schemes for KZG commitment

**Configuration**:
- Experiments: All 36 experiments
- Extractors:
  - `CsvExtractor`: Extracts `output.csv`
  - `IgnoreExtractor`: Ignores stdout/stderr logs
  - `YamlExtractor`: Extracts runtime information
- Transformers:
  - `MergeRowsTransformer`: Merges data from multiple sources
  - `OsirisPreprocessTransformer`: Preprocessing specific to the Artemis/Osiris framework
- Loaders:
  - `CsvSummaryLoader`: Summary CSV output
  - `OsirisFactorLoader`: Factor analysis
  - `LargeTableLoader`: Detailed table generation
  - `MyCustomColumnCrossPlotLoader`: Custom plotting

### 3. Grid IPA Pipeline

**Purpose**: Generate grid plots comparing different CP-SNARK schemes for IPA commitment

**Configuration**: Similar to Grid KZG, but uses `MyCustomBrokenColumnCrossPlotLoader` for specialized IPA visualization.

### 4. Memory Pipeline

**Purpose**: Analyze memory consumption across experiments

**Configuration**:
- Same extractors and transformers as Grid pipelines
- Focuses on `max_rss` (maximum resident set size) metric
- Converts kibibytes to GB for reporting

## Metrics Tracked

### Primary Metrics (output.csv)

1. **prover_time_sec**
   - Prover execution time in seconds
   - Converted to minutes (÷60) for visualization
   - Unit: min

2. **verifier_time_sec**
   - Verifier execution time in seconds
   - Aggregated: mean and stddev over 5 repetitions
   - Unit: sec

3. **proof_size_bytes**
   - Size of generated proof
   - Converted to kilobytes (÷1000) for visualization
   - Unit: kB

4. **prover_time_sec_abs_factor_vs_nocom**
   - Absolute proving time difference compared to no-commitment baseline
   - Used for factor analysis
   - Unit: min

### Secondary Metrics (runtime_info.yaml)

1. **wall_time**
   - Total wall-clock time (format: HH:MM:SS)
   - Measured by `/usr/bin/time`

2. **max_rss**
   - Maximum resident set size (memory usage)
   - Measured in kibibytes, converted to GB (÷976563)
   - Unit: GB

## Visualization Configuration

### Plot Types

1. **Grid Plots** (grid_kzg, grid_ipa)
   - Rows: PC type (kzg, ipa)
   - Columns: Models
   - Bars: CP-SNARK schemes
   - Separate figures for each metric

2. **Memory Plots**
   - Same grid structure
   - Focuses on max_rss metric

### Data Filtering

Allowed combinations for visualization:
- Models: mnist, resnet18, dlrm, mobilenet, vgg, diffusion, gpt2
- CP-SNARKs: no_com, poly, pedersen, cp_link+, cp_link_fast, cp_link, poseidon
- PC types: kzg, ipa

## Running Experiments

### Local Execution

```bash
# Build
rustup override set nightly
cargo build --release

# Run single experiment
./src/bin/test.zsh <model> <pc_type> <cpsnark> <num_verifier_reps> <code_dir>

# Example: Artemis with MNIST on KZG
./src/bin/test.zsh mnist kzg poly 5 .
```

### Remote Execution with DoE-Suite

#### Quick Start

Make sure the working directory is at `./doe-suite`, and that the variables in `.envrc` are loaded.

```bash
# Navigate to doe-suite directory
cd doe-suite

# Load environment variables
source .envrc

# Run a specific suite
make run suite=<suite-name> id=new

# Example: Run all polynomial experiments on small models
make run suite=poly-small id=new

# Monitor progress
make status suite=<suite-name> id=last

# Process results
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview"

# Clean up resources
make clean
```

### Accessing Results

Experiment results are stored in:
- Raw results: `doe-suite-results/<suite-name>_<suite-id>/`
- Processed data: `doe-suite-results-super/`
- Plots: `doe-suite-results-super/paper_plots/`

Current result directories:
- `doe-suite-results-super/` - Main results
- `doe-suite-results-super-sp/` - Split plot variants
- `doe-suite-results-usenix/` - USENIX submission results

## Design Files Organization

### Individual Experiment Designs

Located in `doe-suite-config/designs/`:

- **Size-specific**: `{scheme}-{size}.yml` (e.g., `poly-small.yml`)
- **Generic templates**: `{scheme}.yml` (e.g., `nocom.yml`, `poly-kzg.yml`)
- **Model-specific**: `model-{model}.yml` (e.g., `model-diffusion.yml`)

### Shared Configuration

- `design_vars/general.yml`: Command templates and shared variables
- `group_vars/{size}/main.yml`: Machine configuration per size tier

## Notes and Considerations

1. **Repetitions**: All experiments run with `n_repetitions: 1` for the prover, but verifier is repeated 5 times for statistical significance.

2. **IPA Experiments**: Some experiments note that "POLY IPA [mnist, resnet, dlrm, mobilenet, vgg] needs to be re-run because of a faster verifier implementation."

3. **Model Artifacts**: Model definitions (msgpack files) are stored in S3 bucket `s3://pps-artemis-artifacts/models` (360MB total).

4. **Setup Parameters**:
   - IPA parameters: ~15GB (stored in `params_ipa/`)
   - KZG parameters: ~64GB (stored in `params_kzg/`)
   - Auto-generated if not present

5. **Suite IDs**: Each experiment run gets a unique Suite ID (Unix timestamp), allowing multiple runs to be compared.

## Key Insights from Configuration

1. **Comprehensive Coverage**: The evaluation covers 7 models × 2 PC schemes × 6 CP-SNARK variants = 84 potential combinations, with 36 actively configured.

2. **Performance Tiers**: Three machine sizes ensure appropriate resources for different model complexities.

3. **Statistical Rigor**: 5 verifier repetitions provide statistical confidence in verification times.

4. **Reproducibility**: DoE-Suite configuration makes experiments fully reproducible with single commands.

5. **Visualization Focus**: Heavy emphasis on comparative visualization with colorblind-friendly palettes and clear labeling.
