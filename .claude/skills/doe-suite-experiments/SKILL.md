---
name: doe-suite-experiments
description: Run remote experiments (and process results) using the doe-suite framework. Use when running experiments remotely or when processing experiment results to create plots or other diagrams.
---

This skill orchestrates the complete workflow for running DoE-Suite experiments:

1. Validates suite configuration and launches remote infrastructure
2. Executes experiments on AWS EC2, ETHZ Euler, or existing remote machines
3. Monitors progress and automatically collects results
4. Processes results with ETL (Extract-Transform-Load) pipelines
5. Generates visualizations and aggregated data
6. Creates and edits suite designs and ETL configurations

## Key Concepts

**DoE-Suite** is a tool for remote experiment management following Design of Experiments (DoE) methodology:

- **Suite**: A collection of experiments defined in a YAML configuration file
- **Experiment**: A set of runs with different configurations
- **Factor**: A parameter that varies between runs (e.g., model, algorithm)
- **Level**: A specific value a factor takes in a run (e.g., mnist, resnet18)
- **Run**: A single execution with a specific combination of factor levels
- **ETL Pipeline**: Extract-Transform-Load pipeline that processes results locally
- **Super-ETL**: Aggregates and processes results across multiple suite runs

## Prerequisites

The doe-suite will report configuration issues when invoked. Key requirements:

- `.envrc` file with environment variables:
  - `DOES_PROJECT_DIR`: Project root directory
  - `DOES_PROJECT_ID_SUFFIX`: Project identifier suffix
  - `DOES_SSH_KEY_NAME`: AWS SSH key name
  - `DOES_CLOUD`: Cloud provider (aws)
  - `AWS_PROFILE`: AWS credentials profile
  - `AWS_REGION`: AWS region to deploy in
- AWS credentials properly configured
- SSH key must exist in the target AWS region

**Working Directory**: All doe-suite commands must be run from the `doe-suite` directory within your project.

## Available Suites

Suites are defined as YAML files in `doe-suite-config/designs/`. To discover available suites:

```bash
# List all suite design files
ls doe-suite-config/designs/*.yml

# Get information about configured suites
make info

# View what a specific suite will execute
make design suite=<suite-name>
```

Each suite typically defines:
- **Host types**: Machine sizes and configurations (e.g., small, medium, large)
- **Factors**: Parameters that vary between runs (e.g., model, algorithm, dataset)
- **Experiments**: Collections of runs with different factor combinations
- **ETL pipelines**: How to process and visualize results

Machine configurations are defined in `doe-suite-config/group_vars/<host-type>/main.yml`.

## Workflow Steps

### 1. Design and Validate Suite (Optional)

Before running experiments, you can validate your suite design:

```bash
# View all commands that will be executed
make design suite=<suite-name>

# Validate design and show with default values
make design-validate suite=<suite-name>
```

This helps catch configuration errors before launching expensive cloud resources.

### 2. Launch Experiment

```bash
source .envrc && make run suite=<suite-name> id=new
```

**What happens automatically:**
1. Creates a new suite ID (Unix timestamp, e.g., 1767685699)
2. Sets up cloud networking (VPC, subnet, security groups, instance)
3. Launches EC2 instance(s) with appropriate size
4. Installs dependencies (Rust, AWS CLI, task spooler)
5. Clones code from GitHub (`main` branch)
6. Builds the project with `cargo build --release`
7. Downloads model artifacts from S3
8. Enqueues all experiment jobs to task spooler
9. Starts execution sequentially

**Expected Duration**:
- Small suites: 15-30 minutes per model
- Medium suites: 1-3 hours per model
- Large suites: 4-12 hours per model

**Options:**
- `id=new`: Start a new run
- `id=last`: Continue the most recent run
- `id=<suite_id>`: Continue a specific run
- `make run-keep`: Keep instances running after completion (don't forget to clean up!)

### 3. Monitor Progress

**Continuously monitor** to detect issues early:

```bash
source .envrc && make run suite=<suite-name> id=last
```

This script terminates once all results are done. 
It keeps checking status of the experiments and downloading results, up until a timeout. 
Then the script can be re-invoked to continue monitoring.

**Output at the end shows:**
- Total jobs completed/remaining (e.g., "3/3 jobs")
- Progress percentage
- ETL pipeline warnings/errors
- Current job status (queued/running/finished)

### 4. Review Raw Results

Results are automatically downloaded to:
```
doe-suite-results/<suite-name>_<suite-id>/exp_<exp-name>/run_<N>/rep_<R>/<host-type>/host_<H>/
```

**Directory structure:**
- `<suite-name>_<suite-id>`: Suite and timestamp ID
- `exp_<exp-name>`: Experiment name from design
- `run_<N>`: Run index (starts at 0) - different factor level combinations
- `rep_<R>`: Repetition index (starts at 0)
- `<host-type>`: Machine type (small/medium/large)
- `host_<H>`: Host index (starts at 0)

**Key files:**
- `output.csv`: Experiment metrics (prover time, verifier time, proof size)
- `runtime_info.yaml`: Wall time and max memory (RSS)
- `stdout.log`: Program output
- `stderr.log`: Compilation warnings (usually normal)

**Example:**
`doe-suite-results/nocom-small_1767685699/exp_poly_kzg_small/run_0/rep_0/small/host_0/output.csv`

### 5. Process Results with ETL

#### Automatic ETL (Per-Suite)

ETL pipelines defined in the suite design run **automatically** when results are downloaded. They process individual suite results.

To manually re-run (useful for tweaking plots):
```bash
make etl suite=<suite-name> id=last

# Or develop/test ETL changes:
make etl-design suite=<suite-name> id=last
```

#### Super-ETL (Cross-Suite Aggregation)

Super-ETL configurations (`doe-suite-config/super_etl/`) aggregate results from **multiple suite runs** and generate comparative visualizations.

**Configuration structure:**
```yaml
$SUITE_ID$:
  suite-name-1: "1767685699"
  suite-name-2: "1748250734"
  suite-name-3: "1748545862"

$PIPELINES$:
  pipeline-name:
    experiments: "*"  # or list specific experiments
    extractors:
      CsvExtractor: {file_regex: ["*.csv"]}
      IgnoreExtractor: {file_regex: ["*.log"]}
    transformers: []
    loaders:
      CsvSummaryLoader: {skip_empty: True}
```

**To add new results:**
1. Update suite IDs in the super-ETL config file (`doe-suite-config/super_etl/<config>.yml`)
2. Run the super-ETL pipeline

```bash
source .envrc && make etl-super config=<config-name> out=<output-dir> pipelines="<pipeline1>,<pipeline2>"
```

**Common output locations:**
- `<output-dir>/<config>/overview/overview.csv`
- `<output-dir>/<config>/<pipeline-name>/*.pdf`
- `<output-dir>/<config>/<pipeline-name>/*.html`
- `<output-dir>/<config>/<pipeline-name>/*.csv`

When you add new results to the super-ETL config, verify that the outputs contain the new data by checking the csv files.

### 6. Cleanup

**Automatic cleanup**: Instances are terminated automatically after suite completion (exit code 0).

**Manual cleanup** (if suite fails or is interrupted):
```bash
make clean
```

**Always verify** in AWS Console that instances are terminated!


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

Experiments may run succesfully but the remote code may have errored, which should be visible in stderr.
If this is the case, the result files should provide more information. 
Double check with me whether your proposed fix makes sense. 
Always commit and push the fix, so it gets pulled on the remote server as well when re-invoking doe-suite

### ETL Pipeline Errors
**Error**: `file=runtime_info.yaml matches no extractor`

**Root cause**: The ETL pipeline encounters a file that doesn't match any configured extractor.

**Solution**:
1. **Ignore the file**: Add it to `IgnoreExtractor` in the suite design:
   ```yaml
   $ETL$:
     overview:
       extractors:
         IgnoreExtractor: {file_regex: ["stdout.log", "stderr.log", "runtime_info.yaml"]}
   ```

2. **Extract the file**: Add appropriate extractor (e.g., `YamlExtractor`) if you need the data:
   ```yaml
   extractors:
     YamlExtractor: {file_regex: ["runtime_info.yaml"]}
   ```

3. **Verify files exist**: Check that the experiment actually produced the expected output files

**Note**: ETL runs both automatically (after each job) and during `make status`. A failing ETL doesn't block experiment execution but prevents result aggregation.

## Understanding Results

### Result Files

Each experiment produces files based on the commands executed and ETL extractors configured:

**Common file types:**
- **CSV files**: Structured data (extracted by `CsvExtractor`)
- **JSON files**: Structured data (extracted by `JsonExtractor`)
- **YAML files**: Configuration/metadata (extracted by `YamlExtractor`)
- **Log files**: stdout/stderr output (usually ignored or extracted separately)
- **Custom files**: Any output your experiment produces

**Runtime metrics** (if using `/usr/bin/time`):
- **Wall time**: Total elapsed time
- **max_rss**: Maximum resident set size (memory usage)
- **CPU usage**: User and system time

### ETL Processing Stages

1. **Extract**: Aggregate raw result files into a pandas DataFrame
   - Each file matches one extractor by regex pattern
   - Files that don't match any extractor cause errors (add to `IgnoreExtractor`)

2. **Transform**: Process and aggregate data
   - `RepAggTransformer`: Aggregate over repetitions (mean, std, etc.)
   - `MergeRowsTransformer`: Combine data from multiple sources
   - Custom transformers: Project-specific data processing

3. **Load**: Generate outputs
   - `CsvSummaryLoader`: Save processed data as CSV
   - Plotting loaders: Generate visualizations
   - Custom loaders: Project-specific outputs

## Important Notes

⚠️ **Cost Warning**: Running experiments on AWS incurs costs. EC2 instances are charged per hour.
- Check [AWS EC2 pricing](https://aws.amazon.com/ec2/pricing/) for your region and instance types
- Set up billing alerts in AWS Console
- Always verify instances are terminated after experiments

💡 **Infrastructure Cleanup**:
- **Automatic**: Instances terminate when suite completes successfully
- **Manual**: Use `make clean` if suite fails or is interrupted
- **Verify**: Always check AWS Console that resources are terminated

🔄 **Retrying Failed Runs**:
- Fix code issues in your repository
- Re-run with `id=new` to create a fresh run
- Or continue with `id=last` if the suite partially completed

📊 **Comparing Results**:
- Use Super-ETL to aggregate results from multiple suite runs
- Update suite IDs in `doe-suite-config/super_etl/<config>.yml`
- Run `make etl-super config=<config> out=<output-dir>`

🐛 **Debugging**:
- Use `make design suite=<suite-name>` to validate before running
- Check `doe-suite-results/<suite>_<id>/` for stdout/stderr logs
- Use `make run-keep` to keep instances alive for debugging (remember to clean up!). 
This is particularly useful when running new code, where we might have bugs that require re-runs.

**Other tips**:
- Check the docs in `doe-suite/docs/` for more details on how to edit ETL workflows, server provisioning and other doe-suite syntax.
- It is possible to run multiple suites in parallel. Just run them in different sub-processes or agents.
- Do not run bash commands with sleep to monitor. `make run suite=<suite-name> id=last` is the correct way to monitor progress.

## Example Workflow

```bash
# 1. Navigate to doe-suite directory
cd doe-suite
source .envrc

# 2. Validate suite design
make design suite=<suite-name>

# 3. Launch experiment
make run suite=<suite-name> id=new

# 4. Monitor progress periodically
make status suite=<suite-name> id=last

# 5. Once complete (X/X jobs), review raw results
ls -lh ../doe-suite-results/<suite-name>_<id>/

# 6. Process results with Super-ETL (if configured)
make etl-super config=<config-name> out=../doe-suite-results-aggregated

# 7. View aggregated results
cat ../doe-suite-results-aggregated/<config>/overview/overview.csv

# 8. Verify cleanup (should be automatic)
# Check AWS Console or run: make clean
```

## Additional Commands

```bash
# Get help and see all available commands
make help

# Get info about configured suites
make info

# Manually trigger ETL on existing results
make etl suite=<suite-name> id=<suite-id>

# Clean up all cloud resources
make clean

# Clean up only cloud resources (keep local results)
make clean-cloud
```
