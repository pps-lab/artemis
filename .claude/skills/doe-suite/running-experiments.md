# Running Experiments

Complete workflow for launching and monitoring remote experiments.

## Overview

This workflow:
1. Validates suite configuration
2. Launches cloud infrastructure (AWS EC2)
3. Builds code and runs experiments
4. Monitors progress and downloads results
5. Cleans up resources

**Expected Duration**: 15 minutes to 12+ hours depending on suite size.

## Workflow Steps

### 1. Validate Suite Design (Optional but Recommended)

Before launching expensive cloud resources, validate your configuration:

```bash
cd doe-suite
source .envrc

# View all commands that will be executed
make design suite=<suite-name>

# Validate design with defaults
make design-validate suite=<suite-name>
```

**What to check:**
- Factor combinations are correct
- Commands reference valid factors
- Host types match machine configs
- Constraints filter properly

### 2. Launch Experiment

```bash
make run suite=<suite-name> id=new
```

**What happens automatically:**

1. **Infrastructure Setup** (5-10 min)
   - Creates suite ID (Unix timestamp, e.g., 1767685699)
   - Sets up VPC, subnet, security groups
   - Launches EC2 instance(s) with appropriate size
   - Waits for SSH access

2. **Environment Setup** (10-15 min)
   - Installs dependencies (Rust, AWS CLI, task spooler)
   - Clones code from GitHub (`main` branch)
   - Builds project: `cargo build --release`
   - Downloads model artifacts from S3

3. **Job Execution** (variable time)
   - Enqueues all experiment jobs to task spooler
   - Executes sequentially
   - Saves results locally on instance

4. **Result Collection** (ongoing)
   - Downloads results as jobs complete
   - Runs ETL pipelines automatically
   - Updates status

5. **Cleanup** (automatic on success)
   - Terminates instances when complete
   - Keeps local results in `doe-suite-results/`

**Command Options:**
- `id=new` - Start a fresh run with new suite ID
- `id=last` - Continue monitoring the most recent run
- `id=<suite_id>` - Continue a specific run
- `make run-keep` - Keep instances running after completion (for debugging)

### 3. Monitor Progress

The `make run` command monitors automatically, but if it times out, continue monitoring:

```bash
make run suite=<suite-name> id=last
```

This script:
- Checks job status on remote instances
- Downloads completed results
- Runs ETL pipelines on downloaded results
- Displays progress (e.g., "3/6 jobs complete")
- Terminates when all jobs are done OR times out

**Output shows:**
- Total jobs completed/remaining (e.g., "5/5 jobs")
- Current job status (queued/running/finished)
- ETL pipeline status (warnings/errors)
- Progress percentage

**Re-invoke if needed** - The monitoring times out after a period. Just re-run to continue monitoring.

### 4. Review Raw Results

Results are automatically downloaded to:

```
doe-suite-results/<suite-name>_<suite-id>/
└── exp_<experiment-name>/
    └── run_<N>/              # Run index (factor combination)
        └── rep_<R>/          # Repetition index (usually 0)
            └── <host-type>/   # Machine size (small/medium/large)
                └── host_<H>/  # Host index (usually 0)
                    ├── output.csv           # Experiment metrics
                    ├── runtime_info.yaml    # Wall time, memory usage
                    ├── stdout.log           # Program output
                    └── stderr.log           # Compilation warnings
```

**Key files:**
- `output.csv` - Main experiment data (prover time, verifier time, proof size)
- `runtime_info.yaml` - Wall-clock time, max memory (RSS)
- `stdout.log` - Execution output
- `stderr.log` - Usually just compilation warnings (normal)

**Example path:**
```
doe-suite-results/poly-small_1767685699/exp_poly_kzg_small/run_0/rep_0/small/host_0/output.csv
```

### 5. Verify Success

Check that experiments completed successfully:

```bash
# List result directories
ls -lh doe-suite-results/<suite-name>_<suite-id>/

# Check for output files
find doe-suite-results/<suite-name>_<suite-id> -name "output.csv"

# View quick summary (if ETL ran)
cat doe-suite-results/<suite-name>_<suite-id>/overview/overview.csv
```

**Success indicators:**
- All expected runs have result directories
- Each run has `output.csv` file
- No error messages in `stdout.log`
- ETL pipelines completed without errors

### 6. Cleanup

**Automatic cleanup** - Instances terminate when suite completes successfully (exit code 0).

**Manual cleanup** (if suite fails or is interrupted):

```bash
make clean          # Terminate all resources
make clean-cloud    # Terminate cloud resources only (keep local results)
```

**Always verify** in AWS Console that instances are terminated!

## Available Suites

Discover available suites:

```bash
# List all suite designs
ls doe-suite-config/designs/*.yml

# Show configured suites
make info

# View specific suite details
make design suite=<suite-name>
```

**Suite naming convention:**
- `<scheme>-<size>.yml` - e.g., `poly-small.yml`
- `<scheme>-<size>-<pc>.yml` - e.g., `poly-small-ipa.yml`

## Machine Configurations

Three tiers of AWS EC2 instances:

| Size   | Instance Type | vCPUs | RAM   | Use Case                          |
|--------|---------------|-------|-------|-----------------------------------|
| Small  | r6i.8xlarge   | 32    | 256GB | mnist, resnet18, dlrm             |
| Medium | r6i.16xlarge  | 64    | 512GB | mobilenet, vgg                    |
| Large  | r6i.32xlarge  | 128   | 1TB   | diffusion, gpt2                   |

All use custom AMI: `ami-0714b2f0040bb3a42` (eu-north-1 Stockholm region).

Configurations: `doe-suite-config/group_vars/<size>/main.yml`

## Debugging Failed Experiments

### Experiments Failed on Remote

If experiments panic or error:

1. **Check stderr/stdout logs:**
   ```bash
   cat doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/stderr.log
   cat doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/stdout.log
   ```

2. **Identify the error** - Look for panics, assertion failures, or error messages

3. **Fix the code locally** - Make fixes in your working directory

4. **Test locally:**
   ```bash
   cargo build --release
   ./src/bin/test.zsh <model> <pc_type> <cpsnark> 5 .
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "Fix: <description>"
   git push
   ```

6. **Re-run the suite:**
   ```bash
   make run suite=<suite-name> id=new
   ```

The remote instance will pull the latest `main` branch code.

### Instance Won't Start

**SSH key errors:**
```
InvalidKeyPair.NotFound: The key pair 'xxx' does not exist
```

**Solution:**
- Verify `DOES_SSH_KEY_NAME` in `.envrc` matches a key in target AWS region
- Check key exists: `aws ec2 describe-key-pairs --region <region>`

**AMI not found:**
```
InvalidAMIID.NotFound: The image id '[xxx]' does not exist
```

**Solution:**
- Verify AMI exists in target region
- Update AMI ID in `group_vars/<size>/main.yml` if using different region

### Build Failures

If `cargo build` fails during setup:

1. **Check that `main` branch builds locally:**
   ```bash
   cargo build --release
   ```

2. **Fix build errors and push:**
   ```bash
   git commit -am "Fix build errors"
   git push
   ```

3. **Re-launch suite:**
   ```bash
   make run suite=<suite-name> id=new
   ```

### Keep Instances Alive for Debugging

Use `make run-keep` to keep instances running after completion:

```bash
make run-keep suite=<suite-name> id=new
```

**Remember to clean up manually!**
```bash
make clean
```

## ETL Pipeline Issues

### File Not Matching Extractor

**Error:**
```
file=runtime_info.yaml matches no extractor
```

**Solution 1: Ignore the file**

Add to suite design ETL config:
```yaml
$ETL$:
  overview:
    extractors:
      IgnoreExtractor: {file_regex: ["*.log", "runtime_info.yaml"]}
```

**Solution 2: Extract the file**

Add appropriate extractor if you need the data:
```yaml
extractors:
  YamlExtractor: {file_regex: ["runtime_info.yaml"]}
```

### ETL Failures Don't Block Experiments

ETL runs automatically after each job and during `make status`. A failing ETL doesn't stop experiment execution, but prevents result aggregation.

Check ETL logs in monitoring output for errors.

## Advanced Usage

### Running Multiple Suites in Parallel

Launch multiple suites concurrently using different agents or sub-processes:

```bash
# In agent 1
make run suite=poly-small id=new

# In agent 2
make run suite=poly-med id=new

# In agent 3
make run suite=apollo-small id=new
```

Each suite runs independently with its own infrastructure.

### Testing Suite Configurations

Before running expensive experiments:

1. **Validate design:**
   ```bash
   make design-validate suite=<suite-name>
   ```

2. **Test commands locally:**
   ```bash
   # Extract command from design output
   ./src/bin/test.zsh mnist kzg poly 5 .
   ```

3. **Use small test suite** - Create a minimal test suite with 1-2 runs

## Additional Commands

```bash
# Get help
make help

# Get info about configured suites
make info

# Show what will be executed
make design suite=<suite-name>

# Monitor specific suite
make status suite=<suite-name> id=<suite-id>

# Manually trigger ETL on existing results
make etl suite=<suite-name> id=<suite-id>

# Clean up all resources
make clean

# Clean up only cloud (keep local results)
make clean-cloud
```

## Cost Considerations

⚠️ **AWS EC2 charges per hour**

**Typical costs** (varies by region):
- r6i.8xlarge: ~$2/hour
- r6i.16xlarge: ~$4/hour
- r6i.32xlarge: ~$8/hour

**Cost management:**
- Set up billing alerts in AWS Console
- Use `make clean` if experiments are taking longer than expected
- Verify instances terminate after completion
- Test locally before running expensive suites

## Next Steps

After collecting results from multiple suites, process them with super-ETL:

[→ Super-ETL Processing Guide](./super-etl.md)

[← Back to main guide](./SKILL.md)
