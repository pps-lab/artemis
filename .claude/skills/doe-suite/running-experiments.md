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
source ../doe-suite-config/.envrc
export PYTHONPATH=$DOES_PROJECT_DIR/doe-suite-config

# View all commands that will be executed
make design suite=<suite-name>

# Validate design with defaults (also imports the custom ETL modules: catches ETL config errors early)
make design-validate suite=<suite-name>
```

Also make sure the branch named by `git_remote_repository.version` in `doe-suite-config/group_vars/all/main.yml`
is pushed: the instances clone it, not your working tree.

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

2. **Environment Setup** (~10 min on a stock Ubuntu AMI)
   - `setup-base` installs apt deps (incl. `time`), rustup, AWS CLI, task spooler
   - Clones the branch in `group_vars/all/main.yml` (`git_remote_repository.version`)
   - Builds with the pinned toolchain: `cargo build --release` (Cargo.lock is committed)
   - Syncs model artifacts from S3

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

#### Launching from Claude Code

`make run` is a blocking Ansible playbook that runs for the whole suite, so it must be started as a background
process and watched through files. What worked for the Oct 2026 runs (scripts live in the session scratchpad;
recreate them, they are three lines each):

```zsh
# launch_suite.zsh <suite>: one blocking make run, logged per suite, exit code appended to status.txt
cd /Users/hidde/Postdoc-newpc/artemis/doe-suite || exit 1
source ../doe-suite-config/.envrc
export PYTHONPATH=$DOES_PROJECT_DIR/doe-suite-config
make run suite=$1 id=new > $S/remote/$1.log 2>&1
echo "$1 exit $?" >> $S/remote/status.txt
```

```zsh
# driver: start the suites 40 s apart (the first one creates the VPC; simultaneous starts race on it), wait for all
: > $S/remote/status.txt
for suite in apollo-small cplink-small cplink_fast-small; do zsh launch_suite.zsh $suite & sleep 40; done
wait; echo ALL_SUITES_FINISHED >> $S/remote/status.txt
```

Run the driver with Bash `run_in_background` (one completion notification when all suites have exited), then arm:

- `Monitor`: `tail -n +1 -f $S/remote/status.txt` — one event per suite exit, success or failure.
- `Monitor`: `tail -f <suite>.log ... | grep -E --line-buffered 'fatal:|FAILED!|UNREACHABLE|Error|Traceback|PLAY RECAP'`
  — surfaces a broken setup (SSH, build, toolchain) within the first minutes instead of after the timeout.

Monitors expire after 30 min; re-arm them on expiry. Do not poll the logs or sleep-loop. When a playbook has to
be killed (TaskStop), run `make clean-cloud` so its instance is terminated, then `id=new` after the fix, or
`id=last` to resume monitoring an interrupted but healthy suite.

#### Manual monitoring

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

All use the stock Ubuntu 24.04 AMI in eu-west-1 (`ami-0445442ed20a2c6f8`); `setup-base` installs everything.
The custom AMI `ami-0714b2f0040bb3a42` only exists in eu-north-1 (Stockholm).

Configurations: `doe-suite-config/group_vars/<size>/main.yml` (AMI, snapshot, instance type) and
`group_vars/all/main.yml` (region, key pair, branch).

Timing (small tier, Oct 2026): ~10 min setup + build, then nocom/poly/lego/apollo ≈ 10–15 min, cplink ≈ 25 min,
poseidon ≈ 45 min (dlrm alone 12 min).

## Suite IDs

Good runs of the eight small KZG suites on branch `benchmark-audit-fixes` (26 Oct 2026, before the Lunar
division fix): poly-small 1793023921, nocom-small 1793024241, poly-ip-small 1793024281, lego-small 1793024321,
apollo-small 1793024361, cplink-small 1793024401, cplink_fast-small 1793024441, poseidon-small 1793024481.
IDs 1793022503–1793023722 are failed attempts (SSH / build / toolchain) and can go with `make clean-result`.
With the `VanishingDivisor` division (27 Oct 2026, commit a731ab5) the Lunar/Apollo suites were re-run and these
supersede the apollo/cplink IDs above: apollo-small 1793096236, cplink-small 1793096275, cplink_fast-small 1793096315.

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

The remote instance pulls the branch named in `group_vars/all/main.yml`, so push that branch (do not develop
on `main`; the evaluation branch is `benchmark-audit-fixes`).

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
      ErrorExpectedFileExtractor: {file_regex: ["stderr.log"], expected_file: "output.csv"}
      IgnoreExtractor: {file_regex: ["stdout.log", "runtime_info.yaml"]}
```

Every result file must match exactly one extractor: `stderr.log` is already claimed by
`ErrorExpectedFileExtractor`, so listing it in `IgnoreExtractor` too fails with "matches multiple extractors".
The per-job ETL error does not stop the suite; after fixing the design run `make etl-design suite=<s> id=<id>`.

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
