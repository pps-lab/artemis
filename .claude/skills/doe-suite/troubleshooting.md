# Troubleshooting Guide

Common issues and solutions when running DoE-Suite experiments.

## Infrastructure Issues

### SSH Key Not Found

**Error:**
```
InvalidKeyPair.NotFound: The key pair 'xxx' does not exist
```

**Cause**: SSH key doesn't exist in target AWS region.

**Solution:**
1. Check existing keys in region:
   ```bash
   aws ec2 describe-key-pairs --region eu-north-1
   ```

2. Verify `DOES_SSH_KEY_NAME` in `.envrc` matches actual key name

3. If key missing, create it:
   ```bash
   aws ec2 create-key-pair --key-name my-key --region eu-north-1
   ```

4. Update `.envrc` and reload:
   ```bash
   export DOES_SSH_KEY_NAME="my-key"
   source .envrc
   ```

### AMI Not Found

**Error:**
```
InvalidAMIID.NotFound: The image id '[ami-xxx]' does not exist
```

**Cause**: AMI doesn't exist in target region.

**Solution:**
1. Check current AMI in `doe-suite-config/group_vars/<size>/main.yml`

2. Verify AMI exists in your region:
   ```bash
   aws ec2 describe-images --image-ids ami-0714b2f0040bb3a42 --region eu-north-1
   ```

3. If different region, update AMI ID in group_vars or change region in `.envrc`

### Region Mismatch

**Error**: Resources not available in region

**Cause**: `AWS_REGION` in `.envrc` doesn't match resources.

**Solution:**
1. Verify region consistency:
   ```bash
   echo $AWS_REGION
   aws ec2 describe-key-pairs --region $AWS_REGION
   ```

2. Update `.envrc` to use correct region:
   ```bash
   export AWS_REGION="eu-north-1"
   source .envrc
   ```

### Invalid AWS Credentials

**Error**: Credentials validation failed

**Cause**: AWS profile not configured or expired.

**Solution:**
1. Check credentials:
   ```bash
   aws sts get-caller-identity --profile your-profile
   ```

2. Reconfigure if needed:
   ```bash
   aws configure --profile your-profile
   ```

3. Update `.envrc`:
   ```bash
   export AWS_PROFILE="your-profile"
   source .envrc
   ```

## Experiment Failures

### Code Build Failures

**Error**: Compilation errors during setup

**Cause**: Code on `main` branch doesn't compile.

**Solution:**
1. Test build locally:
   ```bash
   cargo build --release
   ```

2. Fix compilation errors

3. Commit and push:
   ```bash
   git add .
   git commit -m "Fix compilation errors"
   git push
   ```

4. Re-run suite:
   ```bash
   make run suite=<suite-name> id=new
   ```

### Experiment Panics

**Error**: `thread 'main' panicked at src/bin/...`

**Cause**: Runtime error in experiment code.

**Solution:**
1. Check error logs:
   ```bash
   cat doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/stderr.log
   cat doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/stdout.log
   ```

2. Reproduce locally:
   ```bash
   cargo build --release
   ./src/bin/test.zsh mnist kzg poly 5 .
   ```

3. Fix the bug, commit, push

4. Re-run suite with `id=new`

**Common causes:**
- Incorrect argument order in `test.zsh`
- Missing model files
- Incorrect factor values
- Out of memory errors

### Missing Output Files

**Error**: `output.csv` not created

**Cause**: Experiment failed to produce expected output.

**Solution:**
1. Check stdout/stderr for errors:
   ```bash
   tail doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/stdout.log
   tail doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/stderr.log
   ```

2. Verify command template in suite design:
   ```bash
   make design suite=<suite-name>
   ```

3. Test command locally with actual values

4. Fix code/command and re-run

### Out of Memory

**Symptom**: Experiment killed, no output

**Cause**: Experiment exceeded available memory.

**Solution:**
1. Check memory usage from successful runs:
   ```bash
   grep max_rss doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/runtime_info.yaml
   ```

2. Use larger machine size:
   - Update suite design to use `medium` or `large` host type
   - Modify `assign_to: [large]` in experiment config

3. Or optimize memory usage in code

## ETL Issues

### File Not Matching Extractor

**Error:**
```
file=runtime_info.yaml matches no extractor
```

**Cause**: File in results doesn't match any configured extractor.

**Solution 1: Ignore the file**

Add to `IgnoreExtractor` in suite design:
```yaml
$ETL$:
  overview:
    extractors:
      IgnoreExtractor: {file_regex: ["*.log", "runtime_info.yaml"]}
```

**Solution 2: Extract the file**

Add appropriate extractor if needed:
```yaml
extractors:
  YamlExtractor: {file_regex: ["runtime_info.yaml"]}
```

### ETL Transform Errors

**Error**: Transformer fails during processing

**Cause**: Unexpected data format or missing columns.

**Solution:**
1. Check raw CSV files:
   ```bash
   head doe-suite-results/<suite>_<id>/exp_*/run_*/rep_*/*/host_*/output.csv
   ```

2. Verify required columns exist

3. Check transformer expects correct data format

4. Update transformer or fix data source

### Empty Results

**Error**: Overview CSV has empty rows

**Cause**: Experiments didn't complete or failed.

**Solution:**
1. Check which experiments are missing:
   ```bash
   cat doe-suite-results-super/<config>/overview/overview.csv
   ```

2. Verify suite results exist:
   ```bash
   ls doe-suite-results/<suite>_<id>/
   ```

3. Re-run missing experiments

4. Re-run super-ETL after completion

## Super-ETL Issues

### Missing Suite Results

**Error**: Cannot find results for suite

**Cause**: Suite ID incorrect or results not present.

**Solution:**
1. List available results:
   ```bash
   ls doe-suite-results/
   ```

2. Verify suite ID in super-ETL config matches directory:
   ```yaml
   $SUITE_ID$:
     poly-small: "1748250734"  # Check this matches poly-small_1748250734/
   ```

3. Update suite ID if incorrect

4. Re-run super-ETL

### Incorrect Aggregation

**Error**: Plots show unexpected values

**Cause**: Data aggregation or filtering issue.

**Solution:**
1. Check overview CSV for raw values:
   ```bash
   cat doe-suite-results-super/<config>/overview/overview.csv
   ```

2. Verify transformer order in pipeline config

3. Check custom loader filters (in `does_etl_custom/etl/custom.py`)

4. Review transformer logic

### Missing Data Points in Plots

**Error**: Some experiments don't appear in plots

**Cause**: Data filtered out by custom loaders.

**Solution:**
1. Check allowed filters in custom loader:
   ```python
   allowed_models = ['mnist', 'resnet18', ...]
   allowed_schemes = ['no_com', 'poly', ...]
   ```

2. Update filters to include missing experiments

3. Re-run super-ETL

## Monitoring Issues

### Status Shows No Progress

**Symptom**: `make status` shows 0/N jobs

**Cause**: Jobs haven't started or task spooler not running.

**Solution:**
1. Check instance is running:
   ```bash
   # AWS Console or
   aws ec2 describe-instances --filters "Name=tag:Name,Values=*<suite-name>*"
   ```

2. SSH to instance (if using `make run-keep`):
   ```bash
   ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
   tsp  # Check task spooler queue
   ```

3. Re-run monitoring:
   ```bash
   make run suite=<suite-name> id=last
   ```

### Timeout During Monitoring

**Symptom**: Script times out before completion

**Cause**: Experiments take longer than timeout threshold.

**Solution:**
Simply re-invoke monitoring:
```bash
make run suite=<suite-name> id=last
```

Monitoring will resume and continue checking progress.

## Cleanup Issues

### Instances Not Terminating

**Symptom**: Instances still running after suite completion

**Cause**: Automatic cleanup failed or suite still running.

**Solution:**
1. Check if suite completed:
   ```bash
   make status suite=<suite-name> id=last
   ```

2. If complete, manually clean up:
   ```bash
   make clean
   ```

3. Verify in AWS Console that instances terminated

### Clean Command Fails

**Error**: Unable to terminate resources

**Cause**: AWS permissions or resource not found.

**Solution:**
1. Check AWS credentials:
   ```bash
   aws sts get-caller-identity --profile $AWS_PROFILE
   ```

2. Manually terminate in AWS Console:
   - EC2 → Instances → Select instance → Terminate

3. Clean up other resources:
   - VPC
   - Security groups
   - Network interfaces

## Debugging Tips

### Keep Instances Alive

Use `make run-keep` to prevent automatic termination:
```bash
make run-keep suite=<suite-name> id=new
```

**Remember to clean up manually!**
```bash
make clean
```

### SSH to Instance

If using `run-keep`, SSH to debug:
```bash
# Get instance IP from AWS Console or:
aws ec2 describe-instances --filters "Name=tag:Name,Values=*<suite-name>*" \
  --query 'Reservations[*].Instances[*].PublicIpAddress' --output text

# SSH
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>

# Check task spooler
tsp
tsp -c  # Show current job output

# Check logs
tail -f /path/to/results/stdout.log
```

### Local Testing

Always test locally before running expensive suites:
```bash
cargo build --release
./src/bin/test.zsh mnist kzg poly 5 .
```

### Validate Designs

Catch errors before launching:
```bash
make design-validate suite=<suite-name>
```

## Getting More Help

- Check DoE-Suite docs: `doe-suite/docs/`
- Review suite design: `make design suite=<suite-name>`
- Examine raw results: `doe-suite-results/<suite>_<id>/`
- Check AWS Console for infrastructure issues

[← Back to main guide](./SKILL.md)
