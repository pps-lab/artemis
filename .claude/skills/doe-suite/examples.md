# Example Workflows

Complete end-to-end examples for common DoE-Suite tasks.

## Example 1: Running a Single Suite

**Scenario**: Run polynomial evaluation experiments on small models.

```bash
# 1. Navigate to doe-suite
cd doe-suite
source .envrc

# 2. Validate design (optional)
make design-validate suite=poly-small

# 3. Launch experiment
make run suite=poly-small id=new
# Note the suite ID from output, e.g., 1748250734

# 4. Monitor progress (re-invoke if timeout)
make run suite=poly-small id=last

# 5. Check completion
# Output should show "5/5 jobs" or similar

# 6. Review results
ls -lh ../doe-suite-results/poly-small_1748250734/
cat ../doe-suite-results/poly-small_1748250734/overview/overview.csv

# 7. Cleanup (automatic, but verify)
# Check AWS Console that instance terminated
```

**Expected time**: 30-60 minutes for small models

## Example 2: Running Multiple Suites in Parallel

**Scenario**: Run experiments for different schemes concurrently.

```bash
# In terminal 1 (or agent 1)
cd doe-suite && source .envrc
make run suite=poly-small id=new

# In terminal 2 (or agent 2)
cd doe-suite && source .envrc
make run suite=apollo-small id=new

# In terminal 3 (or agent 3)
cd doe-suite && source .envrc
make run suite=nocom-small id=new

# Each runs independently
# Monitor each separately:
make run suite=poly-small id=last    # Terminal 1
make run suite=apollo-small id=last  # Terminal 2
make run suite=nocom-small id=last   # Terminal 3
```

**Benefit**: Reduces total wall-clock time by running in parallel.

## Example 3: Aggregating Results with Super-ETL

**Scenario**: Generate comparative plots from multiple completed suites.

```bash
# 1. Collect suite IDs from completed runs
ls ../doe-suite-results/
# poly-small_1748250734/
# apollo-small_1748545862/
# nocom-small_1746596630/

# 2. Update super-ETL config
vim doe-suite-config/super_etl/poly_plots.yml

# Add to $SUITE_ID$ section:
#   poly-small: "1748250734"
#   apollo-small: "1748545862"
#   nocom-small: "1746596630"

# 3. Run overview pipeline to check completeness
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview"

# 4. Review overview
cat ../doe-suite-results-super/poly_plots/overview/overview.csv
# Check for empty rows or missing experiments

# 5. Generate plots if overview looks good
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="grid_kzg,grid_ipa"

# 6. View results
open ../doe-suite-results-super/poly_plots/grid_kzg/*.pdf
open ../doe-suite-results-super/poly_plots/grid_ipa/*.pdf
```

## Example 4: Debugging a Failed Experiment

**Scenario**: Experiment panicked and produced no output.

```bash
# 1. Check error logs
cat doe-suite-results/poly-small_1748250734/exp_poly_kzg/run_0/rep_0/small/host_0/stderr.log
# Shows: thread 'main' panicked at 'index out of bounds: ...'

# 2. Reproduce locally
cargo build --release
./src/bin/test.zsh mnist kzg poly 5 .
# Error reproduced!

# 3. Fix the bug
vim src/utils/barycentric.rs
# Fix array indexing issue

# 4. Test fix locally
cargo build --release
./src/bin/test.zsh mnist kzg poly 5 .
# Success!

# 5. Commit and push
git add src/utils/barycentric.rs
git commit -m "Fix: array indexing in barycentric evaluation"
git push

# 6. Re-run suite
make run suite=poly-small id=new
# New suite ID: 1748260000

# 7. Monitor to completion
make run suite=poly-small id=last

# 8. Verify success
cat doe-suite-results/poly-small_1748260000/exp_poly_kzg/run_0/rep_0/small/host_0/output.csv
# Shows results!
```

## Example 5: Keeping Instance Alive for Debugging

**Scenario**: New code might have bugs, want to debug interactively.

```bash
# 1. Launch with run-keep
make run-keep suite=poly-small id=new
# Suite ID: 1748270000

# 2. Get instance IP from AWS Console
# Or use:
aws ec2 describe-instances --filters "Name=tag:Name,Values=*poly-small*" \
  --query 'Reservations[*].Instances[*].PublicIpAddress' --output text

# 3. SSH to instance
ssh -i ~/.ssh/your-key.pem ubuntu@54.123.45.67

# 4. Check task spooler on instance
tsp           # List jobs
tsp -c        # Show current job output
tsp -t        # Show job info

# 5. Check logs in real-time
tail -f /home/ubuntu/doe-suite-results/poly-small_1748270000/exp_*/run_*/rep_*/*/host_*/stdout.log

# 6. If you need to fix code:
# Exit SSH, fix locally, commit, push
exit
vim src/bin/test.zsh
git commit -am "Fix command arguments"
git push

# Back on instance
ssh -i ~/.ssh/your-key.pem ubuntu@54.123.45.67
cd /home/ubuntu/artemis
git pull
cargo build --release
# Manually re-run failed experiment
./src/bin/test.zsh mnist kzg poly 5 /home/ubuntu/artemis

# 7. When done, exit and cleanup
exit
make clean
```

## Example 6: Creating a New Suite Design

**Scenario**: Create suite for testing new CP-SNARK scheme on medium models.

```bash
# 1. Copy existing suite as template
cd doe-suite-config/designs
cp poly-med.yml myscheme-med.yml

# 2. Edit suite design
vim myscheme-med.yml

# Update:
# - Experiment names
# - Algorithm factors
# - Constraints to filter for new scheme
# - Host types if different size needed

# Example changes:
# $FACTORS$:
#   algorithm:
#     levels: [myscheme, poly]  # Add new algorithm
#
# $EXPERIMENTS$:
#   myscheme_kzg_med:
#     design: factorial
#     factors: [model, algorithm, pc_type]
#     constraints: |
#       algorithm == 'myscheme' and pc_type == 'kzg'
#     assign_to: [medium]
#     n_repetitions: 1

# 3. Validate design
make design-validate suite=myscheme-med

# 4. Test locally first
cd ../..
./src/bin/test.zsh mobilenet kzg myscheme 5 .

# 5. Launch suite
make run suite=myscheme-med id=new

# 6. Monitor and collect results
make run suite=myscheme-med id=last
```

## Example 7: Updating Existing Super-ETL Results

**Scenario**: Add newly completed suite to existing super-ETL plots.

```bash
# 1. New suite completed
# poly-large_1748280000

# 2. Check current super-ETL config
cat doe-suite-config/super_etl/poly_plots.yml

# 3. Add new suite ID
vim doe-suite-config/super_etl/poly_plots.yml

# Find poly-large entry and update:
# $SUITE_ID$:
#   poly-large: "1748280000"  # Updated from old ID

# 4. Re-run super-ETL
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="all"

# 5. Verify new data appears
cat ../doe-suite-results-super/poly_plots/overview/overview.csv | grep poly-large
# Should show data from suite 1748280000

# 6. Check plots updated
open ../doe-suite-results-super/poly_plots/grid_kzg/*.pdf
# Verify poly-large bars appear with new data
```

## Example 8: Running Experiments for All Sizes

**Scenario**: Run complete evaluation across small, medium, and large models.

```bash
cd doe-suite
source .envrc

# Launch all sizes in parallel (in separate terminals/agents)

# Terminal 1: Small models
make run suite=poly-small id=new &
make run suite=apollo-small id=new &
make run suite=nocom-small id=new &

# Terminal 2: Medium models
make run suite=poly-med id=new &
make run suite=apollo-med id=new &
make run suite=nocom-med id=new &

# Terminal 3: Large models
make run suite=poly-large id=new &
make run suite=apollo-large id=new &
make run suite=nocom-large id=new &

# Monitor all (after initial launch)
make run suite=poly-small id=last
make run suite=poly-med id=last
make run suite=poly-large id=last
# ... repeat for apollo and nocom

# Collect all suite IDs
ls ../doe-suite-results/ | grep -o '[0-9]\{10\}'

# Update super-ETL config with all IDs
vim doe-suite-config/super_etl/poly_plots.yml

# Generate complete results
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="all"
```

**Expected time**:
- Small: 1-2 hours each
- Medium: 3-6 hours each
- Large: 8-16 hours each

**Total**: ~24-48 hours if run in parallel

## Example 9: Comparing KZG vs IPA

**Scenario**: Run same schemes with both polynomial commitment types.

```bash
cd doe-suite
source .envrc

# Run KZG variants
make run suite=poly-small id=new      # Suite ID: 1748250734
make run suite=apollo-small id=new    # Suite ID: 1748545862

# Run IPA variants
make run suite=poly-small-ipa id=new  # Suite ID: 1748252802
make run suite=apollo-small-ipa id=new  # Suite ID: 1748545900

# Update super-ETL config
vim doe-suite-config/super_etl/poly_plots.yml

# Add all IDs:
# $SUITE_ID$:
#   poly-small: "1748250734"
#   poly-small-ipa: "1748252802"
#   apollo-small: "1748545862"
#   apollo-small-ipa: "1748545900"

# Generate comparative plots
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="grid_kzg,grid_ipa"

# Plots now show KZG vs IPA comparison
open ../doe-suite-results-super/poly_plots/grid_kzg/*.pdf
open ../doe-suite-results-super/poly_plots/grid_ipa/*.pdf
```

## Example 10: Quick Sanity Check

**Scenario**: Test new code changes don't break existing experiments.

```bash
# 1. Create minimal test suite
vim doe-suite-config/designs/test-quick.yml

# Minimal config with 1-2 fast runs:
# $EXPERIMENTS$:
#   quick_test:
#     design: factorial
#     factors: [model]
#     constraints: |
#       model == 'mnist'  # Fastest model
#     assign_to: [small]
#     n_repetitions: 1

# 2. Validate
make design-validate suite=test-quick

# 3. Run (should complete in ~15 min)
make run suite=test-quick id=new

# 4. Check results
cat doe-suite-results/test-quick_*/exp_*/run_*/rep_*/*/host_*/output.csv

# If successful, proceed with full suite runs
```

## Tips for All Workflows

**Before launching:**
- Validate design: `make design-validate suite=<name>`
- Test locally first
- Check AWS billing alerts are set

**During execution:**
- Monitor regularly with `make run suite=<name> id=last`
- Check AWS Console for unexpected costs
- Review logs if jobs fail

**After completion:**
- Verify all results collected
- Run super-ETL for comparative analysis
- Clean up manually if auto-cleanup failed
- Backup result directories if needed

[← Back to main guide](./SKILL.md)
