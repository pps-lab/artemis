---
name: doe-suite
description: Run remote experiments and process results using doe-suite. Use for launching experiments on AWS/Euler or aggregating results across multiple suite runs.
---

This skill provides two main workflows for the DoE-Suite framework:

## Quick Reference

**Working Directory**: All commands run from `doe-suite/`

```bash
cd doe-suite && source ../doe-suite-config/.envrc && export PYTHONPATH=$DOES_PROJECT_DIR/doe-suite-config

# Running experiments (push the branch named in group_vars/all/main.yml first)
make run suite=<suite-name> id=new

# Processing results across suites
make etl-super config=<config-name> out=<output-dir> pipelines="<pipeline1>,<pipeline2>"
```

**Current setup (Oct 2026)**: eu-west-1, stock Ubuntu 24.04 AMI set up by `setup-base`, key pair `hidde-mbp`,
project suffix `hly`, instances clone branch `benchmark-audit-fixes`. Details and the gotchas behind each item:
[prerequisites](./prerequisites.md). Good small-KZG suite IDs: [running-experiments](./running-experiments.md#suite-ids).

## Main Workflows

### 1. [Running Experiments](./running-experiments.md)

Launch and monitor remote experiments on AWS EC2, ETHZ Euler, or existing machines.

**Key tasks:**
- Launch experiment suites on cloud infrastructure
- Monitor progress and download results
- Debug failed experiments
- Cleanup resources

[→ Complete experiment workflow](./running-experiments.md)

### 2. [Super-ETL Processing](./super-etl.md)

Aggregate and visualize results from multiple suite runs.

**Key tasks:**
- Update suite IDs in super-ETL configs
- Run ETL pipelines to generate plots and tables
- Verify aggregated results
- Customize visualizations

[→ Complete super-ETL guide](./super-etl.md)

## Core Concepts

**DoE-Suite** manages remote experiments following Design of Experiments methodology:

- **Suite**: Collection of experiments (YAML in `doe-suite-config/designs/`)
- **Experiment**: Set of runs with different factor combinations
- **Factor**: Parameter that varies (e.g., model, algorithm)
- **Suite ID**: Unique timestamp identifier for each run (e.g., 1767685699)
- **ETL Pipeline**: Per-suite result processing (automatic)
- **Super-ETL**: Cross-suite aggregation (manual)

[→ Detailed concepts](./concepts.md)

## Prerequisites

Required `.envrc` variables:
- `DOES_PROJECT_DIR`, `DOES_PROJECT_ID_SUFFIX`, `DOES_SSH_KEY_NAME`
- `DOES_CLOUD=aws`, `AWS_PROFILE`, `AWS_REGION`

SSH key must exist in target AWS region.

[→ Setup guide](./prerequisites.md)

## Common Commands

```bash
# List available suites
ls doe-suite-config/designs/*.yml

# Validate suite design
make design-validate suite=<suite-name>

# Launch new experiment
make run suite=<suite-name> id=new

# Monitor (keeps checking until done)
make run suite=<suite-name> id=last

# Process results across suites
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview,grid_kzg"

# Cleanup
make clean
```

## Need Help?

- **[Troubleshooting](./troubleshooting.md)** - Common issues and solutions
- **[Examples](./examples.md)** - Complete workflow examples
- **DoE-Suite Docs**: `doe-suite/docs/` - Framework documentation

## Launching and Monitoring from Claude Code

`make run` blocks for the whole suite (10 min setup + build, then 10–45 min for a small suite), so never run it
in the foreground. The pattern that works (see [running-experiments](./running-experiments.md#launching-from-claude-code)):

1. One launcher script per suite that sources `.envrc`, sets `PYTHONPATH`, runs `make run suite=$1 id=new` with
   its output redirected to a per-suite log, and appends `"<suite> exit <code>"` to a shared `status.txt`.
2. A driver (Bash `run_in_background`) that starts the launchers 40 s apart (the first creates the VPC) and
   `wait`s, so one completion notification arrives when every suite has exited.
3. Two `Monitor`s: `tail -f status.txt` (one event per suite exit, covers success and failure), and a
   `tail -f` of the playbook logs filtered on `fatal:|FAILED!|UNREACHABLE|Error|Traceback|PLAY RECAP` to catch a
   broken setup early. Re-arm on the 30-minute expiry; no sleep loops, no polling of the log files.
4. After any suite exit ≠ 0 or a killed playbook: `make clean-cloud`, read the log, fix, commit, push, relaunch.
   Confirm with `aws ec2 describe-instances --filters Name=instance-state-name,Values=running` that nothing is
   left running.

## Important Notes

⚠️ **Cost Warning**: AWS EC2 charges per hour (~$2/h per small instance). Verify cleanup after every run.

💡 **Parallel Execution**: One `make run` process per suite, started ~40 s apart; 8 small suites in parallel worked.

🐛 **Debugging**: Use `make run-keep` to keep instances alive for debugging. Failed-attempt result dirs stay in
`doe-suite-results/` until `make clean-result`.

📊 **Monitoring**: background process + status file + `Monitor` (above). `make run suite=<s> id=last` resumes a
suite whose playbook was interrupted.

🔄 **Retrying**: Fix code, commit, **push the branch the instances clone**, then re-run with `id=new`.

🧮 **ETL**: if a job's ETL errored (extractor mismatch), fix the design and rerun `make etl-design suite=<s> id=<id>`
for that suite; it does not need the instances.
