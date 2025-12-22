---
name: doe-suite
description: Run remote experiments and process results using doe-suite. Use for launching experiments on AWS/Euler or aggregating results across multiple suite runs.
---

This skill provides two main workflows for the DoE-Suite framework:

## Quick Reference

**Working Directory**: All commands run from `doe-suite/`

```bash
cd doe-suite && source .envrc

# Running experiments
make run suite=<suite-name> id=new

# Processing results across suites
make etl-super config=<config-name> out=<output-dir> pipelines="<pipeline1>,<pipeline2>"
```

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

## Important Notes

⚠️ **Cost Warning**: AWS EC2 charges per hour. Set billing alerts and verify cleanup.

💡 **Parallel Execution**: Run multiple suites in parallel using different agents/sub-processes.

🐛 **Debugging**: Use `make run-keep` to keep instances alive for debugging.

📊 **Monitoring**: `make run` handles monitoring properly (don't use bash sleep loops).

🔄 **Retrying**: Fix code, commit, push, then re-run with `id=new` or continue with `id=last`.
