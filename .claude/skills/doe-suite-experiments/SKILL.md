---
name: doe-suite-experiments
description: Run remote experiments (and process results) using the doe-suite framework. Use when running experiments remotely or when processing experiment results to create plots or other diagrams.
---

This skill orchestrates the complete workflow for running DoE-Suite experiments on remote infrastructure (AWS EC2, ETHZ Euler, or existing machines).

## Quick Start

**Working Directory**: All commands must be run from the `doe-suite` directory.

```bash
cd doe-suite
source .envrc

# Launch experiment
make run suite=<suite-name> id=new

# Monitor progress
make run suite=<suite-name> id=last

# Process aggregated results
make etl-super config=<config-name> out=<output-dir> pipelines="<pipeline1>,<pipeline2>"

# Cleanup (automatic after success, manual if needed)
make clean
```

## Core Workflow

1. **[Design & Validate](./workflow.md#1-design-and-validate-suite)** - Validate suite configuration before running
2. **[Launch Experiment](./workflow.md#2-launch-experiment)** - Start remote infrastructure and execute jobs
3. **[Monitor Progress](./workflow.md#3-monitor-progress)** - Track execution and download results
4. **[Review Results](./workflow.md#4-review-raw-results)** - Examine raw experiment outputs
5. **[Process with ETL](./etl-guide.md)** - Aggregate and visualize results
6. **[Cleanup](./workflow.md#6-cleanup)** - Terminate cloud resources

## Key Concepts

**DoE-Suite** manages remote experiments following Design of Experiments methodology:

- **Suite**: Collection of experiments (YAML config in `doe-suite-config/designs/`)
- **Experiment**: Set of runs with different configurations
- **Factor**: Parameter that varies (e.g., model, algorithm)
- **Level**: Specific value a factor takes (e.g., mnist, resnet18)
- **ETL Pipeline**: Extract-Transform-Load for processing results
- **Super-ETL**: Aggregates results across multiple suite runs

[→ Detailed concepts and terminology](./concepts.md)

## Prerequisites

Required environment variables in `.envrc`:
- `DOES_PROJECT_DIR`, `DOES_PROJECT_ID_SUFFIX`, `DOES_SSH_KEY_NAME`
- `DOES_CLOUD`, `AWS_PROFILE`, `AWS_REGION`

[→ Complete prerequisites and setup guide](./prerequisites.md)

## Available Suites

Discover suites:
```bash
ls doe-suite-config/designs/*.yml  # List all designs
make info                           # Show configured suites
make design suite=<suite-name>      # View specific suite details
```

[→ Understanding suite configurations](./concepts.md#suite-structure)

## Common Commands

```bash
# Validate design
make design-validate suite=<suite-name>

# Launch new run
make run suite=<suite-name> id=new

# Continue monitoring
make run suite=<suite-name> id=last

# Keep instances alive for debugging
make run-keep suite=<suite-name> id=new

# Process results across suites
make etl-super config=<config-name> out=<output-dir> pipelines="overview"

# Cleanup
make clean          # All resources
make clean-cloud    # Only cloud (keep local results)
```

[→ All available commands](./workflow.md#additional-commands)

## Need Help?

- **[Troubleshooting Guide](./troubleshooting.md)** - Common issues and solutions
- **[ETL Guide](./etl-guide.md)** - Processing and visualizing results
- **[Example Workflows](./examples.md)** - Complete end-to-end examples
- **[DoE-Suite Docs](../../../doe-suite/docs/)** - Framework documentation

## Important Notes

⚠️ **Cost Warning**: AWS EC2 charges per hour. Set up billing alerts and verify cleanup.

💡 **Parallel Execution**: Run multiple suites in parallel using different sub-processes or agents.

🐛 **Debugging**: Use `make run-keep` to keep instances alive (remember to clean up!).

📊 **Monitoring**: Use `make run` (not bash sleep loops) for proper progress monitoring.

🔄 **Retrying**: Fix code issues, commit, push, then re-run with `id=new` or continue with `id=last`.

[→ Detailed notes and best practices](./workflow.md#important-notes)
