# Key Concepts

## DoE-Suite Fundamentals

**DoE-Suite** is a framework for remote experiment management following Design of Experiments (DoE) methodology.

### Core Terms

- **Suite**: Collection of experiments defined in YAML (in `doe-suite-config/designs/`)
- **Experiment**: Set of runs with different factor combinations
- **Factor**: Parameter that varies (e.g., model, algorithm, pc_type)
- **Level**: Specific value a factor takes (e.g., mnist, resnet18, dlrm)
- **Run**: Single execution with specific factor level combination
- **Repetition**: Multiple runs with same factors (for statistical significance)
- **Suite ID**: Unique timestamp identifier for each suite run (e.g., 1767685699)

### Factor Types

**Simple Factor**: Independent values
```yaml
model:
  type: simple
  levels: [mnist, resnet18, dlrm]
```

**Nested Factor**: Values depend on other factors
```yaml
algorithm:
  type: nested
  levels: [poly, pedersen]
  nested_in: [model]
```

**Crossed Factors**: All combinations tested (factorial design)

### Design Types

- **Factorial**: Test all combinations of factor levels
- **Fractional Factorial**: Test subset of combinations
- **Custom**: Specific combinations defined by constraints

Example constraint:
```yaml
constraints: |
  algorithm in ['poly', 'pedersen'] and pc_type == 'kzg'
```

## Suite Structure

A complete suite design (`doe-suite-config/designs/<suite>.yml`):

```yaml
$GENERAL$:
  defaults_file: design_vars/general.yml
  command: "./test.zsh {model} {pc_type} {cpsnark} 5 {code_dir}"

$HOST_TYPES$:
  small:
    n_hosts: 1
    source_group_vars: small  # References group_vars/small/main.yml

$FACTORS$:
  model:
    type: simple
    levels: [mnist, resnet18, dlrm]
  algorithm:
    type: simple
    levels: [poly, pedersen]

$EXPERIMENTS$:
  poly_kzg_small:
    design: factorial
    factors: [model, algorithm]
    constraints: |
      algorithm == 'poly'
    assign_to: [small]
    n_repetitions: 1

$ETL$:
  overview:
    experiments: "*"
    extractors:
      CsvExtractor: {file_regex: ["output.csv"]}
    transformers: []
    loaders:
      CsvSummaryLoader: {skip_empty: True}
```

## Result Organization

Results are stored hierarchically:

```
doe-suite-results/
└── <suite-name>_<suite-id>/
    └── exp_<experiment-name>/
        └── run_<N>/              # Run index (0, 1, 2, ...)
            └── rep_<R>/          # Repetition index (usually 0)
                └── <host-type>/   # Machine size (small/medium/large)
                    └── host_<H>/  # Host index (usually 0)
                        ├── output.csv
                        ├── runtime_info.yaml
                        ├── stdout.log
                        └── stderr.log
```

**Example path:**
```
doe-suite-results/poly-small_1748250734/exp_poly_kzg/run_0/rep_0/small/host_0/output.csv
```

## ETL Processing

### Regular ETL (Per-Suite)

Processes individual suite results **automatically** when results are downloaded.

Defined in suite design (`$ETL$` section).

**Manual trigger:**
```bash
make etl suite=<suite-name> id=<suite-id>
```

### Super-ETL (Cross-Suite)

Aggregates results from **multiple suite runs** for comparative analysis.

Defined in `doe-suite-config/super_etl/<config>.yml`.

**Manual execution:**
```bash
make etl-super config=<config> out=<output-dir> pipelines="<list>"
```

### Processing Stages

1. **Extract**: Read files into pandas DataFrame
   - Match files by regex patterns
   - Parse CSV, JSON, YAML, etc.

2. **Transform**: Process and aggregate
   - Aggregate repetitions (mean, std)
   - Merge data from multiple sources
   - Custom transformations

3. **Load**: Generate outputs
   - Save processed CSVs
   - Generate plots (PDF, HTML)
   - Custom visualizations

## Command Templating

Commands use factor values as template variables:

```yaml
command: "./test.zsh {model} {pc_type} {algorithm} 5 {code_dir}"
```

At runtime for `model=mnist, pc_type=kzg, algorithm=poly`:
```bash
./test.zsh mnist kzg poly 5 /home/ubuntu/artemis
```

## Execution Flow

1. **Infrastructure Setup** (5-10 min)
   - Create VPC, subnet, security groups
   - Launch EC2 instances
   - Wait for SSH access

2. **Environment Setup** (10-15 min)
   - Install dependencies
   - Clone code (main branch)
   - Build project
   - Download artifacts

3. **Job Execution** (variable)
   - Enqueue runs to task spooler
   - Execute sequentially
   - Save results locally

4. **Result Collection** (ongoing)
   - Download results as jobs complete
   - Run ETL pipelines
   - Update status

5. **Cleanup** (automatic on success)
   - Terminate instances
   - Keep local results

[← Back to main guide](./SKILL.md)
