# Super-ETL Processing

Complete guide for aggregating and visualizing results from multiple suite runs.

## Overview

**Super-ETL** aggregates results from multiple suite runs to generate comparative visualizations and tables.

**Key difference from regular ETL:**
- **Regular ETL**: Processes individual suite results (automatic)
- **Super-ETL**: Combines results across multiple suites (manual)

## Quick Start

```bash
cd doe-suite
source .envrc

# Update suite IDs in config
vim doe-suite-config/super_etl/poly_plots.yml

# Run super-ETL
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview,grid_kzg,grid_ipa"

# View results
cat ../doe-suite-results-super/poly_plots/overview/overview.csv
open ../doe-suite-results-super/poly_plots/grid_kzg/*.pdf
```

## Workflow

### 1. Identify Suite IDs

After running experiments, collect the suite IDs for aggregation:

```bash
# List completed suite runs
ls -d doe-suite-results/*/ | grep -o '[0-9]\{10\}'

# Or check from experiment names
ls doe-suite-results/
```

**Example suite IDs:**
- `poly-small_1748250734` → Suite ID: `1748250734`
- `apollo-small_1748545862` → Suite ID: `1748545862`
- `nocom-small_1746596630` → Suite ID: `1746596630`

### 2. Update Super-ETL Configuration

Super-ETL configs are in `doe-suite-config/super_etl/`.

**Example: `poly_plots.yml`**

```yaml
$SUITE_ID$:
  # No commitment baseline
  nocom-small: "1746596630"
  nocom-small-ipa: "1748257682"
  nocom-med: "1746992843"
  nocom-med-ipa: "1748257665"

  # Artemis (Horner)
  poly-small: "1748250734"
  poly-small-ipa: "1748252802"
  poly-med: "1746987414"
  poly-med-ipa: "1748252748"

  # Apollo
  apollo-small: "1748545862"
  apollo-med: "1748545870"
  apollo-large: "1748545880"

$PIPELINES$:
  overview:
    experiments: "*"
    extractors:
      ErrorExpectedFileExtractor: {file_regex: ["output.csv"]}
      IgnoreExtractor: {file_regex: ["stdout.log", "runtime_info.yaml"]}
      CsvExtractor: {file_regex: ["*.csv"]}
    transformers: []
    loaders:
      CsvSummaryLoader: {skip_empty: True}

  grid_kzg:
    experiments: "*"
    extractors:
      CsvExtractor: {file_regex: ["output.csv"]}
      IgnoreExtractor: {file_regex: ["*.log"]}
      YamlExtractor: {file_regex: ["runtime_info.yaml"]}
    transformers:
      - MergeRowsTransformer
      - OsirisPreprocessTransformer
    loaders:
      - CsvSummaryLoader: {skip_empty: True}
      - OsirisFactorLoader
      - LargeTableLoader
      - MyCustomColumnCrossPlotLoader
```

**To add new results:**

1. Run experiments and note suite IDs
2. Update `$SUITE_ID$` section with new IDs
3. Run super-ETL (next step)

### 3. Run Super-ETL

Execute the super-ETL pipeline:

```bash
make etl-super config=<config-name> out=<output-dir> pipelines="<pipeline1>,<pipeline2>"
```

**Parameters:**
- `config` - Name of config file (without `.yml`) in `doe-suite-config/super_etl/`
- `out` - Output directory (typically `../doe-suite-results-super`)
- `pipelines` - Comma-separated list of pipelines to run (or "all")

**Examples:**

```bash
# Run all pipelines
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="all"

# Run specific pipelines
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview,grid_kzg"

# Run only overview
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview"
```

**What happens:**
1. Reads suite IDs from config
2. Loads results from `doe-suite-results/<suite>_<id>/`
3. Extracts data using configured extractors
4. Transforms and aggregates data
5. Generates plots, tables, and CSVs in output directory

### 4. Review Outputs

Results are saved to `<output-dir>/<config-name>/`:

```
doe-suite-results-super/poly_plots/
├── overview/
│   └── overview.csv              # Summary of all experiments
├── grid_kzg/
│   ├── grid.csv                  # Aggregated data
│   ├── $metrics$=proof_size.pdf  # Proof size plot
│   ├── $metrics$=proof_size.html # Interactive plot
│   ├── $metrics$=prover_time.pdf # Prover time plot
│   └── $metrics$=verifier_time.pdf
└── grid_ipa/
    └── ...
```

**Key files:**
- `overview/overview.csv` - Quick summary with all experiments
- `<pipeline>/*.csv` - Processed data tables
- `<pipeline>/*.pdf` - Publication-quality plots
- `<pipeline>/*.html` - Interactive visualizations

### 5. Verify Results

**Check overview for completeness:**

```bash
cat doe-suite-results-super/poly_plots/overview/overview.csv
```

**Verify your new results appear:**
- Check suite names match expected experiments
- Verify suite IDs are correct
- Look for any empty rows (indicates missing data)

**Check plots:**

```bash
# View PDFs
open doe-suite-results-super/poly_plots/grid_kzg/*.pdf

# Or view HTML (interactive)
open doe-suite-results-super/poly_plots/grid_kzg/*.html
```

**Verify plots contain:**
- All expected experiments
- Correct labels and legends
- Proper data aggregation (e.g., mean ± stddev)
- No missing bars or data points

## Super-ETL Configuration Structure

### Suite ID Mapping

Maps suite names to their IDs:

```yaml
$SUITE_ID$:
  suite-name: "suite_id"
  poly-small: "1748250734"
  apollo-med: "1748545870"
```

**Notes:**
- Suite name must match design file name (without `.yml`)
- Suite ID is the timestamp from result directory
- You can have multiple IDs for the same suite name (latest is used)

### Pipeline Configuration

Each pipeline defines how to process data:

```yaml
$PIPELINES$:
  pipeline-name:
    experiments: "*"  # or list specific experiments
    extractors:
      ExtractorName: {file_regex: ["pattern.csv"]}
    transformers:
      - TransformerName
    loaders:
      - LoaderName: {option: value}
```

**Components:**

#### Extractors
Read files into pandas DataFrame:
- `CsvExtractor` - Extract CSV files
- `JsonExtractor` - Extract JSON files
- `YamlExtractor` - Extract YAML files
- `IgnoreExtractor` - Ignore specific files (prevents errors)
- `ErrorExpectedFileExtractor` - Check for expected files

#### Transformers
Process and aggregate data:
- `RepAggTransformer` - Aggregate over repetitions (mean, std)
- `MergeRowsTransformer` - Combine data from multiple sources
- `OsirisPreprocessTransformer` - Project-specific preprocessing
- Custom transformers - Project-specific logic

#### Loaders
Generate outputs:
- `CsvSummaryLoader` - Save processed CSV
- `OsirisFactorLoader` - Factor analysis
- `LargeTableLoader` - Detailed tables
- `MyCustomColumnCrossPlotLoader` - Grid plots for KZG
- `MyCustomBrokenColumnCrossPlotLoader` - Grid plots for IPA
- Custom loaders - Project-specific visualizations

## Common Pipelines

### Overview Pipeline

**Purpose**: Quick summary with error detection

```yaml
overview:
  experiments: "*"
  extractors:
    ErrorExpectedFileExtractor: {file_regex: ["output.csv"]}
    IgnoreExtractor: {file_regex: ["stdout.log", "runtime_info.yaml"]}
    CsvExtractor: {file_regex: ["*.csv"]}
  transformers: []
  loaders:
    CsvSummaryLoader: {skip_empty: True}
```

**Output**: `overview/overview.csv` with all experiments

**Use when**: Checking if all experiments completed successfully

### Grid Plot Pipelines

**Purpose**: Comparative visualizations across schemes/models

```yaml
grid_kzg:
  experiments: "*"
  extractors:
    CsvExtractor: {file_regex: ["output.csv"]}
    IgnoreExtractor: {file_regex: ["*.log"]}
    YamlExtractor: {file_regex: ["runtime_info.yaml"]}
  transformers:
    - MergeRowsTransformer
    - OsirisPreprocessTransformer
  loaders:
    - CsvSummaryLoader
    - MyCustomColumnCrossPlotLoader
```

**Outputs**:
- Grid plots with rows=PC type, columns=models
- Separate figures for each metric
- PDF and HTML formats

**Use when**: Comparing performance across different configurations

### Memory Pipeline

**Purpose**: Memory usage analysis

Similar to grid plots but focuses on `max_rss` metric from `runtime_info.yaml`.

## Metrics Tracked

### Primary Metrics (from output.csv)

1. **prover_time_sec**
   - Prover execution time in seconds
   - Converted to minutes (÷60) in plots
   - Unit: min

2. **verifier_time_sec**
   - Verifier execution time in seconds
   - Aggregated: mean and stddev over repetitions
   - Unit: sec

3. **proof_size_bytes**
   - Size of generated proof
   - Converted to kilobytes (÷1000) in plots
   - Unit: kB

4. **prover_time_sec_abs_factor_vs_nocom**
   - Absolute proving time difference vs baseline
   - Used for factor analysis
   - Unit: min

### Secondary Metrics (from runtime_info.yaml)

1. **wall_time**
   - Total wall-clock time (HH:MM:SS)
   - Measured by `/usr/bin/time`

2. **max_rss**
   - Maximum resident set size (memory)
   - Measured in kibibytes, converted to GB (÷976563)
   - Unit: GB

## Troubleshooting

### Missing Suite Results

**Error**: Cannot find results for suite

**Solution:**
- Verify suite ID is correct
- Check results exist: `ls doe-suite-results/<suite>_<id>/`
- Ensure suite name matches design file name

### Extractor Errors

**Error**: `file=<filename> matches no extractor`

**Solution:**
- Add file to `IgnoreExtractor` if not needed
- Add appropriate extractor if data is needed

Example:
```yaml
extractors:
  IgnoreExtractor: {file_regex: ["*.log", "debug.txt"]}
```

### Empty Plots or Missing Data

**Problem**: Plots generated but missing data points

**Solution:**
1. Check overview CSV for missing experiments
2. Verify suite IDs are correct in config
3. Check that result files exist in raw results
4. Review ETL logs for transformer errors

### Incorrect Data Aggregation

**Problem**: Values don't match expected results

**Solution:**
1. Check transformer order in pipeline
2. Verify `RepAggTransformer` settings (mean vs median)
3. Review custom transformer logic
4. Check raw CSVs in result directories

## Customizing Visualizations

### Modifying Plot Loaders

Plot loaders are defined in `doe-suite-config/does_etl_custom/etl/custom.py`.

**Common customizations:**
- Color schemes (colorblind-friendly palettes)
- Label formatting
- Figure sizes
- Legend positioning
- Data filtering (which schemes/models to include)

Example filters in custom loaders:
```python
# Filter models
allowed_models = ['mnist', 'resnet18', 'dlrm', 'mobilenet', 'vgg']

# Filter CP-SNARK schemes
allowed_schemes = ['no_com', 'poly', 'pedersen', 'cp_link+', 'poseidon']

# Filter PC types
allowed_pc = ['kzg', 'ipa']
```

### Creating New Pipelines

1. **Add pipeline to super-ETL config:**
   ```yaml
   $PIPELINES$:
     my_custom_plot:
       experiments: "*"
       extractors: [...]
       transformers: [...]
       loaders: [...]
   ```

2. **Run specific pipeline:**
   ```bash
   make etl-super config=poly_plots out=../doe-suite-results-super pipelines="my_custom_plot"
   ```

## Best Practices

### Organizing Super-ETL Configs

**Strategy 1: One config per analysis type**
- `poly_plots.yml` - Polynomial evaluation schemes
- `apollo_plots.yml` - Apollo vs Lunar comparison
- `pc_comparison.yml` - KZG vs IPA comparison

**Strategy 2: One config per submission**
- `usenix_plots.yml` - USENIX submission plots
- `journal_plots.yml` - Journal submission plots

### Versioning Results

Keep multiple result directories for different versions:
```
doe-suite-results-super/        # Latest
doe-suite-results-super-v1/     # First version
doe-suite-results-super-final/  # Final version
doe-suite-results-usenix/       # USENIX submission
```

### Verifying New Results

After updating suite IDs and running super-ETL:

1. **Check overview CSV:**
   ```bash
   cat doe-suite-results-super/poly_plots/overview/overview.csv | grep <new-suite>
   ```

2. **Compare with previous results:**
   ```bash
   diff doe-suite-results-super/poly_plots/overview/overview.csv \
        doe-suite-results-super-v1/poly_plots/overview/overview.csv
   ```

3. **Verify plots updated:**
   ```bash
   ls -lt doe-suite-results-super/poly_plots/grid_kzg/*.pdf
   ```

4. **Check new data points appear in plots visually**

## Example Workflow

```bash
# 1. Run multiple experiments (from running-experiments.md)
make run suite=poly-small id=new    # Gets ID: 1748250734
make run suite=apollo-small id=new  # Gets ID: 1748545862
make run suite=nocom-small id=new   # Gets ID: 1746596630

# 2. Update super-ETL config
vim doe-suite-config/super_etl/poly_plots.yml
# Add:
#   poly-small: "1748250734"
#   apollo-small: "1748545862"
#   nocom-small: "1746596630"

# 3. Run super-ETL overview to check completeness
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="overview"

# 4. Review overview
cat ../doe-suite-results-super/poly_plots/overview/overview.csv

# 5. Generate plots if overview looks good
make etl-super config=poly_plots out=../doe-suite-results-super pipelines="grid_kzg,grid_ipa"

# 6. Review plots
open ../doe-suite-results-super/poly_plots/grid_kzg/*.pdf
```

## Additional Commands

```bash
# List available super-ETL configs
ls doe-suite-config/super_etl/*.yml

# Run all pipelines
make etl-super config=<config> out=<out-dir> pipelines="all"

# Run single pipeline
make etl-super config=<config> out=<out-dir> pipelines="overview"

# Check ETL custom code
cat doe-suite-config/does_etl_custom/etl/custom.py
```

## Next Steps

- Customize plot loaders for publication-quality figures
- Create new pipelines for specific analyses
- Export data for external plotting tools

[← Back to main guide](./SKILL.md)
[← Running Experiments](./running-experiments.md)
