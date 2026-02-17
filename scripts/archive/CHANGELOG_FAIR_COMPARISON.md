# Fair Comparison Upgrade - Changelog

## Version: Fair Comparison Release (January 23, 2026)

### Problem Addressed

When processing ~120 lakes on JASMIN HPC, either DINEOF or DINCAE can fail to complete due to:
- Job timeout (walltime exceeded)
- Memory exhaustion
- Other HPC resource issues

This results in some lakes having only DINEOF results, others having only DINCAE results, and some having both. Previously, analysis scripts could inadvertently compare different pools of lakes for each method, leading to unfair and potentially misleading statistical comparisons.

### Solution Implemented

1. **New Module: `scripts/completion_check.py`**
   - Core utility for checking which lakes have completed processing for both methods
   - Determines completion by checking for `*_dineof.nc` and `*_dincae.nc` files in `post/{lake_id}/{alpha}/`
   - Provides functions for fair comparison filtering across all analysis scripts
   - Can be run standalone to check experiment completion status

2. **Unique Output Directories**
   - All analysis scripts now generate timestamped output folders by default
   - Format: `{prefix}_{YYYYMMDD_HHMMSS}_{hash6}`
   - Prevents overwriting previous analysis results
   - Can be disabled with `--no-unique-dir` flag

3. **Modified Scripts**

   | Script | Changes |
   |--------|---------|
   | `analyze_insitu_validation.py` | Added `--no-fair-comparison`, `--no-unique-dir` flags; applies fair comparison filter to all statistics |
   | `run_insitu_validation.py` | Added `--no-fair-comparison`, `--alpha` flags; filters lake discovery |
   | `run_cv_validation.py` | Added `--no-fair-comparison`, `--alpha` flags; filters lake discovery |
   | `cv_validation.py` | Added `--no-fair-comparison` flag; filters lake auto-discovery |
   | `insitu_validation_5_questions.py` | Added `--no-unique-dir` flag; validates input data has both methods |

### Usage Examples

```bash
# Standard analysis with fair comparison (RECOMMENDED)
python analyze_insitu_validation.py --run_root /path/to/experiment

# Check completion status of an experiment
python completion_check.py --run-root /path/to/experiment

# Save detailed exclusion log
python completion_check.py --run-root /path/to/experiment --save-log /path/to/output/

# Disable fair comparison (include all available data)
python analyze_insitu_validation.py --run_root /path/to/experiment --no-fair-comparison

# Use legacy non-timestamped output folder
python analyze_insitu_validation.py --run_root /path/to/experiment --no-unique-dir
```

### Output Files

When fair comparison is enabled, additional files are generated:

| File | Description |
|------|-------------|
| `excluded_lakes_log.csv` | List of excluded lakes with reasons |
| `analysis_metadata.txt` | Summary of fair comparison status and included/excluded lakes |

### Completion Detection Logic

A lake is considered "complete" for a method if:
- For DINEOF: File matching `*_dineof.nc` exists in `post/{lake_id}/{alpha}/`
- For DINCAE: File matching `*_dincae.nc` exists in `post/{lake_id}/{alpha}/`

Lakes are only included in fair comparison analyses when BOTH methods have completed.

### Backward Compatibility

- All new features are opt-out via `--no-fair-comparison` and `--no-unique-dir` flags
- Scripts continue to work even if `completion_check.py` is not available (with warning)
- Existing workflows can be maintained while opting into new features

### Key Functions in completion_check.py

```python
# Get lakes where both methods completed
lake_ids, summary = get_fair_comparison_lakes(run_root, alpha=None, verbose=True)

# Filter a DataFrame to fair comparison set
df_filtered = filter_dataframe_to_fair_comparison(df, lake_ids, 'lake_id_cci')

# Generate unique output directory path
output_dir = generate_unique_output_dir(base_dir, "insitu_validation_analysis")

# Save exclusion log
log_path = save_exclusion_log(summary, output_dir)
```

### Benefits

1. **Scientific Rigor**: All method comparisons now use identical sample pools
2. **Reproducibility**: Unique output directories preserve all analysis runs
3. **Transparency**: Exclusion logs document exactly which lakes were removed and why
4. **Flexibility**: Can be disabled when needed (e.g., for debugging individual lakes)
