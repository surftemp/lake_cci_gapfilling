#!/usr/bin/env python3
"""
Standalone In-Situ Validation Runner (v8)

Runs the InsituValidationStep on existing pipeline results without
re-running the full pipeline.

NEW in v8: Separated stats displayed in plot titles
    - Difference panels now show: RMSE (all), RMSE (obs), RMSE (gap-fill)
    - Easy comparison of reconstruction quality on each subset
    - Both summary and yearly plots updated

NEW in v7: Visual distinction in plots for observed vs gap-fill
    - Observed points in GREEN (circles)
    - Gap-filled points in ORANGE (squares)

NEW in v6: Fixed time decoding for prepared.nc

NEW in v5: Enhanced diagnostics for observed/missing split

NEW in v4: Fixed coordinate mismatch bug

NEW in v3: Observed/missing split analysis

NEW in v2: Supports multiple selection CSVs with cascading fallback.

Usage:
    # Single lake
    python run_insitu_validation.py --run-root /path/to/experiment --lake-id 4503
    
    # All lakes in experiment
    python run_insitu_validation.py --run-root /path/to/experiment --all
    
    # Specific lakes
    python run_insitu_validation.py --run-root /path/to/experiment --lake-ids 4503 3007 1234
    
    # Custom selection CSVs (in priority order)
    python run_insitu_validation.py --run-root /path/to/experiment --all \
        --selection-csvs /path/to/2010.csv /path/to/2007.csv /path/to/2018.csv
    
    # Disable observed/missing split (faster, backward compatible output)
    python run_insitu_validation.py --run-root /path/to/experiment --all \
        --no-split-observed-missing

Author: Shaerdan / NCEO / University of Reading
"""

import argparse
import os
import sys
from glob import glob
from dataclasses import dataclass
from typing import Optional, List

# Add pipeline src to path for importing insitu_validation from post_steps
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts/ -> pipeline root
SRC_PATH = os.path.join(PIPELINE_ROOT, "src", "processors", "postprocessor", "post_steps")

if os.path.exists(SRC_PATH):
    sys.path.insert(0, SRC_PATH)
else:
    # Fallback: try current directory
    sys.path.insert(0, SCRIPT_DIR)

from insitu_validation import InsituValidationStep, INSITU_CONFIG

# Import completion check utilities for fair comparison filtering
try:
    from completion_check import (
        get_fair_comparison_lakes,
        save_exclusion_log,
        CompletionSummary
    )
    HAS_COMPLETION_CHECK = True
except ImportError:
    HAS_COMPLETION_CHECK = False



@dataclass
class MockPostContext:
    """Minimal context to satisfy InsituValidationStep requirements."""
    lake_id: int
    output_path: str
    experiment_config_path: Optional[str] = None
    lake_path: Optional[str] = None
    dineof_input_path: Optional[str] = None
    dineof_output_path: Optional[str] = None
    output_html_folder: Optional[str] = None
    climatology_path: Optional[str] = None


def find_lakes_in_experiment(run_root: str) -> List[int]:
    """Find all lake IDs that have post-processing results."""
    post_dir = os.path.join(run_root, "post")
    if not os.path.exists(post_dir):
        print(f"Error: post/ directory not found in {run_root}")
        return []
    
    lake_ids = []
    for lake_folder in sorted(os.listdir(post_dir)):
        lake_path = os.path.join(post_dir, lake_folder)
        if os.path.isdir(lake_path):
            # Handle both "4503" and "000004503" formats
            try:
                lake_id = int(lake_folder.lstrip('0') or '0')
                if lake_id > 0:
                    lake_ids.append(lake_id)
            except ValueError:
                continue
    
    return lake_ids


def find_alpha_folders(run_root: str, lake_id: int) -> List[str]:
    """Find all alpha folders for a lake."""
    # Try both padded and unpadded lake folder names
    lake_str_padded = f"{lake_id:09d}"
    lake_str_plain = str(lake_id)
    
    for lake_str in [lake_str_padded, lake_str_plain]:
        post_dir = os.path.join(run_root, "post", lake_str)
        if os.path.exists(post_dir):
            break
    else:
        return []
    
    alphas = []
    for folder in sorted(os.listdir(post_dir)):
        folder_path = os.path.join(post_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("a"):
            alphas.append(folder)
    
    return alphas


def get_post_dir(run_root: str, lake_id: int, alpha: str) -> Optional[str]:
    """Get the post directory path, handling both naming conventions."""
    lake_str_padded = f"{lake_id:09d}"
    lake_str_plain = str(lake_id)
    
    for lake_str in [lake_str_padded, lake_str_plain]:
        post_dir = os.path.join(run_root, "post", lake_str, alpha)
        if os.path.exists(post_dir):
            return post_dir
    
    return None


def run_validation_for_lake(run_root: str, lake_id: int, config: dict) -> bool:
    """Run in-situ validation for a single lake."""
    
    # Find alpha folders
    alphas = find_alpha_folders(run_root, lake_id)
    if not alphas:
        print(f"  No alpha folders found for lake {lake_id}")
        return False
    
    success = False
    for alpha in alphas:
        post_dir = get_post_dir(run_root, lake_id, alpha)
        if post_dir is None:
            print(f"  Post directory not found for {alpha}")
            continue
        
        # Check if there are any output files
        nc_files = glob(os.path.join(post_dir, "*.nc"))
        if not nc_files:
            print(f"  No NetCDF files in {post_dir}")
            continue
        
        # Create mock context
        # output_path is used to derive the post_dir in the step
        mock_output = os.path.join(post_dir, "dummy.nc")
        ctx = MockPostContext(
            lake_id=lake_id,
            output_path=mock_output,
        )
        
        print(f"\n  Processing lake {lake_id} / {alpha}...")
        
        # Run the validation step
        step = InsituValidationStep(config=config)
        
        if step.should_apply(ctx, None):
            step.apply(ctx, None)
            success = True
        else:
            print(f"  Skipped (prerequisites not met)")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Run in-situ validation on existing pipeline results (v8 - stats in plots)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single lake (using default paths)
    python run_insitu_validation.py --run-root /path/to/exp1 --lake-id 4503
    
    # All lakes, loading paths from experiment config
    python run_insitu_validation.py --run-root /path/to/exp1 --all \\
        --config-file /path/to/exp1_baseline.json
    
    # Override with custom selection CSVs (searched in order)
    python run_insitu_validation.py --run-root /path/to/exp1 --lake-id 4503 \\
        --selection-csvs /path/to/2010.csv /path/to/2007.csv /path/to/2018.csv
    
    # Disable observed/missing split analysis (faster, less output)
    python run_insitu_validation.py --run-root /path/to/exp1 --all \\
        --no-split-observed-missing

Output:
    Results are saved to: {run_root}/post/{lake_id}/{alpha}/insitu_cv_validation/
    
    With --split-observed-missing (default), CSV files include additional rows:
      - data_type='reconstruction_observed': Points where original was observed
      - data_type='reconstruction_missing': Points where original was missing (pure gap-fill)
        """
    )
    
    parser.add_argument("--run-root", required=True,
                        help="Base directory of the experiment (e.g., /path/to/anomaly-20251126-exp1)")
    
    # Lake selection (mutually exclusive)
    lake_group = parser.add_mutually_exclusive_group(required=True)
    lake_group.add_argument("--lake-id", type=int, 
                            help="Process single lake by ID")
    lake_group.add_argument("--lake-ids", type=int, nargs="+",
                            help="Process multiple lakes by ID")
    lake_group.add_argument("--all", action="store_true",
                            help="Process all lakes in the experiment")
    
    # Fair comparison filtering
    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Disable fair comparison filtering (process lakes even if one method missing)")
    parser.add_argument("--alpha", default=None,
                        help="Specific alpha slug for fair comparison check (e.g., 'a1000')")
    
    # Configuration - can load from experiment JSON or override individually
    parser.add_argument("--config-file",
                        help="Path to experiment JSON config file (reads insitu_validation section)")
    parser.add_argument("--buoy-dir", 
                        help="Path to buoy data directory (overrides config file)")
    
    # NEW: Support both single CSV (legacy) and multiple CSVs (v2)
    csv_group = parser.add_mutually_exclusive_group()
    csv_group.add_argument("--selection-csv",
                           help="Path to single lake selection CSV (legacy mode)")
    csv_group.add_argument("--selection-csvs", nargs="+",
                           help="Paths to selection CSVs in priority order (v2 mode)")
    
    parser.add_argument("--distance-threshold", type=float,
                        default=INSITU_CONFIG["distance_threshold"],
                        help="Max distance (degrees) for grid-buoy matching")
    
    parser.add_argument("--quality-threshold", type=int, default=None,
                        help="Quality level threshold for satellite observations (default: 3, or from config)")
    
    # NEW: Observed/missing split analysis
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--split-observed-missing", action="store_true", dest="split_obs_miss",
                             default=None,
                             help="Enable observed/missing split analysis using prepared.nc (default: enabled)")
    split_group.add_argument("--no-split-observed-missing", action="store_false", dest="split_obs_miss",
                             help="Disable observed/missing split analysis")
    
    args = parser.parse_args()
    
    # Validate run_root
    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)
    
    # Build config - start with defaults
    config = INSITU_CONFIG.copy()
    
    # Variable to track if we found quality_threshold in experiment config
    quality_threshold_from_config = None
    
    # Try to load from experiment config file if provided
    if args.config_file:
        if os.path.exists(args.config_file):
            try:
                import json
                with open(args.config_file, 'r') as f:
                    exp_config = json.load(f)
                
                # First check insitu_validation section
                insitu_section = exp_config.get("insitu_validation", {})
                for key in config.keys():
                    if key in insitu_section:
                        config[key] = insitu_section[key]
                
                # Also try to get quality_threshold from preprocessing_options
                # (this is what was used during preprocessing)
                preprocessing_opts = exp_config.get("preprocessing_options", {})
                if "quality_threshold" in preprocessing_opts:
                    quality_threshold_from_config = preprocessing_opts["quality_threshold"]
                elif "quality_threshold" in insitu_section:
                    quality_threshold_from_config = insitu_section["quality_threshold"]
                
                print(f"Loaded config from: {args.config_file}")
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        else:
            print(f"Warning: Config file not found: {args.config_file}")
    
    # CLI arguments override config file
    if args.buoy_dir:
        config["buoy_dir"] = args.buoy_dir
    
    # Handle selection CSV arguments
    if args.selection_csvs:
        # v2 mode: multiple CSVs
        valid_csvs = [c for c in args.selection_csvs if os.path.exists(c)]
        if not valid_csvs:
            print(f"Error: None of the provided selection CSVs exist")
            sys.exit(1)
        config["selection_csvs"] = valid_csvs
        config["selection_csv"] = None  # Clear legacy option
        print(f"Using {len(valid_csvs)} selection CSV(s) in priority order")
    elif args.selection_csv:
        # Legacy mode: single CSV
        if not os.path.exists(args.selection_csv):
            print(f"Error: Selection CSV not found: {args.selection_csv}")
            sys.exit(1)
        config["selection_csv"] = args.selection_csv
        config["selection_csvs"] = None  # Clear v2 option
        print(f"Using single selection CSV (legacy mode)")
    
    config["distance_threshold"] = args.distance_threshold
    
    # Set quality_threshold: CLI > config file (preprocessing_options or insitu_validation) > default
    if args.quality_threshold is not None:
        config["quality_threshold"] = args.quality_threshold
        print(f"Using quality_threshold={args.quality_threshold} from command line")
    elif quality_threshold_from_config is not None:
        config["quality_threshold"] = quality_threshold_from_config
        print(f"Using quality_threshold={quality_threshold_from_config} from experiment config")
    else:
        config["quality_threshold"] = INSITU_CONFIG.get("quality_threshold", 3)
        print(f"Using default quality_threshold={config['quality_threshold']}")
    
    # Set split_observed_missing: CLI > config file > default (True)
    if args.split_obs_miss is not None:
        config["split_observed_missing"] = args.split_obs_miss
        status = "enabled" if args.split_obs_miss else "disabled"
        print(f"Observed/missing split analysis: {status} (from command line)")
    else:
        config["split_observed_missing"] = config.get("split_observed_missing", True)
        status = "enabled" if config["split_observed_missing"] else "disabled"
        print(f"Observed/missing split analysis: {status} (default)")
    
    # Determine which lakes to process
    completion_summary = None
    
    if args.all:
        # When processing all lakes, apply fair comparison filter by default
        if HAS_COMPLETION_CHECK and not args.no_fair_comparison:
            print("=" * 60)
            print("FAIR COMPARISON MODE: Getting lakes with both methods complete")
            print("=" * 60)
            
            lake_ids, completion_summary = get_fair_comparison_lakes(
                args.run_root, args.alpha, verbose=True
            )
            
            if not lake_ids:
                print("WARNING: No lakes found with both DINEOF and DINCAE complete!")
                print("Falling back to all lakes in post/ directory...")
                lake_ids = find_lakes_in_experiment(args.run_root)
        else:
            lake_ids = find_lakes_in_experiment(args.run_root)
            if not args.no_fair_comparison and not HAS_COMPLETION_CHECK:
                print("Note: completion_check module not available, processing all lakes")
        
        print(f"Found {len(lake_ids)} lakes to process")
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        lake_ids = [args.lake_id]
    
    if not lake_ids:
        print("No lakes to process")
        sys.exit(0)
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"In-Situ Validation Runner (v8)")
    print(f"{'='*60}")
    print(f"Run root: {args.run_root}")
    print(f"Lakes to process: {len(lake_ids)}")
    print(f"Fair comparison: {'DISABLED' if args.no_fair_comparison else 'ENABLED'}")
    print(f"Buoy dir: {config['buoy_dir']}")
    
    # Show selection CSV info
    if config.get("selection_csvs"):
        print(f"Selection CSVs ({len(config['selection_csvs'])} files, searched in order):")
        for i, csv_path in enumerate(config['selection_csvs'], 1):
            print(f"  {i}. {os.path.basename(csv_path)}")
    elif config.get("selection_csv"):
        print(f"Selection CSV: {config['selection_csv']}")
    
    print(f"Quality threshold: >= {config['quality_threshold']} (for observation filtering)")
    split_status = "ENABLED" if config.get("split_observed_missing", True) else "DISABLED"
    print(f"Observed/missing split: {split_status}")
    print(f"{'='*60}\n")
    
    # Save exclusion log if we have completion summary
    if completion_summary is not None:
        log_path = save_exclusion_log(completion_summary, args.run_root, 
                                      filename="insitu_validation_excluded_lakes.csv")
        print(f"Exclusion log saved: {log_path}\n")
    
    success_count = 0
    skip_count = 0
    
    for lake_id in lake_ids:
        print(f"\n[Lake {lake_id}]")
        if run_validation_for_lake(args.run_root, lake_id, config):
            success_count += 1
        else:
            skip_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Lakes with validation: {success_count}")
    print(f"Lakes skipped (no buoy data): {skip_count}")
    print(f"Output location: {{run_root}}/post/{{lake_id}}/{{alpha}}/insitu_cv_validation/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()