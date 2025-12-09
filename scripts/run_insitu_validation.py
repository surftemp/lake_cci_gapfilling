#!/usr/bin/env python3
"""
Standalone In-Situ Validation Runner (v2)

Runs the InsituValidationStep on existing pipeline results without
re-running the full pipeline.

NEW in v2: Supports multiple selection CSVs with cascading fallback.
    - Lakes are matched against each selection CSV in order
    - First CSV containing the lake_id_cci is used
    - Ensures lakes with in-situ data from different time periods are all captured

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

# Try to import v2 first, fall back to original
try:
    from insitu_validation_v2 import InsituValidationStep, INSITU_CONFIG
    print("[Runner] Using insitu_validation_v2 (multi-CSV support)")
except ImportError:
    try:
        from insitu_validation import InsituValidationStep, INSITU_CONFIG
        print("[Runner] Using insitu_validation (single-CSV mode)")
    except ImportError as e:
        print(f"Error: Could not import insitu_validation: {e}")
        print(f"Looked in: {SRC_PATH}")
        print(f"Make sure insitu_validation.py or insitu_validation_v2.py exists in src/processors/postprocessor/post_steps/")
        sys.exit(1)


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
        description="Run in-situ validation on existing pipeline results (v2 - multi-CSV support)",
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

Output:
    Results are saved to: {run_root}/post/{lake_id}/{alpha}/insitu_cv_validation/
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
    
    args = parser.parse_args()
    
    # Validate run_root
    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)
    
    # Build config - start with defaults
    config = INSITU_CONFIG.copy()
    
    # Try to load from experiment config file if provided
    if args.config_file:
        if os.path.exists(args.config_file):
            try:
                import json
                with open(args.config_file, 'r') as f:
                    exp_config = json.load(f)
                insitu_section = exp_config.get("insitu_validation", {})
                for key in config.keys():
                    if key in insitu_section:
                        config[key] = insitu_section[key]
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
    config["quality_threshold"] = INSITU_CONFIG.get("quality_threshold", 3)
    
    # Determine which lakes to process
    if args.all:
        lake_ids = find_lakes_in_experiment(args.run_root)
        print(f"Found {len(lake_ids)} lakes in {args.run_root}")
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        lake_ids = [args.lake_id]
    
    if not lake_ids:
        print("No lakes to process")
        sys.exit(0)
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"In-Situ Validation Runner (v2)")
    print(f"{'='*60}")
    print(f"Run root: {args.run_root}")
    print(f"Lakes to process: {len(lake_ids)}")
    print(f"Buoy dir: {config['buoy_dir']}")
    
    # Show selection CSV info
    if config.get("selection_csvs"):
        print(f"Selection CSVs ({len(config['selection_csvs'])} files, searched in order):")
        for i, csv_path in enumerate(config['selection_csvs'], 1):
            print(f"  {i}. {os.path.basename(csv_path)}")
    elif config.get("selection_csv"):
        print(f"Selection CSV: {config['selection_csv']}")
    
    print(f"{'='*60}\n")
    
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