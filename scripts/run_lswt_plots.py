#!/usr/bin/env python3
"""
Standalone LSWT Plots Runner

Runs the LSWTPlotsStep on existing pipeline results without re-running the full pipeline.
Generates time series plots including yearly breakdowns.

Usage:
    # Single lake
    python run_lswt_plots.py --run-root /path/to/experiment --lake-id 4503
    
    # All lakes in experiment
    python run_lswt_plots.py --run-root /path/to/experiment --all
    
    # Specific lakes
    python run_lswt_plots.py --run-root /path/to/experiment --lake-ids 4503 3007 1234
    
    # With original time series and climatology paths
    python run_lswt_plots.py --run-root /path/to/experiment --lake-id 4503 \
        --lake-ts-template "/path/to/LAKE{lake_id9}-*.nc" \
        --climatology-template "/path/to/LAKE{lake_id9}_REC.nc"

Output:
    Results are saved to: {run_root}/post/{lake_id}/{alpha}/plots/

Author: Shaerdan / NCEO / University of Reading
"""

import argparse
import os
import sys
from glob import glob
from dataclasses import dataclass
from typing import Optional, List

# Add pipeline src to path for importing lswt_plots from post_steps
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts/ -> pipeline root
SRC_PATH = os.path.join(PIPELINE_ROOT, "src", "processors", "postprocessor", "post_steps")

if os.path.exists(SRC_PATH):
    sys.path.insert(0, SRC_PATH)
else:
    # Fallback: try current directory
    sys.path.insert(0, SCRIPT_DIR)

try:
    from lswt_plots import LSWTPlotsStep
except ImportError as e:
    print(f"Error: Could not import lswt_plots: {e}")
    print(f"Looked in: {SRC_PATH}")
    print(f"Make sure lswt_plots.py exists in src/processors/postprocessor/post_steps/")
    sys.exit(1)


# Default paths on JASMIN
DEFAULT_LAKE_TS_TEMPLATE = "/gws/smf/j04/cds_c3s_lakes/LAURA/TIME_SERIES_PL_L3C/PER_LAKE_TIME_SERIES/LAKE_TS/v2.6.1-146-gfe50b81_RES120_CCIv2.1/LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-fv01.0.nc"
DEFAULT_CLIMATOLOGY_TEMPLATE = "/gws/ssde/j25b/cds_c3s_lakes/users/LAURA/TIME_SERIES_PL_L3C/PER_LAKE_CLIM_REC/LAKE_CLIM_REC/v2.6.1-146-gfe50b81_RES120_CCIv2.1_1995_2020/LAKE{lake_id9}_REC.nc"


@dataclass
class MockPostContext:
    """Minimal context to satisfy LSWTPlotsStep requirements."""
    lake_id: int
    output_path: str
    climatology_path: Optional[str] = None
    experiment_config_path: Optional[str] = None
    lake_path: Optional[str] = None
    dineof_input_path: Optional[str] = None
    dineof_output_path: Optional[str] = None
    output_html_folder: Optional[str] = None


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


def find_file_from_template(template: str, lake_id: int) -> Optional[str]:
    """
    Find a file using a template with {lake_id9} placeholder.
    
    Args:
        template: Path template with {lake_id9} placeholder (9-digit padded lake ID)
        lake_id: Lake ID to substitute
    
    Returns:
        Path to found file or None
    """
    lake_id9 = f"{lake_id:09d}"
    pattern = template.replace("{lake_id9}", lake_id9)
    
    matches = glob(pattern)
    if matches:
        return matches[0]
    
    # Also try unpadded
    pattern_unpadded = template.replace("{lake_id9}", str(lake_id))
    matches = glob(pattern_unpadded)
    if matches:
        return matches[0]
    
    return None


def run_plots_for_lake(run_root: str, lake_id: int, 
                       lake_ts_template: str, climatology_template: str) -> bool:
    """Run LSWT plots for a single lake."""
    
    # Find alpha folders
    alphas = find_alpha_folders(run_root, lake_id)
    if not alphas:
        print(f"  No alpha folders found for lake {lake_id}")
        return False
    
    # Find original time series file
    original_ts_path = find_file_from_template(lake_ts_template, lake_id)
    if original_ts_path:
        print(f"  Found original TS: {os.path.basename(original_ts_path)}")
    else:
        print(f"  Original time series not found (template: {lake_ts_template})")
    
    # Find climatology file
    climatology_path = find_file_from_template(climatology_template, lake_id)
    if climatology_path:
        print(f"  Found climatology: {os.path.basename(climatology_path)}")
    else:
        print(f"  Climatology not found (template: {climatology_template})")
    
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
            climatology_path=climatology_path,
        )
        
        print(f"\n  Processing lake {lake_id} / {alpha}...")
        
        # Run the plotting step
        step = LSWTPlotsStep(original_ts_path=original_ts_path)
        
        if step.should_apply(ctx, None):
            step.apply(ctx, None)
            success = True
        else:
            print(f"  Skipped (prerequisites not met)")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Run LSWT time series plots on existing pipeline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single lake (using default paths)
    python run_lswt_plots.py --run-root /path/to/exp1 --lake-id 4503
    
    # All lakes
    python run_lswt_plots.py --run-root /path/to/exp1 --all
    
    # Custom original time series and climatology paths
    python run_lswt_plots.py --run-root /path/to/exp1 --lake-id 4503 \\
        --lake-ts-template "/custom/path/LAKE{lake_id9}-*.nc" \\
        --climatology-template "/custom/path/LAKE{lake_id9}_REC.nc"

Output:
    Results are saved to: {run_root}/post/{lake_id}/{alpha}/plots/
    
    Generated plots include:
    - Individual series: Original, DINEOF, DINEOF_filtered, DINEOF_interp, 
                         DINEOF_filtered_interp, DINCAE
    - Climatology (DOY-based)
    - Comparisons: DINEOF vs DINEOF_filtered, Original vs DINEOF, etc.
    - Yearly versions of all the above
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
    
    # Path templates
    parser.add_argument("--lake-ts-template", default=DEFAULT_LAKE_TS_TEMPLATE,
                        help="Template for original lake time series files. "
                             "Use {lake_id9} for 9-digit padded lake ID. "
                             f"Default: {DEFAULT_LAKE_TS_TEMPLATE}")
    parser.add_argument("--climatology-template", default=DEFAULT_CLIMATOLOGY_TEMPLATE,
                        help="Template for climatology files. "
                             "Use {lake_id9} for 9-digit padded lake ID. "
                             f"Default: {DEFAULT_CLIMATOLOGY_TEMPLATE}")
    
    args = parser.parse_args()
    
    # Validate run_root
    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)
    
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
    print(f"LSWT Plots Runner")
    print(f"{'='*60}")
    print(f"Run root: {args.run_root}")
    print(f"Lakes to process: {len(lake_ids)}")
    print(f"Lake TS template: {args.lake_ts_template}")
    print(f"Climatology template: {args.climatology_template}")
    print(f"{'='*60}\n")
    
    success_count = 0
    skip_count = 0
    
    for lake_id in lake_ids:
        print(f"\n[Lake {lake_id}]")
        if run_plots_for_lake(args.run_root, lake_id, 
                              args.lake_ts_template, args.climatology_template):
            success_count += 1
        else:
            skip_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Lakes with plots: {success_count}")
    print(f"Lakes skipped: {skip_count}")
    print(f"Output location: {{run_root}}/post/{{lake_id}}/{{alpha}}/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()