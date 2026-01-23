#!/usr/bin/env python3
"""
submit_insitu_validation_jobs.py - Submit SLURM jobs to run in-situ validation in parallel

Location: lake_cci_gapfilling-main/scripts/

Usage:
    # Using default selection CSVs (2010, 2007, 2018, 2020 in priority order)
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20251126-c84211-exp1

    # Using config file for paths
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20251126-c84211-exp1 \\
        --config-file /path/to/configs/exp1_baseline.json

    # Custom selection CSVs (searched in order - first match wins)
    python submit_insitu_validation_jobs.py \\
        --run-root /path/to/exp1 \\
        --selection-csvs /path/to/2010.csv /path/to/2007.csv /path/to/2018.csv

    # Dry run (preview without submitting)
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20251126-c84211-exp1 \\
        --dry-run

    # Custom lake list
    python submit_insitu_validation_jobs.py \\
        --run-root /path/to/exp1 \\
        --lake-ids 4503 3007 1234
    
    # Disable fair comparison filtering (process all lakes)
    python submit_insitu_validation_jobs.py \\
        --run-root /path/to/exp1 \\
        --no-fair-comparison

File structure expected:
    lake_cci_gapfilling-main/
    ├── scripts/
    │   ├── run_insitu_validation.py      <- standalone runner
    │   ├── completion_check.py           <- fair comparison utilities
    │   └── submit_insitu_validation_jobs.py  <- this script
    └── src/processors/postprocessor/post_steps/
        └── insitu_validation.py          <- core validation logic

Author: Shaerdan / NCEO / University of Reading
"""
import argparse
import os
import subprocess
from datetime import datetime

# Import completion check for fair comparison filtering
try:
    from completion_check import get_fair_comparison_lakes, save_exclusion_log
    HAS_COMPLETION_CHECK = True
except ImportError:
    HAS_COMPLETION_CHECK = False

# Lake IDs from exp1_baseline (default list)
DEFAULT_LAKE_IDS = [
    2, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 20, 21, 25, 26, 35, 44, 51, 52, 81,
    84, 88, 91, 133, 146, 165, 170, 171, 174, 177, 198, 223, 227, 236, 257, 295,
    298, 310, 311, 314, 318, 319, 321, 327, 335, 375, 380, 395, 423, 443, 463,
    488, 492, 505, 513, 523, 546, 571, 586, 604, 648, 674, 679, 688, 829, 893,
    905, 917, 920, 1046, 1115, 1146, 1196, 1204, 1241, 1307, 1437, 1555, 1771,
    1774, 1801, 1902, 1951, 1955, 1975, 2099, 2965, 3007, 4503, 6785, 12471,
    13377, 300000138, 300000140, 300000311, 300000579, 300000853, 300000918,
    300001118, 300001141, 300001612, 300001625, 300001645, 300002883, 300004634,
    300004816, 300004882, 300004974, 300009807, 300010187, 300010196, 300010537,
    300010539, 300010540, 300010866, 300012284, 300012336, 300012341, 300015376,
    300016649
]

# Default selection CSVs in priority order (first match wins)
_SELECTION_CSV_DIR = "/home/users/shaerdan/general_purposes/insitu_cv"
DEFAULT_SELECTION_CSVS = [
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2010_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2007_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2018_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2020_selection.csv",
]


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for parallel in-situ validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all lakes with default selection CSVs (2010, 2007, 2018, 2020)
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1

    # Run all lakes with config file
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1 \\
        --config-file /path/to/exp1_baseline.json

    # Custom selection CSVs (searched in priority order)
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1 \\
        --selection-csvs /path/to/2010.csv /path/to/2007.csv /path/to/2018.csv

    # Run specific lakes only
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1 \\
        --lake-ids 4503 3007

    # Dry run to preview
    python submit_insitu_validation_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1 \\
        --dry-run
        """
    )
    
    parser.add_argument("--run-root", required=True, 
                        help="Path to run root directory")
    parser.add_argument("--config-file", default=None,
                        help="Path to experiment JSON config file (optional)")
    parser.add_argument("--lake-ids", type=int, nargs="+", default=None,
                        help="Specific lake IDs to process (default: all lakes)")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print commands without submitting")
    parser.add_argument("--script-dir", default=None, 
                        help="Directory containing run_insitu_validation.py (default: current dir)")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for SLURM logs (default: {script_dir}/logs_insitu)")
    
    # Selection CSV options (mutually exclusive with config-file for CSV paths)
    parser.add_argument("--selection-csvs", nargs="+", default=None,
                        help="Selection CSVs in priority order (first match wins). "
                             "Default: 2010, 2007, 2018, 2020 selection files")
    
    parser.add_argument("--quality-threshold", type=int, default=None,
                        help="Quality level threshold for satellite observations. "
                             "If not specified, reads from config file or uses default (3)")
    
    # Fair comparison filtering
    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Disable fair comparison filtering (submit jobs for all lakes, "
                             "not just those with both DINEOF and DINCAE complete)")
    parser.add_argument("--alpha", default=None,
                        help="Alpha slug for fair comparison check (e.g., 'a1000')")
    
    # SLURM options
    parser.add_argument("--partition", default="standard", help="SLURM partition")
    parser.add_argument("--qos", default="long", help="SLURM QoS")
    parser.add_argument("--account", default="eocis_chuk", help="SLURM account")
    parser.add_argument("--time", default="00:60:00", help="Time limit per job")
    parser.add_argument("--mem", default="64G", help="Memory per job")
    
    args = parser.parse_args()
    
    # Determine directories
    script_dir = args.script_dir or os.getcwd()
    log_dir = args.log_dir or os.path.join(script_dir, "logs_insitu")
    
    # Determine lake list with optional fair comparison filtering
    completion_summary = None
    
    if args.lake_ids:
        # User specified explicit lake IDs - use as-is
        lake_ids = args.lake_ids
        print(f"Using user-specified lake list: {len(lake_ids)} lakes")
    elif HAS_COMPLETION_CHECK and not args.no_fair_comparison:
        # Apply fair comparison filtering
        print("=" * 60)
        print("FAIR COMPARISON MODE: Filtering to lakes with both methods complete")
        print("=" * 60)
        
        fair_lake_ids, completion_summary = get_fair_comparison_lakes(
            args.run_root, args.alpha, verbose=True
        )
        
        if not fair_lake_ids:
            print("ERROR: No lakes found with both DINEOF and DINCAE complete!")
            print("Use --no-fair-comparison to process all lakes regardless")
            return 1
        
        # Intersect with DEFAULT_LAKE_IDS to only process known lakes
        lake_ids = [lid for lid in DEFAULT_LAKE_IDS if lid in fair_lake_ids]
        
        if not lake_ids:
            print("ERROR: None of the default lakes have both methods complete!")
            print(f"Fair comparison lakes: {fair_lake_ids[:10]}...")
            return 1
        
        excluded_count = len(DEFAULT_LAKE_IDS) - len(lake_ids)
        print(f"Lakes after fair comparison filter: {len(lake_ids)} (excluded {excluded_count})")
    else:
        # No fair comparison - use default list
        lake_ids = DEFAULT_LAKE_IDS
        if not args.no_fair_comparison and not HAS_COMPLETION_CHECK:
            print("Note: completion_check module not available, processing all lakes")
    
    # Verify run_insitu_validation.py exists
    validation_script = os.path.join(script_dir, "run_insitu_validation.py")
    if not os.path.exists(validation_script):
        print(f"ERROR: run_insitu_validation.py not found in {script_dir}")
        print("Make sure run_insitu_validation.py is in the scripts/ directory")
        return 1
    
    # Determine selection CSVs to use
    selection_csvs = args.selection_csvs if args.selection_csvs else DEFAULT_SELECTION_CSVS
    
    # Validate selection CSVs exist
    valid_csvs = []
    for csv_path in selection_csvs:
        if os.path.exists(csv_path):
            valid_csvs.append(csv_path)
        else:
            print(f"WARNING: Selection CSV not found: {csv_path}")
    
    if not valid_csvs:
        print("ERROR: No valid selection CSVs found!")
        return 1
    
    # Build selection CSV argument string for the runner
    selection_csv_arg = "--selection-csvs " + " ".join(valid_csvs)
    
    # Build config argument (only if config file is provided)
    config_arg = ""
    if args.config_file:
        if os.path.exists(args.config_file):
            config_arg = f"--config-file {os.path.abspath(args.config_file)}"
        else:
            print(f"WARNING: Config file not found: {args.config_file}")
    
    # Build quality threshold argument
    quality_arg = ""
    if args.quality_threshold is not None:
        quality_arg = f"--quality-threshold {args.quality_threshold}"
    
    # Create log directory
    if not args.dry_run:
        os.makedirs(log_dir, exist_ok=True)
    
    # Print summary
    print("=" * 60)
    print("In-Situ Validation SLURM Job Submission")
    print("=" * 60)
    print(f"Run root:    {args.run_root}")
    print(f"Script dir:  {script_dir}")
    print(f"Log dir:     {log_dir}")
    print(f"Config file: {args.config_file or '(not specified)'}")
    print(f"Fair comparison: {'DISABLED' if args.no_fair_comparison else 'ENABLED'}")
    print(f"Selection CSVs ({len(valid_csvs)} files, searched in order):")
    for i, csv_path in enumerate(valid_csvs, 1):
        print(f"  {i}. {os.path.basename(csv_path)}")
    if args.quality_threshold is not None:
        print(f"Quality threshold: >= {args.quality_threshold}")
    else:
        print(f"Quality threshold: (from config or default=3)")
    print(f"Lakes:       {len(lake_ids)}")
    print(f"SLURM:       partition={args.partition}, qos={args.qos}, time={args.time}, mem={args.mem}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")
    print("=" * 60)
    
    # Save exclusion log if we have completion summary
    if completion_summary is not None and not args.dry_run:
        log_path = save_exclusion_log(completion_summary, args.run_root,
                                      filename="insitu_job_excluded_lakes.csv")
        print(f"Exclusion log saved: {log_path}")
    
    # Custom SLURM template with user options
    # Note: selection_csv_arg is inserted directly (not as a format placeholder)
    # because it contains spaces and multiple paths
    slurm_template = f"""#!/bin/bash
#SBATCH --job-name=insitu_{{lake_id}}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --account={args.account}
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --output={{log_dir}}/insitu_{{lake_id9}}.out
#SBATCH --error={{log_dir}}/insitu_{{lake_id9}}.err

source ~/.bashrc
conda activate lake_cci_gapfilling

cd {{script_dir}}

python run_insitu_validation.py \\
    --run-root {{run_root}} \\
    --lake-id {{lake_id}} \\
    {selection_csv_arg} {quality_arg} {{config_arg}}
"""
    
    submitted = 0
    skipped = 0
    
    for lake_id in lake_ids:
        lake_id9 = f"{lake_id:09d}"
        
        # Check if post directory exists for this lake
        post_dir = os.path.join(args.run_root, "post", lake_id9)
        if not os.path.exists(post_dir):
            # Try unpadded
            post_dir = os.path.join(args.run_root, "post", str(lake_id))
            if not os.path.exists(post_dir):
                print(f"[SKIP] Lake {lake_id9}: no post directory found")
                skipped += 1
                continue
        
        script_content = slurm_template.format(
            lake_id=lake_id,
            lake_id9=lake_id9,
            run_root=args.run_root,
            script_dir=script_dir,
            log_dir=log_dir,
            config_arg=config_arg,
        )
        
        if args.dry_run:
            print(f"[DRY] Would submit job for lake {lake_id9}")
            submitted += 1
            continue
        
        # Submit via sbatch with heredoc
        result = subprocess.run(
            ["sbatch"],
            input=script_content,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"[OK] Submitted job {job_id} for lake {lake_id9}")
            submitted += 1
        else:
            print(f"[ERROR] Failed to submit job for lake {lake_id9}: {result.stderr}")
    
    # Summary
    print("=" * 60)
    print(f"Jobs submitted: {submitted}")
    print(f"Lakes skipped:  {skipped}")
    print(f"Log directory:  {log_dir}")
    print("=" * 60)
    
    if not args.dry_run and submitted > 0:
        print(f"\nMonitor jobs with: squeue -u $USER | grep insitu")
        print(f"Check logs in: {log_dir}/")
    
    return 0


if __name__ == "__main__":
    exit(main())