#!/usr/bin/env python3
"""
submit_lswt_plots_jobs.py - Submit SLURM jobs to run LSWT plots in parallel

Location: lake_cci_gapfilling-main/scripts/

Usage:
    # Using default paths
    python submit_lswt_plots_jobs.py \\
        --run-root /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20251126-c84211-exp1

    # Custom lake time series and climatology paths
    python submit_lswt_plots_jobs.py \\
        --run-root /path/to/exp1 \\
        --lake-ts-template "/custom/path/LAKE{lake_id9}-*.nc" \\
        --climatology-template "/custom/path/LAKE{lake_id9}_REC.nc"

    # Dry run (preview without submitting)
    python submit_lswt_plots_jobs.py \\
        --run-root /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20251126-c84211-exp1 \\
        --dry-run

    # Custom lake list
    python submit_lswt_plots_jobs.py \\
        --run-root /path/to/exp1 \\
        --lake-ids 4503 3007 1234

File structure expected:
    lake_cci_gapfilling-main/
    ├── scripts/
    │   ├── run_lswt_plots.py             <- standalone runner
    │   └── submit_lswt_plots_jobs.py     <- this script
    └── src/processors/postprocessor/post_steps/
        └── lswt_plots.py                 <- core plotting logic

Author: Shaerdan / NCEO / University of Reading
"""
import argparse
import os
import subprocess
from datetime import datetime

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

# Default path templates
DEFAULT_LAKE_TS_TEMPLATE = "/gws/smf/j04/cds_c3s_lakes/LAURA/TIME_SERIES_PL_L3C/PER_LAKE_TIME_SERIES/LAKE_TS/v2.6.1-146-gfe50b81_RES120_CCIv2.1/LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-fv01.0.nc"
DEFAULT_CLIMATOLOGY_TEMPLATE = "/gws/ssde/j25b/cds_c3s_lakes/users/LAURA/TIME_SERIES_PL_L3C/PER_LAKE_CLIM_REC/LAKE_CLIM_REC/v2.6.1-146-gfe50b81_RES120_CCIv2.1_1995_2020/LAKE{lake_id9}_REC.nc"


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for parallel LSWT plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all lakes with default paths
    python submit_lswt_plots_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1

    # Custom paths
    python submit_lswt_plots_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1 \\
        --lake-ts-template "/custom/LAKE{lake_id9}-*.nc"

    # Run specific lakes only
    python submit_lswt_plots_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1 \\
        --lake-ids 4503 3007

    # Dry run to preview
    python submit_lswt_plots_jobs.py \\
        --run-root /gws/.../anomaly-20251126-c84211-exp1 \\
        --dry-run
        """
    )
    
    parser.add_argument("--run-root", required=True, 
                        help="Path to run root directory")
    parser.add_argument("--lake-ids", type=int, nargs="+", default=None,
                        help="Specific lake IDs to process (default: all lakes)")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print commands without submitting")
    parser.add_argument("--script-dir", default=None, 
                        help="Directory containing run_lswt_plots.py (default: current dir)")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for SLURM logs (default: {script_dir}/logs_lswt_plots)")
    
    # Path template options
    parser.add_argument("--lake-ts-template", default=DEFAULT_LAKE_TS_TEMPLATE,
                        help="Template for original lake time series files. "
                             f"Default: {DEFAULT_LAKE_TS_TEMPLATE}")
    parser.add_argument("--climatology-template", default=DEFAULT_CLIMATOLOGY_TEMPLATE,
                        help="Template for climatology files. "
                             f"Default: {DEFAULT_CLIMATOLOGY_TEMPLATE}")
    
    # Quality threshold
    parser.add_argument("--quality-threshold", type=int, default=3,
                        help="Quality level threshold for observation filtering. "
                             "Only observations with quality_level >= threshold are included. "
                             "Default: 3")
    
    # SLURM options
    parser.add_argument("--partition", default="standard", help="SLURM partition")
    parser.add_argument("--qos", default="short", help="SLURM QoS")
    parser.add_argument("--account", default="eocis_chuk", help="SLURM account")
    parser.add_argument("--time", default="00:30:00", help="Time limit per job")
    parser.add_argument("--mem", default="32G", help="Memory per job")
    
    args = parser.parse_args()
    
    # Determine directories
    script_dir = args.script_dir or os.getcwd()
    log_dir = args.log_dir or os.path.join(script_dir, "logs_lswt_plots")
    
    # Determine lake list
    lake_ids = args.lake_ids if args.lake_ids else DEFAULT_LAKE_IDS
    
    # Verify run_lswt_plots.py exists
    plots_script = os.path.join(script_dir, "run_lswt_plots.py")
    if not os.path.exists(plots_script):
        print(f"ERROR: run_lswt_plots.py not found in {script_dir}")
        print("Make sure run_lswt_plots.py is in the scripts/ directory")
        return 1
    
    # Create log directory
    if not args.dry_run:
        os.makedirs(log_dir, exist_ok=True)
    
    # Print summary
    print("=" * 60)
    print("LSWT Plots SLURM Job Submission")
    print("=" * 60)
    print(f"Run root:           {args.run_root}")
    print(f"Script dir:         {script_dir}")
    print(f"Log dir:            {log_dir}")
    print(f"Lake TS:            {args.lake_ts_template}")
    print(f"Climatology:        {args.climatology_template}")
    print(f"Quality threshold:  >= {args.quality_threshold}")
    print(f"Lakes:              {len(lake_ids)}")
    print(f"SLURM:              partition={args.partition}, qos={args.qos}, time={args.time}, mem={args.mem}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")
    print("=" * 60)
    
    # SLURM template
    slurm_template = f"""#!/bin/bash
#SBATCH --job-name=lswt_plot_{{lake_id}}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --account={args.account}
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --output={{log_dir}}/lswt_plot_{{lake_id9}}.out
#SBATCH --error={{log_dir}}/lswt_plot_{{lake_id9}}.err

source ~/.bashrc
conda activate lake_cci_gapfilling

cd {{script_dir}}

python run_lswt_plots.py \\
    --run-root {{run_root}} \\
    --lake-id {{lake_id}} \\
    --lake-ts-template "{args.lake_ts_template}" \\
    --climatology-template "{args.climatology_template}" \\
    --quality-threshold {args.quality_threshold}
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
        print(f"\nMonitor jobs with: squeue -u $USER | grep lswt_plot")
        print(f"Check logs in: {log_dir}/")
    
    return 0


if __name__ == "__main__":
    exit(main())