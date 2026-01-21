#!/usr/bin/env python3
"""
submit_cv_validation_jobs.py - Submit SLURM jobs for parallel CV validation

Automatically aggregates results after all jobs complete using SLURM dependency.

Usage:
    python submit_cv_validation_jobs.py --run-root /gws/.../exp3

Author: Shaerdan / NCEO / University of Reading
Date: December 2024
"""
import argparse
import os
import subprocess       

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


def find_lakes_with_cv_data(run_root: str) -> list:
    """Find lakes that have clouds_index.nc (CV points)."""
    prepared_dir = os.path.join(run_root, "prepared")
    if not os.path.exists(prepared_dir):
        return []
    
    lake_ids = []
    for lake_folder in sorted(os.listdir(prepared_dir)):
        lake_path = os.path.join(prepared_dir, lake_folder)
        clouds_file = os.path.join(lake_path, "clouds_index.nc")
        
        if os.path.isdir(lake_path) and os.path.exists(clouds_file):
            try:
                lake_id = int(lake_folder.lstrip('0') or '0')
                if lake_id > 0:
                    lake_ids.append(lake_id)
            except ValueError:
                continue
    
    return lake_ids


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for parallel CV validation with automatic aggregation"
    )
    
    parser.add_argument("--run-root", required=True, help="Path to run root directory")
    parser.add_argument("--lake-ids", type=int, nargs="+", default=None,
                        help="Specific lake IDs (default: auto-detect)")
    parser.add_argument("--use-default-lakes", action="store_true",
                        help="Use default lake list instead of auto-detecting")
    parser.add_argument("--dry-run", action="store_true", help="Preview without submitting")
    parser.add_argument("--script-dir", default=None, 
                        help="Directory containing run_cv_validation.py")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for per-lake CSVs (default: {run_root}/cv_results)")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for SLURM logs (default: {run_root}/logs_cv)")
    parser.add_argument("--master-csv", default=None,
                        help="Final aggregated CSV (default: {run_root}/cv_results_all.csv)")
    
    # SLURM options
    parser.add_argument("--partition", default="standard", help="SLURM partition")
    parser.add_argument("--qos", default="long", help="SLURM QoS")
    parser.add_argument("--account", default="eocis_chuk", help="SLURM account")
    parser.add_argument("--time", default="00:30:00", help="Time limit per job")
    parser.add_argument("--mem", default="32G", help="Memory per job")
    
    args = parser.parse_args()
    
    script_dir = args.script_dir or os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(args.run_root, "cv_results")
    log_dir = args.log_dir or os.path.join(args.run_root, "logs_cv")
    master_csv = args.master_csv or os.path.join(args.run_root, "cv_results_all.csv")
    
    # Determine lake list
    if args.lake_ids:
        lake_ids = args.lake_ids
    elif args.use_default_lakes:
        lake_ids = DEFAULT_LAKE_IDS
    else:
        lake_ids = find_lakes_with_cv_data(args.run_root)
        if not lake_ids:
            print(f"No lakes with CV data found in {args.run_root}/prepared/")
            return 1
    
    # Create directories
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("CV Validation SLURM Job Submission")
    print("=" * 60)
    print(f"Run root:    {args.run_root}")
    print(f"Output dir:  {output_dir}")
    print(f"Master CSV:  {master_csv}")
    print(f"Lakes:       {len(lake_ids)}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")
    print("=" * 60)
    
    # Template for per-lake CV jobs
    cv_job_template = """#!/bin/bash
#SBATCH --job-name=cv_{lake_id}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={log_dir}/cv_{lake_id9}.out
#SBATCH --error={log_dir}/cv_{lake_id9}.err

source ~/.bashrc
conda activate lake_cci_gapfilling

cd {script_dir}

python run_cv_validation.py \\
    --run-root {run_root} \\
    --lake-id {lake_id} \\
    --output {output_file} \\
    -q
"""
    
    # Submit per-lake jobs, collect job IDs
    job_ids = []
    submitted = 0
    skipped = 0
    
    for lake_id in lake_ids:
        lake_id9 = f"{lake_id:09d}"
        
        # Check prerequisites
        prepared_dir = os.path.join(args.run_root, "prepared", lake_id9)
        if not os.path.exists(prepared_dir):
            prepared_dir = os.path.join(args.run_root, "prepared", str(lake_id))
            if not os.path.exists(prepared_dir):
                print(f"[SKIP] Lake {lake_id}: no prepared directory")
                skipped += 1
                continue
        
        clouds_file = os.path.join(prepared_dir, "clouds_index.nc")
        if not os.path.exists(clouds_file):
            print(f"[SKIP] Lake {lake_id}: no clouds_index.nc")
            skipped += 1
            continue
        
        dineof_dir = os.path.join(args.run_root, "dineof", lake_id9)
        dincae_dir = os.path.join(args.run_root, "dincae", lake_id9)
        if not os.path.exists(dineof_dir) and not os.path.exists(dincae_dir):
            print(f"[SKIP] Lake {lake_id}: no DINEOF or DINCAE results")
            skipped += 1
            continue
        
        output_file = os.path.join(output_dir, f"cv_{lake_id9}.csv")
        
        script_content = cv_job_template.format(
            lake_id=lake_id, lake_id9=lake_id9, run_root=args.run_root,
            script_dir=script_dir, log_dir=log_dir, output_file=output_file,
            partition=args.partition, qos=args.qos, account=args.account,
            time=args.time, mem=args.mem,
        )
        
        if args.dry_run:
            print(f"[DRY] Would submit job for lake {lake_id}")
            job_ids.append("DRY_JOB")
            submitted += 1
            continue
        
        result = subprocess.run(["sbatch"], input=script_content, text=True, capture_output=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"[OK] Submitted job {job_id} for lake {lake_id}")
            submitted += 1
        else:
            print(f"[ERROR] Failed for lake {lake_id}: {result.stderr}")
    
    print("-" * 60)
    print(f"CV jobs submitted: {submitted}")
    print(f"Lakes skipped:     {skipped}")
    
    # Submit aggregation job with dependency on all CV jobs
    if job_ids and not args.dry_run:
        dependency_str = ":".join(job_ids)
        
        agg_job_template = """#!/bin/bash
#SBATCH --job-name=cv_aggregate
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --account={account}
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output={log_dir}/cv_aggregate.out
#SBATCH --error={log_dir}/cv_aggregate.err
#SBATCH --dependency=afterany:{dependency}

source ~/.bashrc
conda activate lake_cci_gapfilling

echo "Aggregating CV results..."

# Inline aggregation script
python3 << 'EOF'
import csv
import glob
import os

input_dir = "{output_dir}"
output_file = "{master_csv}"

pattern = os.path.join(input_dir, "cv_*.csv")
csv_files = sorted(glob.glob(pattern))

if not csv_files:
    print(f"No CV result files found in {{input_dir}}")
    exit(1)

print(f"Found {{len(csv_files)}} CSV files")

all_rows = []
header = None

for csv_file in csv_files:
    try:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
            for row in reader:
                if row:
                    all_rows.append(row)
    except Exception as e:
        print(f"Warning: Could not read {{csv_file}}: {{e}}")

if not all_rows:
    print("No data rows found")
    exit(1)

# Sort by lake_id
try:
    all_rows.sort(key=lambda x: int(x[0]))
except:
    pass

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"Aggregated {{len(all_rows)}} lakes to: {{output_file}}")

# Summary
n_dineof = sum(1 for r in all_rows if len(r) > 2 and r[2] != "")
n_dincae = sum(1 for r in all_rows if len(r) > 7 and r[7] != "")
n_verified = sum(1 for r in all_rows if len(r) > 6 and r[6] == "True")
n_both = sum(1 for r in all_rows if len(r) > 7 and r[2] != "" and r[7] != "")

print(f"\\nSummary:")
print(f"  Total lakes:     {{len(all_rows)}}")
print(f"  With DINEOF:     {{n_dineof}} ({{n_verified}} verified)")
print(f"  With DINCAE:     {{n_dincae}}")
print(f"  With both:       {{n_both}}")
EOF

echo "Done! Results in {master_csv}"
"""
        
        agg_script = agg_job_template.format(
            partition=args.partition, qos=args.qos, account=args.account,
            log_dir=log_dir, output_dir=output_dir, master_csv=master_csv,
            dependency=dependency_str
        )
        
        result = subprocess.run(["sbatch"], input=agg_script, text=True, capture_output=True)
        
        if result.returncode == 0:
            agg_job_id = result.stdout.strip().split()[-1]
            print(f"\n[OK] Submitted aggregation job {agg_job_id} (depends on all CV jobs)")
            print(f"     Will automatically create: {master_csv}")
        else:
            print(f"\n[ERROR] Failed to submit aggregation job: {result.stderr}")
    
    elif args.dry_run:
        print(f"\n[DRY] Would submit aggregation job depending on {len(job_ids)} CV jobs")
        print(f"      Would create: {master_csv}")
    
    print("=" * 60)
    print(f"\nMonitor: squeue -u $USER | grep cv_")
    print(f"Logs:    {log_dir}/")
    print(f"Output:  {master_csv} (created automatically after jobs complete)")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())