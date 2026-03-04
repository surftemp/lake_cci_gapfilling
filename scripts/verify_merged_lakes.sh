#!/bin/bash
# ===========================================================================
# Verify merged lake files — SLURM parallel (one task per lake×experiment)
# ===========================================================================
set -euo pipefail

BASE="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN"
VERIFY_DIR="${BASE}/verify_merged"
mkdir -p ${VERIFY_DIR}/logs

# Build job list: one line per (experiment_root, lake_id)
cat > ${VERIFY_DIR}/job_list.txt << 'EOF'
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	2
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	5
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	6
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	8
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	9
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	11
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	12
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	13
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	15
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	16
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	17
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	2
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	5
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	6
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	8
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	9
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	11
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	12
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	13
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	15
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	16
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	17
EOF

N_JOBS=$(wc -l < ${VERIFY_DIR}/job_list.txt)
echo "Submitting ${N_JOBS} verification tasks"

# Write Python verification script separately to avoid heredoc variable mixing
cat > ${VERIFY_DIR}/verify_lake.py << 'PYSCRIPT'
import xarray as xr
import glob
import sys

exp_root = sys.argv[1]
lake_id = int(sys.argv[2])
exp_name = exp_root.rstrip("/").split("/")[-1]

base = f"{exp_root}/post/{lake_id:09d}/a1000/"
files = sorted(glob.glob(base + "*.nc"))

if not files:
    print(f"FAIL {exp_name} lake {lake_id}: NO FILES")
    sys.exit(1)

if len(files) != 6:
    print(f"FAIL {exp_name} lake {lake_id}: {len(files)}/6 files")
    for f in files:
        print(f"  found: {f.split('filled_fine_')[-1]}")
    sys.exit(1)

all_ok = True
for f in files:
    ds = xr.open_dataset(f)
    tf = ds["temp_filled"]
    n = tf.sizes["time"]
    q = n // 4
    counts = [int(tf.isel(time=slice(i * q, (i + 1) * q)).count().values) for i in range(4)]
    ds.close()
    name = f.split("filled_fine_")[-1]
    ok = all(c > 0 for c in counts)
    status = "OK" if ok else "BROKEN"
    print(f"  {status} {name}: {counts}")
    if not ok:
        all_ok = False

tag = "PASS" if all_ok else "FAIL"
print(f"{tag} {exp_name} lake {lake_id}: {len(files)} files")
PYSCRIPT

sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=verify_merged
#SBATCH --array=1-${N_JOBS}
#SBATCH -o ${VERIFY_DIR}/logs/verify_%a.out
#SBATCH -e ${VERIFY_DIR}/logs/verify_%a.err
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=short

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${VERIFY_DIR}/job_list.txt)
EXP_ROOT=\$(echo "\$LINE" | cut -f1)
LAKE_ID=\$(echo "\$LINE" | cut -f2)

python3 -u ${VERIFY_DIR}/verify_lake.py \$EXP_ROOT \$LAKE_ID
EOF

echo ""
echo "After jobs finish, check all results:"
echo "  grep -h 'PASS\|FAIL' ${VERIFY_DIR}/logs/verify_*.out | sort"
