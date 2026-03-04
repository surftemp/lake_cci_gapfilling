#!/bin/bash
# ===========================================================================
# Comprehensive verification of ALL merged lake files + plot cleanup
# ===========================================================================
# 1. Checks merge dir AND main run files for all 11 lakes × 2 experiments
# 2. Validates every .nc file can be opened and has data in all 4 quarters
# 3. Compares file sizes between merge dir and main run (detect bad copies)
# 4. Deletes old plots/insitu results so retrofit starts clean
# ===========================================================================
set -euo pipefail

BASE="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN"
VERIFY_DIR="${BASE}/verify_merged_v2"
mkdir -p ${VERIFY_DIR}/logs

# =========================================================================
# PART 1: Clean old plots + insitu validation for all 11 lakes
# =========================================================================
echo "========================================"
echo "CLEANING OLD PLOTS + INSITU RESULTS"
echo "========================================"

LAKES="2 5 6 8 9 11 12 13 15 16 17"

for EXP_ROOT in \
  "${BASE}/anomaly-20260131-219f0d-exp0_baseline_both" \
  "${BASE}/anomaly-20260131-97cfac-exp1_no_ice_both"; do

  EXP_NAME=$(basename $EXP_ROOT)
  echo ""
  echo "--- ${EXP_NAME} ---"

  for LAKE in $LAKES; do
    LAKE9=$(printf '%09d' $LAKE)
    POST_DIR="${EXP_ROOT}/post/${LAKE9}/a1000"

    # Delete plots directory
    PLOTS_DIR="${POST_DIR}/plots"
    if [ -d "$PLOTS_DIR" ]; then
      N_PLOTS=$(find "$PLOTS_DIR" -type f 2>/dev/null | wc -l)
      rm -rf "$PLOTS_DIR"
      echo "  Lake $LAKE: deleted plots/ ($N_PLOTS files)"
    fi

    # Delete insitu validation directory
    INSITU_DIR="${POST_DIR}/insitu_cv_validation"
    if [ -d "$INSITU_DIR" ]; then
      N_INSITU=$(find "$INSITU_DIR" -type f 2>/dev/null | wc -l)
      rm -rf "$INSITU_DIR"
      echo "  Lake $LAKE: deleted insitu_cv_validation/ ($N_INSITU files)"
    fi
  done
done

echo ""
echo "Plot cleanup complete."
echo ""

# =========================================================================
# PART 2: Build verification job list
# =========================================================================
# Each line: merge_dir_root \t main_run_root \t lake_id
cat > ${VERIFY_DIR}/job_list.txt << 'EOF'
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	2
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	5
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	6
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	8
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	9
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	11
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	12
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	13
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	15
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	16
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp0_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both	17
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	2
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	5
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	6
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	8
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	9
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	11
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	12
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	13
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	15
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	16
/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/exp1_baseline_large_merged_4splits	/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both	17
EOF

N_JOBS=$(wc -l < ${VERIFY_DIR}/job_list.txt)
echo "========================================"
echo "SUBMITTING ${N_JOBS} VERIFICATION TASKS"
echo "========================================"

# =========================================================================
# PART 3: Python verification script
# =========================================================================
cat > ${VERIFY_DIR}/verify_lake.py << 'PYSCRIPT'
"""
Comprehensive verification of a single lake's merged files.

Checks:
  1. Merge dir: all 6 .nc files exist
  2. Merge dir: every file opens without error
  3. Merge dir: every file has data in all 4 time quarters
  4. Main run: all 6 .nc files exist
  5. Main run: file sizes match merge dir (detect truncated copies)
  6. Main run: every file opens without error
  7. Main run: every file has data in all 4 time quarters
  8. CV npz files exist in merge dir

Prints one-line summary: PASS or FAIL with details.
"""
import xarray as xr
import numpy as np
import glob
import sys
import os

merge_root = sys.argv[1]
main_root = sys.argv[2]
lake_id = int(sys.argv[3])
lake_id9 = f"{lake_id:09d}"

exp_name = main_root.rstrip("/").split("/")[-1]
errors = []
warnings = []

EXPECTED_FILES = [
    "dincae.nc",
    "dincae_interp_full.nc",
    "dineof.nc",
    "dineof_eof_filtered.nc",
    "dineof_eof_filtered_interp_full.nc",
    "dineof_eof_interp_full.nc",
]


def check_nc_file(fpath, label):
    """Open a .nc file and verify all 4 time quarters have data."""
    if not os.path.exists(fpath):
        return None, f"{label}: FILE MISSING"

    fsize = os.path.getsize(fpath)
    if fsize == 0:
        return 0, f"{label}: EMPTY FILE (0 bytes)"

    try:
        ds = xr.open_dataset(fpath)
    except Exception as e:
        return fsize, f"{label}: CANNOT OPEN ({e})"

    if "temp_filled" not in ds:
        ds.close()
        return fsize, f"{label}: no temp_filled variable"

    tf = ds["temp_filled"]
    n = tf.sizes["time"]
    q = max(n // 4, 1)

    quarter_counts = []
    for i in range(4):
        try:
            chunk = tf.isel(time=slice(i * q, (i + 1) * q))
            count = int(chunk.count().values)
            quarter_counts.append(count)
        except Exception as e:
            ds.close()
            return fsize, f"{label}: READ ERROR at Q{i+1} ({e})"

    ds.close()

    empty_quarters = [i for i, c in enumerate(quarter_counts) if c == 0]
    if empty_quarters:
        qnames = [f"Q{i+1}" for i in empty_quarters]
        return fsize, f"{label}: EMPTY QUARTERS {','.join(qnames)} counts={quarter_counts}"

    return fsize, None


# ===========================================================================
# Check merge directory
# ===========================================================================
merge_post = f"{merge_root}/post/{lake_id9}/a1000"
merge_sizes = {}

print(f"[{exp_name}] Lake {lake_id}")
print(f"  Merge dir: {merge_post}")

for suffix in EXPECTED_FILES:
    fname = f"LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-filled_fine_{suffix}"
    fpath = os.path.join(merge_post, fname)
    fsize, err = check_nc_file(fpath, f"merge/{suffix}")
    merge_sizes[suffix] = fsize
    if err:
        errors.append(err)
        print(f"    FAIL {suffix}: {err}")
    else:
        print(f"    OK   {suffix} ({fsize:,} bytes)")

# ===========================================================================
# Check main run directory
# ===========================================================================
main_post = f"{main_root}/post/{lake_id9}/a1000"
print(f"  Main run:  {main_post}")

for suffix in EXPECTED_FILES:
    fname = f"LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-filled_fine_{suffix}"
    fpath = os.path.join(main_post, fname)
    fsize, err = check_nc_file(fpath, f"main/{suffix}")
    if err:
        errors.append(err)
        print(f"    FAIL {suffix}: {err}")
    else:
        # Compare size with merge dir
        merge_size = merge_sizes.get(suffix)
        if merge_size is not None and fsize != merge_size:
            msg = f"main/{suffix}: SIZE MISMATCH (merge={merge_size:,}, main={fsize:,})"
            errors.append(msg)
            print(f"    FAIL {suffix}: {msg}")
        else:
            print(f"    OK   {suffix} ({fsize:,} bytes, matches merge)")

# ===========================================================================
# Check CV npz files in merge dir
# ===========================================================================
print(f"  CV files:")
for method in ["dineof", "dincae"]:
    npz = f"{merge_root}/{method}/{lake_id9}/a1000/cv_pairs_{method}.npz"
    if os.path.exists(npz):
        try:
            data = np.load(npz)
            n_points = int(np.sum(~np.isnan(data["diff"])))
            print(f"    OK   cv_pairs_{method}.npz ({n_points:,} points)")
        except Exception as e:
            errors.append(f"cv_{method}: CORRUPT ({e})")
            print(f"    FAIL cv_pairs_{method}.npz: {e}")
    else:
        errors.append(f"cv_{method}: MISSING")
        print(f"    FAIL cv_pairs_{method}.npz: MISSING")

# ===========================================================================
# Check no stale plots remain
# ===========================================================================
n_png = 0
plots_dir = os.path.join(main_post, "plots")
if os.path.isdir(plots_dir):
    n_png = len(glob.glob(os.path.join(plots_dir, "**", "*"), recursive=True))
insitu_dir = os.path.join(main_post, "insitu_cv_validation")
has_insitu = os.path.isdir(insitu_dir)
if n_png > 0:
    warnings.append(f"main run still has plots/ dir ({n_png} files)")
    print(f"  WARN plots/ still in main run post dir ({n_png} files)")
if has_insitu:
    warnings.append("main run still has insitu_cv_validation/")
    print(f"  WARN insitu_cv_validation/ still in main run post dir")

# ===========================================================================
# Summary
# ===========================================================================
if errors:
    print(f"FAIL {exp_name} lake {lake_id}: {len(errors)} errors")
    for e in errors:
        print(f"  - {e}")
else:
    warn_str = f" ({len(warnings)} warnings)" if warnings else ""
    print(f"PASS {exp_name} lake {lake_id}{warn_str}")
PYSCRIPT

# =========================================================================
# PART 4: Submit SLURM array
# =========================================================================
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=verify_v2
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
MERGE_ROOT=\$(echo "\$LINE" | cut -f1)
MAIN_ROOT=\$(echo "\$LINE" | cut -f2)
LAKE_ID=\$(echo "\$LINE" | cut -f3)

python3 -u ${VERIFY_DIR}/verify_lake.py \$MERGE_ROOT \$MAIN_ROOT \$LAKE_ID
EOF
)

echo "Submitted: ${JOB_ID} (${N_JOBS} tasks)"
echo ""
echo "After jobs finish:"
echo "  # Summary of all results:"
echo "  grep -h '^PASS\|^FAIL' ${VERIFY_DIR}/logs/verify_*.out | sort"
echo ""
echo "  # Show only failures:"
echo "  grep -h '^FAIL' ${VERIFY_DIR}/logs/verify_*.out | sort"
echo ""
echo "  # Full details of any failure:"
echo "  for i in \$(seq 1 ${N_JOBS}); do grep -q '^FAIL' ${VERIFY_DIR}/logs/verify_\${i}.out 2>/dev/null && echo '=== Task '\$i' ===' && cat ${VERIFY_DIR}/logs/verify_\${i}.out; done"
