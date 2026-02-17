#!/bin/bash
# ===========================================================================
# In-Situ CV Assembly — Parallel SLURM launcher
# ===========================================================================
#
# Submits two jobs:
#   1. Array job: one task per lake with buoy data (parallel extraction)
#   2. Merge job: waits for all extractions, then concatenates + stats
#
# Usage:
#   bash run_insitu_cv_parallel.sh
#
# Note: Only ~26 lakes have buoy data, so this is fast even sequential.
#       Parallel mode is provided for consistency and in case of expansion.
# ===========================================================================

set -euo pipefail

# === CONFIGURATION ===
REPO=/home/users/shaerdan/lake_cci_gapfilling
RUN_ROOT=/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-97cfac-exp1_no_ice_both
ALPHA=a1000
OUTPUT_DIR=${RUN_ROOT}/insitu_cv_assembly
SCRIPT=${REPO}/scripts/assemble_insitu_cv.py
ACCOUNT=eocis_chuk
PARTITION=standard
QOS=standard
# =====================

# Create output + log directories
mkdir -p ${OUTPUT_DIR}/per_lake
mkdir -p ${OUTPUT_DIR}/logs

# --- Step 1: Generate lake list (only lakes with buoy data) ---
echo "Generating lake list (lakes with buoy data only)..."
source activate lake_cci_gapfilling 2>/dev/null || conda activate lake_cci_gapfilling 2>/dev/null || true

python -u ${SCRIPT} \
    --run-root ${RUN_ROOT} \
    --alpha ${ALPHA} \
    --output-dir ${OUTPUT_DIR} \
    --phase list-lakes \
    --all \
    > ${OUTPUT_DIR}/lake_list.txt

N_LAKES=$(wc -l < ${OUTPUT_DIR}/lake_list.txt)
echo "Found ${N_LAKES} lakes with buoy data"

if [ "${N_LAKES}" -eq 0 ]; then
    echo "ERROR: No lakes with buoy data found."
    exit 1
fi

# --- Step 2: Submit array job (one task per lake) ---
ARRAY_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=insitu_cv_extract
#SBATCH --array=1-${N_LAKES}
#SBATCH -o ${OUTPUT_DIR}/logs/extract_%a.out
#SBATCH -e ${OUTPUT_DIR}/logs/extract_%a.err
#SBATCH --mem=2G
#SBATCH -t 0:15:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}

source activate lake_cci_gapfilling 2>/dev/null || conda activate lake_cci_gapfilling 2>/dev/null || true


LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${OUTPUT_DIR}/lake_list.txt)
echo "Task \${SLURM_ARRAY_TASK_ID}: Lake \${LAKE_ID}"
echo "Started: \$(date)"

python -u ${SCRIPT} --run-root ${RUN_ROOT} --alpha ${ALPHA} --output-dir ${OUTPUT_DIR} --phase extract --lake-id \${LAKE_ID}

echo "Finished: \$(date)"
EOF
)

echo "Submitted array job: ${ARRAY_JOB_ID} (${N_LAKES} tasks)"

# --- Step 3: Submit merge job (runs after all extractions complete) ---
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=insitu_cv_merge
#SBATCH -o ${OUTPUT_DIR}/logs/merge.out
#SBATCH -e ${OUTPUT_DIR}/logs/merge.err
#SBATCH --mem=4G
#SBATCH -t 0:15:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}

source activate lake_cci_gapfilling 2>/dev/null || conda activate lake_cci_gapfilling 2>/dev/null || true


echo "Merge job started: \$(date)"
echo "Merging per-lake CSVs from ${OUTPUT_DIR}/per_lake/"

python -u ${SCRIPT} --run-root ${RUN_ROOT} --alpha ${ALPHA} --output-dir ${OUTPUT_DIR} --phase merge

echo "Merge complete: \$(date)"
EOF
)

echo "Submitted merge job: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"

echo ""
echo "=============================="
echo "  IN-SITU CV PARALLEL PLAN"
echo "=============================="
echo "  Lakes:       ${N_LAKES}"
echo "  Extract job: ${ARRAY_JOB_ID} (array 1-${N_LAKES}, 2G each, 15min)"
echo "  Merge job:   ${MERGE_JOB_ID} (after extract, 4G, 15min)"
echo "  Logs:        ${OUTPUT_DIR}/logs/"
echo "  Output:      ${OUTPUT_DIR}/"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel ${ARRAY_JOB_ID} ${MERGE_JOB_ID}"
