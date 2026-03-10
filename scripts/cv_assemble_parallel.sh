#!/bin/bash
# ===========================================================================
# CV Assembly — Parallel SLURM launcher (satellite + insitu, both experiments)
# ===========================================================================
#
# For each experiment × assembly type, submits:
#   1. Array job: one task per lake (parallel extraction)
#   2. Merge job (all_lakes): stats from all per-lake CSVs
#   3. Merge job (large_lakes): stats from 11 large lakes only
#
# Usage:
#   bash cv_assemble_parallel.sh
# ===========================================================================
# Activate conda on login node (before strict mode, conda scripts use unset vars)
source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

set -euo pipefail

# === CONFIGURATION ===
REPO=/home/users/shaerdan/lake_cci_gapfilling
BASE=/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN
ALPHA=a1000
ACCOUNT=eocis_chuk
PARTITION=standard
QOS=long
LARGE_LAKES="2 5 6 8 9 11 12 13 15 16 17"
# =====================

EXP_ROOTS=(
    "${BASE}/anomaly-20260131-219f0d-exp0_baseline_both"
    "${BASE}/anomaly-20260131-97cfac-exp1_no_ice_both"
)

ASSEMBLIES=(satellite insitu)

# Clean old assembly + stats files to prevent stale data / duplicate headers
for EXP_ROOT in "${EXP_ROOTS[@]}"; do
    for ASSEMBLY in "${ASSEMBLIES[@]}"; do
        OUTPUT_DIR=${EXP_ROOT}/${ASSEMBLY}_cv_assembly
        rm -f ${OUTPUT_DIR}/${ASSEMBLY}_cv_assembly_*.csv
        rm -f ${OUTPUT_DIR}/${ASSEMBLY}_cv_assembly_*.parquet
    done
done

for EXP_ROOT in "${EXP_ROOTS[@]}"; do
    EXP_NAME=$(basename "${EXP_ROOT}")

    for ASSEMBLY in "${ASSEMBLIES[@]}"; do
        if [ "$ASSEMBLY" = "satellite" ]; then
            SCRIPT=${REPO}/scripts/assemble_satellite_cv.py
            EXTRA="--mode physical"
        else
            SCRIPT=${REPO}/scripts/assemble_insitu_cv.py
            EXTRA=""
        fi

        OUTPUT_DIR=${EXP_ROOT}/${ASSEMBLY}_cv_assembly
        mkdir -p ${OUTPUT_DIR}/per_lake
        mkdir -p ${OUTPUT_DIR}/logs

        echo ""
        echo "======================================================================"
        echo "  ${EXP_NAME} / ${ASSEMBLY}"
        echo "======================================================================"

        # --- Step 1: Generate lake list ---
        echo "Generating lake list..."
        python -u ${SCRIPT} \
            --run-root ${EXP_ROOT} \
            --alpha ${ALPHA} \
            ${EXTRA} \
            --output-dir ${OUTPUT_DIR} \
            --phase list-lakes \
            --all \
            | grep -E '^[0-9]+$' > ${OUTPUT_DIR}/lake_list_all.txt

        N_LAKES=$(wc -l < ${OUTPUT_DIR}/lake_list_all.txt)
        echo "Found ${N_LAKES} lakes"

        if [ "${N_LAKES}" -eq 0 ]; then
            echo "WARNING: No lakes found, skipping."
            continue
        fi

        # --- Step 2: Submit array job (one task per lake) ---
        ARRAY_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_ext_${EXP_NAME:0:4}
#SBATCH --array=1-${N_LAKES}
#SBATCH -o ${OUTPUT_DIR}/logs/extract_%a.out
#SBATCH -e ${OUTPUT_DIR}/logs/extract_%a.err
#SBATCH --mem=192G
#SBATCH -t 2-00:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${OUTPUT_DIR}/lake_list_all.txt)
echo "Task \${SLURM_ARRAY_TASK_ID}: Lake \${LAKE_ID}"
echo "Started: \$(date)"

python -u ${SCRIPT} \
    --run-root ${EXP_ROOT} \
    --alpha ${ALPHA} \
    ${EXTRA} \
    --output-dir ${OUTPUT_DIR} \
    --phase extract \
    --lake-id \${LAKE_ID}

echo "Finished: \$(date)"
EOF
        )
        echo "  Extract array: ${ARRAY_JOB_ID} (${N_LAKES} tasks)"

        # --- Step 3: Merge ALL lakes ---
        MERGE_ALL_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_merge_all_${EXP_NAME:0:4}
#SBATCH -o ${OUTPUT_DIR}/logs/merge_all.out
#SBATCH -e ${OUTPUT_DIR}/logs/merge_all.err
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

echo "Merge (all_lakes) started: \$(date)"
python -u ${SCRIPT} \
    --run-root ${EXP_ROOT} \
    --alpha ${ALPHA} \
    ${EXTRA} \
    --output-dir ${OUTPUT_DIR} \
    --phase merge \
    --subset-tag all_lakes
echo "Merge (all_lakes) complete: \$(date)"
EOF
        )
        echo "  Merge all_lakes: ${MERGE_ALL_ID} (depends on ${ARRAY_JOB_ID})"

        # --- Step 4: Merge LARGE LAKES only ---
        MERGE_LARGE_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_merge_lg_${EXP_NAME:0:4}
#SBATCH -o ${OUTPUT_DIR}/logs/merge_large.out
#SBATCH -e ${OUTPUT_DIR}/logs/merge_large.err
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

echo "Merge (large_lakes) started: \$(date)"
python -u ${SCRIPT} \
    --run-root ${EXP_ROOT} \
    --alpha ${ALPHA} \
    ${EXTRA} \
    --output-dir ${OUTPUT_DIR} \
    --phase merge \
    --lake-ids ${LARGE_LAKES} \
    --subset-tag large_lakes
echo "Merge (large_lakes) complete: \$(date)"
EOF
        )
        echo "  Merge large_lakes: ${MERGE_LARGE_ID} (depends on ${ARRAY_JOB_ID})"

    done
done

echo ""
echo "======================================================================"
echo "  ALL JOBS SUBMITTED"
echo "======================================================================"
echo "  Monitor: squeue -u \$USER"
echo ""