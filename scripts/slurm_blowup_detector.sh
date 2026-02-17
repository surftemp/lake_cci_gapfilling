#!/bin/bash
#SBATCH --job-name=blowup_detect
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/users/shaerdan/lake_cci_gapfilling
#SBATCH -o logs/blowup_%A_%a.out
#SBATCH -e logs/blowup_%A_%a.err

# ============================================================
# Parallel blowup detection via SLURM array jobs
#
# Usage:
#   # 1. Generate the lake list first:
#   ls /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both/dineof/ \
#     | sort > /home/users/shaerdan/lake_cci_gapfilling/blowup_lake_list.txt
#
#   # 2. Count lakes and submit:
#   N=$(wc -l < blowup_lake_list.txt)
#   sbatch --array=1-${N} scripts/slurm_blowup_detector.sh
#
#   # 3. After all jobs finish, assemble:
#   python scripts/blowup_detector.py \
#     --exp-dir /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both \
#     --assemble-only
# ============================================================

set -euo pipefail

# --- Config ---
EXP_DIR="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both"
LAKE_LIST="/home/users/shaerdan/lake_cci_gapfilling/blowup_lake_list.txt"
DETECTION_K="4.0"

# --- Activate environment ---
source activate lake_cci_gapfilling 2>/dev/null || conda activate lake_cci_gapfilling 2>/dev/null || true

# --- Get lake ID for this array task ---
LAKE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${LAKE_LIST}")

if [ -z "${LAKE_ID}" ]; then
    echo "ERROR: No lake ID found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "=== Task ${SLURM_ARRAY_TASK_ID}: Lake ${LAKE_ID} ==="
echo "Start: $(date)"

# --- Run detector ---
python scripts/blowup_detector.py \
    --exp-dir "${EXP_DIR}" \
    --lake-id "${LAKE_ID}" \
    --k "${DETECTION_K}"

echo "End: $(date)"
