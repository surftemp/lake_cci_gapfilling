#!/bin/bash
#SBATCH --job-name=assemble_sat_cv
#SBATCH -o assemble_sat_cv.out
#SBATCH -e assemble_sat_cv.err
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=long

source activate lake_cci_gapfilling 2>/dev/null || conda activate lake_cci_gapfilling 2>/dev/null || true

REPO=/home/users/shaerdan/lake_cci_gapfilling
RUN_ROOT=/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260131-219f0d-exp0_baseline_both

cd $REPO/scripts

echo "=========================================="
echo "Satellite CV Assembly"
echo "Started: $(date)"
echo "Run root: $RUN_ROOT"
echo "=========================================="

python -u assemble_satellite_cv.py \
    --run-root $RUN_ROOT \
    --all \
    --mode physical

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
