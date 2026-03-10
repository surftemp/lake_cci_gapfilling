BASE="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN"
REPO=/home/users/shaerdan/lake_cci_gapfilling
LARGE_LAKES="2 5 6 8 9 11 12 13 15 16 17"

# Wait for: merge_interp jobs + copy job + current extract jobs to all finish
# Then:

for EXP_ROOT in \
  "${BASE}/anomaly-20260131-219f0d-exp0_baseline_both" \
  "${BASE}/anomaly-20260131-97cfac-exp1_no_ice_both"; do

  for ASSEMBLY in satellite insitu; do
    if [ "$ASSEMBLY" = "satellite" ]; then
      SCRIPT=${REPO}/scripts/assemble_satellite_cv.py
      EXTRA="--mode physical"
    else
      SCRIPT=${REPO}/scripts/assemble_insitu_cv.py
      EXTRA=""
    fi

    OUTPUT_DIR=${EXP_ROOT}/${ASSEMBLY}_cv_assembly
    printf '%s\n' ${LARGE_LAKES} > ${OUTPUT_DIR}/lake_list_large.txt

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_ext_lg
#SBATCH --array=1-11
#SBATCH -o ${OUTPUT_DIR}/logs/extract_large_%a.out
#SBATCH -e ${OUTPUT_DIR}/logs/extract_large_%a.err
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=long

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${OUTPUT_DIR}/lake_list_large.txt)
echo "Lake \${LAKE_ID}, started: \$(date)"
python -u ${SCRIPT} --run-root ${EXP_ROOT} ${EXTRA} \
  --output-dir ${OUTPUT_DIR} --phase extract --lake-id \${LAKE_ID}
echo "Done: \$(date)"
EOF

  done
done