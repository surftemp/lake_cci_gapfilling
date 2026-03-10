BASE="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN"
REPO="/home/users/shaerdan/lake_cci_gapfilling"
HOMEROOT="/home/users/shaerdan/lake_cci_gapfilling/logs"
LOGDIR="${BASE}/pipeline_chain_logs"
mkdir -p ${LOGDIR}

# Exp0: lakes 11-17 remaining (lake 11 may be partial, redo it)
printf '%s\n' 11 12 13 15 16 17 > ${LOGDIR}/exp0_remaining.txt

MERGE_EXP0=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=merge_interp_exp0
#SBATCH --array=1-6
#SBATCH -o ${LOGDIR}/merge_exp0_%a.out
#SBATCH -e ${LOGDIR}/merge_exp0_%a.err
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${LOGDIR}/exp0_remaining.txt)
echo "Lake \${LAKE_ID} started: \$(date)"
python -u ${REPO}/scripts/merge_segments.py \
  --manifest ${BASE}/exp0_baseline_large_merged_4splits/segment_manifest.json \
  --lake-id \${LAKE_ID} --file-filter interp_full
echo "Done: \$(date)"
EOF
)
echo "Merge exp0 remaining: ${MERGE_EXP0} (6 tasks)"

# Exp1: all 11 lakes
printf '%s\n' 2 5 6 8 9 11 12 13 15 16 17 > ${LOGDIR}/exp1_all.txt

MERGE_EXP1=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=merge_interp_exp1
#SBATCH --array=1-11
#SBATCH -o ${LOGDIR}/merge_exp1_%a.out
#SBATCH -e ${LOGDIR}/merge_exp1_%a.err
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${LOGDIR}/exp1_all.txt)
echo "Lake \${LAKE_ID} started: \$(date)"
python -u ${REPO}/scripts/merge_segments.py \
  --manifest ${BASE}/exp1_baseline_large_merged_4splits/segment_manifest.json \
  --lake-id \${LAKE_ID} --file-filter interp_full
echo "Done: \$(date)"
EOF
)
echo "Merge exp1 all: ${MERGE_EXP1} (11 tasks)"

# Copy (after both merges)
COPY_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_EXP0}:${MERGE_EXP1} <<EOF
#!/bin/bash
#SBATCH --job-name=copy_interp
#SBATCH --array=1-11
#SBATCH -o ${LOGDIR}/copy_%a.out
#SBATCH -e ${LOGDIR}/copy_%a.err
#SBATCH --mem=4G
#SBATCH -t 1:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=short

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${LOGDIR}/exp1_all.txt)
L=\$(printf '%09d' \$LAKE_ID)
echo "Copy lake \$LAKE_ID started: \$(date)"
cp -f ${BASE}/exp0_baseline_large_merged_4splits/post/\${L}/a1000/*interp_full.nc \
      ${BASE}/anomaly-20260131-219f0d-exp0_baseline_both/post/\${L}/a1000/
cp -f ${BASE}/exp1_baseline_large_merged_4splits/post/\${L}/a1000/*interp_full.nc \
      ${BASE}/anomaly-20260131-97cfac-exp1_no_ice_both/post/\${L}/a1000/
echo "Done: \$(date)"
EOF
)
echo "Copy: ${COPY_JOB} (11 tasks)"

# Retrofit (after copy)
LARGE_LAKES="2 5 6 8 9 11 12 13 15 16 17"

for EXP in exp0 exp1; do
  if [ "$EXP" = "exp0" ]; then
    RUN_ROOT="${BASE}/anomaly-20260131-219f0d-exp0_baseline_both"
    CONFIG="configs/exp0_baseline.json"
  else
    RUN_ROOT="${BASE}/anomaly-20260131-97cfac-exp1_no_ice_both"
    CONFIG="configs/exp1_no_ice.json"
  fi

  RETRO=$(sbatch --parsable --dependency=afterok:${COPY_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=retrofit_${EXP}
#SBATCH --array=1-11
#SBATCH -o ${LOGDIR}/retrofit_${EXP}_%a.out
#SBATCH -e ${LOGDIR}/retrofit_${EXP}_%a.err
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${LOGDIR}/exp1_all.txt)
echo "Retrofit ${EXP} lake \${LAKE_ID} started: \$(date)"
cd ${REPO}
python -u scripts/retrofit_post.py \
  --run-root ${RUN_ROOT} --lake-ids \${LAKE_ID} --config ${CONFIG}
echo "Done: \$(date)"
EOF
  )
  echo "Retrofit ${EXP}: ${RETRO} (11 tasks)"
done

# CV extract large lakes (parallel with retrofit, after copy)
for EXP_ROOT in \
  "${BASE}/anomaly-20260131-219f0d-exp0_baseline_both" \
  "${BASE}/anomaly-20260131-97cfac-exp1_no_ice_both"; do

  EXP_SHORT=$(basename ${EXP_ROOT} | cut -c1-4)

  for ASSEMBLY in satellite insitu; do
    if [ "$ASSEMBLY" = "satellite" ]; then
      SCRIPT=${REPO}/scripts/assemble_satellite_cv.py
      EXTRA="--mode physical"
    else
      SCRIPT=${REPO}/scripts/assemble_insitu_cv.py
      EXTRA=""
    fi

    OUTPUT_DIR=${EXP_ROOT}/${ASSEMBLY}_cv_assembly
    mkdir -p ${OUTPUT_DIR}/logs

    EXTRACT=$(sbatch --parsable --dependency=afterok:${COPY_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_ext_${EXP_SHORT}
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

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${LOGDIR}/exp1_all.txt)
echo "Lake \${LAKE_ID} started: \$(date)"
python -u ${SCRIPT} --run-root ${EXP_ROOT} ${EXTRA} \
  --output-dir ${OUTPUT_DIR} --phase extract --lake-id \${LAKE_ID}
echo "Done: \$(date)"
EOF
    )
    echo "${ASSEMBLY} extract (${EXP_SHORT}): ${EXTRACT}"

    # Stats after extract
    for TAG in all_lakes large_lakes; do
      if [ "$TAG" = "large_lakes" ]; then
        LAKE_ARG="--lake-ids ${LARGE_LAKES}"
      else
        LAKE_ARG=""
      fi

      sbatch --dependency=afterok:${EXTRACT} <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_stats_${TAG:0:3}_${EXP_SHORT}
#SBATCH -o ${OUTPUT_DIR}/logs/stats_${TAG}.out
#SBATCH -e ${OUTPUT_DIR}/logs/stats_${TAG}.err
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=short

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

echo "Stats ${TAG} started: \$(date)"
python -u ${SCRIPT} --run-root ${EXP_ROOT} ${EXTRA} \
  --output-dir ${OUTPUT_DIR} --phase merge \
  ${LAKE_ARG} --subset-tag ${TAG}
echo "Done: \$(date)"
EOF
    done
  done
done

echo ""
echo "All jobs submitted. Go to sleep."
echo "Monitor: squeue -u \$USER"