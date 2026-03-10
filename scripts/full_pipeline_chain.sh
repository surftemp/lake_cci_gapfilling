#!/bin/bash
# ===========================================================================
# Full pipeline: merge interp → copy → retrofit + CV assembly (parallel)
# ===========================================================================
source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling
set -euo pipefail

BASE="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN"
REPO="/home/users/shaerdan/lake_cci_gapfilling"
LARGE_LAKES="2 5 6 8 9 11 12 13 15 16 17"
ACCOUNT=eocis_chuk

EXP0_MAIN="${BASE}/anomaly-20260131-219f0d-exp0_baseline_both"
EXP1_MAIN="${BASE}/anomaly-20260131-97cfac-exp1_no_ice_both"
EXP0_MERGE="${BASE}/exp0_baseline_large_merged_4splits"
EXP1_MERGE="${BASE}/exp1_baseline_large_merged_4splits"

LOGDIR="${BASE}/pipeline_chain_logs"
mkdir -p ${LOGDIR}

echo "======================================================================"
echo "  STEP 1: Merge interp files (2 jobs, one per experiment)"
echo "======================================================================"

MERGE_EXP0=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=merge_interp_exp0
#SBATCH -o ${LOGDIR}/merge_interp_exp0.out
#SBATCH -e ${LOGDIR}/merge_interp_exp0.err
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

echo "Merge interp exp0 started: \$(date)"
python -u ${REPO}/scripts/merge_segments.py \
  --manifest ${EXP0_MERGE}/segment_manifest.json \
  --file-filter interp_full
echo "Merge interp exp0 done: \$(date)"
EOF
)
echo "  Merge exp0: ${MERGE_EXP0}"

MERGE_EXP1=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=merge_interp_exp1
#SBATCH -o ${LOGDIR}/merge_interp_exp1.out
#SBATCH -e ${LOGDIR}/merge_interp_exp1.err
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

echo "Merge interp exp1 started: \$(date)"
python -u ${REPO}/scripts/merge_segments.py \
  --manifest ${EXP1_MERGE}/segment_manifest.json \
  --file-filter interp_full
echo "Merge interp exp1 done: \$(date)"
EOF
)
echo "  Merge exp1: ${MERGE_EXP1}"

echo ""
echo "======================================================================"
echo "  STEP 2: Copy interp files to main run (after both merges)"
echo "======================================================================"

COPY_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_EXP0}:${MERGE_EXP1} <<EOF
#!/bin/bash
#SBATCH --job-name=copy_interp
#SBATCH -o ${LOGDIR}/copy_interp.out
#SBATCH -e ${LOGDIR}/copy_interp.err
#SBATCH --mem=4G
#SBATCH -t 2:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --qos=short

echo "Copy started: \$(date)"
for lake in ${LARGE_LAKES}; do
    L=\$(printf '%09d' \$lake)
    cp -f ${EXP0_MERGE}/post/\${L}/a1000/*interp_full.nc \
          ${EXP0_MAIN}/post/\${L}/a1000/
    cp -f ${EXP1_MERGE}/post/\${L}/a1000/*interp_full.nc \
          ${EXP1_MAIN}/post/\${L}/a1000/
    echo "  Lake \$lake done"
done

echo "Verify exp0 lake 9:"
python3 -c "
import xarray as xr, glob
for f in sorted(glob.glob('${EXP0_MAIN}/post/000000009/a1000/*interp_full.nc')):
    ds = xr.open_dataset(f)
    print(f'  {f.split(\"filled_fine_\")[-1]}: time={ds.sizes[\"time\"]}')
    ds.close()
"
echo "Copy done: \$(date)"
EOF
)
echo "  Copy: ${COPY_JOB} (depends on ${MERGE_EXP0}, ${MERGE_EXP1})"

echo ""
echo "======================================================================"
echo "  STEP 3: Retrofit (after copy) — 2 array jobs"
echo "======================================================================"

# Write lake list for retrofit array jobs
printf '%s\n' ${LARGE_LAKES} > ${LOGDIR}/large_lake_list.txt
N_LARGE=11

RETROFIT_EXP0=$(sbatch --parsable --dependency=afterok:${COPY_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=retrofit_exp0
#SBATCH --array=1-${N_LARGE}
#SBATCH -o ${LOGDIR}/retrofit_exp0_%a.out
#SBATCH -e ${LOGDIR}/retrofit_exp0_%a.err
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${LOGDIR}/large_lake_list.txt)
echo "Retrofit exp0 lake \${LAKE_ID} started: \$(date)"
cd ${REPO}
python -u scripts/retrofit_post.py \
  --run-root ${EXP0_MAIN} \
  --lake-ids \${LAKE_ID} \
  --config configs/exp0_baseline.json
echo "Done: \$(date)"
EOF
)
echo "  Retrofit exp0: ${RETROFIT_EXP0} (${N_LARGE} tasks)"

RETROFIT_EXP1=$(sbatch --parsable --dependency=afterok:${COPY_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=retrofit_exp1
#SBATCH --array=1-${N_LARGE}
#SBATCH -o ${LOGDIR}/retrofit_exp1_%a.out
#SBATCH -e ${LOGDIR}/retrofit_exp1_%a.err
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --qos=high

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${LOGDIR}/large_lake_list.txt)
echo "Retrofit exp1 lake \${LAKE_ID} started: \$(date)"
cd ${REPO}
python -u scripts/retrofit_post.py \
  --run-root ${EXP1_MAIN} \
  --lake-ids \${LAKE_ID} \
  --config configs/exp1_no_ice.json
echo "Done: \$(date)"
EOF
)
echo "  Retrofit exp1: ${RETROFIT_EXP1} (${N_LARGE} tasks)"

echo ""
echo "======================================================================"
echo "  STEP 4: CV Assembly extract for large lakes (parallel with retrofit)"
echo "======================================================================"

for EXP_ROOT in "${EXP0_MAIN}" "${EXP1_MAIN}"; do
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
    mkdir -p ${OUTPUT_DIR}/per_lake ${OUTPUT_DIR}/logs
    printf '%s\n' ${LARGE_LAKES} > ${OUTPUT_DIR}/lake_list_large.txt

    EXTRACT_JOB=$(sbatch --parsable --dependency=afterok:${COPY_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_ext_lg_${EXP_SHORT}
#SBATCH --array=1-${N_LARGE}
#SBATCH -o ${OUTPUT_DIR}/logs/extract_large_%a.out
#SBATCH -e ${OUTPUT_DIR}/logs/extract_large_%a.err
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --account=${ACCOUNT}
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
    )
    echo "  ${ASSEMBLY} extract (${EXP_SHORT}): ${EXTRACT_JOB} (${N_LARGE} tasks)"

    # Merge stats jobs — depend on BOTH current extract AND the earlier full extract
    # (the full extract from cv_assemble_parallel.sh should be done by now,
    #  but these large lake extracts overwrite per-lake CSVs with correct interp data)

    MERGE_ALL_STATS=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_stats_all_${EXP_SHORT}
#SBATCH -o ${OUTPUT_DIR}/logs/merge_all_v2.out
#SBATCH -e ${OUTPUT_DIR}/logs/merge_all_v2.err
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --qos=short

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

echo "Stats all_lakes started: \$(date)"
python -u ${SCRIPT} --run-root ${EXP_ROOT} ${EXTRA} \
  --output-dir ${OUTPUT_DIR} --phase merge --subset-tag all_lakes
echo "Done: \$(date)"
EOF
    )

    MERGE_LARGE_STATS=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB} <<EOF
#!/bin/bash
#SBATCH --job-name=${ASSEMBLY}_stats_lg_${EXP_SHORT}
#SBATCH -o ${OUTPUT_DIR}/logs/merge_large_v2.out
#SBATCH -e ${OUTPUT_DIR}/logs/merge_large_v2.err
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --qos=short

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

echo "Stats large_lakes started: \$(date)"
python -u ${SCRIPT} --run-root ${EXP_ROOT} ${EXTRA} \
  --output-dir ${OUTPUT_DIR} --phase merge \
  --lake-ids ${LARGE_LAKES} --subset-tag large_lakes
echo "Done: \$(date)"
EOF
    )
    echo "  ${ASSEMBLY} stats all (${EXP_SHORT}): ${MERGE_ALL_STATS}"
    echo "  ${ASSEMBLY} stats large (${EXP_SHORT}): ${MERGE_LARGE_STATS}"

  done
done

echo ""
echo "======================================================================"
echo "  ALL JOBS SUBMITTED — go to sleep"
echo "======================================================================"
echo "  Logs: ${LOGDIR}/"
echo "  Monitor: squeue -u \$USER"
echo ""
echo "  Chain:"
echo "    merge interp → copy → retrofit (parallel)"
echo "                       → cv extract large lakes → stats"
echo ""
