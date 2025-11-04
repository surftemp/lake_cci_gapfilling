#!/bin/bash
set -euo pipefail
module purge || true

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

# Extract lake_id for this task
LAKE_ID=$(python -c "
import json, itertools
with open('$CONF') as f:
    conf = json.load(f)
ds = conf.get('dataset_options', {})
lakes = ds.get('custom_lake_ids', [])
vers = ds.get('versions_to_process', ['v2'])
alphas = ds.get('alpha_values', conf.get('dineof_parameters', {}).get('alpha_values', [0.1]))
grid = list(itertools.product(lakes, vers, alphas))
if $SLURM_ARRAY_TASK_ID < len(grid):
    print(grid[$SLURM_ARRAY_TASK_ID][0])
else:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")

# Log filenames (split stdout/stderr)
OUTFILE="${LOGS_DIR}/${STAGE}_lake${LAKE_ID}_row${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="${LOGS_DIR}/${STAGE}_lake${LAKE_ID}_row${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.err"

# Redirect stdout/stderr separately
exec 1> "$OUTFILE"
exec 2> "$ERRFILE"

echo "[$(date)] Starting ${STAGE} stage for lake_id=${LAKE_ID}, row=${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "STDOUT: $OUTFILE"
echo "STDERR: $ERRFILE"
echo ""

python lswtctl.py exec --config "$CONF" --row "$SLURM_ARRAY_TASK_ID" --stage "${STAGE}"

echo ""
echo "[$(date)] Completed ${STAGE} stage for lake_id=${LAKE_ID}, row=${SLURM_ARRAY_TASK_ID}"
