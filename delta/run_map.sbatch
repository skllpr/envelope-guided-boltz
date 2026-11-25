#!/bin/bash
# Slurm array task: one map per task, one GPU. Submitted by delta/submit.sh (A40s) and delta/rerun_oom.sh (H200s),
# which set the partition, account, resources and the list file. Task N runs line N of $TARGET_LIST.
#SBATCH --job-name=envguide
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR}"
source "${REPO}/delta/config.sh"
use_env
cd "${REPO}" || exit 1

EMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TARGET_LIST}" | tr -d '[:space:]')
[ -z "${EMD}" ] && { echo "no target on line ${SLURM_ARRAY_TASK_ID} of ${TARGET_LIST}"; exit 0; }
ENVGUIDE_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export ENVGUIDE_GPU
echo "${EMD} on $(hostname), ${ENVGUIDE_GPU}"
"${ENV_PY}" delta/run_map.py "${EMD}" --samples "${SAMPLES}" --steps "${STEPS}" --scale "${SCALE}"
