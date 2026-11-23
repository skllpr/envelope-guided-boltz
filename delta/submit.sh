#!/bin/bash
# Submit every map in delta/targets.txt (or a list file given as $1) to the A40 partition, one GPU per map,
# plus a CPU analysis job that starts when the whole array has finished.
#   bash delta/submit.sh                 # all targets
#   bash delta/submit.sh my_list.txt     # a subset (one EMD id per line)
#   TEST=1 bash delta/submit.sh          # first map only, 1 sample, 20 steps: checks the setup for ~0.1 SU
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO}/delta/config.sh"
check_config
cd "${REPO}"
mkdir -p logs
LIST="${1:-delta/targets.txt}"
N=$(grep -c . "${LIST}")
TIME="06:00:00"
if [ "${TEST:-0}" = "1" ]; then
    N=1; export SAMPLES=1 STEPS=20; TIME="00:45:00"
    # separate run folder (weights linked in), so the short test is not taken as a finished run later
    TEST_CACHE="${ENVGUIDE_DATA}/boltz_cache_test"
    rm -rf "${TEST_CACHE}"; mkdir -p "${TEST_CACHE}"
    for f in "${BOLTZ_CACHE}"/*; do
        [ "$(basename "${f}")" = "envguide_runs" ] || ln -s "${f}" "${TEST_CACHE}/"
    done
    export BOLTZ_CACHE="${TEST_CACHE}"
    echo "TEST run: $(head -1 "${LIST}") only, 1 sample, 20 steps -> ${TEST_CACHE}/envguide_runs"
fi

# A40 node = 4 GPUs, 64 cores, ~250 GB: asking for <= 1/4 of the cores and memory keeps the charge at 1 GPU.
# gpuA40x4 is charged at 0.5 SU per GPU-hour (A100 = 1.0).
JOB=$(sbatch --parsable --export=ALL,TARGET_LIST="${LIST}" \
    --account="${GPU_ACCOUNT}" --partition=gpuA40x4 --cpus-per-task=16 --mem=60g --time="${TIME}" \
    --array="1-${N}%${MAX_PARALLEL}" delta/run_map.sbatch)
echo "GPU array ${JOB}: ${N} maps on gpuA40x4"
if [ "${TEST:-0}" != "1" ]; then
    AJOB=$(sbatch --parsable --dependency=afterany:"${JOB}" --account="${CPU_ACCOUNT}" delta/analyze.sbatch)
    echo "analysis ${AJOB} runs after the array finishes"
fi
echo "watch: squeue -u $(id -un)    logs: logs/    status: \$BOLTZ_CACHE/envguide_runs/status/"
echo "then:  bash delta/rerun_oom.sh   (maps that ran out of GPU memory -> H200)"
