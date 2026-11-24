#!/bin/bash
# Submit every map in delta/targets.txt (or a list file given as $1), one GPU per map: A40s for maps up to
# H200_MIN_TOKENS residues, H200s above that; plus a CPU analysis job that starts when all of them have finished.
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

# Split by size: maps over H200_MIN_TOKENS residues would run out of memory on a 48 GB A40, so they go straight
# to an H200 (3 SU per GPU-hour) instead of failing on an A40 (0.5 SU per GPU-hour) first.
python3 "${REPO}/delta/split_targets.py" "${LIST}" "${H200_MIN_TOKENS}" "${N}"

DEPS=""
if [ -s delta/targets_a40.txt ]; then
    NA=$(grep -c . delta/targets_a40.txt)
    # A40 node = 4 GPUs, 64 cores, ~250 GB: <= 1/4 of the cores and memory keeps the charge at 1 GPU.
    J=$(sbatch --parsable --export=ALL,TARGET_LIST=delta/targets_a40.txt \
        --account="${GPU_ACCOUNT}" --partition=gpuA40x4 --cpus-per-task=16 --mem=60g --time="${TIME}" \
        --array="1-${NA}%${MAX_PARALLEL}" delta/run_map.sbatch)
    echo "A40 array ${J}: ${NA} maps (<= ${H200_MIN_TOKENS} residues) on gpuA40x4"
    DEPS="${J}"
fi
if [ -s delta/targets_h200.txt ]; then
    NH=$(grep -c . delta/targets_h200.txt)
    # H200 node = 8 GPUs, ~96 cores, ~2 TB: 12 cores + 200 GB stays within one GPU's share.
    J=$(sbatch --parsable --export=ALL,TARGET_LIST=delta/targets_h200.txt \
        --account="${GPU_ACCOUNT}" --partition=gpuH200x8 --cpus-per-task=12 --mem=200g --time="${TIME}" \
        --array="1-${NH}%8" delta/run_map.sbatch)
    echo "H200 array ${J}: ${NH} maps (> ${H200_MIN_TOKENS} residues) on gpuH200x8"
    DEPS="${DEPS:+${DEPS}:}${J}"
fi
if [ "${TEST:-0}" != "1" ] && [ -n "${DEPS}" ]; then
    AJOB=$(sbatch --parsable --dependency=afterany:"${DEPS}" --account="${CPU_ACCOUNT}" delta/analyze.sbatch)
    echo "analysis ${AJOB} runs after the GPU jobs finish"
fi
echo "watch: squeue -u $(id -un)    logs: logs/    status: \$BOLTZ_CACHE/envguide_runs/status/"
echo "then:  bash delta/rerun_oom.sh   (maps that still ran out of memory on an A40 -> H200)"
