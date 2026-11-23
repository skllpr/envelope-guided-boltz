#!/bin/bash
# After the A40 array: rerun the maps that ran out of GPU memory on an H200 (141 GB), then re-analyse.
#   bash delta/rerun_oom.sh            # H200 for out-of-memory maps
#   FAILED=1 bash delta/rerun_oom.sh   # also retry maps that failed for other reasons (on an A40 again)
# H200 time is charged at 3 SU per GPU-hour (6x an A40), so only out-of-memory maps go there.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO}/delta/config.sh"
check_config
cd "${REPO}"
mkdir -p logs
STATUS="${BOLTZ_CACHE}/envguide_runs/status"
python3 - "${STATUS}" <<'EOF'
import glob, json, os, sys
oom, failed, ok, missing = [], [], [], []
targets = [l.strip() for l in open("delta/targets.txt") if l.strip()]
for emd in targets:
    p = os.path.join(sys.argv[1], f"{emd}.json")
    if not os.path.exists(p):
        missing.append(emd); continue
    vals = list(json.load(open(p))["modes"].values())
    if any("out of GPU memory" in v for v in vals): oom.append(emd)
    elif len(vals) < 3 or any(v.startswith("FAILED") for v in vals): failed.append(emd)
    else: ok.append(emd)
open("delta/oom_targets.txt", "w").write("".join(e + "\n" for e in oom))
open("delta/failed_targets.txt", "w").write("".join(e + "\n" for e in failed + missing))
print(f"done {len(ok)}, out of memory {len(oom)}, failed {len(failed)}, no status (still running / never ran) {len(missing)}")
EOF

JOBS=""
if [ -s delta/oom_targets.txt ]; then
    N=$(grep -c . delta/oom_targets.txt)
    # H200 node = 8 GPUs, ~96 cores, ~2 TB: 12 cores + 200 GB stays within one GPU's share.
    J=$(sbatch --parsable --export=ALL,TARGET_LIST=delta/oom_targets.txt \
        --account="${GPU_ACCOUNT}" --partition=gpuH200x8 --cpus-per-task=12 --mem=200g --time=06:00:00 \
        --array="1-${N}%8" delta/run_map.sbatch)
    echo "H200 array ${J}: ${N} maps"; JOBS="${J}"
fi
if [ "${FAILED:-0}" = "1" ] && [ -s delta/failed_targets.txt ]; then
    N=$(grep -c . delta/failed_targets.txt)
    J=$(sbatch --parsable --export=ALL,TARGET_LIST=delta/failed_targets.txt \
        --account="${GPU_ACCOUNT}" --partition=gpuA40x4 --cpus-per-task=16 --mem=60g --time=06:00:00 \
        --array="1-${N}%${MAX_PARALLEL}" delta/run_map.sbatch)
    echo "A40 retry array ${J}: ${N} maps"; JOBS="${JOBS:+${JOBS}:}${J}"
fi
if [ -n "${JOBS}" ]; then
    A=$(sbatch --parsable --dependency=afterany:"${JOBS}" --account="${CPU_ACCOUNT}" delta/analyze.sbatch)
    echo "analysis ${A} runs after them"
else
    echo "nothing to rerun"
fi
