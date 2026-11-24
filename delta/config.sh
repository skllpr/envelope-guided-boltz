# shellcheck shell=bash
# Settings for running on NCSA Delta. Edit the two lines marked EDIT, then every delta/ script sources this file.

# EDIT: your Delta project code: the part before "-delta-gpu" in the `accounts` output (e.g. bbka).
export DELTA_PROJECT="${DELTA_PROJECT:-CHANGE_ME}"
# EDIT: a folder with room for the Boltz weights (~6 GB) and the runs (a few GB per 100 maps).
#       /projects/<project>/<user> is the usual choice; your home directory is too small.
export ENVGUIDE_DATA="${ENVGUIDE_DATA:-/projects/${DELTA_PROJECT}/${USER:-$(id -un)}/envguide}"

export GPU_ACCOUNT="${DELTA_PROJECT}-delta-gpu"
export CPU_ACCOUNT="${DELTA_PROJECT}-delta-cpu"
export CONDA_DIR="${CONDA_DIR:-${ENVGUIDE_DATA}/miniforge3}"
export ENV_NAME="envguide"
export BOLTZ_CACHE="${BOLTZ_CACHE:-${ENVGUIDE_DATA}/boltz_cache}"   # weights; runs go to $BOLTZ_CACHE/envguide_runs (run_boltz.py)
export MPLBACKEND=Agg

# Run settings (the pilot used these)
export SAMPLES="${SAMPLES:-5}"
export STEPS="${STEPS:-100}"
export SCALE="${SCALE:-1.0}"
export MAX_PARALLEL="${MAX_PARALLEL:-25}"   # array tasks running at once
export H200_MIN_TOKENS="${H200_MIN_TOKENS:-2500}"   # bigger maps skip the A40s (48 GB) and run on H200s

activate_env() {
    # shellcheck disable=SC1091
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
}

check_config() {
    if [ "${DELTA_PROJECT}" = "CHANGE_ME" ]; then
        echo "Set DELTA_PROJECT in delta/config.sh first (run 'accounts' on Delta to see it)." >&2
        exit 1
    fi
}
