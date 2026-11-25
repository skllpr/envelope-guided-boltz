# shellcheck shell=bash
# Settings for running on NCSA Delta. Edit the line marked EDIT; every delta/ script sources this file.

# EDIT: your Delta project code: the part before "-delta-gpu" in the `accounts` output (e.g. bbka).
export DELTA_PROJECT="${DELTA_PROJECT:-CHANGE_ME}"
DELTA_USER="${USER:-$(id -un)}"
# Where things go, per the Delta data-management guide:
#   /work/hdd/<project>/<user>   job I/O (scratch volume, 1 TB, not purged): this repo, the Boltz weights and runs
#   /projects/<project>/<user>   software (500 GB): the conda environment and conda's package cache
#   $HOME (/u/<user>, 100 GB)    nothing from this pipeline; it is not meant for job I/O
export ENVGUIDE_DATA="${ENVGUIDE_DATA:-/work/hdd/${DELTA_PROJECT}/${DELTA_USER}/envguide}"
# Conda comes from Delta's miniforge3-python module; the environment is created by path (--prefix) on /projects,
# because conda's default (~/.conda) is in $HOME. Set CONDA_DIR to use your own Miniforge install instead.
export CONDA_MODULE="${CONDA_MODULE:-miniforge3-python}"
export CONDA_DIR="${CONDA_DIR:-}"
export ENV_PREFIX="${ENV_PREFIX:-/projects/${DELTA_PROJECT}/${DELTA_USER}/conda/envs/envguide}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/projects/${DELTA_PROJECT}/${DELTA_USER}/conda/pkgs}"

export GPU_ACCOUNT="${DELTA_PROJECT}-delta-gpu"
export CPU_ACCOUNT="${DELTA_PROJECT}-delta-cpu"
export BOLTZ_CACHE="${BOLTZ_CACHE:-${ENVGUIDE_DATA}/boltz_cache}"   # weights; runs go to $BOLTZ_CACHE/envguide_runs (run_boltz.py)
export MPLBACKEND=Agg
# Keep pip/torch/triton caches out of $HOME (torch alone is several GB)
export PIP_CACHE_DIR="${ENVGUIDE_DATA}/cache/pip"
export XDG_CACHE_HOME="${ENVGUIDE_DATA}/cache"
export TRITON_CACHE_DIR="${ENVGUIDE_DATA}/cache/triton"
export TORCH_HOME="${ENVGUIDE_DATA}/cache/torch"

# Run settings (the pilot used these)
export SAMPLES="${SAMPLES:-5}"
export STEPS="${STEPS:-100}"
export SCALE="${SCALE:-1.0}"
export MAX_PARALLEL="${MAX_PARALLEL:-25}"   # array tasks running at once
export H200_MIN_TOKENS="${H200_MIN_TOKENS:-2500}"   # bigger maps skip the A40s (48 GB) and run on H200s

# conda's and Lmod's shell code trips `set -u`, so it is switched off while they run
_with_nounset_off() {
    local had_u=0
    case $- in *u*) had_u=1; set +u ;; esac
    "$@"; local rc=$?
    [ "${had_u}" = 1 ] && set -u
    return ${rc}
}

_load_conda() {
    if [ -n "${CONDA_DIR}" ]; then
        # shellcheck disable=SC1091
        source "${CONDA_DIR}/etc/profile.d/conda.sh"
    else
        if ! type module >/dev/null 2>&1; then  # non-login shells may not have Lmod's function yet
            # shellcheck disable=SC1090,SC1091
            for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh; do
                [ -f "${f}" ] && source "${f}" && break
            done
        fi
        module load "${CONDA_MODULE}" || { echo "could not load module ${CONDA_MODULE}; see 'module avail'" >&2; return 1; }
        eval "$(conda shell.bash hook)"
    fi
}

load_conda() { _with_nounset_off _load_conda; }

activate_env() {
    load_conda || return 1
    _with_nounset_off conda activate "${ENV_PREFIX}"
}

check_config() {
    if [ -z "${DELTA_PROJECT}" ] || [ "${DELTA_PROJECT}" = "CHANGE""_ME" ]; then
        echo "Set DELTA_PROJECT in delta/config.sh first (run 'accounts' on Delta to see it)." >&2
        exit 1
    fi
    local repo
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
    case "${repo}" in
        /u/*|"${HOME}"/*)
            echo "This repo is under your home directory; jobs write into it (scores, figures, logs, unzipped maps)." >&2
            echo "Delta's guide says home is not for job I/O. Clone it under /work/hdd/${DELTA_PROJECT}/${DELTA_USER}/ instead." >&2
            exit 1 ;;
    esac
}
