#!/bin/bash
# CPU job: score every model against the deposited structure and the envelope, redraw figures, build the
# showcase data (same as modal_pilot.analyze). Submitted by submit.sh / rerun_oom.sh with --account=<proj>-delta-cpu.
#SBATCH --job-name=envguide-analyze
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR}"
source "${REPO}/delta/config.sh"
use_env
cd "${REPO}"
"${ENV_PY}" envguide/score.py
"${ENV_PY}" envguide/figures.py || echo "figures.py failed (scores are still saved)"
"${ENV_PY}" pilot_analysis.py --work "${REPO}" --jobs 16
echo "scores: envguide/scores.csv   figures: envguide/figures/   analysis: analysis/"
