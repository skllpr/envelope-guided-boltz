#!/bin/bash
# One-time setup on a Delta login node (needs internet, uses no allocation):
#   bash delta/setup_env.sh
# Installs Miniforge + a Python 3.11 env with Boltz-2 2.2.1, builds US-align, downloads the Boltz-2 weights and
# lays out the folders the pipeline expects (maps_unzipped/, deposited/, external/USalign/ at the repo root).
# Safe to rerun: finished steps are skipped.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO}/delta/config.sh"
check_config
mkdir -p "${ENVGUIDE_DATA}" "${BOLTZ_CACHE}"

# 1. Miniforge (own copy, so nothing depends on Delta's module versions)
INSTALLER="/tmp/miniforge_$(id -un).sh"
if [ ! -x "${CONDA_DIR}/bin/conda" ]; then
    curl -fsSL -o "${INSTALLER}" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "${INSTALLER}"
fi
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# 2. Python env (PyTorch wheels bring their own CUDA runtime; Delta's driver supports it)
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"
pip install -q "boltz==2.2.1" gemmi mrcfile matplotlib pandas pyyaml
python -c "import boltz, torch; print('boltz ok, torch', torch.__version__)"

# 3. US-align (scoring)
mkdir -p "${REPO}/external"
if [ ! -x "${REPO}/external/USalign/USalign" ]; then
    rm -rf "${REPO}/external/USalign"
    git clone -q --depth 1 https://github.com/pylelab/USalign.git "${REPO}/external/USalign"
    (cd "${REPO}/external/USalign" && g++ -O3 -ffast-math -o USalign USalign.cpp -lm)
fi

# 4. Boltz-2 weights + CCD into BOLTZ_CACHE (once; compute jobs then run offline for this part)
python - <<'EOF'
import os
from pathlib import Path
from boltz.main import download_boltz2
download_boltz2(Path(os.environ["BOLTZ_CACHE"]))
print("weights in", os.environ["BOLTZ_CACHE"])
EOF

# 5. Folder layout the scripts expect (same as the Modal containers)
bash "${REPO}/delta/prepare_inputs.sh"
echo "setup done"
