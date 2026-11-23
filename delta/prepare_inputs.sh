#!/bin/bash
# Unzip the maps and link the deposited models where run_boltz.py / score.py / pilot_analysis.py look for them.
# Rerun after adding targets (setup_env.sh calls it too). No allocation used.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
mkdir -p maps_unzipped
n=0
for gz in data/maps/*.map.gz; do
    out="maps_unzipped/$(basename "${gz}" .gz)"
    if [ ! -s "${out}" ]; then
        gunzip -c "${gz}" > "${out}.part" && mv "${out}.part" "${out}"
        n=$((n + 1))
    fi
done
[ -e deposited ] || ln -s data/deposited deposited
if [ ! -f delta/targets.txt ]; then  # default: every map in targets.csv
    tail -n +2 envguide/targets.csv | cut -d, -f1 > delta/targets.txt
fi
missing=0
while read -r emd; do
    [ -z "${emd}" ] && continue
    for f in "maps_unzipped/${emd}.map" "envguide/inputs/${emd}.yaml"; do
        [ -s "${f}" ] || { echo "missing ${f}" >&2; missing=1; }
    done
done < delta/targets.txt
echo "unzipped ${n} new maps; $(grep -c . delta/targets.txt) targets in delta/targets.txt"
exit ${missing}
