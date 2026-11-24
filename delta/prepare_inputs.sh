#!/bin/bash
# Download missing maps/models, unzip the maps and link the deposited models where run_boltz.py / score.py / pilot_analysis.py look for them.
# Rerun after adding targets (setup_env.sh calls it too). No allocation used.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
mkdir -p maps_unzipped data/maps data/deposited
if [ ! -f delta/targets.txt ]; then  # default: every map in targets.csv
    tail -n +2 envguide/targets.csv | cut -d, -f1 > delta/targets.txt
fi
# Maps and models not in git (maps are large) are downloaded here; Delta login nodes can reach EMDB and RCSB.
while IFS=, read -r emd pdb _; do
    [ "${emd}" = "emdb" ] && continue
    grep -qx "${emd}" delta/targets.txt || continue
    if [ ! -s "data/maps/${emd}.map.gz" ]; then
        num="${emd#EMD-}"
        echo "downloading ${emd} map"
        curl -fsSL -o "data/maps/${emd}.map.gz.part" \
            "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/${emd}/map/emd_${num}.map.gz" \
            && mv "data/maps/${emd}.map.gz.part" "data/maps/${emd}.map.gz" || echo "  failed: ${emd} map" >&2
    fi
    if [ ! -s "data/deposited/${pdb}-assembly1.cif" ]; then
        echo "downloading ${pdb} assembly 1"
        curl -fsSL -o "data/deposited/${pdb}-assembly1.cif" "https://files.rcsb.org/download/${pdb}-assembly1.cif" \
            || echo "  failed: ${pdb} model" >&2
    fi
done < envguide/targets.csv
n=0
for gz in data/maps/*.map.gz; do
    out="maps_unzipped/$(basename "${gz}" .gz)"
    if [ ! -s "${out}" ]; then
        gunzip -c "${gz}" > "${out}.part" && mv "${out}.part" "${out}"
        n=$((n + 1))
    fi
done
[ -e deposited ] || ln -s data/deposited deposited
missing=0
while read -r emd; do
    [ -z "${emd}" ] && continue
    for f in "maps_unzipped/${emd}.map" "envguide/inputs/${emd}.yaml"; do
        [ -s "${f}" ] || { echo "missing ${f}" >&2; missing=1; }
    done
done < delta/targets.txt
echo "unzipped ${n} new maps; $(grep -c . delta/targets.txt) targets in delta/targets.txt"
exit ${missing}
