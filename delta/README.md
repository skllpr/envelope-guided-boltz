# Running on NCSA Delta (ACCESS)

The same pipeline as `modal_pilot.py` (plain, envelope-guided and confidence-aware Boltz-2 per map, then scoring
and analysis), run as Slurm jobs: one A40 per map, out-of-memory maps rerun on an H200, analysis on a CPU node.

## 0. On the Mac: pick the next 87 maps (needs EMDB + RCSB, no allocation)
```bash
pip install gemmi
python delta/select_targets.py --n 87 --dry-run                    # see what would be picked
python delta/select_targets.py --n 87 --local-maps /Volumes/KINGSTON/EMDB_negative_stain_maps
git add data envguide/inputs envguide/targets.csv delta/targets.txt delta/selection_log.csv
git commit -m "Add Delta benchmark targets" && git push
```
Rules: EMDB negative stain with a fitted PDB model; protein only; 100-2,500 residues, <= 20 chains in assembly 1;
one map per model and per set of chain sequences; the 13 pilot maps are kept. If fewer than 87 pass, the log
(`delta/selection_log.csv`) says why each map was dropped. Maps are copied from the Kingston drive when found
there, otherwise downloaded.

## 1. On Delta, once: install (login node, no allocation)
```bash
ssh <user>@login.delta.ncsa.illinois.edu
git clone https://github.com/skllpr/envelope-guided-boltz.git && cd envelope-guided-boltz
accounts                      # shows <project>-delta-gpu / <project>-delta-cpu
nano delta/config.sh          # set DELTA_PROJECT (and ENVGUIDE_DATA if /projects/<project>/<user> is not right)
bash delta/setup_env.sh       # Miniforge + Boltz-2 2.2.1 + US-align + weights (~15-30 min)
```

## 2. MSAs (login node, no allocation)
```bash
source delta/config.sh && activate_env
python delta/fetch_msas.py    # ColabFold server, one map at a time; rerun until it reports all maps ready
```
Run it inside `tmux` so it survives a dropped connection. Maps that still lack an MSA fetch it inside the GPU job.

## 3. Check the setup on one map (~0.1 SU)
```bash
TEST=1 bash delta/submit.sh   # first map, 1 sample, 20 steps, separate run folder
squeue -u $USER; tail -f logs/envguide_*.out
```

## 4. Full run
```bash
bash delta/submit.sh          # every map in delta/targets.txt on gpuA40x4 (25 at a time) + analysis job after
bash delta/rerun_oom.sh       # when the array is done: out-of-memory maps -> gpuH200x8, then re-analysis
jobcharge -a <project>-delta-gpu -d 7    # SU actually charged; divide by maps for the cost per map
```
Rerunning `submit.sh` or `rerun_oom.sh` is safe: finished modes of each map are skipped.

## Where things go
- `$ENVGUIDE_DATA/boltz_cache/envguide_runs/<mode>/<EMD>/`: Boltz outputs and logs (as on Modal)
- `$ENVGUIDE_DATA/boltz_cache/envguide_runs/status/<EMD>.json`: per-map status, GPU and run time
- `envguide/scores.csv`, `envguide/figures/`, `analysis/`: scoring, figures, showcase data
- `logs/`: Slurm output

## Cost (estimate; replace with the `jobcharge` numbers after the first batch)
gpuA40x4 is charged 0.5 SU per GPU-hour, gpuH200x8 3 SU. At ~1.5 A40-hours per map (3 modes x 5 samples x
100 steps), 100 maps is ~75 SU, plus ~45 SU if ~15 maps need an H200. Jobs ask for 16 cores / 60 GB on an A40
(12 / 200 GB on an H200), within one GPU's share of the node, so each is charged as one GPU.

## Files
| file | runs on | what |
|---|---|---|
| `select_targets.py` | Mac | picks maps, downloads maps + models, writes inputs and `targets.txt` |
| `config.sh` | - | project, paths, run settings (samples, steps, scale, parallel tasks) |
| `setup_env.sh`, `prepare_inputs.sh` | login node | install; unzip maps and link models (rerun after adding maps) |
| `fetch_msas.py` | login node | MSAs from the ColabFold server into `envguide/msa/` |
| `submit.sh`, `run_map.sbatch`, `run_map.py` | GPU | one map per array task (port of `modal_pilot._run_map`) |
| `rerun_oom.sh` | GPU | H200 reruns (and `FAILED=1` A40 retries) |
| `analyze.sbatch` | CPU | `score.py`, `figures.py`, `pilot_analysis.py` |
