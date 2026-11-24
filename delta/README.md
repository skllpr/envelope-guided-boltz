# Running on NCSA Delta (ACCESS)

The same pipeline as `modal_pilot.py` (plain, envelope-guided and confidence-aware Boltz-2 per map, then scoring
and analysis), run as Slurm jobs: one GPU per map (A40 up to 2,500 residues, H200 above; maps that still run out
of memory on an A40 are rerun on an H200), analysis on a CPU node.

## 1. On Delta, once: install (login node, no allocation)
```bash
ssh <user>@login.delta.ncsa.illinois.edu
git clone -b delta-slurm https://github.com/skllpr/envelope-guided-boltz.git && cd envelope-guided-boltz
accounts                      # shows <project>-delta-gpu / <project>-delta-cpu
nano delta/config.sh          # set DELTA_PROJECT (and ENVGUIDE_DATA if /projects/<project>/<user> is not right)
bash delta/setup_env.sh       # Miniforge + Boltz-2 2.2.1 + US-align + weights (~15-30 min)
```

## 2. Pick the benchmark maps and fetch MSAs (login node, no allocation)
```bash
source delta/config.sh && activate_env
python delta/select_targets.py --dry-run      # what would be picked (up to 64 new + the 13 pilot maps)
python delta/select_targets.py                # downloads maps + models from EMDB/RCSB, writes inputs
bash delta/prepare_inputs.sh                  # unzip maps, link models
python delta/fetch_msas.py                    # ColabFold server, one map at a time; rerun until all ready
```
Selection rules: EMDB negative stain with a fitted PDB model; protein only; 100-4,000 residues and <= 20 chains in
assembly 1; one map per model and per set of chain sequences. About 77 maps pass in total (only 141 negative-stain
maps have a fitted model at all). `delta/selection_log.csv` says why each map was dropped. Maps over
2,500 residues (`H200_MIN_TOKENS`) run on H200s. Run the MSA step inside `tmux` so it survives a dropped
connection; maps still lacking an MSA fetch it inside the GPU job.

Without network (e.g. a sandbox), the script can work from the dataset folder on disk instead:
`--offline --emdb-csv <dir>/manifest.csv --local-models <dir>/deposited --local-maps <dir>/maps`.

## 3. Check the setup on one map (~0.1 SU)
```bash
TEST=1 bash delta/submit.sh   # first map, 1 sample, 20 steps, separate run folder
squeue -u $USER; tail -f logs/envguide_*.out
```

## 4. Full run
```bash
bash delta/submit.sh          # every map: A40s (25 at a time) / H200s (8 at a time) by size, + analysis job after
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
100 steps), ~64 maps <= 2,500 residues is ~50 SU; ~13 maps of 2,500-4,000 residues on H200s at maybe ~1.5 hours each
is ~60 SU; ~110 SU in all, before setup and reruns. Jobs ask for 16 cores / 60 GB on an A40
(12 / 200 GB on an H200), within one GPU's share of the node, so each is charged as one GPU.

## Files
| file | runs on | what |
|---|---|---|
| `select_targets.py` | login node | picks maps, downloads maps + models, writes inputs and `targets.txt` |
| `config.sh` | - | project, paths, run settings (samples, steps, scale, parallel tasks) |
| `setup_env.sh`, `prepare_inputs.sh` | login node | install; unzip maps and link models (rerun after adding maps) |
| `fetch_msas.py` | login node | MSAs from the ColabFold server into `envguide/msa/` |
| `submit.sh`, `split_targets.py`, `run_map.sbatch`, `run_map.py` | GPU | one map per array task, A40 or H200 by size (port of `modal_pilot._run_map`) |
| `rerun_oom.sh` | GPU | H200 reruns (and `FAILED=1` A40 retries) |
| `analyze.sbatch` | CPU | `score.py`, `figures.py`, `pilot_analysis.py` |
