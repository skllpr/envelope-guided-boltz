# Envelope-guided Boltz-2

Steering Boltz-2 structure prediction toward a ~20 Å negative-stain (NS) EM envelope, with **confidence-aware
guidance**: the envelope is applied at full strength when plain Boltz-2 is unsure how the chains assemble, and
switched off when it is already confident.

## How it works
- `envguide/envelope.py`: turns an NS map into a differentiable envelope (contour enclosing the expected protein
  volume) and scores a model's fit to it (correlation of blurred residue density), with a global rigid fit.
- `envguide/run_boltz.py`: runs Boltz-2 (2.2.1). In guided mode, at each diffusion step the predicted structure is
  registered to the envelope and nudged (per-chain rigid moves + a small per-residue term) to raise the fit.
  Boltz itself is not retrained.
- `adaptive.py`: confidence-aware guidance. Plain Boltz-2 runs first; its top model's ipTM sets the strength:
  full below 0.5, off above 0.8, linear in between (the usual AlphaFold-Multimer ipTM bands).
- `envguide/score.py`: TM-score vs the deposited model (US-align, chain mapping) and envelope fit for every model.
- `pilot_analysis.py`: superposes each mode's top-confidence model on the deposited model and places it in the
  envelope for the figures.
- `showcase.py`: marimo notebook with the results and figures.

## Data (13 maps)
`envguide/targets.csv` lists each EMDB negative-stain map with its deposited fitted model (PDB). Inputs:
`data/maps/` (EMDB maps, gzipped), `data/deposited/` (RCSB assembly 1 of each PDB entry, the answer key) and
`envguide/inputs/` (Boltz input YAMLs with the protein sequences). EMDB and PDB data are public (CC0).

## Reproduce
Runs on [Modal](https://modal.com) (one GPU container per map, all maps in parallel; results on a Modal Volume).

```bash
pip install -r requirements.txt
modal setup
modal run modal_pilot.py::check
modal run --detach modal_pilot.py::main
modal run modal_pilot.py::fetch
marimo edit showcase.py
```

`main` runs plain, envelope-guided and confidence-aware Boltz-2 (5 samples, 100 steps each) on all 13 maps, then
scores and analyses everything. `fetch` downloads the results into `modal_results/`, which `showcase.py` reads.
Each finished run is checkpointed on the volume, so rerunning `main` resumes where it stopped.

## Results (13 maps, top-confidence model, TM-score vs deposited)
The executed notebook with all tables and figures is in `results/`: `showcase.html` (open in a browser; includes
the interactive 3D view) and `showcase.ipynb` (renders on GitHub).

`figures/` holds the final figures. `figure1_abc.png` is Figure 1: (a) every map, plain vs confidence-aware
guidance (`fig1a_tm_all_maps.png`); (b) EMD-1149 and (c) EMD-2712, plain vs confidence-aware top model over the
deposited structure (`fig1b_emd1149.png`, `fig1c_emd2712_confidence_aware.png`). The other files are every figure
`showcase.py` exports (overview, plus structures / envelope / samples for each map).

**Official figures** for the write-up are in `figures/official/`: `figure1.png` (the a/b/c figure above),
`figure2.png` (method), `figure3.png` (confidence-aware guidance and ipTM per map) and `figure4.png` (EMD-1149
placed in the negative-stain envelope). Figures 2-4 carry their captions; `make_figures.py` rebuilds them
(edit the caption text in `FIGURES` and rerun).

Mean TM-score 0.485 plain → 0.523 confidence-aware (+0.04). Largest gains: EMD-1149 (12-chain cage)
0.43 → 0.75 and EMD-2712 0.53 → 0.81. Guidance is switched off on the maps plain Boltz-2 is confident on
(e.g. EMD-6339, which always-on guidance had pushed from 0.98 to 0.57).

Caveats: a pilot of 13 maps × 5 samples; most deposited models predate Boltz-2's training cutoff.
