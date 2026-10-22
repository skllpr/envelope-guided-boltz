"""Envelope-guided Boltz-2 pilot on Modal, all maps at once.

Pipeline (orchestrated on Modal, so `--detach` carries it through to the end):
  1. prepare   download the Boltz weights into the volume once (parallel downloads would collide)
  2. run_map   one GPU container per map, all maps concurrently; inside each, plain then guided, so both modes
               reuse the same MSA, then the confidence-aware run (adaptive.py). The volume is committed after each
               run. Maps that run out of memory on the L40S are rerun on an A100-80GB (finished modes skipped).
  3. analyze   one CPU container: score every model (score.py), figures (figures.py), showcase data
               (pilot_analysis.py). Scoring happens here, not per run, because concurrent containers appending
               to one scores.csv would lose rows.

Everything persists on the Modal Volume "envguide-pilot":
  boltz_cache/                  weights + envguide_runs/<mode>/<EMD>/ (Boltz outputs, logs, guide_log.json)
  ns/msa/  ns/scores.csv  ns/figures/  ns/analysis/
A dead or preempted container loses at most the run in progress; rerunning skips finished runs.

  pip install modal && modal setup                      # once: log in through the browser
  modal run modal_pilot.py::check                       # GPU + install check (~1-2 min after the image build)
  modal run --detach modal_pilot.py::main               # full pilot + analysis; keeps running if this terminal closes
  modal run modal_pilot.py::analyze_only                # redo scoring + analysis on whatever runs are finished
  modal run modal_pilot.py::fetch                       # download results + analysis into modal_results/
  marimo edit showcase.py                               # look at them

Options for the full run: --targets EMD-6339,EMD-1777 --samples 5 --steps 100 --scale 1.0
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = "L40S"            # 48 GB; every map starts here
GPU_BIG = "A100-80GB"   # maps that run out of memory on GPU are rerun here automatically
TARGETS = ["EMD-1495", "EMD-6339", "EMD-1777", "EMD-25708", "EMD-40983", "EMD-1149", "EMD-2281", "EMD-5755", "EMD-7013", "EMD-15486", "EMD-8235", "EMD-2355", "EMD-2712"]
MODES = ["unguided", "guided"]
VOL = "/vol"
SAVED = f"{VOL}/ns"     # results that persist (msa, scores, figures, analysis)
WORK = "/root/ns"       # code + inputs, copied per container from the repo (local disk, no write races)

app = modal.App("envguide-pilot")
vol = modal.Volume.from_name("envguide-pilot", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("g++", "git")
    .pip_install("boltz==2.2.1", "gemmi", "mrcfile", "matplotlib", "pandas")
    .run_commands("git clone -q --depth 1 https://github.com/pylelab/USalign.git /opt/USalign"
                  " && cd /opt/USalign && g++ -O3 -ffast-math -o USalign USalign.cpp -lm")
    .env({"BOLTZ_CACHE": f"{VOL}/boltz_cache", "MPLBACKEND": "Agg"})
    .add_local_dir(os.path.join(HERE, "envguide"), "/bundle/ns/envguide")
    .add_local_dir(os.path.join(HERE, "data", "deposited"), "/bundle/ns/deposited")
    .add_local_dir(os.path.join(HERE, "data", "maps"), "/bundle/maps")
    .add_local_file(os.path.join(HERE, "pilot_analysis.py"), "/bundle/pilot_analysis.py")
    .add_local_file(os.path.join(HERE, "adaptive.py"), "/root/adaptive.py")
)
RUNS = f"{VOL}/boltz_cache/envguide_runs"


def setup():
    """Unpack code + inputs locally and point the result paths the scripts use at the volume."""
    import glob, gzip, shutil
    os.makedirs(RUNS, exist_ok=True)
    shutil.copytree("/bundle/ns", WORK, dirs_exist_ok=True, symlinks=True)
    os.makedirs(f"{WORK}/maps_unzipped", exist_ok=True)
    for gz in glob.glob("/bundle/maps/*.map.gz"):
        out = f"{WORK}/maps_unzipped/{os.path.basename(gz)[:-3]}"
        if not os.path.exists(out):
            with gzip.open(gz) as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
    os.makedirs(f"{WORK}/external/USalign", exist_ok=True)
    shutil.copy2("/opt/USalign/USalign", f"{WORK}/external/USalign/USalign")
    for local, saved, is_dir in [(f"{WORK}/envguide/msa", f"{SAVED}/msa", True),
                                 (f"{WORK}/envguide/figures", f"{SAVED}/figures", True),
                                 (f"{WORK}/analysis", f"{SAVED}/analysis", True),
                                 (f"{WORK}/envguide/scores.csv", f"{SAVED}/scores.csv", False)]:
        if is_dir:
            os.makedirs(saved, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(saved), exist_ok=True)
        if not os.path.islink(local):
            os.symlink(saved, local)  # score.py creates scores.csv through the link on first write


def has_preds(mode, emd):
    import glob
    return bool(glob.glob(f"{RUNS}/{mode}/{emd}/boltz_results_*/predictions/*/*_model_*.cif"))


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=600)
def check():
    import subprocess
    setup()
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
    subprocess.run(["python", "-c", "import boltz, torch; print('torch', torch.__version__, 'cuda',"
                    " torch.cuda.is_available())"])
    print("saved runs:", sorted(f"{m}/{e}" for m in MODES + ["guided_adaptive"] for e in TARGETS if has_preds(m, e)))


@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def prepare():
    from pathlib import Path
    from boltz.main import download_boltz2
    setup()
    download_boltz2(Path(os.environ["BOLTZ_CACHE"]))
    vol.commit()


OOM = "FAILED (out of GPU memory)"


def _run_map(emd, samples, steps, scale):
    """Plain then guided Boltz-2 for one map (same MSA for both); checkpoint after each."""
    import random, shutil, subprocess, sys, time
    setup()
    status = {}
    for mode in MODES:
        if has_preds(mode, emd):
            status[f"{emd} {mode}"] = "done (checkpoint)"
            continue
        log = f"{RUNS}/{mode}/{emd}.log"
        os.makedirs(os.path.dirname(log), exist_ok=True)
        t0 = time.time()
        for attempt in range(4):
            shutil.rmtree(f"{RUNS}/{mode}/{emd}", ignore_errors=True)  # partial output from an interrupted run
            print(f"=== {emd} {mode} started {time.strftime('%H:%M:%S')}" + (f" (retry {attempt})" if attempt else ""),
                  flush=True)
            with open(log, "w") as f:
                subprocess.run([sys.executable, f"{WORK}/envguide/run_boltz.py", emd, "--mode", mode,
                                "--samples", str(samples), "--seed", "0", "--accelerator", "gpu",
                                "--steps", str(steps), "--scale", str(scale)], cwd=WORK, stdout=f,
                               stderr=subprocess.STDOUT)
            # The public ColabFold MSA server rate-limits concurrent requests and then returns an error page
            # instead of the result archive; Boltz skips the input ("Failed to process ... not a gzip file").
            # Retry after a growing, jittered wait so the containers do not hit the server in lockstep.
            if has_preds(mode, emd) or "could not be opened successfully" not in open(log).read():
                break
            wait = 90 * (attempt + 1) + random.randint(0, 60)
            print(f"=== {emd} {mode}: MSA server error, retrying in {wait}s", flush=True)
            time.sleep(wait)
        ok = has_preds(mode, emd)
        if ok:
            status[f"{emd} {mode}"] = f"done in {time.time() - t0:.0f}s"
        else:  # Boltz can exit 0 without predictions, so check the files
            text = open(log).read()
            status[f"{emd} {mode}"] = OOM if ("out of memory" in text or "OutOfMemoryError" in text) else "FAILED"
            print(f"!!! last lines of {log}:\n" + "".join(text.splitlines(True)[-25:]), flush=True)
        print(f"=== {emd} {mode}: {status[f'{emd} {mode}']}", flush=True)
        vol.commit()  # checkpoint

    # Confidence-aware guidance (adaptive.py): strength set by the plain run's ipTM; reuses a run when it can.
    # Skipped while plain or guided is missing (e.g. out of memory here; the big-GPU rerun then does it).
    import adaptive
    if all(has_preds(m, emd) for m in MODES) and not has_preds(adaptive.NAME, emd):
        p = adaptive.plan(RUNS, emd, scale)
        print(f"=== {emd} adaptive: plain ipTM {p['plain_iptm']:.2f} -> strength {p['strength']:.2f} "
              f"({'reuse ' + p['source'] if p['source'] != 'run' else 'new run'})", flush=True)
        if p["source"] != "run":
            adaptive.reuse(RUNS, emd, p)
        else:
            shutil.rmtree(f"{RUNS}/{adaptive.NAME}/{emd}", ignore_errors=True)
            log = f"{RUNS}/{adaptive.NAME}/{emd}.log"
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "w") as f:
                subprocess.run([sys.executable, f"{WORK}/envguide/run_boltz.py", emd, "--mode", "guided",
                                "--tag", "adaptive", "--samples", str(samples), "--seed", "0", "--accelerator", "gpu",
                                "--steps", str(steps), "--scale", str(p["scale"])], cwd=WORK, stdout=f,
                               stderr=subprocess.STDOUT)
        ok = has_preds(adaptive.NAME, emd)
        status[f"{emd} adaptive"] = f"strength {p['strength']:.2f}, {'done' if ok else 'FAILED'}"
        if ok:
            adaptive.record(RUNS, emd, p)
        vol.commit()
    return status


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=8 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_map(emd, samples, steps, scale):
    return _run_map(emd, samples, steps, scale)


@app.function(image=image, gpu=GPU_BIG, volumes={VOL: vol}, timeout=8 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_map_big(emd, samples, steps, scale):
    return _run_map(emd, samples, steps, scale)


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=12 * 3600)
def analyze():
    """Score all models, redraw figures, build the showcase data (pilot_analysis.py) -> ns/analysis/."""
    import subprocess, sys
    setup()
    subprocess.run([sys.executable, f"{WORK}/envguide/score.py"], cwd=WORK, check=True)
    vol.commit()
    subprocess.run([sys.executable, f"{WORK}/envguide/figures.py"], cwd=WORK)
    subprocess.run([sys.executable, "/bundle/pilot_analysis.py", "--work", WORK, "--jobs", "8"], cwd=WORK, check=True)
    vol.commit()


@app.function(image=image, volumes={VOL: vol}, timeout=24 * 3600)
def pilot_all(targets, samples, steps, scale):
    """Runs on Modal so a detached run finishes the whole pipeline without this terminal."""
    prepare.remote()
    results = list(run_map.starmap([(e, samples, steps, scale) for e in targets], return_exceptions=True))
    for emd, r in zip(targets, results):
        print(emd, r if isinstance(r, dict) else f"ERROR: {r!r}", flush=True)
    big = [e for e, r in zip(targets, results) if isinstance(r, dict) and OOM in r.values()]
    if big:  # finished modes are skipped, so only what ran out of memory is redone
        print(f"rerunning on {GPU_BIG} (out of memory on {GPU}): {big}", flush=True)
        for emd, r in zip(big, run_map_big.starmap([(e, samples, steps, scale) for e in big], return_exceptions=True)):
            print(emd, r if isinstance(r, dict) else f"ERROR: {r!r}", flush=True)
    analyze.remote()


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def results_zip():
    import shutil, tempfile
    keep = tempfile.mkdtemp()
    for name in ["scores.csv", "figures", "msa", "analysis"]:
        src = os.path.join(SAVED, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(keep, name))
        elif os.path.exists(src):
            shutil.copy(src, keep)
    if os.path.isdir(RUNS):
        shutil.copytree(RUNS, os.path.join(keep, "runs"), ignore=shutil.ignore_patterns("*.npz", "processed", "msa_*"))
    return open(shutil.make_archive(os.path.join(tempfile.mkdtemp(), "pilot_results"), "zip", keep), "rb").read()


def download():
    import io, zipfile
    out = os.path.join(HERE, "modal_results")
    zipfile.ZipFile(io.BytesIO(results_zip.remote())).extractall(out)
    print("results in", out)


@app.local_entrypoint()
def main(targets: str = ",".join(TARGETS), samples: int = 5, steps: int = 100, scale: float = 1.0):
    pilot_all.remote(targets.split(","), samples, steps, scale)
    download()


@app.local_entrypoint()
def analyze_only():
    analyze.remote()
    download()


@app.local_entrypoint()
def fetch():
    download()
