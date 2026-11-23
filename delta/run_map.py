"""Run one map through the full pipeline on the GPU this job holds (Delta port of modal_pilot._run_map).

  python delta/run_map.py EMD-1149 --samples 5 --steps 100 --scale 1.0

Plain then guided Boltz-2 (same MSA), then the confidence-aware run (adaptive.py). Finished modes are skipped,
so rerunning a map (e.g. on an H200 after running out of memory on an A40) only redoes what is missing.
Status goes to $BOLTZ_CACHE/envguide_runs/status/<EMD>.json; exit code 3 means out of GPU memory.
"""
import argparse, glob, json, os, random, shutil, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RUNS = os.path.join(os.environ["BOLTZ_CACHE"], "envguide_runs")  # same as run_boltz.RUNS
MODES = ["unguided", "guided"]
OOM = "FAILED (out of GPU memory)"


def has_preds(mode, emd):
    return bool(glob.glob(f"{RUNS}/{mode}/{emd}/boltz_results_*/predictions/*/*_model_*.cif"))


def run_boltz(emd, args, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(ROOT, "envguide", "run_boltz.py"), emd, "--accelerator", "gpu",
                        "--seed", "0", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return open(log).read()


def is_oom(text):
    return "out of memory" in text or "OutOfMemoryError" in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emd")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    emd, common = a.emd, ["--samples", str(a.samples), "--steps", str(a.steps)]
    status = dict(emd=emd, host=socket.gethostname(), gpu=os.environ.get("ENVGUIDE_GPU", ""),
                  job=os.environ.get("SLURM_JOB_ID", ""), modes={})

    for mode in MODES:
        if has_preds(mode, emd):
            status["modes"][mode] = "done (checkpoint)"
            continue
        t0, log = time.time(), f"{RUNS}/{mode}/{emd}.log"
        for attempt in range(4):
            shutil.rmtree(f"{RUNS}/{mode}/{emd}", ignore_errors=True)  # partial output of an interrupted run
            print(f"=== {emd} {mode} started {time.strftime('%H:%M:%S')}" + (f" (retry {attempt})" if attempt else ""),
                  flush=True)
            text = run_boltz(emd, ["--mode", mode, "--scale", str(a.scale), *common], log)
            # MSA server errors (only if fetch_msas.py was skipped): Boltz skips the input; wait and retry
            if has_preds(mode, emd) or "could not be opened successfully" not in text:
                break
            time.sleep(90 * (attempt + 1) + random.randint(0, 60))
        if has_preds(mode, emd):
            status["modes"][mode] = f"done in {time.time() - t0:.0f}s"
        else:  # Boltz can exit 0 without predictions, so check the files
            status["modes"][mode] = OOM if is_oom(text) else "FAILED"
            print(f"!!! last lines of {log}:\n" + "".join(text.splitlines(True)[-25:]), flush=True)
        print(f"=== {emd} {mode}: {status['modes'][mode]}", flush=True)

    import adaptive
    if all(has_preds(m, emd) for m in MODES):
        if has_preds(adaptive.NAME, emd):
            status["modes"]["adaptive"] = "done (checkpoint)"
        else:
            t0 = time.time()
            p = adaptive.plan(RUNS, emd, a.scale)
            print(f"=== {emd} adaptive: plain ipTM {p['plain_iptm']:.2f} -> strength {p['strength']:.2f} "
                  f"({'reuse ' + p['source'] if p['source'] != 'run' else 'new run'})", flush=True)
            if p["source"] != "run":
                adaptive.reuse(RUNS, emd, p)
                text = ""
            else:
                shutil.rmtree(f"{RUNS}/{adaptive.NAME}/{emd}", ignore_errors=True)
                text = run_boltz(emd, ["--mode", "guided", "--tag", "adaptive", "--scale", str(p["scale"]), *common],
                                 f"{RUNS}/{adaptive.NAME}/{emd}.log")
            if has_preds(adaptive.NAME, emd):
                adaptive.record(RUNS, emd, p)
                status["modes"]["adaptive"] = f"strength {p['strength']:.2f}, done in {time.time() - t0:.0f}s"
            else:
                status["modes"]["adaptive"] = OOM if is_oom(text) else "FAILED"

    os.makedirs(f"{RUNS}/status", exist_ok=True)
    json.dump(status, open(f"{RUNS}/status/{emd}.json", "w"), indent=1)
    print(json.dumps(status), flush=True)
    vals = list(status["modes"].values())
    if OOM in vals:
        return 3
    return 0 if len(vals) == 3 and not any(v.startswith("FAILED") for v in vals) else 1


if __name__ == "__main__":
    sys.exit(main())
