"""Showcase data for the envelope-guided pilot (runs on the Modal volume after the pilot; read by showcase.py).

  python pilot_analysis.py --work /vol/ns [--base unguided --guided guided]

Per map that has both modes, writes <work>/analysis/<EMD>/:
  deposited.cif, <mode>_top.cif   top-confidence model of each mode superposed on the deposited model (US-align -mm 1)
  envelope.npz                    NS envelope grid + CA coords of deposited / each mode's top model in the map frame
  guide_log.json                  envelope CC per diffusion step (guided)
and <work>/analysis/summary.json with per-map metrics, including how far apart the two modes' models are
(TM-score between them) compared with sample-to-sample spread within each mode.
"""
import argparse, csv, glob, itertools, json, os, re, shutil, subprocess, sys

import numpy as np


def usalign(us, model, ref, sup=None):
    """TM-score normalised by ref, RMSD, aligned length. With sup, writes model superposed on ref to sup.cif."""
    cmd = [us, model, ref, "-mm", "1", "-ter", "0"] + (["-o", sup] if sup else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600).stdout
    tms = [float(m) for m in re.findall(r"^TM-score= ([\d.]+)", out, re.M)]
    m = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([\d.]+)", out)
    if len(tms) < 2 or not m:
        raise RuntimeError(f"US-align failed on {model} vs {ref}:\n{out[-500:]}")
    return dict(tm=tms[1], rmsd=float(m.group(2)), aligned=int(m.group(1)))


def ca(path):
    import gemmi
    s = gemmi.read_structure(path)
    xyz, chain = [], []
    for ch in s[0]:
        for r in ch:
            for a in r:
                if a.name == "CA":
                    xyz.append([a.pos.x, a.pos.y, a.pos.z]); chain.append(ch.name)
    return np.array(xyz, dtype=np.float32), np.array(chain)


def model_path(runs, run, emd, name):
    hits = glob.glob(os.path.join(runs, run, emd, "boltz_results_*", "predictions", "*", name))
    return hits[0] if hits else None


def mean_pairwise_tm(us, paths):
    tms = [usalign(us, a, b)["tm"] for a, b in itertools.combinations(paths, 2)]
    return (float(np.mean(tms)), float(np.min(tms))) if tms else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--runs", default=None, help="default: $BOLTZ_CACHE/envguide_runs")
    ap.add_argument("--base", default="unguided")
    ap.add_argument("--guided", default="guided")
    ap.add_argument("--adaptive", default="guided_adaptive", help="confidence-aware run (adaptive.py), if present")
    ap.add_argument("--jobs", type=int, default=1, help="maps analysed in parallel")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.work, "envguide"))

    runs = a.runs or os.environ.get("ENVGUIDE_RUNS") or os.path.join(os.environ["BOLTZ_CACHE"], "envguide_runs")
    us = os.path.join(a.work, "external", "USalign", "USalign")
    out_root = os.path.join(a.work, "analysis")
    os.makedirs(out_root, exist_ok=True)
    targets = {r["emdb"]: r for r in csv.DictReader(open(os.path.join(a.work, "envguide", "targets.csv")))}
    scores = list(csv.DictReader(open(os.path.join(a.work, "envguide", "scores.csv"))))
    modes = [a.base, a.guided]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = {}
    for emd in sorted({r["emdb"] for r in scores}):
        by_mode = {m: [r for r in scores if r["emdb"] == emd and r["run"] == m] for m in modes + [a.adaptive]}
        if all(by_mode[m] for m in modes):
            jobs[emd] = (a, emd, by_mode, targets[emd], runs, us, out_root)
    summary, failed = [], []
    with ProcessPoolExecutor(max(1, a.jobs)) as ex:
        futs = {ex.submit(analyse_map, *args): emd for emd, args in jobs.items()}
        for f in as_completed(futs):
            emd = futs[f]
            try:  # one bad map (e.g. a US-align failure) must not stop the rest
                summary.append(f.result())
                print(emd, flush=True)
            except Exception as e:
                print(f"!!! {emd} skipped: {e!r}", flush=True)
                failed.append(emd)
                continue
            summary.sort(key=lambda r: r["emdb"])
            json.dump(dict(base=a.base, guided=a.guided, adaptive=a.adaptive if any(a.adaptive in r for r in summary)
                           else None, maps=summary, failed=failed), open(os.path.join(out_root, "summary.json"), "w"),
                      indent=1)  # rewritten per map, so partial results survive an interruption
    print(f"analysed {len(summary)} maps -> {out_root}" + (f"; skipped {failed}" if failed else ""))


def analyse_map(a, emd, by_mode, t, runs, us, out_root):
    import torch
    if a.jobs > 1:
        torch.set_num_threads(1)  # parallelism comes from the process pool
    from envelope import Envelope, global_fit
    modes = [a.base, a.guided]
    map_modes = modes + ([a.adaptive] if by_mode[a.adaptive] else [])
    out = os.path.join(out_root, emd)
    os.makedirs(out, exist_ok=True)
    ref = os.path.join(a.work, "deposited", f"{t['pdb']}-assembly1.cif")
    shutil.copy(ref, os.path.join(out, "deposited.cif"))
    print(emd, flush=True)

    row = dict(emdb=emd, pdb=t["pdb"], tokens=int(t["tokens"]), chains=int(t["chains"]), year=int(t["year"]),
               resolution=float(t["resolution"]), homo_oligomer=int(t["homo_oligomer"]))
    tops = {}
    for m in map_modes:
        rs = by_mode[m]
        f = lambda k: [float(r[k]) for r in rs]
        top = max(rs, key=lambda r: float(r["confidence"]))
        top_cc = max(rs, key=lambda r: float(r["env_cc"]))
        tops[m] = model_path(runs, m, emd, top["model"])
        usalign(us, tops[m], ref, sup=os.path.join(out, f"{m}_top"))  # writes <m>_top.cif
        # sample spread; the adaptive run is usually a copy of plain or guided, so take that mode's spread
        src = ""
        if m == a.adaptive:
            info = os.path.join(runs, a.adaptive, emd, "adaptive.json")
            src = json.load(open(info)).get("source") if os.path.exists(info) else "run"
        if src in row:
            within_mean, within_min = row[src]["within_tm_mean"], row[src]["within_tm_min"]
        else:
            within_mean, within_min = mean_pairwise_tm(us, [model_path(runs, m, emd, r["model"]) for r in rs])
        row[m] = dict(n=len(rs), top_model=top["model"], tm_top_conf=float(top["tm_ref"]),
                      rmsd_top_conf=float(top["rmsd"]), conf_top=float(top["confidence"]),
                      iptm_top=float(top["iptm"]), cc_top_conf=float(top["env_cc"]),
                      tm_top_cc=float(top_cc["tm_ref"]), tm_best=max(f("tm_ref")), tm_mean=float(np.mean(f("tm_ref"))),
                      cc_mean=float(np.mean(f("env_cc"))), within_tm_mean=within_mean, within_tm_min=within_min)
    # how different are the two modes' answers, relative to their own sample spread?
    row["cross_tm"] = usalign(us, tops[a.guided], tops[a.base])["tm"]
    row["delta_tm_top_conf"] = row[a.guided]["tm_top_conf"] - row[a.base]["tm_top_conf"]
    row["delta_tm_best"] = row[a.guided]["tm_best"] - row[a.base]["tm_best"]
    if a.adaptive in map_modes:
        row["delta_tm_adaptive"] = row[a.adaptive]["tm_top_conf"] - row[a.base]["tm_top_conf"]
        info = os.path.join(runs, a.adaptive, emd, "adaptive.json")
        if os.path.exists(info):
            row["adaptive_plan"] = json.load(open(info))

    # envelope + models in the map frame: fit the deposited model to the envelope once, then apply the same
    # transform to the superposed tops (they share the deposited frame), so all three sit in the same envelope
    env = Envelope(os.path.join(a.work, "maps_unzipped", f"{emd}.map"), float(t["mass_da"]), float(t["resolution"]))
    X, chains = ca(ref)
    R, tr, cc = global_fit(env, torch.from_numpy(X)[None])
    R, tr = R[0].numpy(), tr[0].numpy()
    row["deposited_env_cc"] = float(cc[0])
    arrays = dict(env=env.env.numpy().astype(np.float32), ax_x=env.axes[0].numpy(), ax_y=env.axes[1].numpy(),
                  ax_z=env.axes[2].numpy(), deposited=X @ R + tr, deposited_chain=chains)
    for m in map_modes:
        Xm, cm = ca(os.path.join(out, f"{m}_top.cif"))
        arrays[m], arrays[f"{m}_chain"] = Xm @ R + tr, cm
    np.savez_compressed(os.path.join(out, "envelope.npz"), **arrays)
    for p in glob.glob(os.path.join(runs, a.guided, emd, "guide_log.json")):
        shutil.copy(p, os.path.join(out, "guide_log.json"))
    for p in glob.glob(os.path.join(out, "*.pml")):
        os.remove(p)
    return row


if __name__ == "__main__":
    main()
