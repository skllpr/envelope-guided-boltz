"""Score every Boltz model under RUNS (see run_boltz.py) against the deposited model and the NS envelope.

  python envguide/score.py            # scores new models only, appends to envguide/scores.csv

Per model: complex TM-score normalised by the deposited model (US-align -mm 1, chain mapping included),
RMSD over aligned residues, envelope CC after a global rigid fit (same fit for every mode, so it is
independent of the guidance's own registration), and Boltz confidence / ipTM / pTM.
"""
import csv, glob, json, os, re, subprocess, sys
import gemmi, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from envelope import Envelope, global_fit  # noqa: E402
from run_boltz import RUNS  # noqa: E402

USALIGN = os.path.join(ROOT, "external", "USalign", "USalign")
OUT = os.path.join(HERE, "scores.csv")
FIELDS = ["run", "emdb", "pdb", "model", "rank", "confidence", "iptm", "ptm", "tm_ref", "rmsd", "aligned",
          "ref_len", "env_cc", "year", "homo_oligomer", "tokens"]


def usalign(model, ref):
    r = subprocess.run([USALIGN, model, ref, "-mm", "1", "-ter", "0", "-outfmt", "2"],
                       capture_output=True, text=True, timeout=1800)
    rows = [l.split("\t") for l in r.stdout.splitlines() if l and not l.startswith("#")]
    if not rows:
        raise RuntimeError(r.stderr[-500:])
    f = rows[0]
    return dict(tm_ref=float(f[3]), rmsd=float(f[4]), ref_len=int(f[9]), aligned=int(f[10]))


def ca_coords(path):
    s = gemmi.read_structure(path)
    return torch.tensor([[a.pos.x, a.pos.y, a.pos.z] for ch in s[0] for r in ch for a in r if a.name == "CA"],
                        dtype=torch.float32)[None]


def main():
    targets = {r["emdb"]: r for r in csv.DictReader(open(os.path.join(HERE, "targets.csv")))}
    done = set()
    if os.path.exists(OUT):
        done = {(r["run"], r["model"]) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    envs = {}
    for cif in sorted(glob.glob(os.path.join(RUNS, "*", "EMD-*", "boltz_results_*", "predictions", "*",
                                             "*_model_*.cif"))):
        run = cif.split(os.sep)[-6]
        emd = cif.split(os.sep)[-5]
        name = os.path.basename(cif)
        if (run, name) in done or "smoke" in run or emd not in targets:
            continue
        t = targets[emd]
        rank = int(re.search(r"_model_(\d+)\.cif", name).group(1))
        conf = json.load(open(cif.replace(name, f"confidence_{name[:-4]}.json")))
        if emd not in envs:
            envs[emd] = Envelope(os.path.join(ROOT, "maps_unzipped", f"{emd}.map"), float(t["mass_da"]),
                                 float(t["resolution"]))
        _, _, cc = global_fit(envs[emd], ca_coords(cif))
        ua = usalign(cif, os.path.join(ROOT, "deposited", f"{t['pdb']}-assembly1.cif"))
        row = dict(run=run, emdb=emd, pdb=t["pdb"], model=name, rank=rank,
                   confidence=round(conf["confidence_score"], 4), iptm=round(conf.get("iptm", float("nan")), 4),
                   ptm=round(conf["ptm"], 4), env_cc=round(float(cc[0]), 4), year=t["year"],
                   homo_oligomer=t["homo_oligomer"], tokens=t["tokens"], **ua)
        w.writerow(row); f.flush()
        print(run, emd, name, "TM", row["tm_ref"], "CC", row["env_cc"], "conf", row["confidence"])
    f.close()


if __name__ == "__main__":
    main()
