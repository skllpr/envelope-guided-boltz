"""Confidence-aware envelope guidance: back off when plain Boltz-2 is already confident.

The plain run's top-confidence ipTM sets the guidance strength:
  ipTM <= IPTM_FULL  -> full scale (Boltz is unsure how the chains assemble; the envelope can help)
  ipTM >= IPTM_OFF   -> no guidance (Boltz is confident; in the pilot, guidance only broke such a map)
  in between         -> linear ramp
Cut-offs follow the usual AlphaFold-Multimer ipTM bands (>0.8 confident, <0.5-0.6 likely wrong), not a fit
to the pilot maps.

Guided ipTM is NOT used as a signal: in the pilot, guidance lowered ipTM on every map, including the one where it
raised TM-score from 0.43 to 0.75, so the confidence head is not a judge of guided structures.

When the ramp gives exactly full or zero strength, the existing guided / plain run is reused (same seed and
settings), so no extra GPU time is spent.
"""
import glob, json, os, shutil

IPTM_FULL, IPTM_OFF = 0.5, 0.8
NAME = "guided_adaptive"  # run name; run_boltz.py --mode guided --tag adaptive writes here


def preds_dir(runs, run, emd):
    hits = glob.glob(os.path.join(runs, run, emd, "boltz_results_*", "predictions", emd))
    return hits[0] if hits and glob.glob(os.path.join(hits[0], "*_model_*.cif")) else None


def top_iptm(runs, run, emd):
    """ipTM of the plain run's top-confidence model."""
    confs = [json.load(open(p)) for p in glob.glob(os.path.join(preds_dir(runs, run, emd), "confidence_*_model_*.json"))]
    top = max(confs, key=lambda c: c["confidence_score"])
    return float(top.get("iptm", top.get("ptm")))


def strength(iptm):
    return min(1.0, max(0.0, (IPTM_OFF - iptm) / (IPTM_OFF - IPTM_FULL)))


def plan(runs, emd, scale):
    """Decide the adaptive run for one map: dict(plain_iptm, strength, scale, source in {unguided, guided, run})."""
    iptm = top_iptm(runs, "unguided", emd)
    f = strength(iptm)
    source = "unguided" if f == 0 else "guided" if f == 1 and preds_dir(runs, "guided", emd) else "run"
    return dict(plain_iptm=iptm, strength=f, scale=scale * f, source=source, iptm_full=IPTM_FULL, iptm_off=IPTM_OFF)


def reuse(runs, emd, p):
    """Copy the plain or guided run as the adaptive result."""
    dst = os.path.join(runs, NAME, emd)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(runs, p["source"], emd), dst,
                    ignore=shutil.ignore_patterns("processed", "lightning_logs", "msa"))


def record(runs, emd, p):
    json.dump(p, open(os.path.join(runs, NAME, emd, "adaptive.json"), "w"), indent=1)
