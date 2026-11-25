"""Precompute MSAs from the ColabFold server on the login node, one map at a time (no allocation used).

  source delta/config.sh && activate_env
  python delta/fetch_msas.py            # every map in delta/targets.txt that has no MSA yet

Writes envguide/msa/<EMD>_<i>.csv, the files run_boltz.py reuses (msa_input), so GPU jobs never wait on the
server and every mode of a map sees the same MSA. Uses Boltz's own MSA call, so the files match what Boltz would
have made. Going one map at a time with a pause keeps us under the public server's rate limit.
"""
import argparse, glob, os, re, shutil, sys, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MSA_DIR = os.path.join(ROOT, "envguide", "msa")
INPUTS = os.path.join(ROOT, "envguide", "inputs")


def sequences(emd):
    """Protein sequences in YAML order; entity i -> <EMD>_<i>.csv (same order Boltz and msa_input use)."""
    text = open(os.path.join(INPUTS, f"{emd}.yaml")).read()
    blocks = text.split("- protein:")[1:]
    return [re.search(r"sequence:\s*(\S+)", b).group(1) for b in blocks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="EMD ids (default: delta/targets.txt)")
    ap.add_argument("--pause", type=float, default=20, help="seconds between maps")
    ap.add_argument("--server", default="https://api.colabfold.com")
    a = ap.parse_args()
    from boltz.main import compute_msa

    emds = a.targets or [l.strip() for l in open(os.path.join(HERE, "targets.txt")) if l.strip()]
    os.makedirs(MSA_DIR, exist_ok=True)
    failed = []
    for k, emd in enumerate(emds):
        seqs = sequences(emd)
        want = [os.path.join(MSA_DIR, f"{emd}_{i}.csv") for i in range(len(seqs))]
        if all(os.path.exists(w) for w in want):
            continue
        print(f"[{k + 1}/{len(emds)}] {emd}: {len(seqs)} entities", flush=True)
        for attempt in range(4):
            [shutil.rmtree(t, ignore_errors=True) for t in glob.glob(os.path.join(MSA_DIR, f"{emd}_*_tmp"))]  # drop a bad cached download
            try:
                compute_msa(data={f"{emd}_{i}": s for i, s in enumerate(seqs)}, target_id=emd,
                            msa_dir=Path(MSA_DIR), msa_server_url=a.server, msa_pairing_strategy="greedy")
                break
            except Exception as e:  # server busy / rate limited: back off and retry
                wait = 90 * (attempt + 1)
                print(f"  {emd}: {e!r}; retrying in {wait}s", flush=True)
                time.sleep(wait)
        for tmp in glob.glob(os.path.join(MSA_DIR, f"{emd}_*_tmp")):
            shutil.rmtree(tmp, ignore_errors=True)
        if not all(os.path.exists(w) for w in want):
            failed.append(emd)
        time.sleep(a.pause)
    done = sum(all(os.path.exists(os.path.join(MSA_DIR, f"{e}_{i}.csv")) for i in range(len(sequences(e))))
               for e in emds)
    print(f"MSAs ready for {done}/{len(emds)} maps" + (f"; failed: {' '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
