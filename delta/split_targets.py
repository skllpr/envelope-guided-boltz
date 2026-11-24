"""Split a target list by size for submit.sh: delta/targets_a40.txt (<= cut residues) and delta/targets_h200.txt.

  python3 delta/split_targets.py delta/targets.txt 2500 [first_n]
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    lst, cut = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    tok = {r["emdb"]: int(r["tokens"]) for r in csv.DictReader(open(os.path.join(ROOT, "envguide", "targets.csv")))}
    emds = [l.strip() for l in open(lst) if l.strip()][:n]
    missing = [e for e in emds if e not in tok]
    if missing:
        sys.exit(f"not in envguide/targets.csv: {' '.join(missing)}")
    small = [e for e in emds if tok[e] <= cut]
    big = [e for e in emds if tok[e] > cut]
    open(os.path.join(HERE, "targets_a40.txt"), "w").write("".join(e + "\n" for e in small))
    open(os.path.join(HERE, "targets_h200.txt"), "w").write("".join(e + "\n" for e in big))
    print(f"{len(small)} maps for A40, {len(big)} for H200 (cut {cut} residues)")


if __name__ == "__main__":
    main()
