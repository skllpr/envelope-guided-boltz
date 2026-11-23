"""Pick the next benchmark maps and build their inputs (run on a machine that can reach EMDB and RCSB, e.g. the Mac).

  pip install gemmi
  python delta/select_targets.py --n 87                                   # pick, download, write inputs
  python delta/select_targets.py --n 87 --local-maps /Volumes/KINGSTON/EMDB_negative_stain_maps
  python delta/select_targets.py --n 87 --dry-run                         # just list what would be picked

Selection (same rules as the dataset manifest, then what Boltz-2 can run on one 48 GB GPU):
  1. EMDB search `staining_category:negative`, no tomography, single-particle only at resolution >= 10 A.
  2. Has a fitted PDB model (the answer key), and that model's RCSB assembly 1 downloads.
  3. Assembly 1 is protein only (no DNA/RNA), <= MAX_ENTITIES distinct chains, <= MAX_CHAINS chains in total,
     MIN_TOKENS..MAX_TOKENS residues in total (the pilot's largest map was 2,408).
  4. Not one of the 13 pilot maps, and not a duplicate: one map per PDB entry and per set of chain sequences.
  If more than --n maps pass, a seeded random draw picks --n of them (stable across reruns).

Writes, for each picked map (same layout as the pilot):
  data/maps/EMD-N.map.gz                 EMDB map (copied from --local-maps when it is there, else downloaded)
  data/deposited/<PDB>-assembly1.cif     RCSB assembly 1 of the fitted model
  envguide/inputs/EMD-N.yaml             Boltz input with the full (SEQRES) sequence of each distinct chain
  envguide/targets.csv                   rows appended (pilot rows kept)
  delta/targets.txt                      every map to run on Delta (pilot + new), one per line
  delta/selection_log.csv                every candidate and why it was kept or dropped
"""
import argparse, csv, glob, gzip, io, os, random, re, shutil, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TARGETS = os.path.join(ROOT, "envguide", "targets.csv")
INPUTS = os.path.join(ROOT, "envguide", "inputs")
MAPS = os.path.join(ROOT, "data", "maps")
DEPOSITED = os.path.join(ROOT, "data", "deposited")

EMDB_SEARCH = ("https://www.ebi.ac.uk/emdb/api/search/staining_category:negative?rows=100000&wt=csv"
               "&fl=emdb_id,title,resolution,structure_determination_method,fitted_pdbs,release_date")
EMDB_MAP = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{n}/map/emd_{n}.map.gz"
RCSB_ASSEMBLY = "https://files.rcsb.org/download/{pdb}-assembly1.cif"

MIN_TOKENS, MAX_TOKENS = 100, 2500
MAX_CHAINS, MAX_ENTITIES = 20, 10
DA_PER_RESIDUE = 110  # mass_da = residues x 110, as in the pilot targets.csv
FIELDS = ["emdb", "pdb", "tokens", "chains", "entities", "mass_da", "resolution", "year", "homo_oligomer",
          "baseline_frac_correct", "baseline_median_rmsd"]
CHAIN_IDS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "envelope-guided-boltz target selection"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(5 * (i + 1))


def emdb_candidates():
    """EMDB negative-stain entries with a fitted model, after the manifest's method/resolution rules."""
    rows = list(csv.DictReader(io.StringIO(get(EMDB_SEARCH).decode())))
    print(f"EMDB staining_category:negative -> {len(rows)} entries", flush=True)
    out = []
    for r in rows:
        method = (r.get("structure_determination_method") or "").lower()
        if "tomograph" in method:
            continue
        try:
            res = float(re.findall(r"[\d.]+", r.get("resolution") or "")[0])
        except IndexError:
            res = None
        if "singleparticle" in method.replace(" ", "").replace("_", "") and (res is None or res < 10):
            continue
        pdbs = sorted({p.upper() for p in re.findall(r"\b[0-9][A-Za-z0-9]{3}\b", r.get("fitted_pdbs") or "")})
        if not pdbs:
            continue
        emd = r["emdb_id"].strip()
        emd = emd if emd.startswith("EMD-") else f"EMD-{emd}"
        year = (re.findall(r"\d{4}", r.get("release_date") or "") or [""])[0]
        out.append(dict(emdb=emd, pdbs=pdbs, resolution=res, year=year, title=r.get("title", "")))
    print(f"  with a fitted model, no tomography, single-particle >= 10 A: {len(out)}", flush=True)
    return out


def describe_assembly(cif_text):
    """Protein entities of an assembly-1 mmCIF -> list of (sequence, copies), or (None, reason)."""
    import gemmi
    doc = gemmi.cif.read_string(cif_text)
    st = gemmi.make_structure_from_block(doc.sole_block())
    st.setup_entities()
    model = st[0]
    copies, seqs = {}, {}
    for ent in st.entities:
        if ent.entity_type != gemmi.EntityType.Polymer:
            continue
        if ent.polymer_type not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            return None, f"non-protein polymer ({ent.polymer_type.name})"
        if not ent.full_sequence:
            return None, f"entity {ent.name} has no sequence"
        seqs[ent.name] = gemmi.one_letter_code(ent.full_sequence).upper()
    for ch in model:
        poly = ch.get_polymer()
        if len(poly) == 0:
            continue
        ent_ids = {r.entity_id for r in poly if r.entity_id}
        if len(ent_ids) != 1:
            return None, f"chain {ch.name}: cannot tell its entity"
        eid = ent_ids.pop()
        if eid not in seqs:
            return None, f"chain {ch.name}: entity {eid} not a protein entity"
        copies[eid] = copies.get(eid, 0) + 1
    ents = [(seqs[e], n) for e, n in copies.items()]
    if not ents:
        return None, "no protein chains"
    for s, _ in ents:
        if s.count("X") > 0.1 * len(s):
            return None, "sequence mostly unknown residues (X)"
    return ents, ""


def yaml_text(ents):
    ids, lines, k = iter(CHAIN_IDS), ["version: 1", "sequences:"], 0
    for seq, n in ents:
        chain = [next(ids) for _ in range(n)]
        lines += ["  - protein:", f"      id: [{', '.join(chain)}]", f"      sequence: {seq}"]
    return "\n".join(lines) + "\n"


def target_row(emd, pdb, ents, res, year):
    tokens = sum(len(s) * n for s, n in ents)
    return dict(emdb=emd, pdb=pdb, tokens=tokens, chains=sum(n for _, n in ents), entities="|".join(
        f"{len(s)}x{n}" for s, n in ents), mass_da=tokens * DA_PER_RESIDUE, resolution=res, year=year,
        homo_oligomer=int(any(n > 1 for _, n in ents)), baseline_frac_correct="", baseline_median_rmsd="")


def find_local_map(local_dir, emd):
    n = emd.split("-")[1]
    for pat in (f"emd_{n}.map.gz", f"EMD-{n}.map.gz", f"emd_{n}.map", f"EMD-{n}.map", f"*{n}*.map*"):
        hits = [h for h in glob.glob(os.path.join(local_dir, "**", pat), recursive=True) if not h.endswith(".part")]
        if hits:
            return hits[0]
    return None


def save_map(emd, local_dir):
    dst = os.path.join(MAPS, f"{emd}.map.gz")
    if os.path.exists(dst):
        return "exists"
    src = find_local_map(local_dir, emd) if local_dir else None
    if src:
        if src.endswith(".gz"):
            shutil.copy(src, dst)
        else:
            with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        return "copied"
    data = get(EMDB_MAP.format(n=emd.split("-")[1]))
    if data is None:
        return None
    open(dst + ".part", "wb").write(data)
    os.replace(dst + ".part", dst)
    return "downloaded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=87, help="new maps to add")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-maps", default=None, help="folder with already-downloaded EMDB maps (searched recursively)")
    ap.add_argument("--dry-run", action="store_true", help="list the picks, write nothing but the log")
    a = ap.parse_args()

    pilot = list(csv.DictReader(open(TARGETS)))
    have = {r["emdb"] for r in pilot}
    seen_pdb = {r["pdb"].upper() for r in pilot}
    seen_seqs = set()
    for r in pilot:
        y = open(os.path.join(INPUTS, f"{r['emdb']}.yaml")).read()
        seen_seqs.add(frozenset(re.findall(r"sequence:\s*(\S+)", y)))

    cands = emdb_candidates()
    random.Random(a.seed).shuffle(cands)  # random draw; stable for a given seed
    log, picked = [], []
    for c in cands:
        if len(picked) >= a.n:
            break
        row = dict(emdb=c["emdb"], pdbs=" ".join(c["pdbs"]), resolution=c["resolution"], year=c["year"],
                   title=c["title"][:120], pdb="", tokens="", status="")
        if c["emdb"] in have:
            row["status"] = "already in targets.csv (pilot)"
            log.append(row)
            continue
        reasons = []
        for pdb in c["pdbs"]:
            if pdb in seen_pdb:
                reasons.append(f"{pdb}: model already used by another map")
                continue
            raw = get(RCSB_ASSEMBLY.format(pdb=pdb))
            if raw is None:
                reasons.append(f"{pdb}: no assembly 1 at RCSB")
                continue
            text = raw.decode()
            ents, why = describe_assembly(text)
            if ents is None:
                reasons.append(f"{pdb}: {why}")
                continue
            t = target_row(c["emdb"], pdb, ents, c["resolution"], c["year"])
            if not MIN_TOKENS <= t["tokens"] <= MAX_TOKENS:
                reasons.append(f"{pdb}: {t['tokens']} residues (limit {MIN_TOKENS}-{MAX_TOKENS})")
                continue
            if t["chains"] > MAX_CHAINS or len(ents) > MAX_ENTITIES:
                reasons.append(f"{pdb}: {t['chains']} chains / {len(ents)} distinct")
                continue
            key = frozenset(s for s, _ in ents)
            if key in seen_seqs:
                reasons.append(f"{pdb}: same chain sequences as a map already picked")
                continue
            if not a.dry_run:
                ok = save_map(c["emdb"], a.local_maps)
                if not ok:
                    reasons.append("map not found at EMDB")
                    break
                open(os.path.join(DEPOSITED, f"{pdb}-assembly1.cif"), "w").write(text)
                open(os.path.join(INPUTS, f"{c['emdb']}.yaml"), "w").write(yaml_text(ents))
            seen_pdb.add(pdb)
            seen_seqs.add(key)
            picked.append(t)
            row.update(pdb=pdb, tokens=t["tokens"], status="picked")
            print(f"  picked {c['emdb']} {pdb} {t['tokens']} residues, {t['chains']} chains", flush=True)
            break
        if not row["status"]:
            row["status"] = "dropped: " + "; ".join(reasons)
        log.append(row)

    with open(os.path.join(HERE, "selection_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]) if log else ["emdb"])
        w.writeheader()
        w.writerows(log)
    print(f"\npicked {len(picked)} of {a.n} requested from {len(cands)} candidates "
          f"(log: delta/selection_log.csv)")
    if len(picked) < a.n:
        print("fewer maps passed the filters than requested; see the log for why the rest were dropped")
    if a.dry_run or not picked:
        return
    with open(TARGETS, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerows(picked)
    names = [r["emdb"] for r in pilot] + [t["emdb"] for t in picked]
    open(os.path.join(HERE, "targets.txt"), "w").write("\n".join(names) + "\n")
    print(f"targets.csv now has {len(names)} maps; delta/targets.txt written")


if __name__ == "__main__":
    sys.exit(main())
