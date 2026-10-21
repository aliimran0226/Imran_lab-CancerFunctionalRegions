"""
Stage 2b — per-residue coordinate export (the "universe" file).

Stage 2 writes only the mutated residues. The clustering model also needs every
other residue in each structure, because its null model scatters the observed
mutations across all possible landing spots — without the non-mutated residues
there is nowhere to scatter them.

So this writes stage2_residues.tsv at a different grain: one row per resolved
residue (not per patient-mutation), covering every residue of each structure,
flagged with whether it was mutated and in how many samples.

  columns: gene, uniprot_acc, branch, structure_source, pdb_id, chain,
           uniprot_pos, pdb_res_num, x, y, z, confidence, is_mutated,
           patient_count
  is_mutated     1 if COSMIC hit this residue, else 0
  patient_count  distinct patient samples mutated here (0 if none)
  confidence     AlphaFold pLDDT for the AF branch; blank for PDB

What counts as a residue in the universe (same rule as Stage 2):
  PDB: chain residues with a Ca coordinate AND a SIFTS UniProt mapping.
  AF:  residues with pLDDT >= PLDDT_MIN.

Re-reads the cached structure files; downloads nothing. Run after Stage 2.

Inputs  (processed/run_<tag>/): stage2_mapped.tsv,
        stage2a_representative_structures.tsv
Cached  (raw/): pdb/*.cif, sifts/xml/*.xml.gz, alphafold/AF-*.cif
Outputs (processed/run_<tag>/): stage2_residues.tsv, stage2_residues_audit.txt

Run:
  python3 stage2b_residues.py                 # all structures
  python3 stage2b_residues.py --limit 20      # quick smoke test (per branch)
  python3 stage2b_residues.py --branch af     # one branch only
"""

from __future__ import annotations

import argparse
import gzip
import os
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

DATA_ROOT = Path(os.environ.get("CFR_DATA_ROOT", "./cfr_data")).resolve()
RUN_TAG = os.environ.get("CFR_RUN_TAG", f"run_{date.today().isoformat()}")

RAW = DATA_ROOT / "raw"
SIFTS_XML_DIR = RAW / "sifts" / "xml"
PDB_DIR = RAW / "pdb"
ALPHAFOLD_DIR = RAW / "alphafold"

PROCESSED_DIR = DATA_ROOT / "processed" / RUN_TAG
STAGE2_MAPPED = PROCESSED_DIR / "stage2_mapped.tsv"
STAGE2A_STRUCTS = PROCESSED_DIR / "stage2a_representative_structures.tsv"

STAGE2B_OUT = PROCESSED_DIR / "stage2_residues.tsv"
STAGE2B_AUDIT = PROCESSED_DIR / "stage2_residues_audit.txt"

PLDDT_MIN = 70.0   # must match Stage 2
SIFTS_NS = "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}"


# Structure parsing — same as Stage 2, kept here so this stage stands alone.

def _as_list(v):
    if v is None:
        return None
    return v if isinstance(v, list) else [v]


def load_ca_coords(cif_path: Path, want_plddt: bool = False):
    """mmCIF -> (ca, plddt), keyed by (auth_chain, auth_resnum). Ca atoms only,
    model 1, primary altloc. plddt filled only when want_plddt."""
    d = MMCIF2Dict(cif_path)
    atom_id = _as_list(d.get("_atom_site.label_atom_id"))
    chain = _as_list(d.get("_atom_site.auth_asym_id")
                     or d.get("_atom_site.label_asym_id"))
    seq = _as_list(d.get("_atom_site.auth_seq_id")
                   or d.get("_atom_site.label_seq_id"))
    xs = _as_list(d.get("_atom_site.Cartn_x"))
    ys = _as_list(d.get("_atom_site.Cartn_y"))
    zs = _as_list(d.get("_atom_site.Cartn_z"))
    altloc = _as_list(d.get("_atom_site.label_alt_id"))
    model = _as_list(d.get("_atom_site.pdbx_PDB_model_num"))
    bfac = _as_list(d.get("_atom_site.B_iso_or_equiv"))

    ca: dict = {}
    plddt: dict = {}
    n = len(atom_id) if atom_id else 0
    first_model = model[0] if model else None
    for i in range(n):
        if model and model[i] != first_model:
            continue
        alt = altloc[i] if altloc else "."
        if alt not in (".", "", "A"):
            continue
        if atom_id[i] != "CA":
            continue
        key = (chain[i], str(seq[i]))
        try:
            ca[key] = (float(xs[i]), float(ys[i]), float(zs[i]))
        except (ValueError, TypeError):
            continue
        if want_plddt and bfac:
            try:
                plddt[key] = float(bfac[i])
            except (ValueError, TypeError):
                pass
    return ca, plddt


def parse_sifts_xml(xml_gz_path: Path, target_chain: str,
                    target_acc: str) -> dict[int, str]:
    """SIFTS XML -> {uniprot_pos -> pdb_auth_resnum} for the chain + accession."""
    with gzip.open(xml_gz_path, "rb") as fh:
        tree = ET.parse(fh)
    root = tree.getroot()
    unp_to_pdb: dict[int, str] = {}
    for residue in root.iter(f"{SIFTS_NS}residue"):
        pdb_num = pdb_chain = unp_num = unp_acc = None
        for xref in residue.findall(f"{SIFTS_NS}crossRefDb"):
            src = xref.get("dbSource")
            if src == "PDB":
                pdb_num = xref.get("dbResNum")
                pdb_chain = xref.get("dbChainId")
            elif src == "UniProt":
                unp_num = xref.get("dbResNum")
                unp_acc = xref.get("dbAccessionId")
        not_observed = False
        for det in residue.findall(f"{SIFTS_NS}residueDetail"):
            if (det.text or "").strip() == "Not_Observed":
                not_observed = True
                break
        if not_observed:
            continue
        if (pdb_chain == target_chain and unp_acc == target_acc
                and unp_num and pdb_num and pdb_num not in ("None", "null")):
            try:
                unp_to_pdb[int(unp_num)] = pdb_num
            except ValueError:
                pass
    return unp_to_pdb


def pdb_residues(pdb_id: str, chain: str, acc: str):
    """(uniprot_pos, pdb_res_num, x, y, z, None) for each chain residue with a
    Ca and a SIFTS mapping. None if a cached file is missing/unreadable."""
    cif = PDB_DIR / f"{pdb_id.upper()}.cif"
    xml = SIFTS_XML_DIR / f"{pdb_id.lower()}.xml.gz"
    if not cif.exists() or not xml.exists():
        return None
    try:
        ca, _ = load_ca_coords(cif)
        unp_to_pdb = parse_sifts_xml(xml, chain, acc)
    except Exception:
        return None
    out = []
    for pos, pdb_num in unp_to_pdb.items():
        coord = ca.get((chain, pdb_num))
        if coord is not None:
            out.append((pos, pdb_num, coord[0], coord[1], coord[2], None))
    return out


def af_residues(acc: str, plddt_min: float):
    """(uniprot_pos, "", x, y, z, pLDDT) for each chain-A residue with
    pLDDT >= cut. None if the cached model is missing."""
    hits = sorted(ALPHAFOLD_DIR.glob(f"AF-{acc}-F1-model_v*.cif"))
    if not hits:
        return None
    try:
        ca, plddt = load_ca_coords(hits[0], want_plddt=True)
    except Exception:
        return None
    out = []
    for (ch, resnum), coord in ca.items():
        if ch != "A":
            continue
        conf = plddt.get((ch, resnum), 0.0)
        if conf < plddt_min:
            continue
        try:
            pos = int(resnum)
        except ValueError:
            continue
        out.append((pos, "", coord[0], coord[1], coord[2], conf))
    return out


class Audit:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def save(self, path: Path):
        path.write_text("\n".join(self.lines) + "\n")


def run_stage2b(plddt_min=PLDDT_MIN, branches=("pdb", "af"), limit=0):
    audit = Audit()
    audit.log("Stage 2b — per-residue coordinate export (the universe file)")
    audit.log(f"DATA_ROOT = {DATA_ROOT}")
    audit.log(f"Run tag   = {RUN_TAG}")
    audit.log(f"pLDDT_min = {plddt_min}")
    audit.log("")

    if not STAGE2_MAPPED.exists():
        raise SystemExit(f"Stage 2 output not found: {STAGE2_MAPPED}\n"
                         "Set CFR_DATA_ROOT / CFR_RUN_TAG and run Stage 2 first.")

    audit.log("Loading Stage 2 mapped mutations...")
    mapped = pd.read_csv(
        STAGE2_MAPPED, sep="\t",
        usecols=["gene", "uniprot_acc", "sample_id", "position",
                 "branch", "structure_source"],
        dtype={"gene": str, "uniprot_acc": str, "sample_id": str,
               "position": int, "branch": str, "structure_source": str},
    )
    audit.log(f"  {len(mapped):,} mapped mutation rows")

    # structure_source -> {uniprot_pos -> distinct patient count}
    counts = (mapped.groupby(["structure_source", "position"])["sample_id"]
              .nunique())
    count_lookup = {}
    for (src, pos), c in counts.items():
        count_lookup.setdefault(src, {})[int(pos)] = int(c)

    # structure_source -> (pdb_id, chain), from Stage 2a
    reps = pd.read_csv(STAGE2A_STRUCTS, sep="\t", dtype=str)
    rep_key = {}
    for _, r in reps.iterrows():
        if pd.isna(r.get("pdb_id")) or pd.isna(r.get("chain")):
            continue
        src = f"PDB:{str(r['pdb_id']).upper()}:{r['chain']}"
        rep_key[src] = (str(r["pdb_id"]), str(r["chain"]))

    meta = (mapped.groupby("structure_source")[["gene", "uniprot_acc", "branch"]]
            .first())

    rows = []
    n_structs = {"pdb": 0, "af": 0}
    n_missing = {"pdb": 0, "af": 0}
    mut_not_in_universe = 0

    for branch in branches:
        srcs = sorted(meta.index[meta["branch"] == branch])
        audit.log("")
        audit.log(f"{branch.upper()} branch: {len(srcs)} structures with mutations")
        done = 0
        for src in srcs:
            if limit and done >= limit:
                break
            gene = meta.loc[src, "gene"]
            acc = meta.loc[src, "uniprot_acc"]

            if branch == "pdb":
                if src not in rep_key:
                    n_missing["pdb"] += 1
                    continue
                pdb_id, chain = rep_key[src]
                residues = pdb_residues(pdb_id, chain, acc)
            else:
                pdb_id, chain = "", ""
                residues = af_residues(acc, plddt_min)

            if not residues:
                n_missing[branch] += 1
                continue

            cmap = count_lookup.get(src, {})
            universe_positions = set()
            for pos, pdb_num, x, y, z, conf in residues:
                universe_positions.add(pos)
                pc = cmap.get(pos, 0)
                rows.append({
                    "gene": gene, "uniprot_acc": acc, "branch": branch,
                    "structure_source": src, "pdb_id": pdb_id, "chain": chain,
                    "uniprot_pos": pos, "pdb_res_num": pdb_num,
                    "x": x, "y": y, "z": z,
                    "confidence": "" if conf is None else round(conf, 2),
                    "is_mutated": 1 if pc > 0 else 0,
                    "patient_count": pc,
                })
            # should be 0: a mutated residue must appear in its own universe
            mut_not_in_universe += len(set(cmap) - universe_positions)
            n_structs[branch] += 1
            done += 1
            if done % 100 == 0:
                audit.log(f"    processed {done} structures...")
        audit.log(f"  {branch.upper()}: {n_structs[branch]} structures exported, "
                  f"{n_missing[branch]} skipped (cached file missing/unreadable)")

    df = pd.DataFrame(rows, columns=[
        "gene", "uniprot_acc", "branch", "structure_source", "pdb_id", "chain",
        "uniprot_pos", "pdb_res_num", "x", "y", "z", "confidence",
        "is_mutated", "patient_count"])
    df = df.sort_values(["branch", "structure_source", "uniprot_pos"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGE2B_OUT, sep="\t", index=False)

    audit.log("")
    audit.log("Results:")
    audit.log(f"  {len(df):,} residue rows written")
    if not df.empty:
        for branch in branches:
            b = df[df["branch"] == branch]
            if b.empty:
                continue
            n_struct = b["structure_source"].nunique()
            n_mut = int((b["is_mutated"] == 1).sum())
            audit.log(f"    {branch.upper():3}: {len(b):,} residues over "
                      f"{n_struct:,} structures; {n_mut:,} mutated, "
                      f"{len(b) - n_mut:,} non-mutated landing spots")
        audit.log(f"  mutated residues not found in any universe: {mut_not_in_universe:,} "
                  "(expect 0; >0 means a structure file changed since Stage 2)")
    audit.log("")
    audit.log(f"Wrote {STAGE2B_OUT}")
    audit.save(STAGE2B_AUDIT)
    audit.log(f"Wrote {STAGE2B_AUDIT}")


def main():
    ap = argparse.ArgumentParser(description="Stage 2b — per-residue universe export")
    ap.add_argument("--plddt-min", type=float, default=PLDDT_MIN,
                    help="AF universe pLDDT cut (default 70)")
    ap.add_argument("--branch", choices=["pdb", "af", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0,
                    help="export at most this many structures PER branch (smoke test)")
    args = ap.parse_args()
    branches = ("pdb", "af") if args.branch == "both" else (args.branch,)
    run_stage2b(plddt_min=args.plddt_min, branches=branches, limit=args.limit)


if __name__ == "__main__":
    main()