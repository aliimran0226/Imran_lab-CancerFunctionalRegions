"""
Stage 2 — map each COSMIC mutation to a 3D coordinate.

Joins Stage 1's mutations with Stage 2a's representative structures and writes
one table where every mutation carries its Ca coordinate. Two branches:

  PDB:       for each representative structure, fetch the mmCIF + SIFTS XML,
             map UniProt position -> PDB residue via SIFTS, read the Ca coord.
  AlphaFold: for every gene, fetch the AF model; AF numbering is UniProt
             numbering, and residues below pLDDT 70 are dropped.

Only the Ca coordinate is stored per (mutation, structure); later stages
re-read the cached mmCIF if they need more. Rows are per patient-sample, not
collapsed. Failures go to a companion file with a typed reason.

Inputs (under DATA_ROOT):
  processed/run_<tag>/stage1_mutations.tsv
  processed/run_<tag>/stage2a_representative_structures.tsv
  raw/uniprot/UP000005640_9606.fasta.gz
  raw/uniprot/human_reviewed_idmap_xrefs.tsv.gz
  cached on demand: raw/pdb/*.cif, raw/sifts/xml/*.xml.gz, raw/alphafold/AF-*.cif
Outputs:
  processed/run_<tag>/stage2_mapped.tsv
  processed/run_<tag>/stage2_failures.tsv
  processed/run_<tag>/stage2_audit.txt
"""

from __future__ import annotations

import gzip
import os
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

DATA_ROOT = Path(os.environ.get("CFR_DATA_ROOT", "./cfr_data")).resolve()
RUN_TAG = os.environ.get("CFR_RUN_TAG", f"run_{date.today().isoformat()}")

RAW = DATA_ROOT / "raw"
UNIPROT_DIR = RAW / "uniprot"
SIFTS_XML_DIR = RAW / "sifts" / "xml"
PDB_DIR = RAW / "pdb"
ALPHAFOLD_DIR = RAW / "alphafold"

PROCESSED_DIR = DATA_ROOT / "processed" / RUN_TAG
STAGE1_MUTS = PROCESSED_DIR / "stage1_mutations.tsv"
STAGE2A_STRUCTS = PROCESSED_DIR / "stage2a_representative_structures.tsv"
FASTA_GZ = UNIPROT_DIR / "UP000005640_9606.fasta.gz"
IDMAP_TSV = UNIPROT_DIR / "human_reviewed_idmap_xrefs.tsv.gz"

STAGE2_OUT = PROCESSED_DIR / "stage2_mapped.tsv"
STAGE2_FAIL = PROCESSED_DIR / "stage2_failures.tsv"
STAGE2_AUDIT = PROCESSED_DIR / "stage2_audit.txt"

PLDDT_MIN = 70.0   # AlphaFold positions below this are excluded

HEADERS = {"User-Agent": "CFR-pipeline/0.1 (research; contact PI)"}
TIMEOUT = 60
RETRIES = 3
RETRY_BACKOFF = 2.0

RCSB_CIF_TMPL = "https://files.rcsb.org/download/{pdb_upper}.cif"
SIFTS_XML_TMPL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_lower}.xml.gz"
ALPHAFOLD_META_TMPL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            return True
        except requests.RequestException as exc:
            last = exc
            time.sleep(RETRY_BACKOFF * attempt)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    print(f"  [download FAIL] {dest.name}: {last}")
    return False


def fetch_pdb_cif(pdb_id: str) -> Path | None:
    dest = PDB_DIR / f"{pdb_id.upper()}.cif"
    return dest if _download(RCSB_CIF_TMPL.format(pdb_upper=pdb_id.upper()), dest) else None


def fetch_sifts_xml(pdb_id: str) -> Path | None:
    dest = SIFTS_XML_DIR / f"{pdb_id.lower()}.xml.gz"
    return dest if _download(SIFTS_XML_TMPL.format(pdb_lower=pdb_id.lower()), dest) else None


def fetch_alphafold_cif(accession: str) -> Path | None:
    existing = list(ALPHAFOLD_DIR.glob(f"AF-{accession}-F1-model_v*.cif"))
    if existing:
        return existing[0]
    try:
        r = requests.get(ALPHAFOLD_META_TMPL.format(accession=accession),
                         headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None
    if not data:
        return None
    rec = data[0] if isinstance(data, list) else data
    cif_url = rec.get("cifUrl")
    if not cif_url:
        return None
    dest = ALPHAFOLD_DIR / cif_url.rsplit("/", 1)[-1]
    return dest if _download(cif_url, dest) else None


SIFTS_NS = "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}"


def parse_sifts_xml(xml_gz_path: Path, target_chain: str,
                    target_acc: str) -> dict[int, str]:
    """SIFTS XML -> {uniprot_resnum -> pdb_auth_resnum} for the given chain and
    accession. Residues not resolved in the structure are skipped."""
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


def _as_list(v):
    # MMCIF2Dict returns a scalar when a loop has a single row
    if v is None:
        return None
    return v if isinstance(v, list) else [v]


def load_ca_coords(cif_path: Path, want_plddt: bool = False):
    """mmCIF -> (ca, rname, plddt), keyed by (auth_chain, auth_resnum). Ca atoms
    only, model 1, primary altloc. plddt is filled only when want_plddt."""
    d = MMCIF2Dict(cif_path)
    atom_id = _as_list(d.get("_atom_site.label_atom_id"))
    comp_id = _as_list(d.get("_atom_site.label_comp_id"))
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
    rname: dict = {}
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
        rname[key] = comp_id[i]
        if want_plddt and bfac:
            try:
                plddt[key] = float(bfac[i])
            except (ValueError, TypeError):
                pass
    return ca, rname, plddt


def load_fasta(path: Path) -> dict[str, str]:
    """UniProt FASTA -> {accession -> sequence}. Header: >sp|ACC|NAME ..."""
    seqs: dict[str, str] = {}
    acc = None
    buf: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def load_stage1(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def load_stage2a(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["sp_beg"] = pd.to_numeric(df["sp_beg"], errors="coerce")
    df["sp_end"] = pd.to_numeric(df["sp_end"], errors="coerce")
    return df


def load_gene_to_acc(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t", compression="gzip",
                     usecols=["Entry", "Gene Names (primary)"], dtype=str)
    df = df.dropna(subset=["Gene Names (primary)"])
    return dict(zip(df["Gene Names (primary)"], df["Entry"]))


class Audit:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, msg=""):
        print(msg)
        self.lines.append(msg)

    def save(self, path: Path):
        path.write_text("\n".join(self.lines) + "\n")


def run_stage2(limit_genes: int | None = None) -> Path:
    audit = Audit()
    audit.log("Stage 2 — Coordinate mapping (PDB + AlphaFold branches)")
    audit.log(f"DATA_ROOT = {DATA_ROOT}")
    audit.log(f"Run tag   = {RUN_TAG}")
    audit.log(f"pLDDT min = {PLDDT_MIN}")
    audit.log("")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    audit.log("Loading inputs...")
    muts = load_stage1(STAGE1_MUTS)
    muts["position"] = pd.to_numeric(muts["position"], errors="coerce").astype("Int64")
    audit.log(f"  {len(muts):,} mutations from Stage 1")

    structs = load_stage2a(STAGE2A_STRUCTS)
    reps = structs[structs["role"] == "representative"].copy()
    audit.log(f"  {len(reps):,} representative structures from Stage 2a")

    fasta = load_fasta(FASTA_GZ)
    audit.log(f"  {len(fasta):,} UniProt sequences (for ref-AA sanity check)")

    gene_to_acc = load_gene_to_acc(IDMAP_TSV)
    audit.log(f"  {len(gene_to_acc):,} gene->accession mappings")
    audit.log("")

    muts_by_gene: dict[str, pd.DataFrame] = {
        g: sub for g, sub in muts.groupby("gene")
    }

    mapped_rows: list[dict] = []
    fail_rows: list[dict] = []

    def fail(mut, source, reason, acc=""):
        fail_rows.append({
            "gene": mut["gene"], "uniprot_acc": acc,
            "sample_id": mut["sample_id"], "position": mut["position"],
            "ref_aa": mut["ref_aa"], "mut_aa": mut["mut_aa"],
            "structure_source": source, "reason": reason,
        })

    def ref_ok(acc, pos, ref_aa):
        seq = fasta.get(acc)
        if not seq:
            return None
        if pos < 1 or pos > len(seq):
            return "position_out_of_range"
        if seq[pos - 1] != ref_aa:
            return "ref_aa_mismatch"
        return "ok"

    # PDB branch — representative structures only
    audit.log("PDB branch (representative structures only)...")
    n_struct = 0
    for _, s in reps.iterrows():
        n_struct += 1
        if limit_genes and n_struct > limit_genes:
            break
        gene = s["gene"]
        acc = s["uniprot_acc"]
        pdb_id = s["pdb_id"]
        chain = s["chain"]
        beg, end = s["sp_beg"], s["sp_end"]
        if gene not in muts_by_gene or pd.isna(beg) or pd.isna(end):
            continue
        gmuts = muts_by_gene[gene]
        in_range = gmuts[(gmuts["position"] >= int(beg))
                         & (gmuts["position"] <= int(end))]
        if in_range.empty:
            continue

        source = f"PDB:{pdb_id.upper()}:{chain}"
        cif = fetch_pdb_cif(pdb_id)
        if cif is None:
            for _, m in in_range.iterrows():
                fail(m, source, "cif_download_failed", acc)
            continue
        xml = fetch_sifts_xml(pdb_id)
        if xml is None:
            for _, m in in_range.iterrows():
                fail(m, source, "sifts_download_failed", acc)
            continue
        try:
            unp_to_pdb = parse_sifts_xml(xml, chain, acc)
            ca, _rname, _ = load_ca_coords(cif)
        except Exception as exc:  # noqa: BLE001
            for _, m in in_range.iterrows():
                fail(m, source, f"parse_error:{type(exc).__name__}", acc)
            continue

        for _, m in in_range.iterrows():
            pos = int(m["position"])
            chk = ref_ok(acc, pos, m["ref_aa"])
            if chk not in ("ok", None):
                fail(m, source, chk, acc)
                continue
            pdb_num = unp_to_pdb.get(pos)
            if pdb_num is None:
                fail(m, source, "sifts_no_mapping", acc)
                continue
            coord = ca.get((chain, pdb_num))
            if coord is None:
                fail(m, source, "pdb_res_not_in_structure", acc)
                continue
            mapped_rows.append({
                "gene": gene, "uniprot_acc": acc, "sample_id": m["sample_id"],
                "position": pos, "ref_aa": m["ref_aa"], "mut_aa": m["mut_aa"],
                "branch": "pdb", "structure_source": source,
                "pdb_res_num": pdb_num,
                "x": coord[0], "y": coord[1], "z": coord[2],
                "confidence": "",
            })

    audit.log(f"  processed {n_struct:,} representative structures")
    audit.log("")

    # AlphaFold branch — every gene (AF numbering == UniProt numbering)
    audit.log("AlphaFold branch (all genes)...")
    genes = sorted(muts_by_gene.keys())
    if limit_genes:
        genes = genes[:limit_genes]
    n_af = 0
    for gene in genes:
        acc = gene_to_acc.get(gene)
        if not acc:
            for _, m in muts_by_gene[gene].iterrows():
                fail(m, "AF:?", "no_accession")
            continue
        n_af += 1
        source = f"AF:{acc}"
        af = fetch_alphafold_cif(acc)
        if af is None:
            for _, m in muts_by_gene[gene].iterrows():
                fail(m, source, "no_alphafold_model", acc)
            continue
        try:
            ca, _rname, plddt = load_ca_coords(af, want_plddt=True)
        except Exception as exc:  # noqa: BLE001
            for _, m in muts_by_gene[gene].iterrows():
                fail(m, source, f"parse_error:{type(exc).__name__}", acc)
            continue

        for _, m in muts_by_gene[gene].iterrows():
            pos = int(m["position"])
            chk = ref_ok(acc, pos, m["ref_aa"])
            if chk not in ("ok", None):
                fail(m, source, chk, acc)
                continue
            key = ("A", str(pos))
            coord = ca.get(key)
            if coord is None:
                fail(m, source, "af_res_missing", acc)
                continue
            pl = plddt.get(key)
            if pl is not None and pl < PLDDT_MIN:
                fail(m, source, "alphafold_low_plddt", acc)
                continue
            mapped_rows.append({
                "gene": gene, "uniprot_acc": acc, "sample_id": m["sample_id"],
                "position": pos, "ref_aa": m["ref_aa"], "mut_aa": m["mut_aa"],
                "branch": "alphafold", "structure_source": source,
                "pdb_res_num": str(pos),
                "x": coord[0], "y": coord[1], "z": coord[2],
                "confidence": f"{pl:.1f}" if pl is not None else "",
            })

    audit.log(f"  processed {n_af:,} genes with an accession")
    audit.log("")

    cols = ["gene", "uniprot_acc", "sample_id", "position", "ref_aa", "mut_aa",
            "branch", "structure_source", "pdb_res_num", "x", "y", "z",
            "confidence"]
    mapped = pd.DataFrame(mapped_rows, columns=cols)
    mapped.to_csv(STAGE2_OUT, sep="\t", index=False)

    fail_cols = ["gene", "uniprot_acc", "sample_id", "position", "ref_aa",
                 "mut_aa", "structure_source", "reason"]
    fails = pd.DataFrame(fail_rows, columns=fail_cols)
    fails.to_csv(STAGE2_FAIL, sep="\t", index=False)

    audit.log("Results:")
    audit.log(f"  {len(mapped):,} mapped (mutation x structure) rows")
    if len(mapped):
        pdb_n = (mapped["branch"] == "pdb").sum()
        af_n = (mapped["branch"] == "alphafold").sum()
        audit.log(f"    PDB branch      : {pdb_n:,}")
        audit.log(f"    AlphaFold branch: {af_n:,}")
        audit.log(f"    distinct genes mapped: {mapped['gene'].nunique():,}")
    audit.log(f"  {len(fails):,} failure rows")
    if len(fails):
        audit.log("  failure reasons:")
        for reason, c in fails["reason"].value_counts().items():
            audit.log(f"    {reason:26s} {c:>9,}")
    audit.log("")
    audit.log(f"Wrote {STAGE2_OUT}")
    audit.log(f"Wrote {STAGE2_FAIL}")
    audit.save(STAGE2_AUDIT)
    audit.log(f"Wrote {STAGE2_AUDIT}")
    return STAGE2_OUT


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_stage2(limit_genes=lim)