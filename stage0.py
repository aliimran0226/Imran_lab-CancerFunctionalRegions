"""Download and cache raw reference data for the pipeline."""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import time
from pathlib import Path

import requests

# Configuration.
DATA_ROOT = Path(os.environ.get("CFR_DATA_ROOT", "./cfr_data")).resolve()

RAW = DATA_ROOT / "raw"
UNIPROT_DIR = RAW / "uniprot"
SIFTS_DIR = RAW / "sifts"
PDB_DIR = RAW / "pdb"
ALPHAFOLD_DIR = RAW / "alphafold"
COSMIC_DIR = RAW / "cosmic"

# Human reviewed reference proteome.
ORGANISM_ID = 9606
HUMAN_REF_PROTEOME = "UP000005640"

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_PROTEOME_FASTA = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/reference_proteomes/Eukaryota/"
    f"{HUMAN_REF_PROTEOME}/{HUMAN_REF_PROTEOME}_{ORGANISM_ID}.fasta.gz"
)

# SIFTS summary and per-entry XML.
SIFTS_SUMMARY_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/"
    "pdb_chain_uniprot.csv.gz"
)
SIFTS_XML_TMPL = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_lower}.xml.gz"
)

# AlphaFold metadata endpoint lets us follow the current model version.
ALPHAFOLD_META_TMPL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"

# PDB mmCIF files are preferred for structure metadata.
RCSB_CIF_TMPL = "https://files.rcsb.org/download/{pdb_upper}.cif"

HEADERS = {"User-Agent": "CFR-pipeline/0.1 (research; contact PI)"}
TIMEOUT = 60
RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, multiplied each retry


# Helpers.

def _ensure_dirs() -> None:
    for d in (UNIPROT_DIR, SIFTS_DIR, PDB_DIR, ALPHAFOLD_DIR, COSMIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, *, skip_if_exists: bool = True) -> bool:
    """Download a URL to disk with retry and atomic replace."""
    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name}")
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(
                url, headers=HEADERS, stream=True, timeout=TIMEOUT
            ) as r:
                if r.status_code == 404:
                    print(f"  [404 ] {url}")
                    return False
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"  [ok   ] {dest.name} ({size_mb:.2f} MB)")
            return True
        except requests.RequestException as exc:
            last_err = exc
            wait = RETRY_BACKOFF * attempt
            print(f"  [retry {attempt}/{RETRIES}] {dest.name}: {exc} "
                  f"(waiting {wait:.0f}s)")
            time.sleep(wait)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    print(f"  [FAIL ] {dest.name}: {last_err}")
    return False


# UniProt downloads.

def download_uniprot_proteome_fasta() -> Path | None:
    """Reviewed human reference proteome FASTA."""
    dest = UNIPROT_DIR / f"{HUMAN_REF_PROTEOME}_{ORGANISM_ID}.fasta.gz"
    ok = _download(UNIPROT_PROTEOME_FASTA, dest)
    return dest if ok else None


def download_uniprot_idmapping_and_xrefs() -> Path | None:
    """Human reviewed UniProt idmapping and xrefs."""
    dest = UNIPROT_DIR / "human_reviewed_idmap_xrefs.tsv.gz"
    params = {
        "query": f"organism_id:{ORGANISM_ID} AND reviewed:true",
        "fields": "accession,gene_primary,gene_names,length,sequence,"
                  "xref_pdb,xref_alphafolddb",
        "format": "tsv",
        "compressed": "true",
    }
    # Build the URL with params via requests' prep (stream endpoint supports GET).
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name}")
        return dest
    req = requests.Request("GET", UNIPROT_STREAM, params=params, headers=HEADERS)
    prepared = requests.Session().prepare_request(req)
    ok = _download(prepared.url, dest)
    return dest if ok else None


def download_uniprot_features_gff() -> Path | None:
    """Human reviewed feature annotations in GFF."""
    dest = UNIPROT_DIR / "human_reviewed_features.gff.gz"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name}")
        return dest
    params = {
        "query": f"organism_id:{ORGANISM_ID} AND reviewed:true",
        "format": "gff",
        "compressed": "true",
    }
    req = requests.Request("GET", UNIPROT_STREAM, params=params, headers=HEADERS)
    prepared = requests.Session().prepare_request(req)
    ok = _download(prepared.url, dest)
    return dest if ok else None


# SIFTS.

def download_sifts_summary() -> Path | None:
    """PDB-to-UniProt summary index."""
    dest = SIFTS_DIR / "pdb_chain_uniprot.csv.gz"
    ok = _download(SIFTS_SUMMARY_URL, dest)
    return dest if ok else None


def download_sifts_xml_for_pdb(pdb_id: str) -> Path | None:
    """Residue-level SIFTS mapping for one PDB entry."""
    pdb_lower = pdb_id.lower()
    dest = SIFTS_DIR / "xml" / f"{pdb_lower}.xml.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = _download(SIFTS_XML_TMPL.format(pdb_lower=pdb_lower), dest)
    return dest if ok else None


# Structure downloads used later in the pipeline.

def download_pdb_cif(pdb_id: str) -> Path | None:
    dest = PDB_DIR / f"{pdb_id.upper()}.cif"
    ok = _download(RCSB_CIF_TMPL.format(pdb_upper=pdb_id.upper()), dest)
    return dest if ok else None


def download_alphafold_cif(accession: str) -> Path | None:
    """Download the current AlphaFold CIF for a UniProt accession."""
    # Reuse a cached AlphaFold model if it already exists.
    existing = list(ALPHAFOLD_DIR.glob(f"AF-{accession}-F1-model_v*.cif"))
    if existing:
        print(f"  [cached] {existing[0].name}")
        return existing[0]

    meta_url = ALPHAFOLD_META_TMPL.format(accession=accession)
    try:
        r = requests.get(meta_url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404:
            print(f"  [no AF ] {accession} — no AlphaFold prediction available")
            return None
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  [FAIL ] AlphaFold metadata for {accession}: {exc}")
        return None

    if not data:
        print(f"  [no AF ] {accession} — empty metadata response")
        return None
    record = data[0] if isinstance(data, list) else data
    cif_url = record.get("cifUrl")
    if not cif_url:
        print(f"  [no AF ] {accession} — no cifUrl in metadata")
        return None

    # Keep the original filename so the model version stays visible locally.
    filename = cif_url.rsplit("/", 1)[-1]
    dest = ALPHAFOLD_DIR / filename
    ok = _download(cif_url, dest)
    return dest if ok else None


# COSMIC is manual and licensed.

def download_cosmic() -> None:
    """COSMIC requires a human-managed licensed download."""
    marker = COSMIC_DIR / "PUT_COSMIC_TSV_HERE.txt"
    COSMIC_DIR.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(download_cosmic.__doc__ or "")
    print(f"  [manual] COSMIC must be placed by hand in {COSMIC_DIR}")


# Orchestration.

def run_bulk_downloads() -> None:
    """Fetch the bulk reference files once."""
    _ensure_dirs()
    print(f"DATA_ROOT = {DATA_ROOT}")
    print("\n== UniProt ==")
    download_uniprot_proteome_fasta()
    download_uniprot_idmapping_and_xrefs()
    download_uniprot_features_gff()
    print("\n== SIFTS (summary index) ==")
    download_sifts_summary()
    print("\n== COSMIC ==")
    download_cosmic()
    print("\nBulk downloads complete. PDB, SIFTS XML, and AlphaFold files are "
          "fetched on demand later.")


if __name__ == "__main__":
    run_bulk_downloads()