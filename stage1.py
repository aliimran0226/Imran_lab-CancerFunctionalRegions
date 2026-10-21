"""
Stage 1 — COSMIC Mutant Census ingestion.

Reads the COSMIC Mutant Census TSV and writes a clean per-mutation table plus
an audit log of what each filter removed.

Input:  DATA_ROOT/raw/cosmic/Cosmic_MutantCensus_Tsv_v102_GRCh38/
            Cosmic_MutantCensus_v102_GRCh38.tsv.gz
Output: DATA_ROOT/processed/run_<date>/stage1_mutations.tsv
        DATA_ROOT/processed/run_<date>/stage1_audit.txt

Filters applied in order: missense only, confirmed-somatic only,
HGVS-parseable, then dedupe so a patient sample counts a mutation once.

This is the Cancer Gene Census file (~700 genes), not the whole genome; for
genome-wide use Cosmic_CompleteMutantExport instead.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ.get("CFR_DATA_ROOT", "./cfr_data")).resolve()

COSMIC_TSV = (
    DATA_ROOT / "raw" / "cosmic"
    / "Cosmic_MutantCensus_Tsv_v102_GRCh38"
    / "Cosmic_MutantCensus_v102_GRCh38.tsv.gz"
)

RUN_TAG = os.environ.get("CFR_RUN_TAG", f"run_{date.today().isoformat()}")
PROCESSED_DIR = DATA_ROOT / "processed" / RUN_TAG
STAGE1_OUT = PROCESSED_DIR / "stage1_mutations.tsv"
STAGE1_AUDIT = PROCESSED_DIR / "stage1_audit.txt"

USECOLS = [
    "GENE_SYMBOL",
    "COSMIC_SAMPLE_ID",
    "TRANSCRIPT_ACCESSION",
    "MUTATION_AA",
    "MUTATION_DESCRIPTION",
    "MUTATION_SOMATIC_STATUS",
]

# Drop "Variant of unknown origin" — no matched normal, so possibly germline.
KEEP_SOMATIC_STATUS = {
    "Confirmed somatic variant",
    "Reported in another cancer sample as somatic",
}

KEEP_MUTATION_TYPE = {"missense_variant"}

# p.<Ref><Pos><Mut>, e.g. p.S1581L
HGVS_SHORT_RE = re.compile(r"^p\.([A-Z])(\d+)([A-Z*])$")


class Audit:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        print(msg)
        self.lines.append(msg)

    def step(self, label: str, before: int, after: int) -> None:
        dropped = before - after
        pct = (dropped / before * 100) if before else 0
        self.log(f"  {label}: {before:>10,} -> {after:>10,}  "
                 f"(dropped {dropped:>9,}, {pct:5.2f}%)")

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def parse_hgvs_short(s: str) -> tuple[str, int, str] | None:
    """Parse 'p.S1581L' -> ('S', 1581, 'L'); None if it doesn't match."""
    if not isinstance(s, str):
        return None
    m = HGVS_SHORT_RE.match(s.strip())
    if not m:
        return None
    ref, pos_str, mut = m.groups()
    if mut == "*" or ref == mut:
        return None
    return ref, int(pos_str), mut


def run_stage1() -> Path:
    audit = Audit()
    audit.log("Stage 1 — COSMIC ingestion")
    audit.log(f"DATA_ROOT = {DATA_ROOT}")
    audit.log(f"Input     = {COSMIC_TSV}")
    audit.log(f"Run tag   = {RUN_TAG}")
    audit.log(f"Output    = {STAGE1_OUT}")
    audit.log("")

    if not COSMIC_TSV.exists():
        raise FileNotFoundError(
            f"COSMIC TSV not found at {COSMIC_TSV}. "
            f"Run Stage 0 and drop the licensed file in place."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    audit.log("Loading COSMIC TSV...")
    df = pd.read_csv(
        COSMIC_TSV,
        sep="\t",
        usecols=USECOLS,
        dtype=str,
        na_filter=False,
        low_memory=False,
    )
    n_raw = len(df)
    audit.log(f"  loaded {n_raw:,} rows, {len(df.columns)} cols")
    audit.log("")

    audit.log("Filtering:")
    before = len(df)
    df = df[df["MUTATION_DESCRIPTION"].isin(KEEP_MUTATION_TYPE)]
    audit.step("missense-only              ", before, len(df))

    before = len(df)
    df = df[df["MUTATION_SOMATIC_STATUS"].isin(KEEP_SOMATIC_STATUS)]
    audit.step("confirmed-somatic status   ", before, len(df))

    parsed = df["MUTATION_AA"].map(parse_hgvs_short)
    before = len(df)
    keep = parsed.notna()
    df = df[keep].copy()
    parsed = parsed[keep]
    audit.step("HGVS parseable             ", before, len(df))

    df["ref_aa"] = [p[0] for p in parsed]
    df["position"] = [p[1] for p in parsed]
    df["mut_aa"] = [p[2] for p in parsed]

    # same sample reporting the same mutation twice counts once
    before = len(df)
    df = df.drop_duplicates(
        subset=["GENE_SYMBOL", "COSMIC_SAMPLE_ID", "position", "mut_aa"],
        keep="first",
    )
    audit.step("deduped (gene,sample,pos)  ", before, len(df))
    audit.log("")

    out = df.rename(columns={
        "GENE_SYMBOL": "gene",
        "COSMIC_SAMPLE_ID": "sample_id",
        "TRANSCRIPT_ACCESSION": "transcript",
    })[["gene", "sample_id", "transcript", "position", "ref_aa", "mut_aa"]]

    n_final = len(out)
    audit.log("Final:")
    audit.log(f"  {n_final:>10,} clean missense mutations")
    audit.log(f"  {out['gene'].nunique():>10,} distinct genes")
    audit.log(f"  {out['sample_id'].nunique():>10,} distinct patient samples")
    audit.log(f"  overall retention: {n_final / n_raw * 100:5.2f}% of raw rows")
    audit.log("")

    out.to_csv(STAGE1_OUT, sep="\t", index=False)
    audit.log(f"Wrote {STAGE1_OUT}")
    audit.save(STAGE1_AUDIT)
    audit.log(f"Wrote {STAGE1_AUDIT}")
    return STAGE1_OUT


if __name__ == "__main__":
    run_stage1()