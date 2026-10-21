# CFR Pipeline — Cancer Functional Regions

Finds 3D-clustered cancer missense mutations in protein structures, as
candidate functional regions, from COSMIC mutations mapped onto PDB and
AlphaFold structures (UniProt + SIFTS as the shared coordinate system).

This repo holds the data-preparation stages (0–2b). They turn raw COSMIC,
UniProt, SIFTS, PDB and AlphaFold inputs into two tables the clustering model
consumes: `stage2_mapped.tsv` (mutations with 3D coordinates) and
`stage2_residues.tsv` (every resolved residue per structure, mutated or not).

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Where data lives

All data lives under one directory, set with an environment variable. This is
the only thing that changes between a laptop and a cloud machine. Nothing in
`cfr_data/` is committed to git.

```bash
export CFR_DATA_ROOT=$(pwd)/cfr_data     # or any path you like
export CFR_RUN_TAG=run_2026-10-19        # output folder name; defaults to today
```

Layout created under `CFR_DATA_ROOT`:

```
cfr_data/
├── raw/         downloaded inputs (never edited)
└── processed/   outputs, one folder per run tag
```

## Getting the data

Most inputs download automatically. COSMIC is licensed and must be fetched by
hand.

1. **COSMIC (manual, licensed).** Register at
   <https://cancer.sanger.ac.uk/cosmic/register>, accept the licence, and
   download the Mutant Census TSV (GRCh38). Place it at:
   `cfr_data/raw/cosmic/Cosmic_MutantCensus_Tsv_v102_GRCh38/Cosmic_MutantCensus_v102_GRCh38.tsv.gz`
   Do not commit or redistribute it.
2. **UniProt + SIFTS (automatic).** `stage0.py` downloads the UniProt human
   proteome FASTA, the id-mapping/xref table, feature tables, and the SIFTS
   summary index into `cfr_data/raw/`.
3. **PDB / AlphaFold / per-entry SIFTS (automatic, on demand).** Fetched and
   cached by `stage2_map_coordinates.py` only for the genes that have
   mutations.

## Running

Run the stages in order, from the repo directory, with the environment
variables set:

```bash
python3 stage0.py                   # download bulk inputs (COSMIC placed by hand)
python3 stage1.py                   # clean COSMIC -> stage1_mutations.tsv
python3 stage2a_structselect.py     # pick representative PDB structures
python3 stage2_map_coordinates.py   # map mutations to 3D coords (downloads structures)
python3 stage2b_residues.py         # per-residue "universe" table for the model
```

Each stage writes its outputs and an `*_audit.txt` log to
`cfr_data/processed/<run_tag>/`. Stage 2 is the long one; it is resumable, so
re-running skips anything already downloaded.

## Stages

| Stage | Script | Output |
|-------|--------|--------|
| 0  | `stage0.py` | downloaded inputs in `raw/` |
| 1  | `stage1.py` | `stage1_mutations.tsv` (clean missense mutations) |
| 2a | `stage2a_structselect.py` | `stage2a_representative_structures.tsv` |
| 2  | `stage2_map_coordinates.py` | `stage2_mapped.tsv`, `stage2_failures.tsv` |
| 2b | `stage2b_residues.py` | `stage2_residues.tsv` (per-residue universe) |

## Notes

- Mutation scope is the COSMIC Cancer Gene Census set (~700 genes). For
  genome-wide, swap in `Cosmic_CompleteMutantExport`.
- Structures come from two independent branches, PDB and AlphaFold; the
  AlphaFold branch drops residues below pLDDT 70.