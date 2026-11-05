# CFR Pipeline — Cancer Functional Regions

Finds regions of protein structures where cancer missense mutations are
**clustered in 3D more tightly than chance would predict** — candidate
functional / druggable sites. It maps COSMIC somatic mutations onto PDB and
AlphaFold structures (using UniProt + SIFTS as the shared coordinate system),
then scores each gene's structure with a **WAP** (Weighted Amino-acid
Proximity) statistic against a permutation null model, following the
CLUMPS / WAP approach of Kamburov et al. (2015).

The pipeline has two halves:

1. **Data preparation (Stages 0–2b)** — turn raw COSMIC/UniProt/SIFTS/PDB/
   AlphaFold inputs into two tidy tables.
2. **Scoring (WAP)** — compute the clustering statistic and a p-value per gene,
   with a CPU implementation and a GPU-accelerated one.

---

## Important: data is not included

**COSMIC is licensed and is never committed to this repository.** Nothing under
`cfr_data/` is redistributable, so you cannot simply clone and run. You must
obtain your own COSMIC download (free for academic/non-commercial use) and place
it by hand; the rest of the inputs download automatically. See
[Getting the data](#getting-the-data).

---

## Requirements

- **Python 3.10+**
- A virtual environment (recommended — see [Environment setup](#environment-setup))
- For GPU scoring: an NVIDIA GPU + a matching CUDA PyTorch build (optional; the
  CPU scorer needs no GPU and no PyTorch)

### Environment setup

Create and activate a virtual environment, then install dependencies.

**CPU only** (data stages + CPU scorer — no PyTorch, lightweight):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
```

**With GPU scoring** (adds PyTorch):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
# Then install the Torch build that matches your CUDA (check with `nvidia-smi`):
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# CUDA 11.8  -> .../whl/cu118      CPU-only torch -> .../whl/cpu
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Why is Torch installed separately?** The correct PyTorch wheel depends on
> your CUDA version, so it can't be pinned in a shared requirements file. The
> GPU scorer falls back to CPU automatically if no GPU is found.

---

## Where data lives

All data lives under one directory, chosen with an environment variable. This is
the **only** thing that changes between a laptop and a cloud machine, which is
what lets the pipeline run identically in both places.

```bash
export CFR_DATA_ROOT=$(pwd)/cfr_data     # any path you like
export CFR_RUN_TAG=run_2026-11-01        # output folder name; defaults to run_<today>
```

Layout created under `CFR_DATA_ROOT`:

```
cfr_data/
├── raw/          downloaded inputs, never edited
│   ├── cosmic/       <- you place the licensed COSMIC file here
│   ├── uniprot/      proteome FASTA, id-mapping, features (auto)
│   ├── sifts/        summary index + per-entry XML (auto)
│   ├── pdb/          mmCIF files, cached on demand (auto)
│   └── alphafold/    AF models, cached on demand (auto)
└── processed/
    └── run_<tag>/    all outputs for one run (TSVs + audit logs)
```

Every stage reads `CFR_DATA_ROOT` and `CFR_RUN_TAG` from the environment, so all
stages in one run write into the same `processed/run_<tag>/` folder.

---

## Getting the data

1. **COSMIC — manual, licensed.** Register at
   <https://cancer.sanger.ac.uk/cosmic/register>, accept the licence, and
   download the **Mutant Census** TSV (GRCh38). Place it exactly here:

   ```
   cfr_data/raw/cosmic/Cosmic_MutantCensus_Tsv_v102_GRCh38/Cosmic_MutantCensus_v102_GRCh38.tsv.gz
   ```

   (Running Stage 0 first creates the folder and leaves a `PUT_COSMIC_TSV_HERE.txt`
   marker to show you where it goes.) The Mutant Census covers the ~700 Cancer
   Gene Census genes; for genome-wide analysis swap in `Cosmic_CompleteMutantExport`.

2. **UniProt + SIFTS — automatic.** `stage0.py` downloads the human proteome
   FASTA, the id-mapping/xref table, feature annotations, and the SIFTS summary
   index into `cfr_data/raw/`.

3. **PDB / AlphaFold / per-entry SIFTS — automatic, on demand.** Fetched and
   cached by `stage2_map_coordinates.py`, only for genes that actually have
   mutations.

---

## Running the pipeline

### The easy way: `run.py`

`run.py` runs every stage in order, sharing one data root and one run tag:

```bash
# Full run, GPU scoring:
python3 run.py --backend gpu

# Full run, CPU scoring, explicit paths and 10^6 permutations:
python3 run.py --backend cpu --data-root $(pwd)/cfr_data --run-tag run_2026-11-01 --iterations 1000000

# Quick smoke test on 20 genes (raw inputs already present):
python3 run.py --skip-stage0 --limit 20

# Data stages only, no scoring:
python3 run.py --no-wap
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--data-root` | `./cfr_data` | sets `CFR_DATA_ROOT` for all stages |
| `--run-tag` | `run_<today>` | output folder under `processed/` |
| `--backend` | `cpu` | `cpu` → `wap_cpu.py`, `gpu` → `wap_gpu.py` |
| `--iterations` | `1000000` | permutations for the WAP null model |
| `--limit` | `0` (all) | cap genes/structures for a smoke test |
| `--skip-stage0` | off | skip the bulk download (inputs already cached) |
| `--no-wap` | off | stop after the data stages |

If any stage exits non-zero, `run.py` stops immediately.

### The manual way: stage by stage

Useful for debugging or re-running one stage. Set the environment variables
once, then run in order:

```bash
export CFR_DATA_ROOT=$(pwd)/cfr_data
export CFR_RUN_TAG=run_2026-11-01

python3 stage0.py                   # download bulk inputs (COSMIC placed by hand)
python3 stage1.py                   # clean COSMIC   -> stage1_mutations.tsv
python3 stage2a_structselect.py     # pick representative PDB structures
python3 stage2_map_coordinates.py   # map mutations to 3D coords (downloads structures)
python3 stage2b_residues.py         # per-residue "universe" table
python3 wap_cpu.py                  # or: python3 wap_gpu.py
```

Stage 2 is the long one (it downloads structures); it is resumable — re-running
skips anything already cached. `stage2_map_coordinates.py` takes an optional
positional gene limit (`python3 stage2_map_coordinates.py 20`); `stage2b_residues.py`
takes `--limit N` and `--branch {pdb,af,both}`.

---

## Stages and outputs

| Stage | Script | Output in `processed/run_<tag>/` |
|-------|--------|----------------------------------|
| 0  | `stage0.py` | bulk inputs downloaded into `raw/` |
| 1  | `stage1.py` | `stage1_mutations.tsv` — clean missense mutations |
| 2a | `stage2a_structselect.py` | `stage2a_representative_structures.tsv` |
| 2  | `stage2_map_coordinates.py` | `stage2_mapped.tsv`, `stage2_failures.tsv` |
| 2b | `stage2b_residues.py` | `stage2_residues.tsv` — per-residue universe |
| WAP | `wap_cpu.py` / `wap_gpu.py` | `WAP_results/` PDF + per-gene p-values |

The two tables the scorer consumes:

- **`stage2_mapped.tsv`** — one row per patient-mutation with its Cα coordinate
  (the *observed* mutations).
- **`stage2_residues.tsv`** — one row per resolved residue of each structure,
  flagged `is_mutated` with a `patient_count` (the *universe* the null model
  scatters mutations across).

Each stage also writes an `*_audit.txt` log recording what each filter kept or
dropped.

---

## Local vs. cloud

Nothing in the code is machine-specific. To run on a cloud VM, install the
requirements, point `CFR_DATA_ROOT` at that machine's disk, place COSMIC there,
and run exactly the same commands. Use `--backend gpu` on a GPU instance and
`--backend cpu` otherwise.

---

## Method notes

- **Mutation filtering (Stage 1):** missense only, confirmed-somatic only,
  HGVS-parseable, deduplicated so one patient sample counts a mutation once.
- **Two structure branches (Stage 2):** PDB (via SIFTS residue mapping) and
  AlphaFold (AF numbering already equals UniProt numbering; residues below
  pLDDT 70 are dropped).
- **WAP score:** for each gene, sums a Gaussian spatial weight over all mutated
  residue pairs, weighted by a saturating function of each residue's mutation
  count, then compares the observed score to a permutation null distribution to
  get a p-value.

Reference: Kamburov et al., *PNAS* 2015 — comprehensive assessment of cancer
missense mutation clustering in protein structures (CLUMPS).

---

## Troubleshooting

- **`COSMIC TSV not found`** — the licensed file isn't at the expected path; see
  [Getting the data](#getting-the-data).
- **`torch.cuda.is_available()` is False** — the installed Torch build doesn't
  match your CUDA, or there's no GPU. Reinstall Torch with the right index URL,
  or just use `--backend cpu`.
- **Stage 2 is slow / interrupted** — it's resumable; re-run the same command and
  it resumes from the cache.
- **`mutated residues not found in any universe` > 0 in the Stage 2b audit** —
  a cached structure file changed between Stage 2 and Stage 2b; re-run Stage 2.

---

## License

Code: see `LICENSE`. **Data is not covered** — COSMIC, UniProt, PDB, SIFTS and
AlphaFold each carry their own terms; obtain and use them under their respective
licences.