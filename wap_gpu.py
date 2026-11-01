"""
WAP (Weighted Amino-acid Proximity) Score Calculator
=====================================================
Bioinformatics tool for calculating the WAP score — a spatial clustering
metric that measures whether cancer somatic mutations are more spatially
concentrated on a protein structure than expected by chance.

Usage
-----
Import this module in another Python file or Jupyter notebook:

    import WAP as wap

    structure_map = wap.load_structure_map_from_pdb("protein.pdb", target_chain="A")
    distance_matrix = wap.calculate_distance_matrix(structure_map)
    exp_matrix = wap.create_expotential_map(distance_matrix)
    res_to_index = wap.build_index_map(structure_map)
    nq_values = wap.calculate_Nq_values_modified_parameters(m=3)
    observed_score = wap.calculate_WAP_Score(exp_matrix, cancer_data, res_to_index, nq_values)

Run the bundled multi-gene pipeline directly::

    python WAP.py

Dependencies
------------
biopython, numpy, pandas, scipy, matplotlib, seaborn, tabulate, requests
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import csv
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from itertools import permutations
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import PDBParser
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base URL for the UniProt REST API
UNIPROT_BASE = "https://rest.uniprot.org"

# Polite delay (seconds) between successive UniProt API calls
SLEEP_BETWEEN_REQUESTS = 0.5

# Mapping from PDB three-letter residue codes to single-letter amino-acid codes.
# Includes standard amino acids plus selenomethionine (MSE), selenocysteine (SEC),
# and pyrrolysine (PYL).
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",  # Selenomethionine → Methionine
    "SEC": "U",  # Selenocysteine
    "PYL": "O",  # Pyrrolysine
}


# ============================================================================
#  UTILITY / DISPLAY
# ============================================================================


def print_with_tabulate(data: dict) -> None:
    """Pretty-print a dictionary as a two-column grid table.

    Parameters
    ----------
    data : dict
        Any dictionary whose keys and values can be cast to strings.
    """
    table_data = [[key, value] for key, value in data.items()]
    print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))


# ============================================================================
#  DATA LOADING — UniProt / PDB
# ============================================================================


def gene_to_accession(gene_name: str, organism_id: int = 9606) -> str | None:
    """Map a gene symbol to a reviewed (Swiss-Prot) UniProt accession.

    Queries the UniProt REST API and returns the primary accession of the
    first reviewed entry that matches the gene name and organism.

    Parameters
    ----------
    gene_name : str
        HGNC gene symbol, e.g. ``"TP53"``.
    organism_id : int, optional
        NCBI taxonomy ID (default ``9606`` = *Homo sapiens*).

    Returns
    -------
    str or None
        UniProt accession string (e.g. ``"P04637"``), or ``None`` if
        no reviewed entry is found or the request fails.
    """
    url = f"{UNIPROT_BASE}/uniprotkb/search"
    params = {
        "query": (
            f"gene_exact:{gene_name} AND "
            f"organism_id:{organism_id} AND "
            "reviewed:true"
        ),
        "fields": "accession,gene_names",
        "format": "json",
        "size": 5,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"UniProt ID mapping failed for {gene_name}: {exc}")
        return None

    # Respect the UniProt API rate limit
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    results = resp.json().get("results", [])
    if not results:
        print(f"No UniProt entry found for gene '{gene_name}'")
        return None

    return results[0]["primaryAccession"]


def fetch_uniprot_sequence(accession: str) -> str:
    """Download the canonical FASTA sequence for a UniProt accession.

    Parameters
    ----------
    accession : str
        UniProt accession, e.g. ``"P04637"``.

    Returns
    -------
    str
        Single-letter amino-acid sequence (no header line, no newlines).

    Raises
    ------
    Exception
        If the HTTP request fails (non-200 status code).
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch UniProt sequence for {accession}. "
            f"Status: {response.status_code}"
        )

    # Strip the FASTA header line; concatenate the remaining sequence lines
    lines = response.text.strip().split("\n")
    return "".join(line.strip() for line in lines[1:])


def get_chain_sequence_and_keys(pdb_file_path: str, target_chain: str = "A"):
    """Extract the resolved one-letter sequence and residue keys for one chain.

    Only standard ATOM records (not HETATM) are included. Residue keys follow
    the format ``"<res_seq><icode>"`` (e.g. ``"127"`` or ``"127A"``).

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file.
    target_chain : str, optional
        Chain identifier to extract (default ``"A"``).

    Returns
    -------
    seq : str
        One-letter amino-acid sequence in file order.
        Unknown residues are represented by ``"X"``.
    ordered_keys : list of str
        Residue keys matching each position in ``seq``.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Protein", pdb_file_path)

    seq_chars, ordered_keys = [], []

    for model in structure:
        for chain in model:
            if chain.get_id() != target_chain:
                continue
            for residue in chain:
                # Skip HETATM records (water, ligands, etc.)
                if residue.id[0] != " ":
                    continue
                one_letter = THREE_TO_ONE.get(residue.get_resname(), "X")
                seq_chars.append(one_letter)
                icode = residue.id[2].strip()
                ordered_keys.append(f"{residue.id[1]}{icode}")
        break  # Use only the first model

    return "".join(seq_chars), ordered_keys


def map_uniprot_to_pdb_by_sequence(
    pdb_file_path: str, target_chain: str, uniprot_accession: str
) -> tuple[dict, dict, int, int]:
    """Align the PDB chain sequence to the UniProt canonical sequence.

    Uses global pairwise alignment (BLOSUM62) to map UniProt residue numbers
    to PDB residue keys. This is robust to engineered mutations, truncations,
    and insertion codes.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file.
    target_chain : str
        Chain identifier to align (e.g. ``"A"``).
    uniprot_accession : str
        UniProt accession for the protein (e.g. ``"P04637"``).

    Returns
    -------
    unp_to_pdb_key : dict
        ``{ uniprot_residue_number_str : pdb_residue_key_str }``
    aa_visual_map : dict
        ``{ uniprot_residue_number_str : {"pdb_res_id", "unp_aa", "pdb_aa"} }``
        Useful for visually confirming the alignment quality.
    len_uniprot : int
        Total length of the UniProt canonical sequence.
    len_pdb : int
        Number of resolved residues in the PDB chain.
    """
    pdb_seq, ordered_keys = get_chain_sequence_and_keys(pdb_file_path, target_chain)
    print(f"Length of the PDB sequence: {len(pdb_seq)}")

    unp_seq = fetch_uniprot_sequence(uniprot_accession)
    print(f"Length of the UniProt sequence: {len(unp_seq)}")

    # Configure the aligner for end-gap-free global alignment (semi-global style)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.end_insertion_score = 0
    aligner.end_deletion_score = 0

    alignment = aligner.align(unp_seq, pdb_seq)[0]

    unp_to_pdb_key: dict = {}
    aa_visual_map: dict = {}
    mismatches = 0

    for (t_start, t_end), (q_start, q_end) in zip(*alignment.aligned):
        for offset in range(t_end - t_start):
            unp_index = t_start + offset
            pdb_index = q_start + offset

            unp_aa = unp_seq[unp_index]
            pdb_aa = pdb_seq[pdb_index]

            if unp_aa != pdb_aa:
                mismatches += 1

            unp_num_str = str(unp_index + 1)
            pdb_res_key = ordered_keys[pdb_index]

            # Map UniProt residue number → PDB residue key (used in calculations)
            unp_to_pdb_key[unp_num_str] = pdb_res_key

            # Visual confirmation map (amino-acid level)
            aa_visual_map[unp_num_str] = {
                "pdb_res_id": pdb_res_key,
                "unp_aa": unp_aa,
                "pdb_aa": pdb_aa,
            }

    status = (
        "clean"
        if mismatches == 0
        else "check for engineered mutations in this construct"
    )
    print(
        f"Sequence alignment: {len(unp_to_pdb_key)} residues mapped, "
        f"{mismatches} mismatches ({status})"
    )

    return unp_to_pdb_key, aa_visual_map, len(unp_seq), len(pdb_seq)


def load_structure_map_from_pdb(pdb_file_path: str, target_chain: str = "A") -> dict:
    """Load Cα (alpha-carbon) 3D coordinates for one chain from a PDB file.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file.
    target_chain : str, optional
        Chain identifier (default ``"A"``).

    Returns
    -------
    dict
        ``{ residue_number_str : [x, y, z] }``
        Only residues with a resolved Cα atom are included.
        Returns an empty dict if the file is not found.
    """
    structure_map: dict = {}
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("Protein", pdb_file_path)

        for model in structure:
            for chain in model:
                if chain.get_id() != target_chain:
                    continue
                for residue in chain:
                    # Skip HETATM records
                    if residue.id[0] != " ":
                        continue
                    if "CA" in residue:
                        res_num = str(residue.id[1])
                        coord = residue["CA"].get_coord()
                        structure_map[res_num] = [
                            float(coord[0]),
                            float(coord[1]),
                            float(coord[2]),
                        ]

        return structure_map

    except FileNotFoundError:
        print(f"Error: PDB File not found -> {pdb_file_path}")
        return {}


def get_mutation_counts_from_tsv(tsv_filepath: str, gene_name: str) -> dict:
    """Read mutation counts for a gene from a mapped TSV file.

    Parameters
    ----------
    tsv_filepath : str
        Path to the TSV file (tab-separated, must have ``"gene"`` and
        ``"position"`` columns).
    gene_name : str
        Gene symbol to filter for (e.g. ``"TP53"``).

    Returns
    -------
    dict
        ``{ residue_position : mutation_count }``
        Returns an empty dict if the gene is not found.
    """
    df = pd.read_csv(tsv_filepath, sep="\t")
    gene_df = df[df["gene"] == gene_name]

    if gene_df.empty:
        print(f"Warning: No data found for {gene_name} in the TSV file.")
        return {}

    # Count how many times each residue position appears (= number of observed mutations)
    return gene_df["position"].value_counts().to_dict()


def calculate_known_data(mutation_count_csv_path: str) -> dict:
    """Parse a COSMIC-style CSV and return missense mutation counts per position.

    The CSV must contain ``"Type"``, ``"Position"``, and ``"Count"`` columns.
    Only rows with ``Type == "Substitution - Missense"`` are included.

    Parameters
    ----------
    mutation_count_csv_path : str
        Path to the mutation-count CSV file.

    Returns
    -------
    dict
        ``{ residue_position_str : total_mutation_count }``
    """
    df = pd.read_csv(mutation_count_csv_path)

    # Keep only missense substitutions
    filtered = df[df["Type"] == "Substitution - Missense"]

    # Aggregate counts per position (multiple CSV rows may share a position)
    final_count = filtered.groupby("Position", as_index=False)["Count"].sum()

    keys = final_count["Position"].astype(int).astype(str)
    values = final_count["Count"].astype(int)

    return dict(zip(keys, values))


def create_structure_map_from_tsv(
    tsv_filepath: str, gene_name: str, structure_source: str
) -> dict:
    """Build a 3D coordinate map from a residues TSV file.

    The TSV must contain ``"gene"``, ``"structure_source"``, ``"uniprot_pos"``,
    ``"x"``, ``"y"``, and ``"z"`` columns.

    Parameters
    ----------
    tsv_filepath : str
        Path to the residues TSV (e.g. ``stage2_residues.tsv``).
    gene_name : str
        Gene symbol to filter for.
    structure_source : str
        Specific structure source identifier to filter for (e.g. a PDB ID).
        Should match the source chosen from the mapped TSV for consistency.

    Returns
    -------
    dict
        ``{ uniprot_position_int : [x, y, z] }``
        Returns an empty dict if no matching rows are found.
    """
    df = pd.read_csv(tsv_filepath, sep="\t")

    # Filter for the requested gene and structure source
    structure_df = df[
        (df["gene"] == gene_name) & (df["structure_source"] == structure_source)
    ]

    if structure_df.empty:
        print(
            f"Error: No residues found for gene '{gene_name}' "
            f"with structure source '{structure_source}'."
        )
        return {}

    # Deduplicate by UniProt position; drop rows with missing coordinates
    coords_df = (
        structure_df[["uniprot_pos", "x", "y", "z"]]
        .dropna()
        .drop_duplicates(subset=["uniprot_pos"])
    )

    aligned_structure_map: dict = {}
    for _, row in coords_df.iterrows():
        pos = int(row["uniprot_pos"])
        aligned_structure_map[pos] = [float(row["x"]), float(row["y"]), float(row["z"])]

    return aligned_structure_map


# ============================================================================
#  DISTANCE / EXPONENTIAL MATRICES
# ============================================================================


def calculate_distance_matrix(structure_map: dict) -> np.ndarray:
    """Compute the squared Euclidean distance matrix for all Cα pairs.

    Parameters
    ----------
    structure_map : dict
        ``{ residue_id : [x, y, z] }`` as returned by
        :func:`load_structure_map_from_pdb` or :func:`create_structure_map_from_tsv`.

    Returns
    -------
    numpy.ndarray
        Symmetric *N × N* matrix of squared distances (Å²),
        where *N* is the number of residues.
    """
    """Vectorized squared Euclidean distance matrix for all Cα pairs (100x faster than nested loops)."""

    coords = np.array(list(structure_map.values()), dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def build_index_map(structure_map: dict) -> dict:
    """Build a residue-ID → matrix-index lookup dictionary.

    Parameters
    ----------
    structure_map : dict
        ``{ residue_id : coordinates }`` — keys must be in the same order
        as used when building the distance matrix.

    Returns
    -------
    dict
        ``{ residue_id : row_or_column_index }``
    """
    return {res_id: idx for idx, res_id in enumerate(structure_map.keys())}


def create_expotential_map(distance_matrix: np.ndarray, sigma: float = 6) -> np.ndarray:
    """Convert a squared-distance matrix into a Gaussian spatial-weight matrix.

    Computes ``exp(-d² / (2σ²))`` element-wise, giving large weights to
    nearby residue pairs and small weights to distant ones.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Symmetric *N × N* matrix of squared Cα distances (Å²).
    sigma : float, optional
        Gaussian width parameter in Ångströms (default ``6``).

    Returns
    -------
    numpy.ndarray
        *N × N* matrix of spatial weights in ``[0, 1]``.
    """
    return np.exp(-distance_matrix / (2 * (sigma**2)))


# ============================================================================
#  N(q) SATURATION FUNCTIONS
# ============================================================================


def calculate_Nq_values(known_cancer_data: dict) -> dict:
    """Compute Hill-function N(q) values for observed mutation counts.

    Uses the standard Hill equation with n=3 and half-saturation at q=2:
    ``N(q) = q³ / (2³ + q³)``

    Parameters
    ----------
    known_cancer_data : dict
        ``{ residue_id : mutation_count }``

    Returns
    -------
    dict
        ``{ mutation_count : N(q) }`` (keyed by count, not residue ID).
    """
    return {
        count: (count**3) / (2**3 + count**3)
        for res_id, count in known_cancer_data.items()
    }


def calculate_Nq_values_modified_parameters(m: float) -> dict:
    """Precompute Hill-function N(q) values with a tuneable Hill coefficient.

    ``N(q) = qᵐ / (2ᵐ + qᵐ)``

    Parameters
    ----------
    m : float
        Hill coefficient (exponent). Common values: ``2`` (softer
        saturation) or ``3`` (original formulation).

    Returns
    -------
    dict
        ``{ count : N(q) }`` for counts 0 … 5000.
        Count 0 maps to 0.0 (no mutations → no weight).
    """
    result = {0: 0.0}
    for i in range(1, 5001):
        result[i] = (i**m) / (2**m + i**m)
    return result


def calculate_Nq_values_logrithmic() -> dict:
    """Precompute logistic-sigmoid N(q) values centred at q=2.

    ``N(q) = 1 / (1 + exp(-1.5 × (q - 2)))``

    Returns
    -------
    dict
        ``{ count : N(q) }`` for counts 0 … 5000.
    """
    result = {0: 0.0}
    for i in range(1, 5001):
        result[i] = 1 / (1 + math.exp(-1.5 * (i - 2)))
    return result


def generate_logistic_nq_dict() -> dict:
    """Precompute log-cubic Hill N(q) values.

    ``N(q) = ln(q)³ / (2³ + ln(q)³)``

    Returns
    -------
    dict
        ``{ count : N(q) }`` for counts 0 … 5000.
        Count 0 maps to 0.0.
    """
    result = {}
    for count in range(5001):
        if count > 0:
            log_count = math.log(count) ** 3
            result[count] = log_count / (2**3 + log_count)
        else:
            result[count] = 0.0
    return result


def calculate_new_nq_values(known_cancer_data: dict) -> dict:
    """Compute N(q) for every integer up to the total observed mutation count.

    ``N(q) = q³ / (8 + q³)``

    Parameters
    ----------
    known_cancer_data : dict
        ``{ residue_id : mutation_count }``

    Returns
    -------
    dict
        ``{ count : N(q) }`` for counts 1 … total_mutations.
    """
    total_mutation_count = sum(known_cancer_data.values())
    return {
        count: (count**3) / (8.0 + count**3)
        for count in range(1, total_mutation_count + 1)
    }


# ============================================================================
#  WAP SCORE CALCULATION
# ============================================================================


def get_pairs(keys) -> list:
    """Return all ordered pairs (permutations of size 2) from *keys*.

    Parameters
    ----------
    keys : iterable
        Residue identifiers.

    Returns
    -------
    list of tuple
        All ``(res_i, res_j)`` pairs where ``i ≠ j``.
    """
    return list(permutations(keys, 2))


def calculate_WAP_Score(
    expotential_matrix: np.ndarray,
    known_cancer_data_dict: dict,
    res_to_index: dict,
    Nq_values: dict,
) -> float:
    """Calculate the observed WAP score for a set of mutated residues.

    The WAP score sums over all ordered residue pairs:

        WAP = Σ_{i≠j} N(q_i) · N(q_j) · exp(-d²_{ij} / 2σ²)

    Parameters
    ----------
    expotential_matrix : numpy.ndarray
        Precomputed Gaussian weight matrix (from :func:`create_expotential_map`).
    known_cancer_data_dict : dict
        ``{ residue_id : mutation_count }`` for mutated residues only.
    res_to_index : dict
        ``{ residue_id : matrix_index }`` from :func:`build_index_map`.
    Nq_values : dict
        ``{ mutation_count : N(q) }`` from any ``calculate_Nq_*`` function.

    Returns
    -------
    float
        The WAP score for the observed mutation pattern.
    """
    pairs = get_pairs(known_cancer_data_dict.keys())
    wap_score = 0.0

    for res1, res2 in pairs:
        N1 = Nq_values[known_cancer_data_dict[res1]]
        N2 = Nq_values[known_cancer_data_dict[res2]]
        exp_term = expotential_matrix[res_to_index[res1]][res_to_index[res2]]
        wap_score += N1 * N2 * exp_term

    return wap_score


def calculate_p_value(random_scores: list, observed_score: float) -> float:
    """Estimate the empirical p-value from a permutation null distribution.

    The p-value is the fraction of random scores that are ≥ the observed score.

    Parameters
    ----------
    random_scores : list of float
        WAP scores generated under the null hypothesis.
    observed_score : float
        WAP score computed from the real mutation data.

    Returns
    -------
    float
        Empirical p-value in ``[0, 1]``.
    """
    arr = np.asarray(random_scores)
    return float(np.mean(arr >= observed_score))


# ============================================================================
#  PERMUTATION / SIMULATION FUNCTIONS
# ============================================================================


def generate_permutation_scores(
    structure_map: dict,
    expotential_map: np.ndarray,
    res_to_index: dict,
    nq_values: dict,
    observed_counts_dict: dict,
    iterations: int = 10**6,
) -> list:
    """Generate a null distribution of WAP scores by shuffling mutation positions.

    For each iteration, the observed mutation counts are randomly assigned to
    residues drawn without replacement from the full structure map.

    Parameters
    ----------
    structure_map : dict
        ``{ residue_id : coordinates }`` — defines the universe of residues.
    expotential_map : numpy.ndarray
        Precomputed Gaussian weight matrix.
    res_to_index : dict
        Residue-ID → matrix-index lookup.
    nq_values : dict
        Precomputed N(q) saturation values.
    observed_counts_dict : dict
        ``{ residue_id : mutation_count }`` — the real observed data.
    iterations : int, optional
        Number of random permutations (default 10⁶).

    Returns
    -------
    list of float
        Random WAP scores forming the null distribution.
    """
    structure_residues = list(structure_map.keys())
    counts_values = list(observed_counts_dict.values())
    random_scores = []

    print(f"Running simulation ({iterations} iterations)...")

    for _ in range(iterations):
        # Pick random positions and shuffle the mutation counts independently
        random_residues = random.sample(structure_residues, len(counts_values))
        shuffled_counts = random.sample(counts_values, len(counts_values))

        random_data_dict = dict(zip(random_residues, shuffled_counts))
        score = calculate_WAP_Score(
            expotential_map, random_data_dict, res_to_index, nq_values
        )
        random_scores.append(score)

    return random_scores


import torch

def generate_permutation_scores_vectorized(
    exp_matrix: np.ndarray,
    nq_values: dict,
    known_cancer_mutation_dict: dict,
    iterations: int = 10**6,
    batch_size: int = None,
) -> list:
    """VRAM-safe, GPU-accelerated null-distribution generator using PyTorch cuBLAS."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = exp_matrix.shape[0]
    counts_values = list(known_cancer_mutation_dict.values())
    k = len(counts_values)

    # 1. Move weights to GPU
    n_vals = np.array([nq_values[c] for c in counts_values], dtype=np.float32)
    n_tensor = torch.tensor(n_vals, device=device)  # Shape: (k,)

    # 2. Move exponential matrix to GPU and zero out the diagonal
    E = torch.tensor(exp_matrix, dtype=torch.float32, device=device)
    E.fill_diagonal_(0.0)

    # 3. Ultra-safe batch sizing: targets ~500 MB max footprint per batch
    # Formula: B * N <= 25,000,000 floats
    safe_batch_size = max(1000, min(50000, int(2.5e7 / max(1, N))))

    random_scores = np.empty(iterations, dtype=np.float32)
    filled = 0

    with torch.no_grad():
        while filled < iterations:
            B = min(safe_batch_size, iterations - filled)

            # --- OPTIMIZATION: topk() replaces argsort() ---
            # Finds top k indices directly without allocating a massive (B, N) sort matrix
            rand_idx = torch.rand((B, N), device=device).topk(k, dim=1).indices

            # Scatter the k weights into a 2D weight matrix W of shape (B, N)
            W = torch.zeros((B, N), dtype=torch.float32, device=device)
            W.scatter_(1, rand_idx, n_tensor.unsqueeze(0).expand(B, k))

            # Compute WAP = row_sum((W @ E) * W) via hardware cuBLAS GEMM
            scores_batch = torch.sum((W @ E) * W, dim=1)

            random_scores[filled : filled + B] = scores_batch.cpu().numpy()
            filled += B

            # Aggressive cleanup for 4GB VRAM cards
            del rand_idx, W, scores_batch

    # Clear residual cache before the next gene
    torch.cuda.empty_cache()

    return random_scores  # (Returning as fast NumPy array)

def calculate_new_mutation_count(known_cancer_data: dict, valid_residues: list) -> dict:
    """Randomly redistribute the total observed mutations across valid residues.

    Samples residues *with* replacement (multinomial-like), preserving the
    total number of mutations but not the number of mutated positions.

    Parameters
    ----------
    known_cancer_data : dict
        ``{ residue_id : mutation_count }`` — original observed data.
    valid_residues : list
        Residue IDs that have resolved 3D coordinates (the sampling universe).

    Returns
    -------
    dict
        ``{ residue_id : new_mutation_count }`` — only residues that received
        at least one mutation are included.
    """
    total_count = sum(known_cancer_data.values())
    new_mutated_residues = random.choices(valid_residues, k=total_count)
    return dict(Counter(new_mutated_residues))


def generate_mutation_technique_verification(
    structure_map: dict,
    expotential_map: np.ndarray,
    res_to_index: dict,
    start_res: int,
    end_res: int,
    total_mutations: int,
    nq_values_dict: dict,
    iterations: int = 10**6,
    batch_size: int = 100000,
) -> list:
    """Vectorized multinomial verification of the null distribution.

    Distributes ``total_mutations`` across residues in ``[start_res, end_res]``
    using a uniform multinomial, then computes the WAP score for each draw.
    Used to verify that the main permutation test is correctly calibrated.

    Parameters
    ----------
    structure_map : dict
        ``{ residue_id : coordinates }`` — only residues in range are used.
    expotential_map : numpy.ndarray
        *M × M* Gaussian weight matrix (M = number of residues in structure).
    res_to_index : dict
        Residue-ID → matrix-index lookup.
    start_res : int
        First residue position in the valid range (inclusive).
    end_res : int
        Last residue position in the valid range (inclusive).
    total_mutations : int
        Total number of mutations to distribute per iteration.
    nq_values_dict : dict
        Precomputed N(q) lookup dictionary.
    iterations : int, optional
        Number of verification draws (default 10⁶).
    batch_size : int, optional
        Batch size for memory-safe processing (default 100 000).

    Returns
    -------
    list of float
        Verification WAP scores.
    """
    # Identify and index residues within the valid range
    valid_residues = [
        res for res in structure_map.keys() if start_res <= int(res) <= end_res
    ]
    num_valid = len(valid_residues)
    valid_indices = [res_to_index[res] for res in valid_residues]
    M = len(structure_map)

    # Build a fast NumPy lookup array from the N(q) dictionary
    max_q = max(nq_values_dict.keys())
    nq_lookup_array = np.zeros(max_q + 1)
    for q, val in nq_values_dict.items():
        nq_lookup_array[q] = val

    print(f"\nRunning Vectorized Verification ({iterations} iterations)...")
    print(f"Logic: Uniform multinomial across {num_valid} valid residues.")
    print(f"Using precomputed N(q) dictionary (max supported count: {max_q}).")

    random_scores = np.zeros(iterations)
    pvals = np.ones(num_valid) / num_valid
    sum_of_max_counts = 0

    for i in range(0, iterations, batch_size):
        current_batch = min(batch_size, iterations - i)

        # Draw mutation counts for this batch from a uniform multinomial
        Q_valid = np.random.multinomial(total_mutations, pvals, size=current_batch)
        sum_of_max_counts += np.sum(np.max(Q_valid, axis=1))

        # Map valid-residue counts into the full structural matrix
        Q_full = np.zeros((current_batch, M), dtype=int)
        Q_full[:, valid_indices] = Q_valid

        # Cap counts at max_q to prevent index-out-of-bounds in the lookup array
        Q_full = np.clip(Q_full, 0, max_q)

        # Vectorized N(q) lookup using array indexing
        N_matrix = nq_lookup_array[Q_full]

        # WAP = (N @ E) · N summed per row, minus the diagonal self-terms
        term1 = np.sum((N_matrix @ expotential_map) * N_matrix, axis=1)
        term2 = np.sum(N_matrix**2, axis=1)
        random_scores[i : i + current_batch] = term1 - term2

    average_max_count = sum_of_max_counts / iterations
    print("\n" + "=" * 60)
    print(" VERIFICATION COMPLETE")
    print("=" * 60)
    print(f" Total Iterations    : {iterations}")
    print(f" Avg Max Mutation    : {average_max_count:.2f}")
    print("=" * 60 + "\n")

    return random_scores.tolist()


def generate_new_permutation_scores(
    structure_map: dict,
    expotential_map: np.ndarray,
    res_to_index: dict,
    nq_values: dict,
    start_res: int,
    end_res: int,
    known_cancer_data_dict: dict,
    iterations: int = 10**6,
    batch_size: int = 100000,
) -> list:
    """Multinomial null-distribution generator with a missing-loop probability sink.

    Distributes mutations across *M* resolved residues plus one "missing loop"
    bucket (absorbs mutations that fall in unresolved regions). Ensures that
    the total mutation count is conserved in every draw.

    Parameters
    ----------
    structure_map : dict
        ``{ residue_id : coordinates }`` — resolved residues only.
    expotential_map : numpy.ndarray
        *M × M* Gaussian weight matrix.
    res_to_index : dict
        Residue-ID → matrix-index lookup.
    nq_values : dict
        Precomputed N(q) lookup dictionary.
    start_res : int
        First position of the protein region (inclusive).
    end_res : int
        Last position of the protein region (inclusive).
    known_cancer_data_dict : dict
        ``{ residue_id : mutation_count }`` — real observed data.
    iterations : int, optional
        Number of permutations (default 10⁶).
    batch_size : int, optional
        Batch size for memory-safe processing (default 100 000).

    Returns
    -------
    list of float
        Null-distribution WAP scores.

    Raises
    ------
    ValueError
        If structural residues lie outside ``[start_res, end_res]``, or if
        the structure is larger than the specified window.
    """
    print(
        f"\nRunning Vectorized Simulation ({iterations} iterations) "
        f"for range {start_res}-{end_res}..."
    )

    # Count only mutations that fall within the specified window
    total_mutations = sum(
        count
        for res, count in known_cancer_data_dict.items()
        if start_res <= int(res) <= end_res
    )

    window_length = end_res - start_res + 1
    M = len(structure_map)

    if M > window_length:
        raise ValueError(
            f"Error: Number of valid 3D residues (M={M}) exceeds "
            f"the defined window length ({window_length})."
        )

    out_of_bounds = [
        res for res in structure_map.keys() if not (start_res <= int(res) <= end_res)
    ]
    if out_of_bounds:
        raise ValueError(
            f"Found structural residues outside {start_res}-{end_res}. "
            "Please filter structure_map first."
        )

    random_scores = np.zeros(iterations)

    # Probability vector: M resolved positions + 1 "missing loop" sink
    pvals = np.ones(M + 1) / window_length
    pvals[-1] = (window_length - M) / window_length  # fraction of unresolved sequence

    for i in range(0, iterations, batch_size):
        current_batch_size = min(batch_size, iterations - i)

        # Draw from multinomial; last column is the missing-loop sink
        Q_matrix_full = np.random.multinomial(
            total_mutations, pvals, size=current_batch_size
        )
        Q_matrix = Q_matrix_full[:, :-1]  # Drop the sink column

        # Integrity check: every run must equal total_mutations
        total_per_run = np.sum(Q_matrix_full, axis=1)
        if not np.all(total_per_run == total_mutations):
            raise ValueError(
                "Simulation Error: A run did not match the expected total mutations!"
            )

        # Print a brief debug summary for the very first batch only
        if i == 0:
            unique_mutated = np.count_nonzero(Q_matrix, axis=1)
            print(f"\n--- [DEBUG] First 5 runs (Target Total: {total_mutations}) ---")
            for run_idx in range(5):
                print(
                    f"Run {run_idx + 1}: "
                    f"{unique_mutated[run_idx]} distinct residues, "
                    f"{np.sum(Q_matrix[run_idx])} in structure, "
                    f"{Q_matrix_full[run_idx, -1]} in missing loops"
                )
            print(
                f"All {iterations} runs verified to maintain "
                f"{total_mutations} mutations.\n"
            )

        # Vectorized N(q) using Hill equation directly (avoids dictionary lookup)
        N_matrix = (Q_matrix**3) / (8.0 + Q_matrix**3)

        # WAP = (N @ E) · N - Σ N_i²  (subtract diagonal self-terms)
        term1 = np.sum((N_matrix @ expotential_map) * N_matrix, axis=1)
        term2 = np.sum(N_matrix**2, axis=1)
        random_scores[i : i + current_batch_size] = term1 - term2

    return random_scores.tolist()


def generate_new_permutation_scores_2(
    structure_map: dict,
    expotential_map: np.ndarray,
    res_to_index: dict,
    nq_values: dict,
    start_res: int,
    end_res: int,
    known_cancer_data_dict: dict,
    iterations: int = 10**6,
    batch_size: int = 100000,
) -> list:
    """Null-distribution generator that preserves the number of unique mutated residues.

    For each permutation, exactly ``num_unique`` residues are selected, and the
    total mutation count is distributed among them (each getting at least one).

    Parameters
    ----------
    structure_map : dict
        ``{ residue_id : coordinates }``
    expotential_map : numpy.ndarray
        *M × M* Gaussian weight matrix.
    res_to_index : dict
        Residue-ID → matrix-index lookup.
    nq_values : dict
        Precomputed N(q) lookup dictionary (unused directly; N(q) is inline).
    start_res : int
        First position of the valid range (inclusive).
    end_res : int
        Last position of the valid range (inclusive).
    known_cancer_data_dict : dict
        ``{ residue_id : mutation_count }`` — real observed data.
    iterations : int, optional
        Number of permutations (default 10⁶).
    batch_size : int, optional
        Batch size for memory-safe processing (default 100 000).

    Returns
    -------
    list of float
        Null-distribution WAP scores.

    Raises
    ------
    ValueError
        If the number of unique mutated residues exceeds the structure size,
        or if ``total_mutations < num_unique``.
    """
    # Only consider mutations within the specified window
    valid_observed = {
        res: count
        for res, count in known_cancer_data_dict.items()
        if start_res <= int(res) <= end_res
    }

    num_unique = len(valid_observed)
    total_mutations = sum(valid_observed.values())
    M = len(structure_map)

    print(f"\nRunning Vectorized Simulation ({iterations} iterations)...")

    if num_unique > M:
        raise ValueError(
            f"Cannot select {num_unique} unique residues from a structure "
            f"with only {M} residues."
        )
    if total_mutations < num_unique:
        raise ValueError(
            "Total mutations must be >= the number of unique mutated residues."
        )

    random_scores = np.zeros(iterations)
    remaining_mutations = total_mutations - num_unique
    pvals = np.ones(num_unique) / num_unique
    sum_of_max_counts = 0

    for i in range(0, iterations, batch_size):
        current_batch_size = min(batch_size, iterations - i)

        # STEP A: Pick num_unique random positions via top-k random projection
        rand_matrix = np.random.rand(current_batch_size, M)
        top_indices = np.argpartition(rand_matrix, -num_unique, axis=1)[:, -num_unique:]

        # STEP B: Distribute remaining mutations among selected positions (each gets ≥1)
        random_extra = np.random.multinomial(
            remaining_mutations, pvals, size=current_batch_size
        )
        final_counts = (
            random_extra + 1
        )  # Guarantee at least 1 mutation per selected residue

        # STEP C: Build the full mutation-count matrix (zeros for unselected residues)
        Q_matrix = np.zeros((current_batch_size, M))
        row_indices = np.arange(current_batch_size)[:, None]
        Q_matrix[row_indices, top_indices] = final_counts

        # Track the maximum mutation count across all runs for diagnostics
        sum_of_max_counts += np.sum(np.max(Q_matrix, axis=1))

        # STEP D: Vectorized N(q) using Hill equation
        N_matrix = (Q_matrix**3) / (8.0 + Q_matrix**3)

        # STEP E: WAP score
        term1 = np.sum((N_matrix @ expotential_map) * N_matrix, axis=1)
        term2 = np.sum(N_matrix**2, axis=1)
        random_scores[i : i + current_batch_size] = term1 - term2

    average_max_count = sum_of_max_counts / iterations
    print(f"\n--- [SIMULATION STATISTICS] ---")
    print(
        f"Average max mutation count across {iterations} runs: {average_max_count:.2f}"
    )
    print("-------------------------------\n")

    return random_scores.tolist()


# ============================================================================
#  PLOTTING
# ============================================================================


def plot_results(
    random_scores: list,
    observed_score: float,
    p_value: float,
    protein_name: str,
    method_name: str,
    dpi: int = 1000,
) -> None:
    """Publication-ready permutation test plot.

    Draws a histogram of the null distribution with a KDE overlay,
    shades the p-value tail, and marks the observed score with a vertical line.

    Parameters
    ----------
    random_scores : list of float
        Null-distribution WAP scores.
    observed_score : float
        Observed WAP score to annotate.
    p_value : float
        Empirical p-value (displayed as text annotation).
    protein_name : str
        Gene/protein name for the plot title.
    method_name : str
        Label for the N(q) function (displayed in title and annotation).
    dpi : int, optional
        Resolution for saved figures (default 1000).
    """
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        f"WAP Score Permutation Test: {protein_name}, Method: {method_name}",
        fontsize=10,
        fontweight="bold",
        pad=15,
    )

    COLOR_HIST = "#8A8A8A"
    COLOR_KDE = "#081D01"
    COLOR_OBS = "#002499"

    # 1. Histogram of null distribution
    sns.histplot(
        random_scores,
        bins="auto",
        stat="density",
        color=COLOR_HIST,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        ax=ax,
        zorder=1,
    )

    data_min = min(random_scores)
    data_max = max(random_scores)

    if np.isclose(data_min, data_max):
        # Edge case: all random scores are identical → skip KDE
        print(
            f"  -> [Warning] KDE skipped for '{method_name}': "
            f"all random scores are identical ({data_min:.4f})."
        )
        x_min = data_min - 0.5
        x_max = max(data_max + 0.5, observed_score + 0.5)
        y_max = ax.get_ylim()[1]

    else:
        # 2. KDE on a subsample (fast for 10⁶ points)
        max_kde_samples = 10000
        kde_data = (
            np.random.choice(random_scores, size=max_kde_samples, replace=False)
            if len(random_scores) > max_kde_samples
            else np.asarray(random_scores)
        )
        kde = gaussian_kde(kde_data)

        x_min = data_min
        x_max = max(data_max, observed_score)
        padding = (x_max - x_min) * 0.05
        x_grid = np.linspace(x_min - padding, x_max + padding, 1000)
        y_grid = kde(x_grid)
        y_max = max(y_grid)

        # 3. KDE line
        ax.plot(
            x_grid,
            y_grid,
            color=COLOR_KDE,
            linewidth=1.5,
            zorder=2,
            label="Null Distribution (KDE)",
        )

        # 4. Shade the p-value tail
        ax.fill_between(
            x_grid,
            y_grid,
            where=(x_grid >= observed_score),
            color=COLOR_OBS,
            alpha=0.3,
            zorder=3,
            label="p-value Area",
        )

    # 5. Observed score vertical line
    ax.axvline(
        observed_score,
        color=COLOR_OBS,
        linestyle="--",
        linewidth=2,
        zorder=4,
        label="Observed Score",
    )

    # 6. Axis labels and spine formatting
    ax.set_xlabel("WAP Score", fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", direction="out", length=5, width=1)
    sns.despine(top=True, right=True)

    # 7. P-value annotation
    text_x_offset = (x_max - x_min) * 0.02 if not np.isclose(x_max, x_min) else 0.05
    ax.text(
        observed_score + text_x_offset,
        y_max * 0.5,
        f"p = {p_value:.5f}",
        color=COLOR_OBS,
        fontsize=12,
        fontweight="bold",
        va="center",
    )

    ax.legend(loc="upper right", frameon=False, fontsize=10)
    plt.tight_layout()
    print(f"Displaying permutation test plot for {method_name}...")
    plt.show()


# ============================================================================
#  MULTI-GENE PIPELINE  (run directly: python WAP.py)
# ============================================================================


def run_gene(
    df_mapped: pd.DataFrame,
    residue_df: pd.DataFrame,
    gene_name: str,
    iterations: int,
) -> tuple | None:
    """Run the full WAP analysis pipeline for a single gene, pooling mutations

    across all structure sources using UniProt coordinates.
    """
    print(f"Loading data for {gene_name} from mapped TSV...")
    gene_df = df_mapped[df_mapped["gene"] == gene_name]

    if gene_df.empty:
        print(f"Error: No data found for {gene_name} in the TSV file. Skipping.")
        return None

    # Load available structure sources for this gene from residues TSV
    available_structures = set(
        residue_df[residue_df["gene"] == gene_name]["structure_source"]
        .dropna()
        .unique()
    )

    if not available_structures:
        print(
            f"Error: No residue structure data found for {gene_name} in residue TSV. Skipping."
        )
        return None

    # Select the structure source present in residues TSV that has the most entries in mapped TSV
    valid_gene_df = gene_df[gene_df["structure_source"].isin(available_structures)]
    if valid_gene_df.empty:
        print(
            f"Error: No matching structure source between mapped TSV and residue TSV for {gene_name}. Skipping."
        )
        return None

    best_structure = valid_gene_df["structure_source"].value_counts().idxmax()
    print(f"Selected 3D structure source: {best_structure}.")

    # Load all resolved Cα coordinates for the chosen structure template
    structure_df = residue_df[
        (residue_df["gene"] == gene_name) & (residue_df["structure_source"] == best_structure)
    ]
    if structure_df.empty:
        print(
            f"Error: No residues found for gene '{gene_name}' "
            f"with structure source '{best_structure}'."
        )
        aligned_structure_map = {}
    else:
        coords_df = (
            structure_df[["uniprot_pos", "x", "y", "z"]]
            .dropna()
            .drop_duplicates(subset=["uniprot_pos"])
        )
        aligned_structure_map = {
            int(row["uniprot_pos"]): [float(row["x"]), float(row["y"]), float(row["z"])]
            for _, row in coords_df.iterrows()
        }

    if not aligned_structure_map:
        print(f"Error: No structure data found for {gene_name}. Skipping.")
        return None

    # Pool mutations across ALL structure sources for this gene based on UniProt position
    all_mutation_counts = gene_df["position"].value_counts()

    # Exclude outlier positions (> 1000 mutations)
    filtered_cancer_data = {
        pos: count for pos, count in all_mutation_counts.items() if count <= 1000
    }

    n_before = len(filtered_cancer_data)

    # Retain only positions that exist in the selected 3D structure map
    filtered_cancer_data = {
        pos: count
        for pos, count in filtered_cancer_data.items()
        if pos in aligned_structure_map
    }

    print(
        f"Pooled mutated positions resolved in structure: {len(filtered_cancer_data)} of {n_before}"
    )

    len_structure = len(aligned_structure_map)
    len_mutated = len(filtered_cancer_data)
    print(f"Total residues in structure: {len_structure}")
    print(f"Total mutated residues     : {len_mutated}")

    if len_mutated < 2:
        print("Error: Not enough valid mutations to calculate WAP score. Skipping.")
        return None

    # Pre-compute shared matrices
    distance_matrix = calculate_distance_matrix(aligned_structure_map)
    res_to_index = build_index_map(aligned_structure_map)
    exp_matrix = create_expotential_map(distance_matrix)

    # Four N(q) saturation functions to compare
    nq_functions = [
        ("Old Hill (m=3)", calculate_Nq_values_modified_parameters(3)),
        ("New Hill (m=2)", calculate_Nq_values_modified_parameters(2)),
        ("Logarithmic", calculate_Nq_values_logrithmic()),
        ("Logistic", generate_logistic_nq_dict()),
    ]

    results = []
    for label, nq_values in nq_functions:
        observed_score = calculate_WAP_Score(
            exp_matrix, filtered_cancer_data, res_to_index, nq_values
        )
        print(f"Observed WAP Score [{label}]: {observed_score:.6f}")

        # Auto-size batch size
        k = len(filtered_cancer_data)
        N = exp_matrix.shape[0]
        batch_size = max(1, min(int(3e8 / (k * k * 8)), int(6e7 / N)))

        print(f"Running {iterations} permutations [{label}]...")
        random_scores = generate_permutation_scores_vectorized(
            exp_matrix, nq_values, filtered_cancer_data, iterations, batch_size
        )

        p_value = calculate_p_value(random_scores, observed_score)
        mean_random = float(np.mean(random_scores))
        print(f"Mean Random [{label}]: {mean_random:.7f} | P-Value: {p_value:.7f}\n")

        results.append(
            {
                "label": label,
                "observed": observed_score,
                "p_value": p_value,
                "mean_random": mean_random,
                "random_scores": random_scores,
            }
        )

    return len_structure, len_mutated, best_structure, results

def add_gene_page(
    pdf: PdfPages,
    gene_name: str,
    len_structure: int,
    len_mutated: int,
    best_structure: str,
    results: list,
    iterations: int,
) -> None:
    """Append a 2×2 histogram page for one gene to an open PDF.

    Each subplot shows the null distribution and observed score for one
    N(q) function.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Open PDF file to write to.
    gene_name : str
        Gene symbol (used in the page header).
    len_structure : int
        Total number of Cα residues in the chosen structure.
    len_mutated : int
        Number of mutated residues with 3D coordinates.
    best_structure : str
        Structure source identifier (e.g. PDB ID).
    results : list of dict
        Output from :func:`run_gene`.
    iterations : int
        Number of permutations (displayed in the header).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"{gene_name}  |  Residues in structure: {len_structure}  |  "
        f"Mutated residues: {len_mutated}\n"
        f"Structure source: {best_structure}  |  Permutations: {iterations}",
        fontsize=10,
    )

    for ax, r in zip(axes.ravel(), results):
        ax.hist(np.asarray(r["random_scores"]), bins=50, color="steelblue", alpha=0.8)
        ax.axvline(
            r["observed"], color="red", linestyle="--", linewidth=2, label="Observed"
        )
        ax.set_title(
            f"{r['label']}\n"
            f"Observed WAP: {r['observed']:.6f}  |  "
            f"P-value: {r['p_value']:.7f}  |  "
            f"Mean random: {r['mean_random']:.6f}",
            fontsize=8,
        )
        ax.set_xlabel("WAP score")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_summary_page(
    pdf: PdfPages,
    summary_rows: list,
    title: str = "Summary",
) -> None:
    """Append a summary table page (or pages) to an open PDF.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Open PDF file to write to.
    summary_rows : list of list
        Rows for the summary table; each row should have 7 fields matching
        the column headers defined inside this function.
    title : str, optional
        Page title (default ``"Summary"``).
    """
    if not summary_rows:
        return

    cols = [
        "Gene",
        "Function",
        "Residues in structure",
        "Mutated residues",
        "Observed WAP",
        "P-value",
        "Mean random",
    ]
    rows_per_page = 40

    for start in range(0, len(summary_rows), rows_per_page):
        chunk = summary_rows[start : start + rows_per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        table = ax.table(cellText=chunk, colLabels=cols, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.3)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    data_root = Path(os.environ.get("CFR_DATA_ROOT", "./cfr_data")).resolve()
    run_tag = os.environ.get("CFR_RUN_TAG", f"run_{datetime.now().strftime('%Y-%m-%d')}")
    processed = data_root / "processed" / run_tag

    tsv_filepath = str(processed / "stage2_mapped.tsv")
    residue_tsv_filepath = str(processed / "stage2_residues.tsv")
    iterations = int(os.environ.get("CFR_ITERATIONS", 10**6))

    df_mapped = pd.read_csv(tsv_filepath, sep="\t")
    residue_df = pd.read_csv(residue_tsv_filepath, sep="\t")

    genes = sorted(residue_df["gene"].dropna().unique().tolist())
    print(genes)
    print(len(genes))

    # genes = [
            # "A1CF", "ABI1", "ABL1", "ABL2", "ACKR3", "ACVR1", "ACVR1B", "ACVR2A", "AFDN", "AFF4",
            # "AKT1", "AKT2", "AKT3", "ALDH2", "ALK", "AMER1", "ANK1", "APC", "APOBEC3B", "AR",
            # "ARAF", "ARHGAP26", "ARHGAP35", "ARHGAP5", "ARHGEF12", "ARID1A", "ARID1B", "ARID2", "ARNT", "ASPSCR1",
            # "ASXL1", "ATIC", "ATM", "ATP1A1", "ATR", "ATRX", "AXIN1", "AXIN2", "B2M", "BAP1",
            # "BARD1", "BAX", "BAZ1A", "BCL10", "BCL11A", "BCL11B", "BCL2", "BCL3", "BCL6", "BCL7A",
            # "BCL9", "BCL9L", "BCLAF1", "BCOR", "BCORL1", "BCR", "BIRC3", "BIRC6", "BLM", "BMPR1A",
            # "BRAF", "BRCA1", "BRCA2", "BRD3", "BRD4", "BRIP1", "BTG2", "BTK", "BUB1B", "CACNA1D",
            # "CALR", "CAMTA1", "CANT1", "CARD11", "CASP3", "CASP8", "CASP9", "CBFA2T3", "CBFB", "CBL",
            # "CBLB", "CBLC", "CCDC6", "CCNB1IP1", "CCNC", "CCND1", "CCND2", "CCND3", "CCNE1", "CCR4",
            # "CCR7", "CD209", "CD274", "CD28", "CD74", "CD79A", "CD79B", "CDC73", "CDH1", "CDH17",
            # "CDK12", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2C", "CDX2", "CEBPA", "CHCHD7",
            # "CHD4", "CHEK2", "CHIC2", "CLIP1", "CLP1", "CLTC", "CNOT3", "CNTNAP2", "COL1A1", "COL2A1",
            # "COL3A1", "COX6C", "CPEB3", "CREB1", "CREBBP", "CRLF2", "CRNKL1", "CRTC1", "CSF1R", "CSF3R",
            # "CTCF", "CTNNA1", "CTNNA2", "CTNNB1", "CTNND1", "CUL3", "CUX1", "CXCR4", "CYLD", "CYP2C8",
            # "CYSLTR2", "DAXX", "DCC", "DCTN1", "DDB2", "DDR2", "DDX10", "DDX3X", "DDX5", "DDX6",
            # "DEK", "DGCR8", "DICER1", "DNAJB1", "DNM2", "DNMT3A", "DROSHA", "EBF1", "EED", "EGFR",
            # "EIF1AX", "EIF3E", "EIF4A2", "ELF3", "ELK4", "ELL", "EML4", "EP300", "EPAS1", "EPHA3",
            # "EPHA7", "EPS15", "ERBB2", "ERBB3", "ERBB4", "ERCC2", "ERCC3", "ERCC4", "ERCC5", "ERG",
            # "ESR1", "ETV1", "ETV4", "ETV5", "ETV6", "EWSR1", "EXT1", "EXT2", "EZH2", "EZR",
            # "FADD", "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG", "FAS", "FAT4", "FBXO11",
            # "FBXW7", "FCGR2B", "FCRL4", "FEN1", "FES", "FEV", "FGFR1", "FGFR2", "FGFR3", "FGFR4",
            # "FH", "FHIT", "FIP1L1", "FLCN", "FLI1", "FLNA", "FLT3", "FLT4", "FNBP1", "FOXA1",
            # "FOXL2", "FOXO1", "FOXO3", "FOXO4", "FOXP1", "FSTL3", "FUBP1", "FUS", "GAS7", "GATA1",
            # "GATA2", "GATA3", "GLI1", "GMPS", "GNA11", "GNAQ", "GNAS", "GOLPH3", "GOPC", "GPC3",
            # "GPHN", "GRIN2A", "GRM3", "GSK3B", "GTF2I", "HERPUD1", "HEY1", "HGF", "HIF1A", "HIP1",
            # "HLA-A", "HMGA1", "HNF1A", "HNRNPA2B1", "HOOK3", "HOXA13", "HRAS", "HSP90AA1", "HSP90AB1", "ID3",
            # "IDH1", "IDH2", "IGF2BP2", "IKBKB", "IKZF1", "IKZF3", "IL2", "IL21R", "IL6ST", "IL7R",
            # "IRF4", "ITGAV", "ITK", "JAK1", "JAK2", "JAK3", "JUN", "KAT6A", "KAT6B", "KAT7",
            # "KDM5A", "KDM5C", "KDM6A", "KDR", "KEAP1", "KIF5B", "KIT", "KLF4", "KLK2", "KMT2A",
            # "KMT2C", "KMT2D", "KNL1", "KRAS", "LARP4B", "LASP1", "LATS1", "LATS2", "LCK", "LCP1",
            # "LIFR", "LMNA", "LMO2", "LSM14A", "LYN", "LZTR1", "MALT1", "MAP2K1", "MAP2K2", "MAP2K4",
            # "MAP3K1", "MAPK1", "MAX", "MB21D2", "MDM2", "MDM4", "MED12", "MEN1", "MET", "MGMT",
            # "MITF", "MLH1", "MLLT1", "MLLT10", "MLLT3", "MPL", "MSH2", "MSH6", "MSI2", "MSN",
            # "MTCP1", "MTOR", "MUC16", "MUTYH", "MYC", "MYCN", "MYD88", "MYH11", "MYH9", "MYO5A",
            # "N4BP2", "NACA", "NBEA", "NBN", "NCKIPSD", "NCOA1", "NCOA2", "NCOR1", "NCOR2", "NDRG1",
            # "NF1", "NF2", "NFATC2", "NFE2L2", "NFIB", "NFKB2", "NKX2-1", "NONO", "NOTCH1", "NOTCH2",
            # "NPM1", "NR2F2", "NR4A3", "NRAS", "NRG1", "NSD1", "NSD2", "NSD3", "NT5C2", "NTHL1",
            # "NTRK1", "NTRK2", "NTRK3", "NUMA1", "NUP214", "NUP98", "NUTM1", "OMD", "P2RY8", "PABPC1",
            # "PAFAH1B2", "PALB2", "PATZ1", "PAX3", "PAX5", "PAX8", "PBX1", "PCBP1", "PCM1", "PDCD1LG2",
            # "PDGFB", "PDGFRA", "PDGFRB", "PHF6", "PHOX2B", "PIK3CA", "PIK3R1", "PIK3R2", "PIM1", "PLCG1",
            # "PLEC", "PML", "PMS1", "PMS2", "POLD1", "POLE", "POLG", "POLQ", "POLR2A", "POT1",
            # "POU2AF1", "POU5F1", "PPARG", "PPM1D", "PPP2R1A", "PRDM1", "PRDM16", "PRDM2", "PREX2", "PRKACA",
            # "PRKAR1A", "PRKCB", "PSIP1", "PTCH1", "PTEN", "PTK6", "PTPN11", "PTPN13", "PTPN6", "PTPRB",
            # "PTPRC", "PTPRD", "PTPRK", "PTPRT", "QKI", "RABEP1", "RAC1", "RAD21", "RAD50", "RAD51",
            # "RAD51B", "RAD51C", "RAD51D", "RAF1", "RALGDS", "RANBP2", "RAP1B", "RAP1GDS1", "RARA", "RASA1",
            # "RB1", "RBM10", "RBM15", "RECQL4", "RET", "RFWD3", "RGS7", "RHOA", "RICTOR", "RMI2",
            # "RNF213", "RNF43", "ROBO2", "ROCK1", "ROS1", "RPL10", "RPL22", "RPL5", "RPN1", "RRAS2",
            # "RSPO2", "RUNX1", "RUNX1T1", "RXRA", "S100A7", "SALL4", "SBDS", "SDC4", "SDHA", "SDHAF2",
            # "SDHB", "SDHC", "SDHD", "SET", "SETD1B", "SETD2", "SETDB1", "SF3B1", "SFPQ", "SGK1",
            # "SIRPA", "SIX1", "SKI", "SMAD2", "SMAD3", "SMAD4", "SMARCA4", "SMARCB1", "SMARCD1", "SMARCE1",
            # "SMC1A", "SMO", "SND1", "SOS1", "SOX2", "SPEN", "SPOP", "SRC", "SRSF2", "SRSF3",
            # "SS18", "SSX1", "STAG1", "STAG2", "STAT3", "STAT5B", "STAT6", "STIL", "STK11", "SUB1",
            # "SUFU", "SUZ12", "SYK", "TAL1", "TBL1XR1", "TBX3", "TCEA1", "TCF12", "TCF3", "TCF7L2",
            # "TCL1A", "TEC", "TENT5C", "TERT", "TET1", "TET2", "TFE3", "TFEB", "TFG", "TFRC",
            # "TGFBR2", "TMPRSS2", "TMSB4X", "TNC", "TNFAIP3", "TNFRSF14", "TNFRSF17", "TNRC18", "TOP1", "TP53",
            # "TP63", "TPM3", "TPR", "TRAF7", "TRIM24", "TRIM33", "TRRAP", "TSC1", "TSC2", "TSHR",
            # "U2AF1", "UBR5", "USP8", "USP9X", "VAV1", "VHL", "WAS", "WIF1", "WNK2", "WRN",
            # "WT1", "WWTR1", "XPA", "XPO1", "YAP1", "YWHAE", "ZBTB16", "ZCCHC8", "ZEB1", "ZFHX3",
            # "ZNRF3"
            # ]

            # genes = [
            # "A1CF", "ABI1", "ABL1", "ABL2", "ACKR3", "ACVR1", "ACVR1B", "ACVR2A", "AFDN", "AFF4",
            # "AKT1", "AKT2", "AKT3", "ALDH2", "ALK", "AMER1", "ANK1", "APC", "APOBEC3B", "AR",
            # "ARAF", "ARHGAP26", "ARHGAP35", "ARHGAP5", "ARHGEF12", "ARID1A", "ARID1B", "ARID2", "ARNT", "ASPSCR1",
            # "ASXL1", "ATIC", "ATM", "ATP1A1", "ATR", "ATRX", "AXIN1", "AXIN2", "B2M", "BAP1",
            # "BARD1", "BAX", "BAZ1A", "BCL10", "BCL11A", "BCL11B", "BCL2", "BCL3", "BCL6", "BCL7A"
            # ]

    output_folder = "WAP_results_GPU"
    output_pdf = os.path.join(
        output_folder,
        f"WAP_all_genes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf",
    )

    os.makedirs(output_folder, exist_ok=True)

    summary_rows = []
    with PdfPages(output_pdf) as pdf:
        for gene_name in genes:
            print("=" * 60)
            print(f"GENE: {gene_name}")
            print("=" * 60)

            out = run_gene(df_mapped, residue_df, gene_name, iterations)
            if out is None:
                continue

            len_structure, len_mutated, best_structure, results = out
            add_gene_page(
                pdf,
                gene_name,
                len_structure,
                len_mutated,
                best_structure,
                results,
                iterations,
            )

            for r in results:
                summary_rows.append(
                    [
                        gene_name,
                        r["label"],
                        len_structure,
                        len_mutated,
                        f"{r['observed']:.6f}",
                        f"{r['p_value']:.7f}",
                        f"{r['mean_random']:.6f}",
                    ]
                )

        add_summary_page(pdf, summary_rows, title="Summary of all genes")

    print(f"\nDone. PDF saved as: {output_pdf}")

if __name__ == "__main__":
    main()