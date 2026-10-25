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
    keys = list(structure_map.keys())
    n = len(keys)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            c1 = structure_map[keys[i]]
            c2 = structure_map[keys[j]]
            dist_sq = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
            # Fill both halves of the symmetric matrix in one assignment
            distance_matrix[i][j] = distance_matrix[j][i] = dist_sq

    return distance_matrix


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
    count_extreme = sum(1 for s in random_scores if s >= observed_score)
    return count_extreme / len(random_scores)


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


def generate_permutation_scores_vectorized(
    exp_matrix: np.ndarray,
    nq_values: dict,
    known_cancer_mutation_dict: dict,
    iterations: int = 10**6,
    batch_size: int = None,
) -> list:
    """Fast vectorized null-distribution generator using NumPy batching.

    For each permutation, ``k`` residue indices are sampled without replacement
    from the *N* structural positions, then the WAP score is computed via
    matrix operations — avoiding Python loops over pairs.

    Parameters
    ----------
    exp_matrix : numpy.ndarray
        *N × N* Gaussian weight matrix.
    nq_values : dict
        Precomputed N(q) values for the observed mutation counts.
    known_cancer_mutation_dict : dict
        ``{ residue_id : mutation_count }`` — real observed data.
    iterations : int, optional
        Number of permutations (default 10⁶).
    batch_size : int or None, optional
        Number of permutations per batch. Auto-selected based on available RAM
        if ``None``.

    Returns
    -------
    list of float
        Null-distribution WAP scores.
    """
    rng = np.random.default_rng()

    N = exp_matrix.shape[0]
    counts_values = list(known_cancer_mutation_dict.values())
    k = len(counts_values)

    # Convert N(q) values to a NumPy array for fast indexing
    n_values = np.array([nq_values[c] for c in counts_values])

    # Auto-size batches to stay within ~300 MB RAM usage
    if batch_size is None:
        batch_size = max(1, int(3e8 / (k * k * 8)))

    random_scores = np.empty(iterations)
    filled = 0

    while filled < iterations:
        this_batch = min(batch_size, iterations - filled)

        # Sample k indices (without replacement) for each permutation in the batch
        idx_pool = np.tile(np.arange(N), (this_batch, 1))
        idx = rng.permuted(idx_pool, axis=1)[:, :k]

        # Extract the (k × k) sub-matrix of exponential weights for each permutation
        E_sub = exp_matrix[idx[:, :, None], idx[:, None, :]]
        diag = exp_matrix[idx, idx]

        # WAP = Σ_{i≠j} N_i · E_ij · N_j  =  (N^T E N) - Σ_i N_i² · E_ii
        scores = np.einsum("i,bij,j->b", n_values, E_sub, n_values) - np.sum(
            (n_values**2) * diag, axis=1
        )

        random_scores[filled : filled + this_batch] = scores
        filled += this_batch

    return random_scores.tolist()


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
    tsv_filepath: str,
    residue_tsv_filepath: str,
    gene_name: str,
    iterations: int,
) -> tuple | None:
    """Run the full WAP analysis pipeline for a single gene.

    Loads cancer mutation data and 3D coordinates, computes the observed WAP
    score, and runs a vectorized permutation test for each of four N(q)
    functions.

    Parameters
    ----------
    tsv_filepath : str
        Path to the mapped mutations TSV (``stage2_mapped.tsv``).
    residue_tsv_filepath : str
        Path to the residues TSV (``stage2_residues.tsv``).
    gene_name : str
        Gene symbol to analyse.
    iterations : int
        Number of permutation iterations per N(q) function.

    Returns
    -------
    tuple or None
        ``(len_structure, len_mutated, best_structure, results)`` on success;
        ``None`` if the gene has insufficient data.

        *results* is a list of dicts — one per N(q) function — each containing:
        ``"label"``, ``"observed"``, ``"p_value"``, ``"mean_random"``,
        ``"random_scores"``.
    """
    print(f"Loading data for {gene_name} from mapped TSV...")
    df = pd.read_csv(tsv_filepath, sep="\t")
    gene_df = df[df["gene"] == gene_name]

    if gene_df.empty:
        print(f"Error: No data found for {gene_name} in the TSV file. Skipping.")
        return None

    # Use the structure source that covers the most mutations
    best_structure = gene_df["structure_source"].value_counts().idxmax()
    print(f"Filtering to structure source: {best_structure}.")

    structure_df = gene_df[gene_df["structure_source"] == best_structure].copy()

    # Build the mutation-count dictionary; discard outlier positions (> 1000 mutations)
    filtered_cancer_data = {
        pos: count
        for pos, count in structure_df["position"].value_counts().items()
        if count <= 1000
    }

    # Load all resolved Cα coordinates for the chosen structure
    aligned_structure_map = create_structure_map_from_tsv(
        residue_tsv_filepath, gene_name, best_structure
    )
    if not aligned_structure_map:
        print(f"Error: No structure data found for {gene_name}. Skipping.")
        return None

    # Keep only mutated positions that are resolved in the structure
    n_before = len(filtered_cancer_data)
    filtered_cancer_data = {
        pos: count
        for pos, count in filtered_cancer_data.items()
        if pos in aligned_structure_map
    }
    print(f"Mutated positions in structure: {len(filtered_cancer_data)} of {n_before}")

    len_structure = len(aligned_structure_map)
    len_mutated = len(filtered_cancer_data)
    print(f"Total residues in structure: {len_structure}")
    print(f"Total mutated residues     : {len_mutated}")

    if len_mutated < 2:
        print("Error: Not enough valid mutations to calculate WAP score. Skipping.")
        return None

    # Pre-compute shared matrices (computed once, reused for all four N(q) functions)
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

        # Auto-size batch to avoid RAM exhaustion for large proteins
        k = len(filtered_cancer_data)
        N = exp_matrix.shape[0]
        batch_size = max(1, min(int(3e8 / (k * k * 8)), int(6e7 / N)))

        print(f"Running {iterations} permutations [{label}]...")
        random_scores = generate_permutation_scores_vectorized(
            exp_matrix, nq_values, filtered_cancer_data, iterations, batch_size
        )

        p_value = calculate_p_value(random_scores, observed_score)
        mean_random = sum(random_scores) / len(random_scores)
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
    """Run the multi-gene WAP analysis pipeline and save results to a PDF.

    Edit the ``GENES`` list and the file paths below to customise the run.
    The PDF is written to ``WAP_results/`` in the current working directory,
    with a timestamp in the filename so previous runs are not overwritten.
    """
    # ------------------------------------------------------------------ #
    #  CONFIGURATION — edit these paths and parameters as needed          #
    # ------------------------------------------------------------------ #
    tsv_filepath = r"data\stage2_mapped.tsv"
    residue_tsv_filepath = r"data\stage2_residues.tsv"
    iterations = 10**4

    genes = [
        "KMT2C",
        "BRAF",
        "ATR",
        "BTK",
        "BRCA1",
        "BRCA2",
        "TP53",
        "PTEN",
        "PALB2",
        "CHEK2",
        "ATM",
        "CDH1",
        "STK11",
        "BARD1",
        "BRIP1",
        "RAD51C",
        "RAD51D",
        "NBN",
        "NF1",
        "CD24",
        "CYP19A1",
        "FGFR2",
        "MAP3K1",
        "MLH1",
    ]

    output_folder = "WAP_results"
    output_pdf = os.path.join(
        output_folder,
        f"WAP_all_genes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf",
    )
    # ------------------------------------------------------------------ #

    os.makedirs(output_folder, exist_ok=True)

    summary_rows = []
    with PdfPages(output_pdf) as pdf:
        for gene_name in genes:
            print("=" * 60)
            print(f"GENE: {gene_name}")
            print("=" * 60)

            out = run_gene(tsv_filepath, residue_tsv_filepath, gene_name, iterations)
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
