import csv
import math
import random
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.stats import gaussian_kde
from Bio.PDB import PDBParser
from Bio.Align import PairwiseAligner, substitution_matrices
import pandas as pd

# --- Missing Constants ---
UNIPROT_BASE = "https://rest.uniprot.org"
SLEEP_BETWEEN_REQUESTS = 0.5


# =========================================================
#                    DATA LOADING
# =========================================================
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
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


from tabulate import tabulate


def print_with_tabulate(data: dict):
    # Convert dictionary items to a list of lists
    table_data = [[key, value] for key, value in data.items()]
    print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))


def gene_to_accession(gene_name: str, organism_id: int = 9606) -> str | None:
    """
    Map a gene symbol to a reviewed (Swiss-Prot) UniProt accession.
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

    time.sleep(SLEEP_BETWEEN_REQUESTS)

    results = resp.json().get("results", [])
    if not results:
        print(f"No UniProt entry found for gene '{gene_name}'")
        return None

    accession = results[0]["primaryAccession"]
    return accession


def get_chain_sequence_and_keys(pdb_file_path, target_chain="A"):
    """
    Returns the resolved one-letter sequence for target_chain, in file order,
    plus the matching PDB residue keys.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Protein", pdb_file_path)
    seq_chars, ordered_keys = [], []
    for model in structure:
        for chain in model:
            if chain.get_id() != target_chain:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                one_letter = THREE_TO_ONE.get(residue.get_resname(), "X")
                seq_chars.append(one_letter)
                icode = residue.id[2].strip()
                ordered_keys.append(f"{residue.id[1]}{icode}")
        break
    return "".join(seq_chars), ordered_keys


def fetch_uniprot_sequence(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch UniProt sequence for {accession}. Status: {response.status_code}"
        )
    lines = response.text.strip().split("\n")
    return "".join(line.strip() for line in lines[1:])


from Bio.Align import PairwiseAligner, substitution_matrices


from Bio.Align import PairwiseAligner, substitution_matrices


def map_uniprot_to_pdb_by_sequence(
    pdb_file_path, target_chain, uniprot_accession
) -> tuple[dict, dict, int, int]:
    """
    Aligns the PDB's resolved sequence directly against the UniProt canonical sequence.
    Returns two dictionaries: the original UniProt-to-PDB key mapping, and an amino acid visual map.
    """
    pdb_seq, ordered_keys = get_chain_sequence_and_keys(pdb_file_path, target_chain)
    print(f"Length of the PDB sequence: {len(pdb_seq)}")

    unp_seq = fetch_uniprot_sequence(uniprot_accession)
    print(f"Length of the UniProt sequence: {len(unp_seq)}")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.end_insertion_score = 0
    aligner.end_deletion_score = 0

    alignment = aligner.align(unp_seq, pdb_seq)[0]

    unp_to_pdb_key = {}
    aa_visual_map = {}

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

            # Dictionary 1: Original UniProt to PDB mapping (Calculations)
            unp_to_pdb_key[unp_num_str] = pdb_res_key

            # Dictionary 2: Amino Acids (Visual Confirmation)
            aa_visual_map[unp_num_str] = {
                "pdb_res_id": pdb_res_key,
                "unp_aa": unp_aa,
                "pdb_aa": pdb_aa,
            }

    print(
        f"Sequence alignment: {len(unp_to_pdb_key)} residues mapped, "
        f"{mismatches} mismatches "
        f"({'clean' if mismatches == 0 else 'check for engineered mutations in this construct'})"
    )

    return unp_to_pdb_key, aa_visual_map, len(unp_seq), len(pdb_seq)


def load_structure_map_from_pdb(pdb_file_path, target_chain="A") -> dict:
    """
    Loads 3D coordinates for Alpha Carbons (CA) directly from a PDB file.
    """
    structure_map = {}
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("Protein", pdb_file_path)

        for model in structure:
            for chain in model:
                if chain.get_id() != target_chain:
                    continue

                for residue in chain:
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


def calculate_distance_matrix(structure_map):
    """
    Calculates the distance matrix for the given structure map.
    """
    keys = list(structure_map.keys())
    num_keys = len(keys)
    distance_matrix = np.zeros((num_keys, num_keys))

    for i in range(num_keys):
        for j in range(num_keys):
            if i != j:
                c1 = structure_map[keys[i]]
                c2 = structure_map[keys[j]]
                distance_sq = (
                    ((c1[0] - c2[0]) ** 2)
                    + ((c1[1] - c2[1]) ** 2)
                    + ((c1[2] - c2[2]) ** 2)
                )
                distance_matrix[i][j] = distance_matrix[j][i] = distance_sq

    return distance_matrix


def build_index_map(structure_map):
    """
    Builds a mapping from residue ID -> row/column index into the distance_matrix.
    """
    return {res_id: idx for idx, res_id in enumerate(structure_map.keys())}


def calculate_Nq_values(known_cancer_data: dict) -> dict:
    return {
        count: (count**3) / (2**3 + count**3)
        for res_id, count in known_cancer_data.items()
    }


def calculate_Nq_values_modified_parameters(m):
    result = {0: 0.0}
    for i in range(1, 5001):
        result[i] = (i**m) / (2**m + i**m)
    return result


def calculate_Nq_values_logrithmic():
    result = {0: 0.0}
    for i in range(1, 5001):
        result[i] = 1 / (1 + math.exp(-1.5 * (i - 2)))
    return result


def generate_logistic_nq_dict():
    """
    Precomputes logistic N(q) values for mutation counts from 0 up to max_count.
    Includes 0 to support fast array lookups for unmutated residues.
    """
    result = {}
    for count in range(5000 + 1):  # Iterates from 0 to 5000
        if count > 0:
            log_count = math.log(count) ** 3
            result[count] = log_count / (2**3 + log_count)
        else:
            result[count] = 0.0

    return result


def create_expotential_map(distance_matrix: np.array, sigma=6) -> np.array:
    """
    distance_sq / (2*sigma^2)) for every pair of structure positions.
    Computed once — distance_matrix never changes across permutations
    """
    return np.exp(-distance_matrix / (2 * (sigma**2)))


def get_mutation_counts_from_tsv(tsv_filepath, gene_name):
    """
    Reads the mapped TSV file and returns a dictionary of
    {residue_position: mutation_count} for the specified gene.
    """
    # Load the TSV file
    df = pd.read_csv(tsv_filepath, sep="\t")

    # Filter for the specific gene
    gene_df = df[df["gene"] == gene_name]

    if gene_df.empty:
        print(f"Warning: No data found for {gene_name} in the TSV file.")
        return {}

    # Count occurrences of each residue position and convert to a dictionary
    mutation_counts = gene_df["position"].value_counts().to_dict()

    return mutation_counts


# =========================================================
#                 WAP SCORE CALCULATION
# =========================================================


def get_pairs(keys):
    return list(permutations(keys, 2))


def calculate_WAP_Score(
    expotential_matrix,
    known_cancer_data_dict: dict,
    res_to_index: dict,
    Nq_values: dict,
) -> float:
    """
    Calculates WAP score using dictionary lookup for coordinates.
    """
    pairs = get_pairs(known_cancer_data_dict.keys())
    wap_score = 0.0

    for res1, res2 in pairs:
        count_1 = known_cancer_data_dict[res1]
        count_2 = known_cancer_data_dict[res2]

        idx1 = res_to_index[res1]
        idx2 = res_to_index[res2]

        # WAP formula terms
        N1 = Nq_values[count_1]
        N2 = Nq_values[count_2]
        exp_term = exp_term = expotential_matrix[idx1][idx2]

        wap_score += N1 * N2 * exp_term

    return wap_score


# =========================================================
#                 PERMUTATION TEST
# =========================================================


def generate_permutation_scores(
    structure_map,
    expotential_map,
    res_to_index,
    nq_values,
    observed_counts_dict,
    iterations=10**6,
):
    """
    Generates random WAP scores by shuffling the OBSERVED mutation counts.
    """
    structure_residues = list(structure_map.keys())
    counts_values = list(observed_counts_dict.values())
    random_scores = []

    print(f"Running simulation ({iterations} iterations)...")

    for _ in range(iterations):
        random_residues = random.sample(structure_residues, len(counts_values))
        shuffled_counts = random.sample(counts_values, len(counts_values))

        random_data_dict = {}
        for res, count in zip(random_residues, shuffled_counts):
            random_data_dict[res] = count

        score = calculate_WAP_Score(
            expotential_map, random_data_dict, res_to_index, nq_values
        )
        random_scores.append(score)

    return random_scores


# =========================================================
#                  PLOTTING FUNCTION
# =========================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde


def plot_results(
    random_scores, observed_score, p_value, protein_name, method_name, dpi=1000
):
    """
    Publication-ready permutation test plot.
    Optimized for large datasets (e.g., 1,000,000 permutations).
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

    # 1. Plot the Histogram (Still uses all 1,000,000 points, which is fast)
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

    # 2. CHECK FOR ZERO VARIANCE to prevent LinAlgError
    if np.isclose(data_min, data_max):
        print(
            f"  -> [Warning] KDE curve skipped for '{method_name}' because all random scores are identical ({data_min:.4f})."
        )

        # Set dummy limits so the text annotation below doesn't break
        x_min = data_min - 0.5
        x_max = max(data_max + 0.5, observed_score + 0.5)
        y_max = ax.get_ylim()[1]

    else:
        # 3. FAST KDE CALCULATION: Subsample the data if it's too large
        max_kde_samples = 10000
        if len(random_scores) > max_kde_samples:
            kde_data = np.random.choice(
                random_scores, size=max_kde_samples, replace=False
            )
        else:
            kde_data = random_scores

        kde = gaussian_kde(kde_data)

        x_min = data_min
        x_max = max(data_max, observed_score)
        padding = (x_max - x_min) * 0.05

        x_grid = np.linspace(x_min - padding, x_max + padding, 1000)
        y_grid = kde(x_grid)
        y_max = max(y_grid)

        # 4. Plot the KDE Line
        ax.plot(
            x_grid,
            y_grid,
            color=COLOR_KDE,
            linewidth=1.5,
            zorder=2,
            label="Null Distribution (KDE)",
        )

        # 5. Shade the P-Value Tail
        ax.fill_between(
            x_grid,
            y_grid,
            where=(x_grid >= observed_score),
            color=COLOR_OBS,
            alpha=0.3,
            zorder=3,
            label=f"p-value Area",
        )

    # 6. Observed Score Vertical Line
    ax.axvline(
        observed_score,
        color=COLOR_OBS,
        linestyle="--",
        linewidth=2,
        zorder=4,
        label="Observed Score",
    )

    # 7. Axis Formatting
    ax.set_xlabel("WAP Score", fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", direction="out", length=5, width=1)
    sns.despine(top=True, right=True)

    # 8. Add P-Value Text Annotation
    p_text = f"p = {p_value:.5f}"

    # Calculate x position dynamically based on whether variance was 0
    text_x_offset = (x_max - x_min) * 0.02 if not np.isclose(x_max, x_min) else 0.05

    ax.text(
        observed_score + text_x_offset,
        y_max * 0.5,
        p_text,
        color=COLOR_OBS,
        fontsize=12,
        fontweight="bold",
        va="center",
    )

    # 9. Legend Formatting
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    plt.tight_layout()

    print(f"Displaying fast permutation test plot for {method_name}...")
    plt.show()


def generate_permutation_scores_vectorized(
    exp_matrix,
    nq_values,
    known_cancer_mutation_dict,
    iterations=10**6,
    batch_size=None,
) -> list:
    rng = np.random.default_rng()

    N = exp_matrix.shape[0]
    counts_values = list(known_cancer_mutation_dict.values())
    k = len(counts_values)
    n_values = np.array([nq_values[c] for c in counts_values])

    if batch_size is None:
        batch_size = max(1, int(3e8 / (k * k * 8)))

    random_scores = np.empty(iterations)
    filled = 0

    while filled < iterations:
        this_batch = min(batch_size, iterations - filled)

        # tile the index range, then shuffle each row independently
        idx_pool = np.tile(np.arange(N), (this_batch, 1))
        idx = rng.permuted(idx_pool, axis=1)[:, :k]

        E_sub = exp_matrix[idx[:, :, None], idx[:, None, :]]
        diag = exp_matrix[idx, idx]

        scores = np.einsum("i,bij,j->b", n_values, E_sub, n_values) - np.sum(
            (n_values**2) * diag, axis=1
        )

        random_scores[filled : filled + this_batch] = scores
        filled += this_batch

    return random_scores.tolist()


def calculate_known_data(mutation_count_csv_path: csv) -> dict:
    unfiltered_mutation_count = pd.read_csv(mutation_count_csv_path)
    # TP53_unfiltered_mutation_count
    # # 1. Filter the rows first while the 'Type' column still exists
    filter_data = unfiltered_mutation_count[
        unfiltered_mutation_count["Type"] == "Substitution - Missense"
    ]

    # 2. Group the filtered data and sum the counts
    final_count = filter_data.groupby("Position", as_index=False)["Count"].sum()

    keys = final_count["Position"].astype(int).astype(str)

    # 2. Ensure 'Count' is an integer
    values = final_count["Count"].astype(int)

    known_cancer_data_dict = dict(zip(keys, values))

    return known_cancer_data_dict


def calculate_p_value(random_scores: list, observed_score: float) -> float:
    count_extreme = sum(1 for s in random_scores if s >= observed_score)
    p_value = count_extreme / len(random_scores)

    return p_value


def generate_mutation_technique_verification(
    structure_map,
    expotential_map,
    res_to_index,
    start_res,
    end_res,
    total_mutations,
    nq_values_dict,  # <--- New parameter
    iterations=10**6,
    batch_size=100000,
):
    # 1. Extract valid residues and their structural matrix indices
    valid_residues = [
        res for res in structure_map.keys() if start_res <= int(res) <= end_res
    ]
    num_valid = len(valid_residues)

    # Get the exact matrix indices for these specific residues
    valid_indices = [res_to_index[res] for res in valid_residues]
    M = len(structure_map)

    # --- PREPARE LOOKUP ARRAY FOR VECTORIZED INDEXING ---
    # Convert the dictionary into a NumPy array for ultra-fast matrix indexing
    max_q = max(nq_values_dict.keys())
    nq_lookup_array = np.zeros(max_q + 1)
    for q, val in nq_values_dict.items():
        nq_lookup_array[q] = val

    print(f"\nRunning Vectorized Verification ({iterations} iterations)...")
    print(
        f"Logic: Pure uniform multinomial distribution across {num_valid} valid residues."
    )
    print(f"Using precomputed N(q) dictionary (max supported count: {max_q}).")

    random_scores = np.zeros(iterations)
    pvals = np.ones(num_valid) / num_valid

    sum_of_max_counts = 0

    # Process in batches to prevent RAM overflow on 10^6 iterations
    for i in range(0, iterations, batch_size):
        current_batch = min(batch_size, iterations - i)

        # 2. Pure 1-step random distribution
        Q_valid = np.random.multinomial(total_mutations, pvals, size=current_batch)

        # Find the maximum mutation count in every run and add it to our running total
        sum_of_max_counts += np.sum(np.max(Q_valid, axis=1))

        # 3. Map the valid counts into the full 3D structural matrix
        Q_full = np.zeros((current_batch, M), dtype=int)
        Q_full[:, valid_indices] = Q_valid

        # Safety clip: if a count randomly exceeds our dictionary max, cap it to avoid IndexError
        Q_full = np.clip(Q_full, 0, max_q)

        # 4. FAST VECTORIZED DICTIONARY LOOKUP
        # Instead of calculating (Q_full**1.5) / ..., we use Q_full as indices for the lookup array
        N_matrix = nq_lookup_array[Q_full]

        # 5. Vectorized calculation of the WAP Score
        term1 = np.sum((N_matrix @ expotential_map) * N_matrix, axis=1)
        term2 = np.sum(N_matrix**2, axis=1)

        batch_scores = term1 - term2
        random_scores[i : i + current_batch] = batch_scores

    # --- 6. PRINT FINAL RESULTS ---
    average_max_count = sum_of_max_counts / iterations

    print("\n" + "=" * 60)
    print(" VERIFICATION COMPLETE")
    print("=" * 60)
    print(f" Total Iterations: {iterations}")
    print(f" Average Maximum Mutation Count: {average_max_count:.2f}")
    print("=" * 60 + "\n")

    return random_scores.tolist()


import random
from collections import Counter


def calculate_new_nq_values(known_cancer_data: dict) -> dict:
    """
    Calculates Nq values for a given total mutation count.

    total_mutation_count: The total number of observed mutations.

    """

    total_mutation_count = sum(known_cancer_data.values())

    return {
        count: (count**3) / (8.0 + count**3)
        for count in range(1, total_mutation_count + 1)
    }


def calculate_new_mutation_count(known_cancer_data: dict, valid_residues: list) -> dict:
    """
    Randomly redistributes the same total number of observed mutations
    strictly across residues that have 3D coordinates.
    """
    total_count = sum(known_cancer_data.values())

    # Use random.choices to pick directly from the list of valid residues
    new_mutated_residues = random.choices(valid_residues, k=total_count)

    return dict(Counter(new_mutated_residues))


import numpy as np


def generate_new_permutation_scores(
    structure_map,
    expotential_map,
    res_to_index,
    nq_values,
    start_res,
    end_res,
    known_cancer_data_dict,
    iterations=10**6,
    batch_size=100000,
):
    print(
        f"\nRunning Vectorized Simulation ({iterations} iterations) for range {start_res}-{end_res}..."
    )

    # --- 1. SETUP & SANITY CHECKS ---
    total_mutations = sum(
        count
        for res, count in known_cancer_data_dict.items()
        if start_res <= int(res) <= end_res
    )

    window_length = end_res - start_res + 1
    M = len(structure_map)

    if M > window_length:
        raise ValueError(
            f"Error: Number of valid 3D residues (M={M}) exceeds the defined window length ({window_length})."
        )

    out_of_bounds_residues = [
        res for res in structure_map.keys() if not (start_res <= int(res) <= end_res)
    ]
    if out_of_bounds_residues:
        raise ValueError(
            f"Found structural residues outside {start_res}-{end_res}. Please filter structure_map first."
        )

    random_scores = np.zeros(iterations)

    # --- 2. SET UP PROBABILITIES ---
    pvals = np.ones(M + 1) / window_length
    pvals[-1] = (window_length - M) / window_length

    # --- 3. PROCESS IN BATCHES ---
    for i in range(0, iterations, batch_size):
        current_batch_size = min(batch_size, iterations - i)

        Q_matrix_full = np.random.multinomial(
            total_mutations, pvals, size=current_batch_size
        )
        Q_matrix = Q_matrix_full[:, :-1]

        # --- VERIFICATION BLOCK ---
        # 1. Count how many distinct residues received at least one mutation
        unique_residues_mutated = np.count_nonzero(Q_matrix, axis=1)

        # 2. Sum the total mutations across the structure and missing loops
        total_mutations_per_run = np.sum(Q_matrix_full, axis=1)

        # 3. Mathematically guarantee every single run equals your exact total
        if not np.all(total_mutations_per_run == total_mutations):
            raise ValueError(
                "Simulation Error: A run did not match the expected total mutations!"
            )

        if i == 0:
            print(
                f"\n--- [DEBUG] Verification of first 5 runs (Target Total: {total_mutations}) ---"
            )
            for run_idx in range(5):
                print(f"Run {run_idx + 1}:")
                print(
                    f"  -> Distinct structural residues mutated: {unique_residues_mutated[run_idx]}"
                )
                print(f"  -> Total mutations in structure: {np.sum(Q_matrix[run_idx])}")
                print(
                    f"  -> Total dropped in missing loops: {Q_matrix_full[run_idx, -1]}"
                )
                print(f"  -> GRAND TOTAL: {total_mutations_per_run[run_idx]}")
            print(
                "------------------------------------------------------------------\n"
            )
            print(
                f"Successfully verified that ALL {iterations} runs strictly maintain {total_mutations} mutations.\n"
            )
        # --------------------------

        # Step C: Vectorized calculation of N(q)
        N_matrix = (Q_matrix**3) / (8.0 + Q_matrix**3)

        # Step D: Compute the WAP Score
        term1 = np.sum((N_matrix @ expotential_map) * N_matrix, axis=1)
        term2 = np.sum(N_matrix**2, axis=1)

        batch_scores = term1 - term2
        random_scores[i : i + current_batch_size] = batch_scores

    return random_scores.tolist()


import numpy as np


def generate_new_permutation_scores_2(
    structure_map,
    expotential_map,
    res_to_index,
    nq_values,
    start_res,
    end_res,
    known_cancer_data_dict,
    iterations=10**6,
    batch_size=100000,
):
    # --- 1. SETUP & SANITY CHECKS ---
    valid_observed_data = {
        res: count
        for res, count in known_cancer_data_dict.items()
        if start_res <= int(res) <= end_res
    }

    num_unique = len(valid_observed_data)
    total_mutations = sum(valid_observed_data.values())
    M = len(structure_map)

    residue_keys = list(structure_map.keys())

    print(f"\nRunning Vectorized Simulation ({iterations} iterations)...")

    if num_unique > M:
        raise ValueError(
            f"Cannot select {num_unique} unique residues from a structure with only {M} residues."
        )
    if total_mutations < num_unique:
        raise ValueError(
            "Total mutations must be >= the number of unique mutated residues."
        )

    random_scores = np.zeros(iterations)
    remaining_mutations = total_mutations - num_unique
    pvals = np.ones(num_unique) / num_unique

    # --- TRACKER FOR THE AVERAGE MAX COUNT ---
    sum_of_max_counts = 0

    # --- 3. PROCESS IN BATCHES ---
    for i in range(0, iterations, batch_size):
        current_batch_size = min(batch_size, iterations - i)

        # STEP A: Pick exactly `num_unique` random structure positions
        rand_matrix = np.random.rand(current_batch_size, M)
        top_indices = np.argpartition(rand_matrix, -num_unique, axis=1)[:, -num_unique:]

        # STEP B: Generate random mutation counts exclusively for those chosen residues
        random_extra_counts = np.random.multinomial(
            remaining_mutations, pvals, size=current_batch_size
        )

        final_counts = random_extra_counts + 1

        # STEP C: Build the Q_matrix
        Q_matrix = np.zeros((current_batch_size, M))
        row_indices = np.arange(current_batch_size)[:, None]
        Q_matrix[row_indices, top_indices] = final_counts

        # --- CALCULATE MAXIMUMS FOR THIS BATCH ---
        # Find the maximum mutation count for every run in this batch instantly
        max_counts_in_batch = np.max(Q_matrix, axis=1)
        # Accumulate the sum of these maximums
        sum_of_max_counts += np.sum(max_counts_in_batch)
        # -----------------------------------------

        # STEP D: Vectorized calculation of N(q)
        N_matrix = (Q_matrix**3) / (8.0 + Q_matrix**3)

        # STEP E: Compute the WAP Score
        term1 = np.sum((N_matrix @ expotential_map) * N_matrix, axis=1)
        term2 = np.sum(N_matrix**2, axis=1)

        batch_scores = term1 - term2
        random_scores[i : i + current_batch_size] = batch_scores

    # --- FINAL CALCULATION ---
    average_max_count = sum_of_max_counts / iterations
    print(f"\n--- [SIMULATION STATISTICS] ---")
    print(
        f"Average Maximum Mutation Count across all {iterations} runs: {average_max_count:.2f}"
    )
    print("-------------------------------\n")

    return random_scores.tolist()


import numpy as np
import pandas as pd
import WAP as wap
import importlib

importlib.reload(wap)


# A lightweight wrapper that tricks WAP.py into treating TSV coordinates like Biopython Residue objects
class MockResidue:
    def __init__(self, x, y, z):
        self.coord = np.array([x, y, z])

    def __getitem__(self, key):
        return self  # Handles calls like residue['CA']

    def __sub__(self, other):
        return np.linalg.norm(self.coord - other.coord)  # Handles distance calculation

    def __contains__(self, key):
        return True


def analyze_gene_wap(
    tsv_filepath,
    gene_name,
    nq_values_dict,
    method_name="Custom Function",
    permutations=10**6,
    run_verification=False,
):
    """
    Calculates the WAP score, P-value, and plots the distribution for a given gene and N(q) function.

    Args:
        tsv_filepath (str): Path to the mapped TSV data file.
        gene_name (str): Name of the gene to analyze (e.g., 'ESR1').
        nq_values_dict (dict): Precomputed dictionary of N(q) values from your chosen function.
        method_name (str): Label for the N(q) function used (used for plotting and print statements).
        permutations (int): Number of permutations to run for the standard test.
        run_verification (bool): Whether to run the optional multinomial verification test.

    Returns:
        dict: A dictionary containing the observed WAP score and the calculated P-value.
    """
    print(f"Loading data for {gene_name} from TSV...")
    df = pd.read_csv(tsv_filepath, sep="\t")
    gene_df = df[df["gene"] == gene_name]

    if gene_df.empty:
        print(f"Error: No data found for {gene_name} in the TSV file. Exiting.")
        return None

    # ISOLATE A SINGLE STRUCTURE SOURCE
    best_structure = gene_df["structure_source"].value_counts().idxmax()
    print(
        f"Filtering to structure source: {best_structure} to ensure accurate 3D distances."
    )

    structure_df = gene_df[gene_df["structure_source"] == best_structure].copy()

    # BUILD THE REQUIRED DICTIONARIES DIRECTLY FROM THE TSV
    counts = structure_df["position"].value_counts()
    filtered_cancer_data = counts[counts <= 1000].to_dict()
    coords_df = structure_df[["position", "x", "y", "z"]].drop_duplicates("position")

    print(
        f"Successfully loaded {len(filtered_cancer_data)} mutated positions with 3D coordinates."
    )

    if len(filtered_cancer_data) < 2:
        print("Error: Not enough valid mapped mutations to calculate WAP score.")
        return None

    # Map each position to our MockResidue so WAP.py accepts it smoothly
    aligned_structure_map = {}
    for _, row in coords_df.iterrows():
        aligned_structure_map[int(row["position"])] = MockResidue(
            row["x"], row["y"], row["z"]
        )

    # CALCULATE MATRICES USING WAP.PY NATIVELY
    distance_matrix = wap.calculate_distance_matrix(aligned_structure_map)
    res_to_index = wap.build_index_map(aligned_structure_map)
    exp_matrix = wap.create_expotential_map(distance_matrix)

    # CALCULATE OBSERVED WAP SCORE
    observed_score = wap.calculate_WAP_Score(
        exp_matrix, filtered_cancer_data, res_to_index, nq_values_dict
    )
    print(f"\nObserved WAP Score ({method_name}): {observed_score:.6f}")

    # RUN PERMUTATION TESTS (Vectorized)
    print(f"Running {permutations} standard permutations for {method_name}...")
    random_scores = wap.generate_permutation_scores_vectorized(
        exp_matrix, nq_values_dict, filtered_cancer_data, permutations
    )

    # CALCULATE P-VALUES
    p_value = wap.calculate_p_value(random_scores, observed_score)
    mean_random = sum(random_scores) / len(random_scores)
    print(
        f"Mean Random Score ({method_name}): {mean_random:.7f} | P-Value: {p_value:.7f}\n"
    )

    # PLOT THE DISTRIBUTIONS
    # wap.plot_results(
    #     random_scores, observed_score, p_value, gene_name, f"TSV Data ({method_name})"
    # )

    results = {
        "gene": gene_name,
        "method": method_name,
        "observed_score": observed_score,
        "p_value": p_value,
    }

    # RUN VECTORIZED MULTINOMIAL VERIFICATION (OPTIONAL)
    if run_verification:
        total_mutations = sum(filtered_cancer_data.values())
        residue_numbers = [int(res) for res in aligned_structure_map.keys()]
        start_res = min(residue_numbers)
        end_res = max(residue_numbers)

        print(f"--- Running Verification: {method_name} ---")
        verif_scores = wap.generate_mutation_technique_verification(
            aligned_structure_map,
            exp_matrix,
            res_to_index,
            start_res,
            end_res,
            total_mutations,
            nq_values_dict,
            10**4,
        )
        p_val_verif = wap.calculate_p_value(verif_scores, observed_score)
        print(f"Verification P-Value ({method_name}): {p_val_verif:.7f}\n")
        # wap.plot_results(
        #     verif_scores,
        #     observed_score,
        #     p_val_verif,
        #     gene_name,
        #     f"Multinomial Verification ({method_name})",
        # )

        results["verification_p_value"] = p_val_verif

    return results
