"""
STIGRN BEELINE Data Loader

Loads and processes benchmark datasets from the BEELINE framework
(Pratapa et al., Nature Methods 2020) for systematic evaluation.

Supports:
- Curated Boolean models (HSC, mCAD, VSC, GSD): BoolODE-simulated expression
- Synthetic networks (LI, LL, CY, BF, BFC, TF): known trajectory topologies
- Real scRNA-seq datasets (hESC, mESC, mDC, hHep, mHSC-E): with ChIP-seq GRNs

Expected directory structure after running setup_beeline.py:
    data/beeline/
    ├── inputs/
    │   ├── curated/
    │   │   ├── HSC/
    │   │   │   ├── ExpressionData.csv
    │   │   │   ├── PseudoTime.csv
    │   │   │   └── refNetwork.csv
    │   │   ├── mCAD/
    │   │   ├── VSC/
    │   │   └── GSD/
    │   ├── synthetic/
    │   │   ├── LI/ ... LL/ ... CY/ ... BF/ ... BFC/ ... TF/
    │   └── real/
    │       ├── hESC/ ... mESC/ ... mDC/ ... hHep/ ... mHSC-E/
"""

import os
import numpy as np
import csv
from typing import Dict, List, Optional, Tuple
import warnings


class BEELINEDataset:
    """
    Container for a single BEELINE benchmark dataset.

    Attributes
    ----------
    name : str
        Dataset name (e.g., 'HSC', 'mESC').
    expression : np.ndarray
        Gene expression matrix (cells × genes).
    gene_names : list of str
        Gene names.
    pseudotime : np.ndarray
        Pseudotime values per cell.
    ground_truth_edges : list of (str, str)
        Ground truth regulatory edges (TF, target).
    ground_truth_adjacency : np.ndarray
        Binary adjacency matrix from ground truth.
    dataset_type : str
        'curated', 'synthetic', or 'real'.
    """

    def __init__(self):
        self.name = ""
        self.expression = None
        self.gene_names = []
        self.pseudotime = None
        self.ground_truth_edges = []
        self.ground_truth_adjacency = None
        self.dataset_type = ""
        self.n_cells = 0
        self.n_genes = 0

    def __repr__(self):
        return (f"BEELINEDataset(name='{self.name}', type='{self.dataset_type}', "
                f"cells={self.n_cells}, genes={self.n_genes}, "
                f"edges={len(self.ground_truth_edges)})")


def load_expression_csv(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load BEELINE ExpressionData.csv.
    Format: rows = genes, columns = cells. First column = gene names.
    """
    gene_names = []
    data_rows = []

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # cell IDs (skip)

        for row in reader:
            gene_names.append(row[0])
            data_rows.append([float(x) for x in row[1:]])

    expression = np.array(data_rows)  # genes × cells
    expression = expression.T  # → cells × genes

    return expression, gene_names


def load_pseudotime_csv(filepath: str) -> np.ndarray:
    """
    Load BEELINE PseudoTime.csv.
    Format: column 1 = cell ID, column 2+ = pseudotime values.
    Uses the first pseudotime column. Handles 'NA' values by interpolation.
    """
    pseudotimes = []
    na_indices = []

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        for i, row in enumerate(reader):
            if len(row) > 1 and row[1].strip().upper() not in ("NA", "NAN", ""):
                try:
                    pseudotimes.append(float(row[1]))
                except ValueError:
                    pseudotimes.append(np.nan)
                    na_indices.append(i)
            else:
                pseudotimes.append(np.nan)
                na_indices.append(i)

    pt = np.array(pseudotimes)

    # Interpolate NAs: use cell index as proxy
    if na_indices:
        valid_mask = ~np.isnan(pt)
        if valid_mask.sum() > 0:
            # Linear interpolation from valid values
            valid_idx = np.where(valid_mask)[0]
            valid_vals = pt[valid_mask]
            pt[na_indices] = np.interp(na_indices, valid_idx, valid_vals)
        else:
            # All NA: fall back to linear spacing
            pt = np.linspace(0, 1, len(pt))

    return pt


def load_reference_network(filepath: str) -> List[Tuple[str, str]]:
    """
    Load BEELINE refNetwork.csv.
    Format: TF, target, type (+/-)
    Returns list of (TF, target) edges.
    """
    edges = []

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            if len(row) >= 2:
                tf = row[0].strip()
                target = row[1].strip()
                edges.append((tf, target))

    return edges


def build_adjacency_from_edges(
    edges: List[Tuple[str, str]],
    gene_names: List[str],
) -> np.ndarray:
    """Convert edge list to binary adjacency matrix."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n = len(gene_names)
    adj = np.zeros((n, n))

    for tf, target in edges:
        i = gene_to_idx.get(tf)
        j = gene_to_idx.get(target)
        if i is not None and j is not None:
            adj[i, j] = 1.0

    return adj


def load_beeline_dataset(
    data_dir: str,
    dataset_name: str,
    dataset_type: str = "curated",
    sample_idx: int = 0,
) -> BEELINEDataset:
    """
    Load a complete BEELINE dataset.

    Parameters
    ----------
    data_dir : str
        Root BEELINE data directory (e.g., 'data/beeline/inputs').
    dataset_name : str
        Dataset name: 'HSC', 'mCAD', 'VSC', 'GSD', 'LI', 'hESC', etc.
    dataset_type : str
        'curated', 'synthetic', or 'real'.
    sample_idx : int
        For curated/synthetic datasets with multiple samples, which to load.

    Returns
    -------
    BEELINEDataset
    """
    ds = BEELINEDataset()
    ds.name = dataset_name
    ds.dataset_type = dataset_type

    base_path = os.path.join(data_dir, dataset_type, dataset_name)

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {base_path}\n"
            f"Run 'python scripts/setup_beeline.py' first to download the data."
        )

    # Look for expression data
    expr_candidates = [
        os.path.join(base_path, "ExpressionData.csv"),
        os.path.join(base_path, f"ExpressionData_{sample_idx}.csv"),
        os.path.join(base_path, "ExpressionData.tsv"),
    ]

    expr_path = None
    for candidate in expr_candidates:
        if os.path.exists(candidate):
            expr_path = candidate
            break

    if expr_path is None:
        raise FileNotFoundError(f"No expression data found in {base_path}")

    # Load expression
    if expr_path.endswith(".tsv"):
        # Handle TSV format
        ds.expression, ds.gene_names = _load_tsv_expression(expr_path)
    else:
        ds.expression, ds.gene_names = load_expression_csv(expr_path)

    ds.n_cells, ds.n_genes = ds.expression.shape

    # Load pseudotime
    pt_candidates = [
        os.path.join(base_path, "PseudoTime.csv"),
        os.path.join(base_path, f"PseudoTime_{sample_idx}.csv"),
    ]

    for candidate in pt_candidates:
        if os.path.exists(candidate):
            ds.pseudotime = load_pseudotime_csv(candidate)
            break

    if ds.pseudotime is None:
        # Generate pseudotime from cell ordering (fallback)
        warnings.warn(f"No pseudotime file found for {dataset_name}, "
                      f"using cell index as proxy")
        ds.pseudotime = np.linspace(0, 1, ds.n_cells)

    # Trim pseudotime if length mismatch
    if len(ds.pseudotime) != ds.n_cells:
        min_len = min(len(ds.pseudotime), ds.n_cells)
        ds.pseudotime = ds.pseudotime[:min_len]
        ds.expression = ds.expression[:min_len, :]
        ds.n_cells = min_len

    # Load ground truth network
    ref_candidates = [
        os.path.join(base_path, "refNetwork.csv"),
        os.path.join(base_path, "GroundTruthNetwork.csv"),
        os.path.join(base_path, "network.csv"),
    ]

    for candidate in ref_candidates:
        if os.path.exists(candidate):
            ds.ground_truth_edges = load_reference_network(candidate)
            ds.ground_truth_adjacency = build_adjacency_from_edges(
                ds.ground_truth_edges, ds.gene_names
            )
            break

    if not ds.ground_truth_edges:
        warnings.warn(f"No ground truth network found for {dataset_name}")

    return ds


def _load_tsv_expression(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """Load tab-separated expression data."""
    gene_names = []
    data_rows = []

    with open(filepath, "r") as f:
        header = f.readline()  # skip
        for line in f:
            parts = line.strip().split("\t")
            gene_names.append(parts[0])
            data_rows.append([float(x) for x in parts[1:]])

    expression = np.array(data_rows).T  # → cells × genes
    return expression, gene_names


def list_available_datasets(data_dir: str) -> Dict[str, List[str]]:
    """
    Scan the data directory and report which BEELINE datasets are available.

    Returns
    -------
    dict
        Keys: 'curated', 'synthetic', 'real'. Values: list of dataset names.
    """
    available = {"curated": [], "synthetic": [], "real": []}

    for dtype in available.keys():
        type_dir = os.path.join(data_dir, dtype)
        if os.path.exists(type_dir):
            for name in sorted(os.listdir(type_dir)):
                full_path = os.path.join(type_dir, name)
                if os.path.isdir(full_path):
                    # Check for expression data
                    has_expr = any(
                        os.path.exists(os.path.join(full_path, f))
                        for f in ["ExpressionData.csv", "ExpressionData.tsv",
                                  "ExpressionData_0.csv"]
                    )
                    if has_expr:
                        available[dtype].append(name)

    return available
