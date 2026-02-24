"""
STIGRN GRN Inference Pipeline

Constructs time-varying gene regulatory networks from scRNA-seq data
by partitioning cells into pseudotime windows and inferring GRNs within
each window using GRNBoost2 or correlation-based methods.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings


def infer_grn_correlation(
    expression: np.ndarray,
    method: str = "pearson",
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Infer a GRN from expression data using correlation.
    Fast fallback when GRNBoost2 is not available.

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix (cells × genes).
    method : str
        'pearson' or 'spearman'.
    threshold : float
        Minimum absolute correlation to retain an edge.

    Returns
    -------
    np.ndarray
        Weighted adjacency matrix (genes × genes).
    """
    if method == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(expression)
    else:
        corr = np.corrcoef(expression.T)

    # Handle NaN (e.g., zero-variance genes)
    corr = np.nan_to_num(corr, nan=0.0)

    # Absolute correlation as edge weight
    adj = np.abs(corr)

    # Remove self-loops
    np.fill_diagonal(adj, 0)

    # Threshold
    adj[adj < threshold] = 0.0

    return adj


def infer_grn_grnboost2(
    expression: np.ndarray,
    gene_names: List[str],
    tf_names: Optional[List[str]] = None,
    threshold: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    Infer a GRN using GRNBoost2 (from arboreto package).

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix (cells × genes).
    gene_names : list of str
        Gene names corresponding to columns of expression.
    tf_names : list of str, optional
        Transcription factor names (subset of gene_names).
        If None, all genes are considered potential regulators.
    threshold : float
        Minimum importance score to retain an edge.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Weighted adjacency matrix (genes × genes).
    """
    try:
        from arboreto.algo import grnboost2
        import pandas as pd
    except ImportError:
        warnings.warn(
            "arboreto not installed. Falling back to correlation-based GRN inference. "
            "Install with: pip install arboreto"
        )
        return infer_grn_correlation(expression)

    # Create DataFrame
    df = pd.DataFrame(expression, columns=gene_names)

    # Run GRNBoost2
    network = grnboost2(
        expression_data=df,
        tf_names=tf_names or gene_names,
        seed=seed,
        verbose=False,
    )

    # Convert to adjacency matrix
    n_genes = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    adj = np.zeros((n_genes, n_genes))

    for _, row in network.iterrows():
        tf_idx = gene_to_idx.get(row["TF"])
        target_idx = gene_to_idx.get(row["target"])
        if tf_idx is not None and target_idx is not None:
            adj[tf_idx, target_idx] = row["importance"]

    # Threshold
    adj[adj < threshold] = 0.0

    return adj


def partition_cells_by_pseudotime(
    pseudotime: np.ndarray,
    n_windows: int = 20,
    overlap_fraction: float = 0.25,
    min_cells_per_window: int = 30,
) -> List[Tuple[np.ndarray, float]]:
    """
    Partition cells into overlapping pseudotime windows.

    Parameters
    ----------
    pseudotime : np.ndarray
        Pseudotime values for each cell (length n_cells).
    n_windows : int
        Number of windows to create.
    overlap_fraction : float
        Fraction of overlap between consecutive windows (0 to 0.5).
    min_cells_per_window : int
        Minimum cells required per window.

    Returns
    -------
    list of (indices, center_pseudotime)
        Each element is a tuple of (cell indices in window, pseudotime center).
    """
    pt_min, pt_max = pseudotime.min(), pseudotime.max()
    pt_range = pt_max - pt_min

    # Window size with overlap
    step = pt_range / n_windows
    window_size = step * (1 + overlap_fraction)

    windows = []
    for i in range(n_windows):
        center = pt_min + step * (i + 0.5)
        low = center - window_size / 2
        high = center + window_size / 2

        mask = (pseudotime >= low) & (pseudotime <= high)
        indices = np.where(mask)[0]

        if len(indices) >= min_cells_per_window:
            windows.append((indices, center))
        else:
            # Try expanding window to meet minimum
            sorted_dists = np.argsort(np.abs(pseudotime - center))
            expanded_indices = sorted_dists[:min_cells_per_window]
            if len(expanded_indices) >= min_cells_per_window:
                windows.append((expanded_indices, center))

    return windows


def construct_grn_trajectory(
    expression: np.ndarray,
    pseudotime: np.ndarray,
    gene_names: List[str],
    n_windows: int = 20,
    overlap_fraction: float = 0.25,
    min_cells_per_window: int = 30,
    inference_method: str = "correlation",
    correlation_threshold: float = 0.3,
    grnboost2_threshold: float = 0.01,
    tf_names: Optional[List[str]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Full pipeline: partition cells by pseudotime, infer GRN in each window.

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix (cells × genes).
    pseudotime : np.ndarray
        Pseudotime values (length n_cells).
    gene_names : list of str
        Gene names.
    n_windows : int
        Number of pseudotime windows.
    overlap_fraction : float
        Window overlap.
    min_cells_per_window : int
        Minimum cells per window.
    inference_method : str
        'correlation', 'spearman', or 'grnboost2'.
    correlation_threshold : float
        Threshold for correlation-based methods.
    grnboost2_threshold : float
        Threshold for GRNBoost2.
    tf_names : list of str, optional
        TF names for GRNBoost2.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'adjacency_matrices': list of np.ndarray
        'timepoints': list of float (pseudotime centers)
        'n_cells_per_window': list of int
        'gene_names': list of str
    """
    # Partition cells
    windows = partition_cells_by_pseudotime(
        pseudotime, n_windows, overlap_fraction, min_cells_per_window
    )

    if verbose:
        print(f"Created {len(windows)} pseudotime windows")

    adjacency_matrices = []
    timepoints = []
    n_cells_list = []

    for i, (indices, center) in enumerate(windows):
        if verbose:
            print(f"  Window {i+1}/{len(windows)}: {len(indices)} cells, "
                  f"pseudotime={center:.3f}")

        # Extract expression for this window
        expr_window = expression[indices, :]

        # Filter zero-variance genes for this window
        gene_var = np.var(expr_window, axis=0)
        valid_genes = gene_var > 1e-10

        if valid_genes.sum() < 3:
            warnings.warn(f"Window {i} has fewer than 3 variable genes, skipping")
            continue

        expr_filtered = expr_window[:, valid_genes]
        names_filtered = [g for g, v in zip(gene_names, valid_genes) if v]

        # Infer GRN
        if inference_method == "grnboost2":
            adj_filtered = infer_grn_grnboost2(
                expr_filtered, names_filtered,
                tf_names=tf_names, threshold=grnboost2_threshold, seed=seed,
            )
        elif inference_method == "spearman":
            adj_filtered = infer_grn_correlation(
                expr_filtered, method="spearman", threshold=correlation_threshold,
            )
        else:
            adj_filtered = infer_grn_correlation(
                expr_filtered, method="pearson", threshold=correlation_threshold,
            )

        # Expand back to full gene set (padding zeros for zero-variance genes)
        adj_full = np.zeros((len(gene_names), len(gene_names)))
        valid_idx = np.where(valid_genes)[0]
        for ii, gi in enumerate(valid_idx):
            for jj, gj in enumerate(valid_idx):
                adj_full[gi, gj] = adj_filtered[ii, jj]

        adjacency_matrices.append(adj_full)
        timepoints.append(center)
        n_cells_list.append(len(indices))

    return {
        "adjacency_matrices": adjacency_matrices,
        "timepoints": timepoints,
        "n_cells_per_window": n_cells_list,
        "gene_names": gene_names,
    }
