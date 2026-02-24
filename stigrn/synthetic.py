"""
STIGRN Synthetic Validation Module

Generates synthetic gene regulatory networks that undergo controlled
bifurcations, for validating STIGRN metrics. Includes both:
1. Direct GRN trajectory simulation (controlled topology changes)
2. Expression simulation from GRN dynamics (ODE-based)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def generate_bifurcating_grn_trajectory(
    n_genes: int = 50,
    n_timepoints: int = 30,
    bifurcation_point: float = 0.6,
    base_connectivity: float = 0.15,
    noise_level: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    Generate a sequence of GRN adjacency matrices that undergo a controlled
    bifurcation (network splits into two modules).

    The network starts as a single connected module and gradually develops
    two communities, with the split completing at the bifurcation point.

    Parameters
    ----------
    n_genes : int
        Number of genes in the network.
    n_timepoints : int
        Number of pseudotime snapshots.
    bifurcation_point : float
        Fraction of pseudotime (0-1) at which bifurcation occurs.
    base_connectivity : float
        Baseline probability of an edge existing.
    noise_level : float
        Magnitude of random weight perturbation at each step.
    seed : int
        Random seed.

    Returns
    -------
    dict
        'adjacency_matrices': list of np.ndarray
        'timepoints': np.ndarray
        'bifurcation_time': float
        'module_assignments': np.ndarray (gene → module label)
        'gene_names': list of str
    """
    rng = np.random.RandomState(seed)

    # Assign genes to two modules
    half = n_genes // 2
    module = np.array([0] * half + [1] * (n_genes - half))
    rng.shuffle(module)

    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    timepoints = np.linspace(0, 1, n_timepoints)

    adjacency_matrices = []

    for t_idx, t in enumerate(timepoints):
        adj = np.zeros((n_genes, n_genes))

        # How far toward bifurcation (0 = undifferentiated, 1 = fully split)
        if t < bifurcation_point:
            split_progress = (t / bifurcation_point) ** 2  # Quadratic ramp
        else:
            split_progress = 1.0

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                same_module = module[i] == module[j]

                if same_module:
                    # Within-module connectivity increases toward bifurcation
                    p = base_connectivity * (1 + 0.5 * split_progress)
                    weight_scale = 1.0 + 0.3 * split_progress
                else:
                    # Between-module connectivity decreases toward bifurcation
                    p = base_connectivity * (1 - 0.8 * split_progress)
                    weight_scale = 1.0 - 0.7 * split_progress

                if rng.random() < p:
                    weight = rng.uniform(0.1, 1.0) * max(weight_scale, 0.05)
                    # Add noise
                    weight += rng.normal(0, noise_level)
                    weight = max(weight, 0.0)
                    adj[i, j] = weight
                    adj[j, i] = weight

        adjacency_matrices.append(adj)

    return {
        "adjacency_matrices": adjacency_matrices,
        "timepoints": timepoints,
        "bifurcation_time": bifurcation_point,
        "module_assignments": module,
        "gene_names": gene_names,
    }


def generate_expression_from_grn(
    adjacency: np.ndarray,
    n_cells: int = 200,
    basal_expression: Optional[np.ndarray] = None,
    noise_scale: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic single-cell expression data from a GRN adjacency matrix
    using a simplified stochastic regulation model.

    For each cell, expression is sampled as:
        x = (I - alpha * A_norm)^{-1} @ (basal + noise)

    where A_norm is the row-normalized adjacency and alpha controls
    the strength of regulation.

    Parameters
    ----------
    adjacency : np.ndarray
        GRN adjacency matrix (genes × genes).
    n_cells : int
        Number of cells to simulate.
    basal_expression : np.ndarray, optional
        Basal expression level per gene. Default: ones.
    noise_scale : float
        Scale of lognormal noise.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Expression matrix (cells × genes).
    """
    rng = np.random.RandomState(seed)
    n_genes = adjacency.shape[0]

    if basal_expression is None:
        basal_expression = np.ones(n_genes)

    # Normalize adjacency
    row_sums = adjacency.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    A_norm = adjacency / row_sums[:, np.newaxis]

    # Regulation strength (must be < 1 for stability)
    alpha = 0.3

    # Propagation matrix
    try:
        prop = np.linalg.inv(np.eye(n_genes) - alpha * A_norm)
    except np.linalg.LinAlgError:
        prop = np.eye(n_genes)

    expression = np.zeros((n_cells, n_genes))
    for c in range(n_cells):
        noise = rng.lognormal(0, noise_scale, n_genes)
        raw = prop @ (basal_expression + noise)
        expression[c, :] = np.maximum(raw, 0)

    return expression


def generate_full_synthetic_dataset(
    n_genes: int = 50,
    n_timepoints: int = 30,
    n_cells_per_window: int = 200,
    bifurcation_point: float = 0.6,
    noise_scale: float = 0.5,
    seed: int = 42,
) -> Dict:
    """
    Generate a complete synthetic dataset: GRN trajectory + expression data.

    Returns
    -------
    dict
        'grn_trajectory': dict from generate_bifurcating_grn_trajectory
        'expression_data': list of np.ndarray (cells × genes per window)
        'all_expression': np.ndarray (all cells concatenated)
        'all_pseudotime': np.ndarray (pseudotime for all cells)
    """
    grn_traj = generate_bifurcating_grn_trajectory(
        n_genes=n_genes,
        n_timepoints=n_timepoints,
        bifurcation_point=bifurcation_point,
        seed=seed,
    )

    expression_data = []
    all_expression = []
    all_pseudotime = []

    for i, (adj, t) in enumerate(
        zip(grn_traj["adjacency_matrices"], grn_traj["timepoints"])
    ):
        expr = generate_expression_from_grn(
            adj,
            n_cells=n_cells_per_window,
            noise_scale=noise_scale,
            seed=seed + i,
        )
        expression_data.append(expr)
        all_expression.append(expr)

        # Assign pseudotime with jitter
        rng = np.random.RandomState(seed + i + 1000)
        pt_jitter = rng.normal(0, 0.01, n_cells_per_window)
        all_pseudotime.append(np.full(n_cells_per_window, t) + pt_jitter)

    return {
        "grn_trajectory": grn_traj,
        "expression_data": expression_data,
        "all_expression": np.vstack(all_expression),
        "all_pseudotime": np.concatenate(all_pseudotime),
    }


def generate_saddle_node_trajectory(
    n_genes: int = 50,
    n_timepoints: int = 40,
    bifurcation_point: float = 0.5,
    seed: int = 42,
) -> Dict:
    """
    Generate a GRN trajectory undergoing a saddle-node-like bifurcation.
    
    Instead of two-module splitting, this models a loss of a hub gene's
    regulatory influence, causing the network to lose a stable state.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    n_timepoints : int
        Number of timepoints.
    bifurcation_point : float
        Fractional timepoint of the bifurcation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Same structure as generate_bifurcating_grn_trajectory.
    """
    rng = np.random.RandomState(seed)
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    timepoints = np.linspace(0, 1, n_timepoints)

    # Designate hub genes (top ~10%)
    n_hubs = max(2, n_genes // 10)
    hub_indices = rng.choice(n_genes, n_hubs, replace=False)
    module = np.zeros(n_genes, dtype=int)
    module[hub_indices] = 1  # Hubs marked

    # Base adjacency: hubs connected to many targets
    base_adj = np.zeros((n_genes, n_genes))
    for hub in hub_indices:
        n_targets = rng.randint(n_genes // 4, n_genes // 2)
        targets = rng.choice(n_genes, n_targets, replace=False)
        for t in targets:
            if t != hub:
                weight = rng.uniform(0.3, 1.0)
                base_adj[hub, t] = weight
                base_adj[t, hub] = weight * 0.3  # Weaker feedback

    # Random background edges
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if base_adj[i, j] == 0 and rng.random() < 0.05:
                w = rng.uniform(0.05, 0.3)
                base_adj[i, j] = w
                base_adj[j, i] = w

    adjacency_matrices = []

    for t_idx, t in enumerate(timepoints):
        adj = base_adj.copy()

        # Approaching bifurcation: hub influence weakens
        if t < bifurcation_point:
            decay = (t / bifurcation_point) ** 1.5
        else:
            decay = 1.0

        for hub in hub_indices:
            adj[hub, :] *= (1 - 0.85 * decay)
            adj[:, hub] *= (1 - 0.85 * decay)

        # Add noise
        noise = rng.normal(0, 0.02, (n_genes, n_genes))
        noise = (noise + noise.T) / 2
        adj = np.maximum(adj + noise, 0)
        np.fill_diagonal(adj, 0)

        adjacency_matrices.append(adj)

    return {
        "adjacency_matrices": adjacency_matrices,
        "timepoints": timepoints,
        "bifurcation_time": bifurcation_point,
        "module_assignments": module,
        "gene_names": gene_names,
    }
