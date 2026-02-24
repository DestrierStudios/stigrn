"""
STIGRN: Spectral Transition Indicators for Gene Regulatory Networks

A framework for detecting early warning signals of critical transitions
in biological systems using spectral properties of time-varying gene
regulatory networks inferred from single-cell transcriptomic data.
"""

from .metrics import STIGRNMetrics, LaplacianSpectrum
from .grn_inference import (
    construct_grn_trajectory,
    infer_grn_correlation,
    partition_cells_by_pseudotime,
)
from .synthetic import (
    generate_bifurcating_grn_trajectory,
    generate_saddle_node_trajectory,
    generate_full_synthetic_dataset,
    generate_expression_from_grn,
)
from .baselines import CSDIndicators, DNBScore
from .evaluation import evaluate_method, compare_methods, compute_auroc

__version__ = "0.1.0"
__author__ = "Nikhil Saxena"

__all__ = [
    "STIGRNMetrics",
    "LaplacianSpectrum",
    "construct_grn_trajectory",
    "infer_grn_correlation",
    "partition_cells_by_pseudotime",
    "generate_bifurcating_grn_trajectory",
    "generate_saddle_node_trajectory",
    "generate_full_synthetic_dataset",
    "generate_expression_from_grn",
]
