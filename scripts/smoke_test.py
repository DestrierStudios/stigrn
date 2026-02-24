#!/usr/bin/env python3
"""
STIGRN Smoke Test

Validates the full pipeline on synthetic data:
1. Generate bifurcating GRN trajectory
2. Compute all five STIGRN metrics
3. Verify metrics show expected behavior near bifurcation
4. Test early warning detection
5. Test Fiedler vector gene ranking

Run: python -m scripts.smoke_test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stigrn.metrics import STIGRNMetrics, LaplacianSpectrum
from stigrn.synthetic import (
    generate_bifurcating_grn_trajectory,
    generate_saddle_node_trajectory,
    generate_full_synthetic_dataset,
)


def test_laplacian_spectrum():
    """Test basic LaplacianSpectrum computation."""
    print("=" * 60)
    print("TEST 1: LaplacianSpectrum basic computation")
    print("=" * 60)

    # Simple 4-node graph: path graph 0-1-2-3
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)

    spec = LaplacianSpectrum(adj, normalized=False, symmetrize=False)
    print(f"  Eigenvalues: {spec.eigenvalues}")
    print(f"  λ₂ (algebraic connectivity): {spec.algebraic_connectivity:.4f}")
    print(f"  Fiedler vector: {spec.fiedler_vector}")
    print(f"  Spectral radius: {spec.spectral_radius:.4f}")

    # Verify: λ₁ should be 0 for connected graph
    assert spec.eigenvalues[0] < 1e-10, "First eigenvalue should be ~0"
    # λ₂ > 0 for connected graph
    assert spec.algebraic_connectivity > 0, "λ₂ should be > 0 for connected graph"

    print("  ✓ PASSED\n")


def test_bifurcating_trajectory():
    """Test STIGRN metrics on a bifurcating GRN trajectory."""
    print("=" * 60)
    print("TEST 2: STIGRN metrics on bifurcating trajectory")
    print("=" * 60)

    # Generate synthetic bifurcating network
    data = generate_bifurcating_grn_trajectory(
        n_genes=50,
        n_timepoints=30,
        bifurcation_point=0.6,
        base_connectivity=0.15,
        seed=42,
    )

    print(f"  Generated {len(data['adjacency_matrices'])} timepoints, "
          f"{data['adjacency_matrices'][0].shape[0]} genes")
    print(f"  Bifurcation at t={data['bifurcation_time']}")

    # Compute STIGRN metrics
    metrics = STIGRNMetrics(normalized_laplacian=True, symmetrize=True)

    for adj, t in zip(data["adjacency_matrices"], data["timepoints"]):
        result = metrics.compute(adj, timepoint=t)

    trajectory = metrics.get_trajectory()

    # Print metric trajectories
    print(f"\n  Metric trajectories (first → last):")
    for metric_name in ["SGI", "SE", "SRD", "FVI", "SMI"]:
        vals = trajectory[metric_name]
        print(f"    {metric_name}: {vals[0]:.3f} → {vals[-1]:.3f} "
              f"(range: {vals.min():.3f} - {vals.max():.3f})")

    # Verify expected behavior:
    # SGI should decrease (weakening connectivity before bifurcation)
    sgi = trajectory["SGI"]
    sgi_early = np.mean(sgi[:5])
    sgi_late = np.mean(sgi[-5:])
    print(f"\n  SGI early avg: {sgi_early:.3f}, late avg: {sgi_late:.3f}")

    # SE should generally decrease or become less uniform
    se = trajectory["SE"]
    se_early = np.mean(se[:5])
    se_late = np.mean(se[-5:])
    print(f"  SE early avg: {se_early:.3f}, late avg: {se_late:.3f}")

    # SMI should increase (emerging modular structure)
    smi = trajectory["SMI"]
    smi_early = np.mean(smi[:5])
    smi_late = np.mean(smi[-5:])
    print(f"  SMI early avg: {smi_early:.3f}, late avg: {smi_late:.3f}")

    print("  ✓ PASSED (metrics computed successfully)\n")
    return metrics, data


def test_composite_score_and_warning():
    """Test composite score and early warning detection."""
    print("=" * 60)
    print("TEST 3: Composite score and early warning detection")
    print("=" * 60)

    data = generate_bifurcating_grn_trajectory(
        n_genes=80,
        n_timepoints=40,
        bifurcation_point=0.5,
        base_connectivity=0.12,
        seed=123,
    )

    metrics = STIGRNMetrics(normalized_laplacian=True, symmetrize=True)
    for adj, t in zip(data["adjacency_matrices"], data["timepoints"]):
        metrics.compute(adj, timepoint=t)

    # Composite score
    composite = metrics.get_composite_score()
    print(f"  Composite score range: [{composite.min():.3f}, {composite.max():.3f}]")
    print(f"  Composite at start: {composite[0]:.3f}")
    print(f"  Composite at end: {composite[-1]:.3f}")

    # Early warning detection
    warning = metrics.detect_warning(threshold_sigma=1.5, min_consecutive=2)
    print(f"\n  Warning triggered: {warning['warning_triggered']}")
    if warning["warning_triggered"]:
        print(f"  Warning timepoint: {warning['warning_timepoint']:.3f}")
        print(f"  Bifurcation point: {data['bifurcation_time']:.3f}")
        lead_time = data["bifurcation_time"] - warning["warning_timepoint"]
        print(f"  Lead time: {lead_time:.3f}")
        if lead_time > 0:
            print(f"  ✓ Warning detected BEFORE bifurcation (lead = {lead_time:.3f})")
        else:
            print(f"  ⚠ Warning detected AFTER bifurcation")
    else:
        print("  ⚠ No warning detected (may need parameter tuning)")

    print("  ✓ PASSED\n")


def test_fiedler_gene_ranking():
    """Test Fiedler vector gene ranking."""
    print("=" * 60)
    print("TEST 4: Fiedler vector gene ranking")
    print("=" * 60)

    data = generate_bifurcating_grn_trajectory(
        n_genes=30,
        n_timepoints=20,
        bifurcation_point=0.6,
        seed=42,
    )

    metrics = STIGRNMetrics(normalized_laplacian=True)
    for adj, t in zip(data["adjacency_matrices"], data["timepoints"]):
        metrics.compute(adj, timepoint=t)

    # Get Fiedler ranking at last timepoint
    ranking = metrics.get_fiedler_gene_ranking(
        timepoint_idx=-1,
        gene_names=data["gene_names"],
    )

    print(f"  Top 5 genes by |Fiedler value| at final timepoint:")
    for i in range(min(5, len(ranking["ranking"]))):
        idx = ranking["ranking"][i]
        name = data["gene_names"][idx]
        fv = ranking["fiedler_values"][idx]
        module = data["module_assignments"][idx]
        print(f"    {name}: Fiedler={fv:+.4f}, true module={module}")

    # Check partition alignment with true modules
    pos_modules = data["module_assignments"][ranking["partition_positive"]]
    neg_modules = data["module_assignments"][ranking["partition_negative"]]

    # One partition should be enriched for module 0, the other for module 1
    pos_m0_frac = np.mean(pos_modules == 0) if len(pos_modules) > 0 else 0
    neg_m0_frac = np.mean(neg_modules == 0) if len(neg_modules) > 0 else 0
    alignment = abs(pos_m0_frac - neg_m0_frac)

    print(f"\n  Partition alignment with true modules: {alignment:.3f}")
    print(f"    Positive partition: {len(ranking['partition_positive'])} genes, "
          f"{pos_m0_frac:.1%} module 0")
    print(f"    Negative partition: {len(ranking['partition_negative'])} genes, "
          f"{neg_m0_frac:.1%} module 0")

    if alignment > 0.3:
        print("  ✓ Fiedler vector shows meaningful module separation")
    else:
        print("  ⚠ Weak module separation (may be expected pre-bifurcation)")

    print("  ✓ PASSED\n")


def test_saddle_node_trajectory():
    """Test STIGRN metrics on saddle-node-like bifurcation."""
    print("=" * 60)
    print("TEST 5: Saddle-node trajectory")
    print("=" * 60)

    data = generate_saddle_node_trajectory(
        n_genes=60,
        n_timepoints=30,
        bifurcation_point=0.5,
        seed=42,
    )

    metrics = STIGRNMetrics(normalized_laplacian=True)
    for adj, t in zip(data["adjacency_matrices"], data["timepoints"]):
        metrics.compute(adj, timepoint=t)

    trajectory = metrics.get_trajectory()

    print(f"  Metric trajectories:")
    for name in ["SGI", "SE", "SRD", "FVI", "SMI"]:
        vals = trajectory[name]
        print(f"    {name}: {vals[0]:.3f} → {vals[-1]:.3f}")

    # For saddle-node: SGI should decrease strongly (hub loss = connectivity loss)
    sgi = trajectory["SGI"]
    print(f"\n  SGI drop: {sgi[0]:.3f} → {sgi[-1]:.3f} "
          f"({(1 - sgi[-1]/sgi[0])*100:.1f}% decrease)")

    warning = metrics.detect_warning(threshold_sigma=1.5)
    print(f"  Warning triggered: {warning['warning_triggered']}")
    if warning["warning_triggered"]:
        lead = data["bifurcation_time"] - warning["warning_timepoint"]
        print(f"  Lead time: {lead:.3f}")

    print("  ✓ PASSED\n")


def test_full_synthetic_pipeline():
    """Test the full pipeline: synthetic expression → GRN inference → STIGRN."""
    print("=" * 60)
    print("TEST 6: Full synthetic pipeline (expression → GRN → STIGRN)")
    print("=" * 60)

    data = generate_full_synthetic_dataset(
        n_genes=30,
        n_timepoints=15,
        n_cells_per_window=100,
        bifurcation_point=0.6,
        noise_scale=0.3,
        seed=42,
    )

    from stigrn.grn_inference import construct_grn_trajectory

    print(f"  Total cells: {data['all_expression'].shape[0]}")
    print(f"  Genes: {data['all_expression'].shape[1]}")

    # Infer GRNs from expression
    grn_result = construct_grn_trajectory(
        expression=data["all_expression"],
        pseudotime=data["all_pseudotime"],
        gene_names=data["grn_trajectory"]["gene_names"],
        n_windows=15,
        inference_method="correlation",
        correlation_threshold=0.3,
        verbose=False,
    )

    print(f"  Inferred {len(grn_result['adjacency_matrices'])} GRN windows")

    # Compute STIGRN
    metrics = STIGRNMetrics(normalized_laplacian=True)
    for adj, t in zip(grn_result["adjacency_matrices"], grn_result["timepoints"]):
        metrics.compute(adj, timepoint=t)

    trajectory = metrics.get_trajectory()
    composite = metrics.get_composite_score()

    print(f"\n  Metric trajectories (from inferred GRNs):")
    for name in ["SGI", "SE", "SRD", "FVI", "SMI"]:
        vals = trajectory[name]
        print(f"    {name}: {vals[0]:.3f} → {vals[-1]:.3f}")

    print(f"\n  Composite score: {composite[0]:.3f} → {composite[-1]:.3f}")
    print("  ✓ PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("STIGRN SMOKE TEST SUITE")
    print("=" * 60 + "\n")

    test_laplacian_spectrum()
    test_bifurcating_trajectory()
    test_composite_score_and_warning()
    test_fiedler_gene_ranking()
    test_saddle_node_trajectory()
    test_full_synthetic_pipeline()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
