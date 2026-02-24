#!/usr/bin/env python3
"""
STIGRN BEELINE Benchmark

Runs STIGRN + baselines on all BEELINE curated Boolean models and computes
formal AUROC/AUPRC evaluation. This produces the core benchmarking results
for the paper (Table 1, Figure 3).

Usage:
    python scripts/run_beeline_benchmark.py [--data-dir data/beeline/inputs]

Prerequisites:
    python scripts/setup_beeline.py   (downloads BEELINE data)
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stigrn.metrics import STIGRNMetrics
from stigrn.baselines import CSDIndicators, DNBScore
from stigrn.grn_inference import construct_grn_trajectory
from stigrn.beeline_loader import load_beeline_dataset, list_available_datasets
from stigrn.evaluation import (
    evaluate_method,
    compare_methods,
    bootstrap_auroc,
    label_transition_windows,
)
from stigrn.visualization import (
    plot_metric_trajectories,
    plot_composite_comparison,
    plot_robustness_heatmap,
)


# ─── Bifurcation points for curated models ───
# These are approximate from the known trajectory structures.
# For bifurcating trajectories, the split occurs around the midpoint.
CURATED_BIFURCATION_POINTS = {
    "HSC": 0.5,    # Hematopoietic stem cell: bifurcates midway
    "mCAD": 0.5,   # Cortical area development: bifurcates midway
    "VSC": 0.5,    # Ventral spinal cord: bifurcates midway
    "GSD": 0.5,    # Gonadal sex determination: bifurcates midway
}


def run_single_benchmark(
    dataset_name: str,
    data_dir: str,
    dataset_type: str = "curated",
    n_windows: int = 20,
    output_dir: str = "results/beeline",
    verbose: bool = True,
) -> dict:
    """
    Run full STIGRN + baseline evaluation on one BEELINE dataset.
    """
    if verbose:
        print(f"\n{'─' * 50}")
        print(f"  Dataset: {dataset_name} ({dataset_type})")
        print(f"{'─' * 50}")

    # ─── Load data ───
    try:
        ds = load_beeline_dataset(data_dir, dataset_name, dataset_type)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    if verbose:
        print(f"  Loaded: {ds}")

    bifurcation_time = CURATED_BIFURCATION_POINTS.get(dataset_name, 0.5)

    # ─── Infer time-varying GRNs ───
    if verbose:
        print(f"  Inferring GRNs ({n_windows} windows)...")

    grn_result = construct_grn_trajectory(
        expression=ds.expression,
        pseudotime=ds.pseudotime,
        gene_names=ds.gene_names,
        n_windows=n_windows,
        overlap_fraction=0.3,
        min_cells_per_window=max(15, ds.n_cells // (n_windows * 2)),
        inference_method="correlation",
        correlation_threshold=0.25,
        verbose=False,
    )

    if len(grn_result["adjacency_matrices"]) < 5:
        print(f"  SKIP: Only {len(grn_result['adjacency_matrices'])} windows "
              f"(need ≥5)")
        return None

    timepoints = np.array(grn_result["timepoints"])

    # Normalize timepoints to [0, 1]
    if timepoints.max() > timepoints.min():
        timepoints_norm = (timepoints - timepoints.min()) / (
            timepoints.max() - timepoints.min()
        )
    else:
        timepoints_norm = np.linspace(0, 1, len(timepoints))

    # ─── Partition expression into matching windows ───
    from stigrn.grn_inference import partition_cells_by_pseudotime
    windows = partition_cells_by_pseudotime(
        ds.pseudotime, n_windows=n_windows, overlap_fraction=0.3,
        min_cells_per_window=max(15, ds.n_cells // (n_windows * 2)),
    )
    # Align window count with GRN count
    windows = windows[:len(grn_result["adjacency_matrices"])]

    # ─── STIGRN ───
    if verbose:
        print(f"  Computing STIGRN metrics...")

    stigrn = STIGRNMetrics()
    for adj, t in zip(grn_result["adjacency_matrices"], timepoints_norm):
        stigrn.compute(adj, timepoint=t)

    stigrn_composite = stigrn.get_composite_score()
    stigrn_eval = evaluate_method(
        stigrn_composite, timepoints_norm, bifurcation_time,
        method_name="STIGRN",
    )

    # ─── DNB ───
    if verbose:
        print(f"  Computing DNB scores...")

    dnb = DNBScore(dominant_group_size=max(5, ds.n_genes // 10))
    for (indices, _), t in zip(windows, timepoints_norm):
        expr_window = ds.expression[indices, :]
        dnb.compute(expr_window, timepoint=t)

    dnb_composite = dnb.get_composite_score()
    dnb_eval = evaluate_method(
        dnb_composite, timepoints_norm, bifurcation_time,
        method_name="DNB",
    )

    # ─── CSD ───
    if verbose:
        print(f"  Computing CSD indicators...")

    csd = CSDIndicators()
    for (indices, _), t in zip(windows, timepoints_norm):
        expr_window = ds.expression[indices, :]
        csd.compute(expr_window, timepoint=t)

    csd_composite = csd.get_composite_score()
    csd_eval = evaluate_method(
        csd_composite, timepoints_norm, bifurcation_time,
        method_name="CSD",
    )

    # ─── Bootstrap CI for AUROC ───
    if verbose:
        print(f"  Computing bootstrap CIs...")

    labels = label_transition_windows(timepoints_norm, bifurcation_time)
    stigrn_boot = bootstrap_auroc(stigrn_composite, labels, n_bootstrap=500)
    dnb_boot = bootstrap_auroc(dnb_composite, labels, n_bootstrap=500)
    csd_boot = bootstrap_auroc(csd_composite, labels, n_bootstrap=500)

    # ─── Print comparison ───
    if verbose:
        print(f"\n  Results for {dataset_name}:")
        print(compare_methods([stigrn_eval, dnb_eval, csd_eval]))
        print(f"\n  AUROC 95% CI:")
        print(f"    STIGRN: {stigrn_boot['mean']:.3f} "
              f"[{stigrn_boot['lower']:.3f}, {stigrn_boot['upper']:.3f}]")
        print(f"    DNB:    {dnb_boot['mean']:.3f} "
              f"[{dnb_boot['lower']:.3f}, {dnb_boot['upper']:.3f}]")
        print(f"    CSD:    {csd_boot['mean']:.3f} "
              f"[{csd_boot['lower']:.3f}, {csd_boot['upper']:.3f}]")

    # ─── Save figure ───
    os.makedirs(output_dir, exist_ok=True)

    plot_composite_comparison(
        timepoints=timepoints_norm,
        stigrn_composite=stigrn_composite,
        dnb_composite=dnb_composite,
        csd_composite=csd_composite,
        bifurcation_time=bifurcation_time,
        stigrn_warning=stigrn_eval.get("lead_time"),
        dnb_warning=dnb_eval.get("lead_time"),
        csd_warning=csd_eval.get("lead_time"),
        title=f"EWS Comparison: {dataset_name}",
        save_path=os.path.join(output_dir, f"comparison_{dataset_name}.png"),
    )

    # Save STIGRN metric trajectories
    trajectory = stigrn.get_trajectory()
    plot_metric_trajectories(
        trajectory, bifurcation_time=bifurcation_time,
        title=f"STIGRN Metrics: {dataset_name}",
        save_path=os.path.join(output_dir, f"metrics_{dataset_name}.png"),
    )

    return {
        "dataset": dataset_name,
        "n_cells": ds.n_cells,
        "n_genes": ds.n_genes,
        "n_windows": len(grn_result["adjacency_matrices"]),
        "stigrn": stigrn_eval,
        "dnb": dnb_eval,
        "csd": csd_eval,
        "stigrn_auroc_ci": stigrn_boot,
        "dnb_auroc_ci": dnb_boot,
        "csd_auroc_ci": csd_boot,
    }


def generate_summary_table(all_results: list, output_dir: str):
    """Generate the paper's Table 1: AUROC comparison across datasets."""
    print("\n" + "=" * 70)
    print("TABLE 1: AUROC Comparison Across BEELINE Curated Models")
    print("=" * 70)

    header = (f"{'Dataset':<10} {'Cells':>6} {'Genes':>6} │ "
              f"{'STIGRN':>12} {'DNB':>12} {'CSD':>12}")
    print(header)
    print("─" * len(header))

    table_data = []

    for r in all_results:
        if r is None:
            continue

        stigrn_str = (f"{r['stigrn']['auroc']:.3f} "
                      f"±{r['stigrn_auroc_ci']['std']:.3f}")
        dnb_str = (f"{r['dnb']['auroc']:.3f} "
                   f"±{r['dnb_auroc_ci']['std']:.3f}")
        csd_str = (f"{r['csd']['auroc']:.3f} "
                   f"±{r['csd_auroc_ci']['std']:.3f}")

        row = (f"{r['dataset']:<10} {r['n_cells']:>6} {r['n_genes']:>6} │ "
               f"{stigrn_str:>12} {dnb_str:>12} {csd_str:>12}")
        print(row)

        table_data.append({
            "dataset": r["dataset"],
            "n_cells": r["n_cells"],
            "n_genes": r["n_genes"],
            "stigrn_auroc": r["stigrn"]["auroc"],
            "dnb_auroc": r["dnb"]["auroc"],
            "csd_auroc": r["csd"]["auroc"],
            "stigrn_lead": r["stigrn"]["lead_time"],
            "dnb_lead": r["dnb"]["lead_time"],
            "csd_lead": r["csd"]["lead_time"],
        })

    print("─" * len(header))

    # Compute means
    valid = [r for r in all_results if r is not None]
    if valid:
        mean_stigrn = np.mean([r["stigrn"]["auroc"] for r in valid])
        mean_dnb = np.mean([r["dnb"]["auroc"] for r in valid])
        mean_csd = np.mean([r["csd"]["auroc"] for r in valid])
        print(f"{'Mean':<10} {'':>6} {'':>6} │ "
              f"{mean_stigrn:>12.3f} {mean_dnb:>12.3f} {mean_csd:>12.3f}")

    # Save as JSON
    json_path = os.path.join(output_dir, "beeline_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(table_data, f, indent=2, default=str)
    print(f"\n  → Saved {json_path}")

    # Generate AUROC heatmap
    if valid:
        methods = ["STIGRN", "DNB", "CSD"]
        datasets = [r["dataset"] for r in valid]
        auroc_matrix = np.array([
            [r["stigrn"]["auroc"], r["dnb"]["auroc"], r["csd"]["auroc"]]
            for r in valid
        ])

        plot_robustness_heatmap(
            auroc_matrix,
            row_labels=datasets,
            col_labels=methods,
            metric_name="AUROC",
            title="AUROC: STIGRN vs Baselines on BEELINE Models",
            save_path=os.path.join(output_dir, "table1_auroc_heatmap.png"),
        )
        print(f"  → Saved table1_auroc_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="STIGRN BEELINE Benchmark")
    parser.add_argument("--data-dir", default="data/beeline/inputs",
                        help="BEELINE data directory")
    parser.add_argument("--output-dir", default="results/beeline",
                        help="Output directory")
    parser.add_argument("--n-windows", type=int, default=20,
                        help="Number of pseudotime windows")
    parser.add_argument("--datasets", nargs="*",
                        default=["HSC", "mCAD", "VSC", "GSD"],
                        help="Datasets to benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("STIGRN BEELINE BENCHMARK")
    print("=" * 60)

    # Check data availability
    available = list_available_datasets(args.data_dir)
    print(f"\nAvailable datasets:")
    for dtype, names in available.items():
        if names:
            print(f"  {dtype}: {', '.join(names)}")

    total = sum(len(v) for v in available.values())
    if total == 0:
        print(f"\nNo datasets found in {args.data_dir}")
        print(f"Run 'python scripts/setup_beeline.py' first.")
        sys.exit(1)

    # Run benchmarks
    all_results = []
    for dataset_name in args.datasets:
        # Determine type
        dtype = "curated"
        for t, names in available.items():
            if dataset_name in names:
                dtype = t
                break

        result = run_single_benchmark(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            dataset_type=dtype,
            n_windows=args.n_windows,
            output_dir=args.output_dir,
        )
        all_results.append(result)

    # Generate summary
    generate_summary_table(all_results, args.output_dir)

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"Results in: {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
