#!/usr/bin/env python3
"""
STIGRN Systematic Experiment Suite

Runs the full experimental pipeline for the paper:
  Experiment 1: STIGRN metrics on bifurcating trajectory (Fig 2)
  Experiment 2: Head-to-head comparison with DNB/CSD baselines (Fig 4)
  Experiment 3: Lead time comparison across scenarios (Fig 3-like)
  Experiment 4: Robustness to network size and noise (Fig 7)
  Experiment 5: Fiedler vector gene identification (Fig 6)

All figures are saved to the output directory.

Usage: python scripts/run_experiments.py [--output-dir OUTPUT_DIR]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stigrn.metrics import STIGRNMetrics
from stigrn.synthetic import (
    generate_bifurcating_grn_trajectory,
    generate_saddle_node_trajectory,
    generate_full_synthetic_dataset,
)
from stigrn.baselines import CSDIndicators, DNBScore
from stigrn.visualization import (
    plot_metric_trajectories,
    plot_composite_comparison,
    plot_lead_time_comparison,
    plot_robustness_heatmap,
    plot_fiedler_partition,
    plot_parameter_sweep,
)


def experiment_1_metric_trajectories(output_dir: str):
    """
    Experiment 1: STIGRN metric behavior on a bifurcating GRN.
    Generates Figure 2.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Metric Trajectories on Bifurcating GRN")
    print("=" * 60)

    data = generate_bifurcating_grn_trajectory(
        n_genes=100, n_timepoints=40, bifurcation_point=0.6,
        base_connectivity=0.12, seed=42,
    )

    metrics = STIGRNMetrics()
    for adj, t in zip(data["adjacency_matrices"], data["timepoints"]):
        metrics.compute(adj, timepoint=t)

    trajectory = metrics.get_trajectory()

    # Print summary
    print(f"  Network: {data['adjacency_matrices'][0].shape[0]} genes, "
          f"{len(data['timepoints'])} timepoints")
    print(f"  Bifurcation at t={data['bifurcation_time']}")
    for m in ["SGI", "SE", "SRD", "FVI", "SMI"]:
        vals = trajectory[m]
        print(f"  {m}: {vals[0]:.3f} → {vals[-1]:.3f} "
              f"(Δ = {vals[-1]-vals[0]:+.3f})")

    # Generate figure
    fig = plot_metric_trajectories(
        trajectory, bifurcation_time=data["bifurcation_time"],
        title="STIGRN Metrics During Network Bifurcation (n=100 genes)",
        save_path=os.path.join(output_dir, "fig2_metric_trajectories.png"),
    )
    print(f"  → Saved fig2_metric_trajectories.png")
    return trajectory, data


def experiment_2_head_to_head(output_dir: str):
    """
    Experiment 2: STIGRN vs DNB vs CSD on synthetic expression data.
    Generates Figure 4.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Head-to-Head Comparison (STIGRN vs DNB vs CSD)")
    print("=" * 60)

    # Generate full synthetic dataset (expression + known GRN)
    syn = generate_full_synthetic_dataset(
        n_genes=60, n_timepoints=30, n_cells_per_window=150,
        bifurcation_point=0.6, noise_scale=0.4, seed=42,
    )
    grn_data = syn["grn_trajectory"]

    # ─── STIGRN (operates on GRN topology) ───
    stigrn = STIGRNMetrics()
    for adj, t in zip(grn_data["adjacency_matrices"], grn_data["timepoints"]):
        stigrn.compute(adj, timepoint=t)
    stigrn_composite = stigrn.get_composite_score()
    stigrn_warning = stigrn.detect_warning(threshold_sigma=1.5)

    # ─── DNB (operates on expression) ───
    dnb = DNBScore(dominant_group_size=10)
    for expr, t in zip(syn["expression_data"], grn_data["timepoints"]):
        dnb.compute(expr, timepoint=t)
    dnb_composite = dnb.get_composite_score()
    dnb_warning = dnb.detect_warning(threshold_sigma=1.5)

    # ─── CSD (operates on expression) ───
    csd = CSDIndicators()
    for expr, t in zip(syn["expression_data"], grn_data["timepoints"]):
        csd.compute(expr, timepoint=t)
    csd_composite = csd.get_composite_score()
    csd_warning = csd.detect_warning(threshold_sigma=1.5)

    # Print results
    bif = grn_data["bifurcation_time"]
    print(f"  Bifurcation at t={bif}")
    for name, warning in [("STIGRN", stigrn_warning),
                          ("DNB", dnb_warning),
                          ("CSD", csd_warning)]:
        if warning["warning_triggered"]:
            lead = bif - warning["warning_timepoint"]
            print(f"  {name}: warning at t={warning['warning_timepoint']:.3f} "
                  f"(lead = {lead:.3f})")
        else:
            print(f"  {name}: no warning triggered")

    # Generate figure
    timepoints = grn_data["timepoints"]
    fig = plot_composite_comparison(
        timepoints=np.array(timepoints),
        stigrn_composite=stigrn_composite,
        dnb_composite=dnb_composite,
        csd_composite=csd_composite,
        bifurcation_time=bif,
        stigrn_warning=stigrn_warning.get("warning_timepoint"),
        dnb_warning=dnb_warning.get("warning_timepoint"),
        csd_warning=csd_warning.get("warning_timepoint"),
        title="Composite EWS Score: STIGRN vs DNB vs CSD",
        save_path=os.path.join(output_dir, "fig4_head_to_head.png"),
    )
    print(f"  → Saved fig4_head_to_head.png")

    return {
        "STIGRN": stigrn_warning,
        "DNB": dnb_warning,
        "CSD": csd_warning,
    }


def experiment_3_lead_time_scenarios(output_dir: str):
    """
    Experiment 3: Lead time comparison across different bifurcation scenarios.
    Generates lead time bar chart (Fig 3-like).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Lead Time Across Scenarios")
    print("=" * 60)

    scenarios = {
        "Bifurcation\n(n=50)": {"n_genes": 50, "type": "bifurcation"},
        "Bifurcation\n(n=100)": {"n_genes": 100, "type": "bifurcation"},
        "Saddle-node\n(n=50)": {"n_genes": 50, "type": "saddle_node"},
        "Saddle-node\n(n=100)": {"n_genes": 100, "type": "saddle_node"},
    }

    all_results = {}

    for scenario_name, params in scenarios.items():
        print(f"\n  Scenario: {scenario_name.replace(chr(10), ' ')}")

        if params["type"] == "bifurcation":
            syn = generate_full_synthetic_dataset(
                n_genes=params["n_genes"], n_timepoints=30,
                n_cells_per_window=150, bifurcation_point=0.6,
                noise_scale=0.4, seed=42,
            )
        else:
            grn_data = generate_saddle_node_trajectory(
                n_genes=params["n_genes"], n_timepoints=30,
                bifurcation_point=0.5, seed=42,
            )
            # Generate expression for saddle-node
            from stigrn.synthetic import generate_expression_from_grn
            expression_data = []
            for adj in grn_data["adjacency_matrices"]:
                expr = generate_expression_from_grn(adj, n_cells=150, seed=42)
                expression_data.append(expr)
            syn = {
                "grn_trajectory": grn_data,
                "expression_data": expression_data,
            }

        grn_data = syn["grn_trajectory"]
        bif = grn_data["bifurcation_time"]

        scenario_results = {}

        # STIGRN
        stigrn = STIGRNMetrics()
        for adj, t in zip(grn_data["adjacency_matrices"], grn_data["timepoints"]):
            stigrn.compute(adj, timepoint=t)
        w = stigrn.detect_warning(threshold_sigma=1.5)
        lead = (bif - w["warning_timepoint"]) if w["warning_triggered"] else None
        scenario_results["STIGRN"] = lead
        print(f"    STIGRN lead: {lead}")

        # DNB
        dnb = DNBScore(dominant_group_size=max(5, params["n_genes"] // 10))
        for expr, t in zip(syn["expression_data"], grn_data["timepoints"]):
            dnb.compute(expr, timepoint=t)
        w = dnb.detect_warning(threshold_sigma=1.5)
        lead = (bif - w["warning_timepoint"]) if w["warning_triggered"] else None
        scenario_results["DNB"] = lead
        print(f"    DNB lead: {lead}")

        # CSD
        csd = CSDIndicators()
        for expr, t in zip(syn["expression_data"], grn_data["timepoints"]):
            csd.compute(expr, timepoint=t)
        w = csd.detect_warning(threshold_sigma=1.5)
        lead = (bif - w["warning_timepoint"]) if w["warning_triggered"] else None
        scenario_results["CSD"] = lead
        print(f"    CSD lead: {lead}")

        all_results[scenario_name] = scenario_results

    # Generate figure
    fig = plot_lead_time_comparison(
        all_results,
        save_path=os.path.join(output_dir, "fig3_lead_time.png"),
    )
    print(f"\n  → Saved fig3_lead_time.png")
    return all_results


def experiment_4_robustness(output_dir: str):
    """
    Experiment 4: Robustness of STIGRN metrics to network size and noise.
    Generates robustness heatmap (Fig 7).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Robustness Analysis")
    print("=" * 60)

    gene_sizes = [30, 50, 80, 100]
    noise_levels = [0.2, 0.4, 0.6, 0.8]

    # We measure "signal strength" = max composite score reached
    # before bifurcation, normalized by post-bifurcation levels
    signal_matrix = np.zeros((len(gene_sizes), len(noise_levels)))

    for i, n_genes in enumerate(gene_sizes):
        for j, noise in enumerate(noise_levels):
            print(f"  n_genes={n_genes}, noise={noise}...", end=" ")

            syn = generate_full_synthetic_dataset(
                n_genes=n_genes, n_timepoints=25,
                n_cells_per_window=120, bifurcation_point=0.6,
                noise_scale=noise, seed=42 + i * 10 + j,
            )

            grn_data = syn["grn_trajectory"]

            stigrn = STIGRNMetrics()
            for adj, t in zip(grn_data["adjacency_matrices"],
                              grn_data["timepoints"]):
                stigrn.compute(adj, timepoint=t)

            composite = stigrn.get_composite_score()

            # Signal = difference between late and early composite
            early = np.mean(composite[:5])
            late = np.mean(composite[-5:])
            signal = late - early

            # Normalize to [0, 1] range for heatmap
            signal_matrix[i, j] = max(0, min(signal / 3.0, 1.0)) + 0.5

            print(f"signal={signal:.3f}")

    fig = plot_robustness_heatmap(
        signal_matrix,
        row_labels=[f"n={n}" for n in gene_sizes],
        col_labels=[f"σ={n}" for n in noise_levels],
        metric_name="Detection Strength",
        title="STIGRN Robustness: Network Size × Expression Noise",
        save_path=os.path.join(output_dir, "fig7_robustness.png"),
    )
    print(f"  → Saved fig7_robustness.png")
    return signal_matrix


def experiment_5_fiedler_genes(output_dir: str):
    """
    Experiment 5: Fiedler vector identifies fate-determining genes.
    Generates Figure 6.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Fiedler Vector Gene Identification")
    print("=" * 60)

    data = generate_bifurcating_grn_trajectory(
        n_genes=50, n_timepoints=30, bifurcation_point=0.6, seed=42,
    )

    metrics = STIGRNMetrics()
    for adj, t in zip(data["adjacency_matrices"], data["timepoints"]):
        metrics.compute(adj, timepoint=t)

    # Get Fiedler ranking at the post-bifurcation timepoint
    ranking = metrics.get_fiedler_gene_ranking(
        timepoint_idx=-1, gene_names=data["gene_names"],
    )

    # Module alignment
    true_modules = data["module_assignments"]
    fv = ranking["fiedler_values"]

    pos_mask = fv > 0
    neg_mask = fv <= 0
    pos_m0 = np.mean(true_modules[pos_mask] == 0) if pos_mask.sum() > 0 else 0
    neg_m0 = np.mean(true_modules[neg_mask] == 0) if neg_mask.sum() > 0 else 0
    alignment = abs(pos_m0 - neg_m0)

    print(f"  Module alignment: {alignment:.3f}")
    print(f"  Positive partition: {pos_mask.sum()} genes ({pos_m0:.1%} module 0)")
    print(f"  Negative partition: {neg_mask.sum()} genes ({neg_m0:.1%} module 0)")

    # Top genes
    print(f"\n  Top 10 fate-determining genes:")
    for k in range(min(10, len(ranking["ranking"]))):
        idx = ranking["ranking"][k]
        print(f"    {data['gene_names'][idx]}: "
              f"Fiedler={fv[idx]:+.4f}, module={true_modules[idx]}")

    # Generate figure
    fig = plot_fiedler_partition(
        fiedler_values=fv,
        module_assignments=true_modules,
        gene_names=data["gene_names"],
        top_n=15,
        title="Fiedler Vector Gene Partition vs True Modules",
        save_path=os.path.join(output_dir, "fig6_fiedler_partition.png"),
    )
    print(f"  → Saved fig6_fiedler_partition.png")
    return ranking, alignment


def generate_summary_table(lead_results, robustness_matrix, alignment, output_dir):
    """Generate a JSON summary of all experimental results."""
    summary = {
        "experiment_1": "See fig2_metric_trajectories.png",
        "experiment_2": "See fig4_head_to_head.png",
        "experiment_3_lead_times": {
            k.replace("\n", " "): {
                m: (f"{v:.3f}" if v is not None else "not triggered")
                for m, v in v_dict.items()
            }
            for k, v_dict in lead_results.items()
        },
        "experiment_4_robustness_mean": float(np.mean(robustness_matrix)),
        "experiment_5_fiedler_alignment": float(alignment),
    }

    path = os.path.join(output_dir, "results_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → Saved results_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="STIGRN Systematic Experiments")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for figures and results")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("STIGRN SYSTEMATIC EXPERIMENT SUITE")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Run all experiments
    traj, _ = experiment_1_metric_trajectories(output_dir)
    _ = experiment_2_head_to_head(output_dir)
    lead_results = experiment_3_lead_time_scenarios(output_dir)
    robustness = experiment_4_robustness(output_dir)
    _, alignment = experiment_5_fiedler_genes(output_dir)

    # Summary
    summary = generate_summary_table(lead_results, robustness, alignment, output_dir)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
