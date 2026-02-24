#!/usr/bin/env python3
"""
Generate all figures for the STIGRN manuscript.

Produces Figures 2-6 matching the final manuscript numbering:
  Figure 1: Framework schematic (static asset, not generated here)
  Figure 2: STIGRN metric trajectories with zoomed SE/SRD
  Figure 3: Robustness heatmap (network size x noise)
  Figure 4: Fiedler vector gene partition (synthetic)
  Figure 5: AUROC heatmap across BEELINE models
  Figure 6: mESC Fiedler vector gene ranking

Usage:
  python scripts/generate_paper_figures.py [--output-dir figures/]

Note: Figure 1 is a conceptual schematic created separately.
      Figures 5 and 6 use published values from Tables 1 and 3.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stigrn.metrics import STIGRNMetrics
from stigrn.synthetic import (
    generate_bifurcating_grn_trajectory,
    generate_full_synthetic_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate STIGRN manuscript figures")
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_figure2(output_dir, seed=42):
    """Figure 2: STIGRN metrics during a controlled network bifurcation."""
    print("Generating Figure 2: Metric trajectories...")
    np.random.seed(seed)

    result = generate_bifurcating_grn_trajectory(
        n_genes=100, n_timepoints=40, bifurcation_point=0.6
    )
    m = STIGRNMetrics()
    for A in result["adjacency_matrices"]:
        m.compute(A)
    traj = m.get_trajectory()

    t = np.linspace(0, 1, 40)
    bif = 0.6
    colors = {"SGI": "#2166AC", "SE": "#4393C3", "SRD": "#D6604D",
              "FVI": "#F4A582", "SMI": "#762A83"}

    fig, axes = plt.subplots(5, 1, figsize=(8, 12), dpi=300, sharex=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    for ax, key in zip(axes, ["SGI", "SE", "SRD", "FVI", "SMI"]):
        vals = traj[key]
        c = colors[key]
        ax.fill_between(t, vals, alpha=0.15, color=c)
        ax.plot(t, vals, color=c, linewidth=2)
        ax.axvline(bif, color="gray", linestyle="--", linewidth=1, label="Bifurcation")
        ax.axvspan(bif - 0.15, bif, alpha=0.08, color="red")
        ax.set_ylabel(key, fontsize=12, fontweight="bold", color=c, rotation=0, labelpad=30)
        ax.tick_params(labelsize=9)

        if key == "SE":
            ax.set_ylim(min(vals) - 0.002, max(vals) + 0.002)
            ax.text(0.02, 0.85, f"Range: {min(vals):.4f} to {max(vals):.4f}",
                    transform=ax.transAxes, fontsize=8, color="gray", style="italic")
        elif key == "SRD":
            ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
            ax.text(0.02, 0.85, f"Range: {min(vals):.3f} to {max(vals):.3f}",
                    transform=ax.transAxes, fontsize=8, color="gray", style="italic")

        panel = chr(65 + ["SGI", "SE", "SRD", "FVI", "SMI"].index(key))
        ax.text(-0.08, 1.0, panel, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.8)
    axes[-1].set_xlabel("Pseudotime", fontsize=11)
    fig.suptitle("STIGRN Metrics During Network Bifurcation (n=100 genes)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = os.path.join(output_dir, "Fig2_metric_trajectories.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    for k in ["SGI", "SE", "SRD", "FVI", "SMI"]:
        v = traj[k]
        print(f"  {k}: min={min(v):.4f} max={max(v):.4f} range={max(v)-min(v):.4f}")


def generate_figure3(output_dir, seed=42):
    """Figure 3: Detection strength across network sizes and noise levels."""
    print("Generating Figure 3: Robustness heatmap...")

    gene_sizes = [30, 50, 80, 100]
    noise_levels = [0.2, 0.4, 0.6, 0.8]
    signal_matrix = np.zeros((len(gene_sizes), len(noise_levels)))

    for i, n_genes in enumerate(gene_sizes):
        for j, noise in enumerate(noise_levels):
            print(f"    n={n_genes}, sigma={noise}...", end=" ")
            syn = generate_full_synthetic_dataset(
                n_genes=n_genes, n_timepoints=25,
                n_cells_per_window=120, bifurcation_point=0.6,
                noise_scale=noise, seed=seed + i * 10 + j,
            )
            grn_data = syn["grn_trajectory"]
            stigrn = STIGRNMetrics()
            for adj in grn_data["adjacency_matrices"]:
                stigrn.compute(adj)
            composite = stigrn.get_composite_score()
            early = np.mean(composite[:5])
            late = np.mean(composite[-5:])
            signal = late - early
            strength = max(0, min(signal / 3.0, 1.0)) + 0.5
            signal_matrix[i, j] = strength
            print(f"strength={strength:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    im = ax.imshow(signal_matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f"\u03c3={s}" for s in noise_levels], fontsize=10)
    ax.set_yticks(range(len(gene_sizes)))
    ax.set_yticklabels([f"n={n}" for n in gene_sizes], fontsize=10)
    for i in range(len(gene_sizes)):
        for j in range(len(noise_levels)):
            ax.text(j, i, f"{signal_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color="white" if signal_matrix[i, j] > 0.9 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Detection Strength", fontsize=10)
    ax.set_title("STIGRN Robustness: Network Size \u00d7 Expression Noise",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "Fig3_robustness.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_figure4(output_dir, seed=42):
    """Figure 4: Fiedler vector partition alignment with true modules."""
    print("Generating Figure 4: Fiedler partition (synthetic)...")
    np.random.seed(seed)

    result = generate_bifurcating_grn_trajectory(
        n_genes=50, n_timepoints=30, bifurcation_point=0.6
    )
    modules = result["module_assignments"]
    gene_names = result["gene_names"]

    m = STIGRNMetrics()
    for A in result["adjacency_matrices"]:
        m.compute(A)
    ranking = m.get_fiedler_gene_ranking(gene_names=gene_names)
    fiedler_vals = ranking["fiedler_values"]
    gene_fiedler = {g: fiedler_vals[i] for i, g in enumerate(gene_names)}

    sorted_items = sorted(gene_fiedler.items(), key=lambda x: x[1])
    names = [g for g, _ in sorted_items]
    values = [v for _, v in sorted_items]
    gene_modules = [modules[gene_names.index(g)] for g in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), dpi=300,
                                     gridspec_kw={"width_ratios": [3, 2]})

    colors = ["#2166AC" if mod == 0 else "#B2182B" for mod in gene_modules]
    ax1.barh(range(len(names)), values, color=colors, height=0.7,
             edgecolor="white", linewidth=0.3)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=6)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Fiedler Vector Value", fontsize=10)
    ax1.set_title("A", fontsize=13, fontweight="bold", loc="left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor="#2166AC", label="Module 0"),
                        Patch(facecolor="#B2182B", label="Module 1")],
               loc="lower right", fontsize=9)

    top15 = sorted(gene_fiedler.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    top_names = [g for g, _ in top15]
    top_vals = [v for _, v in top15]
    top_mods = [modules[gene_names.index(g)] for g in top_names]
    top_colors = ["#2166AC" if mod == 0 else "#B2182B" for mod in top_mods]
    ax2.barh(range(len(top_names)), [abs(v) for v in top_vals],
             color=top_colors, height=0.6, edgecolor="white", linewidth=0.3)
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("|Fiedler Value|", fontsize=10)
    ax2.set_title("B", fontsize=13, fontweight="bold", loc="left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    pos_mods = [modules[gene_names.index(g)] for g, v in gene_fiedler.items() if v >= 0]
    neg_mods = [modules[gene_names.index(g)] for g, v in gene_fiedler.items() if v < 0]
    from collections import Counter
    pos_maj = Counter(pos_mods).most_common(1)[0][0]
    correct = sum(1 for mod in pos_mods if mod == pos_maj)
    correct += sum(1 for mod in neg_mods if mod != pos_maj)
    alignment = correct / len(gene_names)

    fig.suptitle(f"Fiedler Vector Gene Partition (alignment: {alignment:.1%})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "Fig4_fiedler_synthetic.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Alignment: {alignment:.1%}")


def generate_figure5(output_dir):
    """Figure 5: AUROC heatmap from Table 1 (BEELINE benchmark)."""
    print("Generating Figure 5: AUROC heatmap...")
    datasets = ["VSC", "GSD", "HSC", "mCAD"]
    methods = ["STIGRN", "DNB", "CSD"]
    auroc = np.array([
        [0.941, 0.588, 0.157],
        [0.608, 0.529, 0.804],
        [0.255, 0.961, 0.549],
        [0.353, 0.275, 0.490],
    ])
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=300)
    im = ax.imshow(auroc, cmap="RdYlGn", vmin=0.1, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for i in range(len(datasets)):
        for j in range(len(methods)):
            val = auroc[i, j]
            best = j == np.argmax(auroc[i])
            weight = "bold" if best else "normal"
            color = "white" if val < 0.35 or val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight=weight, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("AUROC", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "Fig5_AUROC_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_figure6(output_dir):
    """Figure 6: Fiedler vector gene ranking from mESC experiment (Table 3)."""
    print("Generating Figure 6: mESC Fiedler ranking...")
    genes_all = [
        ("Pdgfra", 0.527), ("Gata6", 0.489), ("Fn1", 0.474),
        ("Pou5f1", 0.241), ("Fgf5", 0.152), ("Otx2", 0.138),
        ("Col4a1", 0.125), ("Lama1", 0.118), ("Sox17", 0.105),
        ("Fgf4", 0.088), ("Fgfr2", 0.072), ("Dnmt3b", 0.045),
        ("Dab2", 0.032), ("Sox2", -0.028), ("Tcf15", -0.065),
        ("Gata4", -0.137), ("Tbx3", -0.152), ("Klf2", -0.154),
        ("Esrrb", -0.156), ("Nanog", -0.161),
    ]
    genes_all.sort(key=lambda x: x[1])
    names_all = [g[0] for g in genes_all]
    values_all = [g[1] for g in genes_all]
    colors_all = [
        "#C0392B" if v < -0.1 else "#E74C3C" if v < 0
        else "#2980B9" if v < 0.2 else "#1A5276" for v in values_all
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300,
                                       gridspec_kw={"width_ratios": [3, 2]})
    ax_a.barh(range(len(names_all)), values_all, color=colors_all,
              edgecolor="white", linewidth=0.3, height=0.7)
    ax_a.set_yticks(range(len(names_all)))
    ax_a.set_yticklabels(names_all, fontsize=8)
    ax_a.axvline(x=0, color="black", linewidth=0.8)
    ax_a.set_xlabel("Fiedler Vector Value", fontsize=10)
    ax_a.set_title("A", fontsize=12, fontweight="bold", loc="left")
    ax_a.text(0.35, 1, "PrE fate \u2192", fontsize=9, color="#1A5276",
              fontweight="bold", ha="center", va="bottom",
              transform=ax_a.get_xaxis_transform())
    ax_a.text(-0.12, 1, "\u2190 Pluripotency", fontsize=9, color="#C0392B",
              fontweight="bold", ha="center", va="bottom",
              transform=ax_a.get_xaxis_transform())
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    top10 = [
        (1, "Pdgfra", "+0.527", "PrE marker"),
        (2, "Gata6", "+0.489", "PrE master TF"),
        (3, "Fn1", "+0.474", "PrE ECM"),
        (4, "Pou5f1", "+0.241", "Pluri./early PrE"),
        (5, "Nanog", "\u22120.161", "Pluri. master TF"),
        (6, "Esrrb", "\u22120.156", "Pluri. TF"),
        (7, "Klf2", "\u22120.154", "Pluri. TF"),
        (8, "Tbx3", "\u22120.152", "Pluri. TF"),
        (9, "Fgf5", "+0.152", "Epiblast"),
        (10, "Gata4", "\u22120.137", "PrE TF"),
    ]
    ax_b.axis("off")
    ax_b.set_title("B", fontsize=12, fontweight="bold", loc="left")
    col_labels = ["Rank", "Gene", "Fiedler", "Role"]
    table_data = [[str(r), g, v, role] for r, g, v, role in top10]
    table = ax_b.table(cellText=table_data, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       colWidths=[0.12, 0.22, 0.22, 0.44])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, 11):
        _, _, val, _ = top10[i - 1]
        bg = "#D6EAF8" if val.startswith("+") else "#FADBD8"
        for j in range(4):
            table[i, j].set_facecolor(bg)
            table[i, j].set_edgecolor("white")

    plt.tight_layout()
    path = os.path.join(output_dir, "Fig6_mESC_fiedler.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("STIGRN Manuscript Figure Generation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    print("Note: Figure 1 (framework schematic) is a static asset")
    print("      created separately and not generated by this script.")
    print()

    generate_figure2(args.output_dir, seed=args.seed)
    print()
    generate_figure3(args.output_dir, seed=args.seed)
    print()
    generate_figure4(args.output_dir, seed=args.seed)
    print()
    generate_figure5(args.output_dir)
    print()
    generate_figure6(args.output_dir)
    print()

    print("=" * 60)
    print("All figures generated successfully.")
    print(f"Files in {args.output_dir}/:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            print(f"  {f}")
    print()
    print("Figure 1 (Fig1_framework.png) must be added separately.")
    print("=" * 60)


if __name__ == "__main__":
    main()
