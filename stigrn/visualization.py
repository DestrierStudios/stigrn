"""
STIGRN Visualization Module

Generates publication-quality figures for the STIGRN paper.
All figures use a consistent style suitable for PLoS Computational Biology.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from typing import Dict, Optional, List, Tuple


# ─── Publication style ───────────────────────────────────────────────
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
}

METRIC_COLORS = {
    "SGI": "#2166AC",    # blue
    "SE":  "#4393C3",    # light blue
    "SRD": "#D6604D",    # red
    "FVI": "#F4A582",    # salmon
    "SMI": "#762A83",    # purple
}

METRIC_LABELS = {
    "SGI": "Spectral Gap Indicator",
    "SE":  "Spectral Entropy",
    "SRD": "Spectral Radius Divergence",
    "FVI": "Fiedler Vector Instability",
    "SMI": "Spectral Modularity Index",
}

METHOD_COLORS = {
    "STIGRN":  "#2166AC",
    "DNB":     "#B2182B",
    "CSD":     "#4DAF4A",
    "Random":  "#999999",
}


def apply_style():
    plt.rcParams.update(STYLE)


def plot_metric_trajectories(
    trajectory: Dict[str, np.ndarray],
    bifurcation_time: float,
    title: str = "STIGRN Metric Trajectories",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 8.5),
) -> plt.Figure:
    """
    Plot all five STIGRN metrics as subpanels with bifurcation line.

    This is Figure 2 of the paper.
    """
    apply_style()
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    timepoints = trajectory["timepoint"]
    metrics = ["SGI", "SE", "SRD", "FVI", "SMI"]
    panel_labels = ["A", "B", "C", "D", "E"]

    for ax, metric, label in zip(axes, metrics, panel_labels):
        values = trajectory[metric]
        color = METRIC_COLORS[metric]

        ax.plot(timepoints, values, color=color, linewidth=2, zorder=3)
        ax.fill_between(timepoints, values, alpha=0.15, color=color, zorder=2)
        ax.axvline(bifurcation_time, color="#333333", linestyle="--",
                   linewidth=1, alpha=0.7, label="Bifurcation", zorder=1)

        ax.set_ylabel(metric, fontweight="bold", color=color)
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

        # Shade the pre-bifurcation warning zone
        ax.axvspan(bifurcation_time - 0.15, bifurcation_time,
                   alpha=0.08, color="red", zorder=0)

        ax.grid(True, alpha=0.2, linewidth=0.5)

    axes[-1].set_xlabel("Pseudotime", fontsize=10)
    axes[0].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_composite_comparison(
    timepoints: np.ndarray,
    stigrn_composite: np.ndarray,
    dnb_composite: np.ndarray,
    csd_composite: np.ndarray,
    bifurcation_time: float,
    stigrn_warning: Optional[float] = None,
    dnb_warning: Optional[float] = None,
    csd_warning: Optional[float] = None,
    title: str = "Early Warning Signal Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4),
) -> plt.Figure:
    """
    Head-to-head comparison of composite scores from STIGRN, DNB, and CSD.

    This is Figure 4 of the paper.
    """
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot composite scores
    ax.plot(timepoints, stigrn_composite, color=METHOD_COLORS["STIGRN"],
            linewidth=2.5, label="STIGRN", zorder=4)
    ax.plot(timepoints, dnb_composite, color=METHOD_COLORS["DNB"],
            linewidth=2, label="DNB", linestyle="-.", zorder=3)
    ax.plot(timepoints, csd_composite, color=METHOD_COLORS["CSD"],
            linewidth=2, label="CSD", linestyle=":", zorder=3)

    # Bifurcation line
    ax.axvline(bifurcation_time, color="#333333", linestyle="--",
               linewidth=1.2, alpha=0.7, label="Bifurcation")

    # Warning markers
    marker_y = ax.get_ylim()[1] * 0.9
    if stigrn_warning is not None:
        ax.axvline(stigrn_warning, color=METHOD_COLORS["STIGRN"],
                   linestyle=":", linewidth=1, alpha=0.7)
        ax.plot(stigrn_warning, marker_y, marker="v", color=METHOD_COLORS["STIGRN"],
                markersize=10, zorder=5)

    if dnb_warning is not None:
        ax.axvline(dnb_warning, color=METHOD_COLORS["DNB"],
                   linestyle=":", linewidth=1, alpha=0.7)
        ax.plot(dnb_warning, marker_y * 0.85, marker="v", color=METHOD_COLORS["DNB"],
                markersize=10, zorder=5)

    if csd_warning is not None:
        ax.axvline(csd_warning, color=METHOD_COLORS["CSD"],
                   linestyle=":", linewidth=1, alpha=0.7)
        ax.plot(csd_warning, marker_y * 0.8, marker="v", color=METHOD_COLORS["CSD"],
                markersize=10, zorder=5)

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Composite EWS Score (z-scored)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_lead_time_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """
    Bar chart comparing lead times across methods and scenarios.

    Parameters
    ----------
    results : dict
        Nested dict: results[scenario][method] = lead_time (float or None)
    """
    apply_style()

    scenarios = list(results.keys())
    methods = ["STIGRN", "DNB", "CSD"]

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, method in enumerate(methods):
        lead_times = []
        for scenario in scenarios:
            lt = results[scenario].get(method)
            lead_times.append(lt if lt is not None else 0)

        bars = ax.bar(x + i * width, lead_times, width,
                      label=method, color=METHOD_COLORS[method], alpha=0.85)

        # Add value labels
        for bar, lt in zip(bars, lead_times):
            if lt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{lt:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Lead Time (pseudotime units)")
    ax.set_title("Early Warning Lead Time Comparison", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_robustness_heatmap(
    robustness_data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    metric_name: str = "AUROC",
    title: str = "Robustness Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """
    Heatmap showing metric performance across parameter variations.

    Parameters
    ----------
    robustness_data : np.ndarray
        Matrix of performance values (rows × cols).
    row_labels, col_labels : list of str
        Labels for rows and columns.
    """
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(robustness_data, cmap="RdYlBu_r", aspect="auto",
                   vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = robustness_data[i, j]
            color = "white" if val > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label=metric_name, shrink=0.8)
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_fiedler_partition(
    fiedler_values: np.ndarray,
    module_assignments: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_n: int = 10,
    title: str = "Fiedler Vector Gene Partition",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4),
) -> plt.Figure:
    """
    Visualize Fiedler vector partition with true module labels.

    This is Figure 6 of the paper.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, width_ratios=[2, 1])

    n_genes = len(fiedler_values)
    sorted_idx = np.argsort(fiedler_values)
    sorted_fv = fiedler_values[sorted_idx]
    sorted_modules = module_assignments[sorted_idx]

    # Left panel: sorted Fiedler values colored by true module
    colors = ["#2166AC" if m == 0 else "#B2182B" for m in sorted_modules]
    ax1.barh(range(n_genes), sorted_fv, color=colors, alpha=0.8, height=0.8)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Genes (sorted by Fiedler value)")
    ax1.set_xlabel("Fiedler vector component")
    ax1.set_title("A  Gene partition", fontweight="bold", loc="left")

    # Add module legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166AC", alpha=0.8, label="Module 0"),
        Patch(facecolor="#B2182B", alpha=0.8, label="Module 1"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    # Right panel: top genes by |Fiedler value|
    top_idx = np.argsort(-np.abs(fiedler_values))[:top_n]
    top_fv = fiedler_values[top_idx]
    top_modules = module_assignments[top_idx]
    if gene_names is not None:
        top_names = [gene_names[i] for i in top_idx]
    else:
        top_names = [f"Gene {i}" for i in top_idx]

    colors_top = ["#2166AC" if m == 0 else "#B2182B" for m in top_modules]
    y_pos = range(len(top_names))
    ax2.barh(y_pos, np.abs(top_fv), color=colors_top, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_names, fontsize=7)
    ax2.set_xlabel("|Fiedler value|")
    ax2.set_title("B  Top fate-determining genes", fontweight="bold", loc="left")
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_parameter_sweep(
    sweep_results: Dict[str, Dict[str, np.ndarray]],
    param_name: str,
    param_values: np.ndarray,
    metric_names: List[str] = None,
    title: str = "Parameter Sensitivity",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4),
) -> plt.Figure:
    """
    Plot how STIGRN metrics vary across a parameter sweep.

    Parameters
    ----------
    sweep_results : dict
        sweep_results[metric_name] = {
            'mean': np.ndarray (len = n_param_values),
            'std': np.ndarray
        }
    param_name : str
        Name of the swept parameter.
    param_values : np.ndarray
        Values of the parameter.
    """
    apply_style()

    if metric_names is None:
        metric_names = list(sweep_results.keys())

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for metric in metric_names:
        if metric not in sweep_results:
            continue
        mean = sweep_results[metric]["mean"]
        std = sweep_results[metric]["std"]
        color = METRIC_COLORS.get(metric, "#333333")

        ax.plot(param_values, mean, color=color, linewidth=2, label=metric)
        ax.fill_between(param_values, mean - std, mean + std,
                        alpha=0.15, color=color)

    ax.set_xlabel(param_name)
    ax.set_ylabel("Metric Value (normalized)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
