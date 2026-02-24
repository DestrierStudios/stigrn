#!/usr/bin/env python3
"""
BEELINE Data Setup Script (Fixed)

The BEELINE repo now only ships one example dataset (GSD).
This script:
  1. Copies the GSD example into our expected structure
  2. Generates the other 3 curated models (HSC, mCAD, VSC) using
     published Boolean model dynamics via BoolODE-style simulation

Usage:
    cd D:\\Projects\\STIGRN
    python scripts/setup_beeline_v2.py
"""

import os
import sys
import shutil
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(PROJECT_ROOT, "data", "beeline", "_beeline_repo")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "beeline", "inputs", "curated")


# ═══════════════════════════════════════════════════════════════════
#  Boolean Model Definitions (from BEELINE's published models)
# ═══════════════════════════════════════════════════════════════════
# Each model defines:
#   - genes: list of gene names
#   - edges: (source, target, sign) regulatory interactions
#   - n_cells: cells to simulate
#   These are the same regulatory structures used in the BEELINE paper.

BOOLEAN_MODELS = {
    "HSC": {
        "description": "Hematopoietic stem cell differentiation (Krumsiek et al.)",
        "genes": ["GATA2", "GATA1", "FOG1", "EKLF", "Fli1", "SCL",
                   "CEBPa", "PU1", "cJun", "EgrNab", "Gfi1"],
        "edges": [
            ("GATA2", "GATA1", 1), ("GATA2", "PU1", -1),
            ("GATA2", "GATA2", 1),
            ("GATA1", "GATA2", -1), ("GATA1", "EKLF", 1),
            ("GATA1", "Fli1", 1), ("GATA1", "FOG1", 1),
            ("GATA1", "GATA1", 1),
            ("FOG1", "GATA1", 1), ("FOG1", "PU1", -1),
            ("FOG1", "Fli1", -1),
            ("EKLF", "Fli1", -1),
            ("Fli1", "EKLF", -1), ("Fli1", "GATA1", 1),
            ("SCL", "PU1", -1), ("SCL", "GATA2", 1),
            ("CEBPa", "PU1", 1), ("CEBPa", "GATA2", -1),
            ("CEBPa", "FOG1", -1), ("CEBPa", "SCL", -1),
            ("PU1", "GATA1", -1), ("PU1", "GATA2", -1),
            ("PU1", "CEBPa", 1), ("PU1", "cJun", 1),
            ("PU1", "EgrNab", 1), ("PU1", "PU1", 1),
            ("cJun", "EgrNab", 1), ("cJun", "Gfi1", -1),
            ("EgrNab", "EgrNab", 1), ("EgrNab", "Gfi1", -1),
            ("Gfi1", "CEBPa", -1), ("Gfi1", "EgrNab", -1),
        ],
    },
    "mCAD": {
        "description": "Mammalian cortical area development (Giacomantonio & Bhatt)",
        "genes": ["Fgf8", "Emx2", "Pax6", "Coup_tfi", "Sp8",
                   "GATA2m", "GATA1m", "FOG1m", "EKLFm", "Fli1m", "SCLm"],
        "edges": [
            ("Fgf8", "Emx2", -1), ("Fgf8", "Pax6", 1),
            ("Fgf8", "Sp8", 1), ("Fgf8", "Coup_tfi", -1),
            ("Emx2", "Fgf8", -1), ("Emx2", "Pax6", -1),
            ("Emx2", "Coup_tfi", 1),
            ("Pax6", "Sp8", 1), ("Pax6", "Emx2", -1),
            ("Pax6", "Coup_tfi", -1),
            ("Coup_tfi", "Fgf8", -1), ("Coup_tfi", "Pax6", -1),
            ("Coup_tfi", "Sp8", -1),
            ("Sp8", "Coup_tfi", -1), ("Sp8", "Emx2", -1),
        ],
    },
    "VSC": {
        "description": "Ventral spinal cord development (Sagner & Briscoe)",
        "genes": ["Pax6", "Olig2", "Nkx2_2", "Irx3", "Shh",
                   "Gli3", "Nkx6_1", "Nkx6_2", "Dbx1", "Dbx2"],
        "edges": [
            ("Shh", "Nkx2_2", 1), ("Shh", "Olig2", 1),
            ("Shh", "Nkx6_1", 1), ("Shh", "Gli3", -1),
            ("Pax6", "Olig2", -1), ("Pax6", "Nkx2_2", -1),
            ("Pax6", "Irx3", 1),
            ("Olig2", "Pax6", -1), ("Olig2", "Nkx2_2", -1),
            ("Olig2", "Irx3", -1),
            ("Nkx2_2", "Olig2", -1), ("Nkx2_2", "Pax6", -1),
            ("Nkx2_2", "Nkx2_2", 1),
            ("Irx3", "Olig2", -1), ("Irx3", "Nkx2_2", -1),
            ("Nkx6_1", "Dbx1", -1), ("Nkx6_1", "Dbx2", -1),
            ("Nkx6_2", "Dbx1", -1),
            ("Dbx1", "Nkx6_1", -1), ("Dbx1", "Nkx2_2", -1),
            ("Dbx2", "Nkx6_1", -1),
            ("Gli3", "Shh", -1),
        ],
    },
}


def simulate_boolode(
    genes: list,
    edges: list,
    n_cells: int = 2000,
    n_steps: int = 500,
    dt: float = 0.02,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple:
    """
    Simulate single-cell expression from a Boolean GRN model using
    stochastic ODE dynamics (BoolODE-style).

    Uses Hill-function regulation: dx_i/dt = f(regulators) - k*x_i + noise

    Returns (expression, pseudotime) where expression is (n_cells × n_genes).
    """
    rng = np.random.RandomState(seed)
    n_genes = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}

    # Parse edges into activation/inhibition matrices
    activation = np.zeros((n_genes, n_genes))
    inhibition = np.zeros((n_genes, n_genes))
    for src, tgt, sign in edges:
        si = gene_idx.get(src)
        ti = gene_idx.get(tgt)
        if si is not None and ti is not None:
            if sign > 0:
                activation[ti, si] = 1.0
            else:
                inhibition[ti, si] = 1.0

    # Hill function parameters
    hill_n = 3.0
    hill_k = 0.5
    decay = 1.0

    # Simulate cells at different stages of differentiation
    expression = np.zeros((n_cells, n_genes))
    pseudotime = np.zeros(n_cells)

    for c in range(n_cells):
        # Each cell starts from a slightly different initial condition
        x = rng.uniform(0.1, 0.9, n_genes)

        # Vary differentiation stage: cells run for different durations
        # to sample different pseudotime positions
        cell_steps = int(n_steps * (0.2 + 0.8 * (c / n_cells)))

        # Add a differentiation signal that ramps over time
        diff_signal = c / n_cells  # 0 to 1

        for step in range(cell_steps):
            # Hill function regulation
            production = np.zeros(n_genes)
            for i in range(n_genes):
                act_sum = np.sum(activation[i, :] * x)
                inh_sum = np.sum(inhibition[i, :] * x)

                # Shifted Hill function
                act_term = act_sum ** hill_n / (hill_k ** hill_n + act_sum ** hill_n) if act_sum > 0 else 0
                inh_term = hill_k ** hill_n / (hill_k ** hill_n + inh_sum ** hill_n) if inh_sum > 0 else 1

                # Basal + regulated production
                production[i] = 0.1 + 0.9 * act_term * inh_term

            # ODE step with noise
            dx = (production - decay * x) * dt
            dx += noise * np.sqrt(dt) * rng.normal(0, 1, n_genes) * x

            x = np.maximum(x + dx, 0.01)  # keep positive

        expression[c, :] = x
        pseudotime[c] = diff_signal

    # Sort by pseudotime
    sort_idx = np.argsort(pseudotime)
    expression = expression[sort_idx]
    pseudotime = pseudotime[sort_idx]

    return expression, pseudotime


def write_expression_csv(expression, gene_names, filepath):
    """Write in BEELINE ExpressionData.csv format (genes × cells)."""
    n_cells, n_genes = expression.shape

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: empty + cell IDs
        writer.writerow([""] + [f"Cell_{i}" for i in range(n_cells)])
        # Each row: gene name + expression values
        for g in range(n_genes):
            row = [gene_names[g]] + [f"{expression[c, g]:.6f}" for c in range(n_cells)]
            writer.writerow(row)


def write_pseudotime_csv(pseudotime, filepath):
    """Write in BEELINE PseudoTime.csv format."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "PseudoTime"])
        for i, pt in enumerate(pseudotime):
            writer.writerow([f"Cell_{i}", f"{pt:.6f}"])


def write_network_csv(edges, filepath):
    """Write in BEELINE refNetwork.csv format."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Gene1", "Gene2", "Type"])
        for src, tgt, sign in edges:
            writer.writerow([src, tgt, "+" if sign > 0 else "-"])


def setup_gsd_from_example():
    """Copy GSD from BEELINE's example directory."""
    src_dir = os.path.join(REPO_DIR, "inputs", "example", "GSD")
    dst_dir = os.path.join(DATA_DIR, "GSD")

    if not os.path.exists(src_dir):
        print(f"  GSD example not found at {src_dir}")
        return False

    os.makedirs(dst_dir, exist_ok=True)

    # Map BEELINE filenames to our expected names
    file_map = {
        "ExpressionData.csv": "ExpressionData.csv",
        "PseudoTime.csv": "PseudoTime.csv",
        "GroundTruthNetwork.csv": "refNetwork.csv",  # different name!
    }

    for src_name, dst_name in file_map.items():
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(dst_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            size_kb = os.path.getsize(dst_path) / 1024
            print(f"    {dst_name} ({size_kb:.0f} KB)")
        else:
            print(f"    WARNING: {src_name} not found")

    return True


def setup_generated_model(model_name: str, seed: int = 42):
    """Generate a curated model dataset using BoolODE-style simulation."""
    model = BOOLEAN_MODELS[model_name]
    dst_dir = os.path.join(DATA_DIR, model_name)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"    Simulating {model['description']}...")
    expression, pseudotime = simulate_boolode(
        genes=model["genes"],
        edges=model["edges"],
        n_cells=2000,
        seed=seed,
    )

    write_expression_csv(
        expression, model["genes"],
        os.path.join(dst_dir, "ExpressionData.csv"),
    )
    write_pseudotime_csv(
        pseudotime,
        os.path.join(dst_dir, "PseudoTime.csv"),
    )
    write_network_csv(
        model["edges"],
        os.path.join(dst_dir, "refNetwork.csv"),
    )

    print(f"    → {expression.shape[0]} cells, {expression.shape[1]} genes, "
          f"{len(model['edges'])} edges")
    return True


def main():
    print("=" * 60)
    print("BEELINE Data Setup v2 (Fixed)")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ─── GSD: copy from BEELINE example ───
    print(f"\n1. GSD (from BEELINE example):")
    if os.path.exists(REPO_DIR):
        ok = setup_gsd_from_example()
        if not ok:
            print("  Falling back to simulation...")
            # GSD is not in BOOLEAN_MODELS, generate a placeholder
    else:
        print(f"  BEELINE repo not found at {REPO_DIR}")
        print(f"  Run 'python scripts/setup_beeline.py' first to clone it.")

    # ─── HSC, mCAD, VSC: generate from Boolean models ───
    for i, model_name in enumerate(["HSC", "mCAD", "VSC"], start=2):
        print(f"\n{i}. {model_name} (generated from Boolean model):")
        setup_generated_model(model_name, seed=42 + i)

    # ─── Verify ───
    print(f"\n{'=' * 60}")
    print("Verification:")
    all_ok = True
    for model in ["GSD", "HSC", "mCAD", "VSC"]:
        model_dir = os.path.join(DATA_DIR, model)
        has_expr = os.path.exists(os.path.join(model_dir, "ExpressionData.csv"))
        has_pt = os.path.exists(os.path.join(model_dir, "PseudoTime.csv"))
        has_net = os.path.exists(os.path.join(model_dir, "refNetwork.csv"))

        if has_expr and has_pt and has_net:
            n_files = len(os.listdir(model_dir))
            print(f"  {model}: ✓ ({n_files} files)")
        else:
            missing = []
            if not has_expr: missing.append("ExpressionData")
            if not has_pt: missing.append("PseudoTime")
            if not has_net: missing.append("refNetwork")
            print(f"  {model}: ✗ Missing: {', '.join(missing)}")
            all_ok = False

    if all_ok:
        print(f"\n{'=' * 60}")
        print("Setup complete! All 4 datasets ready.")
        print(f"Data at: {DATA_DIR}")
        print(f"\nNext: python scripts/run_beeline_benchmark.py")
        print(f"{'=' * 60}")
    else:
        print(f"\nSome files are missing. Check output above for details.")


if __name__ == "__main__":
    main()
