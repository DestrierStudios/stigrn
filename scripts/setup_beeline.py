#!/usr/bin/env python3
"""
BEELINE Data Setup Script

Downloads and organizes the BEELINE benchmark datasets for STIGRN evaluation.

The BEELINE framework (Pratapa et al., Nature Methods 2020) provides:
  - 4 curated Boolean GRN models (HSC, mCAD, VSC, GSD)
  - 6 synthetic trajectory networks (LI, LL, CY, BF, BFC, TF)
  - Ground truth networks for each

Usage:
    cd D:\\Projects\\STIGRN
    python scripts/setup_beeline.py

This will create:
    data/beeline/inputs/curated/{HSC,mCAD,VSC,GSD}/
    data/beeline/inputs/synthetic/{LI,LL,CY,BF,BFC,TF}/
"""

import os
import sys
import subprocess
import shutil


BEELINE_REPO = "https://github.com/Murali-group/Beeline.git"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "beeline")


def check_git():
    """Verify git is available."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def clone_beeline(target_dir: str):
    """Shallow clone the BEELINE repository."""
    if os.path.exists(target_dir):
        print(f"  BEELINE repo already exists at {target_dir}")
        return

    print(f"  Cloning BEELINE repository (shallow)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", BEELINE_REPO, target_dir],
        check=True,
    )
    print(f"  Done.")


def organize_data(beeline_clone_dir: str, output_dir: str):
    """
    Copy relevant data files from the BEELINE clone into our project structure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ─── Curated Boolean models ───
    curated_src = os.path.join(beeline_clone_dir, "inputs", "Curated")
    curated_dst = os.path.join(output_dir, "inputs", "curated")

    curated_models = ["HSC", "mCAD", "VSC", "GSD"]

    for model in curated_models:
        src = os.path.join(curated_src, model)
        dst = os.path.join(curated_dst, model)

        if not os.path.exists(src):
            # Try alternate paths used in different BEELINE versions
            alt_src = os.path.join(beeline_clone_dir, "inputs", "curated", model)
            if os.path.exists(alt_src):
                src = alt_src
            else:
                print(f"  WARNING: {model} not found at {src}, skipping")
                continue

        os.makedirs(dst, exist_ok=True)

        # Copy expression, pseudotime, and reference network files
        for pattern in ["ExpressionData", "PseudoTime", "refNetwork",
                        "GeneOrdering"]:
            for fname in os.listdir(src):
                if fname.startswith(pattern):
                    shutil.copy2(os.path.join(src, fname), dst)

        n_files = len(os.listdir(dst))
        print(f"  {model}: {n_files} files copied")

    # ─── Synthetic networks ───
    synthetic_src = os.path.join(beeline_clone_dir, "inputs", "Synthetic")
    synthetic_dst = os.path.join(output_dir, "inputs", "synthetic")

    synthetic_models = ["LI", "LL", "CY", "BF", "BFC", "TF"]

    for model in synthetic_models:
        src = os.path.join(synthetic_src, model)
        dst = os.path.join(synthetic_dst, model)

        if not os.path.exists(src):
            alt_src = os.path.join(beeline_clone_dir, "inputs", "synthetic", model)
            if os.path.exists(alt_src):
                src = alt_src
            else:
                # Synthetic names sometimes include suffixes
                found = False
                parent = os.path.dirname(src)
                if os.path.exists(parent):
                    for d in os.listdir(parent):
                        if d.startswith(model):
                            src = os.path.join(parent, d)
                            found = True
                            break
                if not found:
                    print(f"  WARNING: {model} not found, skipping")
                    continue

        os.makedirs(dst, exist_ok=True)

        for fname in os.listdir(src):
            if any(fname.startswith(p) for p in
                   ["ExpressionData", "PseudoTime", "refNetwork", "GeneOrdering"]):
                shutil.copy2(os.path.join(src, fname), dst)

        n_files = len(os.listdir(dst))
        print(f"  {model}: {n_files} files copied")


def verify_setup(output_dir: str):
    """Check that key files are in place."""
    print("\n  Verification:")
    expected = {
        "curated": ["HSC", "mCAD", "VSC", "GSD"],
        "synthetic": ["LI", "LL", "CY", "BF", "BFC", "TF"],
    }

    all_ok = True
    for dtype, models in expected.items():
        for model in models:
            expr_path = os.path.join(output_dir, "inputs", dtype, model,
                                     "ExpressionData.csv")
            # Also check for numbered variants
            expr_path_0 = os.path.join(output_dir, "inputs", dtype, model,
                                        "ExpressionData_0.csv")
            exists = os.path.exists(expr_path) or os.path.exists(expr_path_0)
            status = "✓" if exists else "✗ MISSING"
            if not exists:
                all_ok = False
            print(f"    {dtype}/{model}: {status}")

    return all_ok


def main():
    print("=" * 60)
    print("BEELINE Data Setup for STIGRN")
    print("=" * 60)

    if not check_git():
        print("\nERROR: git not found. Please install git first.")
        print("  Windows: https://git-scm.com/download/win")
        sys.exit(1)

    clone_dir = os.path.join(DATA_DIR, "_beeline_repo")

    print(f"\nStep 1: Clone BEELINE repository")
    clone_beeline(clone_dir)

    print(f"\nStep 2: Organize data files")
    organize_data(clone_dir, DATA_DIR)

    print(f"\nStep 3: Verify setup")
    ok = verify_setup(DATA_DIR)

    if ok:
        print(f"\n{'=' * 60}")
        print("Setup complete! Data is ready at:")
        print(f"  {DATA_DIR}")
        print(f"{'=' * 60}")
    else:
        print(f"\nSome datasets are missing. The BEELINE repo structure may have")
        print(f"changed. Check: {clone_dir}")
        print(f"You may need to manually copy files into the expected structure.")
        print(f"See stigrn/beeline_loader.py for expected paths.")

    # Offer to clean up the clone
    print(f"\nThe BEELINE repo clone is at: {clone_dir}")
    print(f"You can delete it after setup to save space (~200 MB).")


if __name__ == "__main__":
    main()
