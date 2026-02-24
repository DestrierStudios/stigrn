# STIGRN: Spectral Transition Indicators for Gene Regulatory Networks

A framework for detecting early warning signals of critical transitions in biological systems using spectral properties of time-varying gene regulatory networks inferred from single-cell transcriptomic data.

**Paper:** Saxena N. Spectral Properties of Gene Regulatory Network Topology Provide Early Warning Signals for Critical Transitions in Cell Fate Decisions. *Submitted to PLoS Computational Biology*, 2026.

## Overview

STIGRN derives five interpretable metrics from the Laplacian eigenvalue spectrum of gene regulatory networks:

| Metric | Name | What It Detects |
|--------|------|-----------------|
| **SGI** | Spectral Gap Indicator | Network connectivity loss |
| **SE** | Spectral Entropy | Regulatory mode consolidation |
| **SRD** | Spectral Radius Divergence | Dominant hub emergence |
| **FVI** | Fiedler Vector Instability | Community structure reorganization |
| **SMI** | Spectral Modularity Index | Emerging modular partitioning |

The Fiedler vector additionally identifies fate-determining genes as a byproduct of the spectral computation.

## Key Results

On the BEELINE benchmark:
- AUROC 0.941 on ventral spinal cord development (vs. 0.588 DNB, 0.157 CSD)
- Only method to trigger an early warning on the gonadal sex determination dataset (real scRNA-seq)
- Fiedler vector recovers the Gata6/Nanog fate-decision axis in ESC-to-PrE differentiation without prior biological knowledge

## Quick Start

```bash
pip install numpy scipy matplotlib
```

```python
from stigrn.synthetic import generate_bifurcating_grn_trajectory
from stigrn.metrics import STIGRNMetrics

# Generate synthetic bifurcating network
data = generate_bifurcating_grn_trajectory(n_genes=50, n_timepoints=30)

# Compute STIGRN metrics
m = STIGRNMetrics()
for A in data["adjacency_matrices"]:
    m.compute(A)

# Get trajectory and detect warnings
trajectory = m.get_trajectory()
warning = m.detect_warning(threshold_sigma=1.5)

# Identify fate-determining genes
ranking = m.get_fiedler_gene_ranking(gene_names=data["gene_names"])
```

## Repository Structure

```
stigrn/
├── stigrn/                  # Core Python package
│   ├── __init__.py
│   ├── metrics.py           # LaplacianSpectrum, STIGRNMetrics (5 metrics + composite)
│   ├── grn_inference.py     # Windowed GRN inference (correlation, GRNBoost2)
│   ├── synthetic.py         # Synthetic bifurcation generators
│   ├── baselines.py         # CSD and DNB baseline implementations
│   ├── evaluation.py        # AUROC, AUPRC, lead time, bootstrap CIs
│   ├── beeline_loader.py    # BEELINE benchmark data loader
│   └── visualization.py     # Publication-quality figure generation
├── scripts/
│   ├── generate_paper_figures.py  # Generates Figs 2-6 for the manuscript
│   ├── smoke_test.py              # 6-test validation suite
│   ├── run_experiments.py         # Synthetic experiments (original)
│   ├── run_beeline_benchmark.py   # BEELINE benchmark (Table 1)
│   ├── run_extended_experiments.py # mESC + combined score (Tables 2-4)
│   ├── setup_beeline_v2.py        # Downloads and prepares BEELINE data
│   └── setup_beeline.py           # Alternative BEELINE setup
├── LICENSE                  # MIT License
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducing the Paper

### 1. Smoke tests
```bash
python scripts/smoke_test.py
```

### 2. Generate all manuscript figures (Figs 2-6)
```bash
python scripts/generate_paper_figures.py --output-dir figures/
```

### 3. BEELINE benchmark (Table 1, reproduces Fig 5 values)
```bash
python scripts/setup_beeline_v2.py   # download and prepare data
python scripts/run_beeline_benchmark.py
```

### 4. mESC and combined score analysis (Tables 2-4, reproduces Fig 6 values)
```bash
python scripts/run_extended_experiments.py
```

### 5. Original synthetic experiments (alternative to generate_paper_figures.py)
```bash
python scripts/run_experiments.py
```

## Dependencies

**Required:** numpy (>=1.24), scipy (>=1.10), matplotlib (>=3.7)

**Optional:** arboreto (for GRNBoost2 inference), scanpy/anndata (for real scRNA-seq processing)

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use STIGRN in your research, please cite:

```
Saxena N. Spectral Properties of Gene Regulatory Network Topology Provide
Early Warning Signals for Critical Transitions in Cell Fate Decisions.
PLoS Computational Biology (submitted), 2026.
```

## Author

Nikhil Saxena, Department of Computer Science, Northeastern University

Contact: saxena.ni@northeastern.edu
