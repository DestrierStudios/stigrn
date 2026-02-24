"""
STIGRN Baseline Early Warning Signal Methods

Implements competing EWS methods for head-to-head comparison:
1. CSD indicators: variance, lag-1 autocorrelation, skewness
2. DNB score: Dynamic Network Biomarker (Chen et al., 2012)

These operate on expression data (not network topology), providing
the contrast that motivates STIGRN's topology-aware approach.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import skew


class CSDIndicators:
    """
    Classical Critical Slowing Down indicators computed from
    gene expression time series in sliding windows.

    Metrics:
        - Variance: average gene-wise variance (increases near transition)
        - Autocorrelation: average lag-1 autocorrelation (increases near transition)
        - Skewness: average absolute skewness (may increase near transition)
    """

    def __init__(self):
        self._results: List[Dict[str, float]] = []
        self._timepoints: List[float] = []

    def compute(
        self,
        expression: np.ndarray,
        timepoint: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute CSD indicators for a single expression window.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix (cells × genes) for this window.
        timepoint : float
            Pseudotime center of the window.

        Returns
        -------
        dict with keys 'variance', 'autocorrelation', 'skewness', 'timepoint'
        """
        n_cells, n_genes = expression.shape

        # Gene-wise variance, averaged across genes
        gene_variances = np.var(expression, axis=0)
        avg_variance = np.mean(gene_variances)

        # Gene-wise lag-1 autocorrelation (using cell ordering as proxy for time)
        autocorrs = []
        for g in range(n_genes):
            x = expression[:, g]
            if np.std(x) < 1e-10:
                autocorrs.append(0.0)
                continue
            x_centered = x - np.mean(x)
            if len(x_centered) < 3:
                autocorrs.append(0.0)
                continue
            c0 = np.sum(x_centered ** 2)
            if c0 < 1e-10:
                autocorrs.append(0.0)
                continue
            c1 = np.sum(x_centered[:-1] * x_centered[1:])
            autocorrs.append(c1 / c0)
        avg_autocorr = np.mean(autocorrs)

        # Gene-wise skewness
        gene_skewness = []
        for g in range(n_genes):
            x = expression[:, g]
            if np.std(x) < 1e-10:
                gene_skewness.append(0.0)
            else:
                gene_skewness.append(abs(skew(x)))
        avg_skewness = np.mean(gene_skewness)

        result = {
            "timepoint": timepoint,
            "variance": avg_variance,
            "autocorrelation": avg_autocorr,
            "skewness": avg_skewness,
        }

        self._results.append(result)
        self._timepoints.append(timepoint)
        return result

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        if not self._results:
            return {}
        keys = self._results[0].keys()
        return {k: np.array([r[k] for r in self._results]) for k in keys}

    def get_composite_score(self) -> np.ndarray:
        """
        Composite CSD score: z-scored sum of variance + autocorrelation.
        Increasing values = approaching transition.
        """
        traj = self.get_trajectory()
        if not traj:
            return np.array([])

        composite = np.zeros(len(self._results))
        for metric in ["variance", "autocorrelation"]:
            vals = traj[metric]
            mu, sigma = np.mean(vals), np.std(vals)
            if sigma > 1e-12:
                composite += (vals - mu) / sigma
            
        return composite / 2.0

    def detect_warning(self, threshold_sigma: float = 2.0, min_consecutive: int = 2) -> Dict:
        composite = self.get_composite_score()
        if len(composite) < 3:
            return {"warning_triggered": False, "warning_timepoint": None,
                    "composite_scores": composite, "threshold": np.nan}

        mu, sigma = np.mean(composite), np.std(composite)
        threshold = mu + threshold_sigma * sigma
        above = composite > threshold

        warning_tp = None
        count = 0
        for i, a in enumerate(above):
            if a:
                count += 1
                if count >= min_consecutive:
                    warning_tp = self._timepoints[i - min_consecutive + 1]
                    break
            else:
                count = 0

        return {
            "warning_triggered": warning_tp is not None,
            "warning_timepoint": warning_tp,
            "composite_scores": composite,
            "threshold": threshold,
        }

    def reset(self):
        self._results.clear()
        self._timepoints.clear()


class DNBScore:
    """
    Dynamic Network Biomarker score (simplified).

    Based on Chen et al. (2012): identifies a dominant group of genes
    with high intra-group correlation, high intra-group standard deviation,
    and low correlation with other genes.

    DNB_score = (SD_in * PCC_in) / PCC_out

    Where:
        SD_in   = average std dev of genes in the dominant group
        PCC_in  = average |correlation| within the dominant group
        PCC_out = average |correlation| between dominant group and others
    """

    def __init__(self, dominant_group_size: int = 10):
        """
        Parameters
        ----------
        dominant_group_size : int
            Number of genes in the candidate DNB group.
            In practice, this is identified dynamically; here we use
            the top-N most variable genes as a simplified proxy.
        """
        self.group_size = dominant_group_size
        self._results: List[Dict[str, float]] = []
        self._timepoints: List[float] = []

    def compute(
        self,
        expression: np.ndarray,
        timepoint: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute DNB score for a single expression window.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix (cells × genes).
        timepoint : float
            Pseudotime center.

        Returns
        -------
        dict with 'dnb_score', 'sd_in', 'pcc_in', 'pcc_out', 'timepoint'
        """
        n_cells, n_genes = expression.shape
        group_size = min(self.group_size, n_genes // 2)

        # Select dominant group: top genes by variance
        gene_vars = np.var(expression, axis=0)
        dominant_idx = np.argsort(-gene_vars)[:group_size]
        other_idx = np.array([i for i in range(n_genes) if i not in dominant_idx])

        if len(other_idx) == 0:
            other_idx = dominant_idx  # fallback

        # SD_in: average std dev of dominant group genes
        sd_in = np.mean(np.std(expression[:, dominant_idx], axis=0))

        # Correlation matrix
        corr = np.corrcoef(expression.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # PCC_in: average |correlation| within dominant group
        pcc_in_vals = []
        for i_idx, i in enumerate(dominant_idx):
            for j_idx, j in enumerate(dominant_idx):
                if i < j:
                    pcc_in_vals.append(abs(corr[i, j]))
        pcc_in = np.mean(pcc_in_vals) if pcc_in_vals else 0.0

        # PCC_out: average |correlation| between dominant and others
        pcc_out_vals = []
        for i in dominant_idx:
            for j in other_idx:
                pcc_out_vals.append(abs(corr[i, j]))
        pcc_out = np.mean(pcc_out_vals) if pcc_out_vals else 1e-6

        # DNB composite score
        if pcc_out < 1e-6:
            pcc_out = 1e-6
        dnb_score = (sd_in * pcc_in) / pcc_out

        result = {
            "timepoint": timepoint,
            "dnb_score": dnb_score,
            "sd_in": sd_in,
            "pcc_in": pcc_in,
            "pcc_out": pcc_out,
        }

        self._results.append(result)
        self._timepoints.append(timepoint)
        return result

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        if not self._results:
            return {}
        keys = self._results[0].keys()
        return {k: np.array([r[k] for r in self._results]) for k in keys}

    def get_composite_score(self) -> np.ndarray:
        """Return z-scored DNB score trajectory."""
        traj = self.get_trajectory()
        if not traj:
            return np.array([])
        vals = traj["dnb_score"]
        mu, sigma = np.mean(vals), np.std(vals)
        if sigma < 1e-12:
            return np.zeros_like(vals)
        return (vals - mu) / sigma

    def detect_warning(self, threshold_sigma: float = 2.0, min_consecutive: int = 2) -> Dict:
        composite = self.get_composite_score()
        if len(composite) < 3:
            return {"warning_triggered": False, "warning_timepoint": None,
                    "composite_scores": composite, "threshold": np.nan}

        mu, sigma = np.mean(composite), np.std(composite)
        threshold = mu + threshold_sigma * sigma
        above = composite > threshold

        warning_tp = None
        count = 0
        for i, a in enumerate(above):
            if a:
                count += 1
                if count >= min_consecutive:
                    warning_tp = self._timepoints[i - min_consecutive + 1]
                    break
            else:
                count = 0

        return {
            "warning_triggered": warning_tp is not None,
            "warning_timepoint": warning_tp,
            "composite_scores": composite,
            "threshold": threshold,
        }

    def reset(self):
        self._results.clear()
        self._timepoints.clear()
