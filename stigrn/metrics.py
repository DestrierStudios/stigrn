"""
STIGRN: Spectral Transition Indicators for Gene Regulatory Networks

Core module implementing five spectral metrics derived from the Laplacian
eigenvalue spectrum of time-varying gene regulatory networks, designed to
detect early warning signals of critical transitions (e.g., cell fate decisions).

Metrics:
    1. SGI  - Spectral Gap Indicator (algebraic connectivity tracking)
    2. SE   - Spectral Entropy (eigenvalue distribution diversity)
    3. SRD  - Spectral Radius Divergence (dominant mode emergence)
    4. FVI  - Fiedler Vector Instability (community reorganization rate)
    5. SMI  - Spectral Modularity Index (emerging modular structure)

Author: Nikhil Saxena (Northeastern University)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from typing import Optional, Dict, Tuple, List
import warnings


class LaplacianSpectrum:
    """
    Compute and store the Laplacian eigenvalue spectrum of a weighted graph.

    Parameters
    ----------
    adjacency : np.ndarray or scipy.sparse matrix
        Weighted adjacency matrix (n x n). Can be directed or undirected.
    normalized : bool, default=True
        If True, compute the normalized Laplacian (symmetric normalization).
    symmetrize : bool, default=True
        If True, symmetrize directed adjacency matrices via (A + A^T) / 2.
    n_eigenvalues : int or None, default=None
        Number of smallest eigenvalues to compute. If None, compute all.
        For large networks, computing a subset is much faster.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        normalized: bool = True,
        symmetrize: bool = True,
        n_eigenvalues: Optional[int] = None,
    ):
        self.n = adjacency.shape[0]
        assert adjacency.shape == (self.n, self.n), "Adjacency must be square"

        # Store original and process
        A = adjacency.copy()

        # Remove self-loops
        if sparse.issparse(A):
            A = A.tolil()
            A.setdiag(0)
            A = A.tocsr()
        else:
            np.fill_diagonal(A, 0)

        # Ensure non-negative weights
        if sparse.issparse(A):
            A.data = np.abs(A.data)
        else:
            A = np.abs(A)

        # Symmetrize if requested (for directed networks)
        if symmetrize:
            A = (A + A.T) / 2.0

        self.adjacency = A

        # Compute degree matrix
        if sparse.issparse(A):
            degrees = np.array(A.sum(axis=1)).flatten()
        else:
            degrees = A.sum(axis=1)

        self.degrees = degrees

        # Compute Laplacian
        if normalized:
            self.laplacian, self.eigenvalues, self.eigenvectors = (
                self._compute_normalized_laplacian(A, degrees, n_eigenvalues)
            )
        else:
            self.laplacian, self.eigenvalues, self.eigenvectors = (
                self._compute_unnormalized_laplacian(A, degrees, n_eigenvalues)
            )

    def _compute_unnormalized_laplacian(
        self, A: np.ndarray, degrees: np.ndarray, n_eig: Optional[int]
    ) -> Tuple:
        """L = D - A"""
        if sparse.issparse(A):
            D = sparse.diags(degrees)
            L = D - A
        else:
            L = np.diag(degrees) - A

        eigenvalues, eigenvectors = self._solve_eigenvalues(L, n_eig)
        return L, eigenvalues, eigenvectors

    def _compute_normalized_laplacian(
        self, A: np.ndarray, degrees: np.ndarray, n_eig: Optional[int]
    ) -> Tuple:
        """L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}"""
        # Handle zero-degree nodes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)

        if sparse.issparse(A):
            D_inv_sqrt = sparse.diags(d_inv_sqrt)
            L_norm = sparse.eye(self.n) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L_norm = np.eye(self.n) - D_inv_sqrt @ A @ D_inv_sqrt

        eigenvalues, eigenvectors = self._solve_eigenvalues(L_norm, n_eig)
        return L_norm, eigenvalues, eigenvectors

    def _solve_eigenvalues(
        self, L, n_eig: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of the Laplacian."""
        if n_eig is not None and n_eig < self.n - 1:
            # Sparse eigenvalue solver (ARPACK) for subset
            if not sparse.issparse(L):
                L = sparse.csr_matrix(L)
            eigenvalues, eigenvectors = eigsh(L, k=n_eig, which="SM")
        else:
            # Full decomposition
            if sparse.issparse(L):
                L = L.toarray()
            eigenvalues, eigenvectors = eigh(L)

        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp near-zero eigenvalues to exactly zero
        eigenvalues = np.maximum(eigenvalues, 0.0)

        return eigenvalues, eigenvectors

    @property
    def algebraic_connectivity(self) -> float:
        """λ₂ - the second smallest eigenvalue (Fiedler value)."""
        if len(self.eigenvalues) < 2:
            return 0.0
        return float(self.eigenvalues[1])

    @property
    def fiedler_vector(self) -> np.ndarray:
        """Eigenvector associated with λ₂."""
        if self.eigenvectors.shape[1] < 2:
            return np.zeros(self.n)
        return self.eigenvectors[:, 1]

    @property
    def spectral_radius(self) -> float:
        """Largest eigenvalue."""
        return float(self.eigenvalues[-1])

    @property
    def nontrivial_eigenvalues(self) -> np.ndarray:
        """All eigenvalues except the trivial zero eigenvalue."""
        return self.eigenvalues[self.eigenvalues > 1e-10]


class STIGRNMetrics:
    """
    Compute the five STIGRN early warning signal metrics from a sequence
    of gene regulatory networks along pseudotime.

    Usage
    -----
    >>> metrics = STIGRNMetrics()
    >>> for t, adjacency in enumerate(grn_sequence):
    ...     result = metrics.compute(adjacency, t)
    ...     print(result)
    >>> summary = metrics.get_trajectory()
    """

    def __init__(
        self,
        normalized_laplacian: bool = True,
        symmetrize: bool = True,
        n_eigenvalues: Optional[int] = None,
    ):
        self.normalized = normalized_laplacian
        self.symmetrize = symmetrize
        self.n_eigenvalues = n_eigenvalues

        # Storage for trajectory
        self._spectra: List[LaplacianSpectrum] = []
        self._timepoints: List[float] = []
        self._results: List[Dict[str, float]] = []

    def compute(
        self, adjacency: np.ndarray, timepoint: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute all five STIGRN metrics for a single GRN snapshot.

        Parameters
        ----------
        adjacency : np.ndarray
            Weighted adjacency matrix of the GRN at this timepoint.
        timepoint : float
            Pseudotime value (for trajectory tracking).

        Returns
        -------
        dict
            Keys: 'SGI', 'SE', 'SRD', 'FVI', 'SMI', 'timepoint'
        """
        spectrum = LaplacianSpectrum(
            adjacency,
            normalized=self.normalized,
            symmetrize=self.symmetrize,
            n_eigenvalues=self.n_eigenvalues,
        )

        result = {
            "timepoint": timepoint,
            "SGI": self._spectral_gap_indicator(spectrum),
            "SE": self._spectral_entropy(spectrum),
            "SRD": self._spectral_radius_divergence(spectrum),
            "FVI": self._fiedler_vector_instability(spectrum),
            "SMI": self._spectral_modularity_index(spectrum),
            "lambda2": spectrum.algebraic_connectivity,
            "lambda_max": spectrum.spectral_radius,
            "n_nodes": spectrum.n,
        }

        self._spectra.append(spectrum)
        self._timepoints.append(timepoint)
        self._results.append(result)

        return result

    def _spectral_gap_indicator(self, spectrum: LaplacianSpectrum) -> float:
        """
        SGI(t) = λ₂(t) / λ₂(t₀)

        Tracks relative algebraic connectivity. Values < 1 indicate
        weakening network connectivity relative to baseline.
        Collapse toward 0 signals approaching network fragmentation.
        """
        lambda2 = spectrum.algebraic_connectivity

        if len(self._spectra) == 0:
            # This is the first timepoint; SGI = 1.0 by definition
            return 1.0

        baseline_lambda2 = self._spectra[0].algebraic_connectivity
        if baseline_lambda2 < 1e-12:
            return 0.0

        return lambda2 / baseline_lambda2

    def _spectral_entropy(self, spectrum: LaplacianSpectrum) -> float:
        """
        SE(t) = -Σ pᵢ log₂(pᵢ)  where pᵢ = λᵢ / Σλⱼ  (for λ > 0)

        Measures diversity of regulatory modes. High entropy = distributed
        regulation; dropping entropy = concentration of regulatory control
        in fewer modes, signaling approach to a bifurcation.
        """
        eigs = spectrum.nontrivial_eigenvalues
        if len(eigs) == 0:
            return 0.0

        total = eigs.sum()
        if total < 1e-12:
            return 0.0

        p = eigs / total
        # Filter out zeros to avoid log(0)
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log2(p))

        # Normalize to [0, 1] by dividing by max possible entropy
        max_entropy = np.log2(len(eigs)) if len(eigs) > 1 else 1.0
        if max_entropy < 1e-12:
            return 0.0

        return entropy / max_entropy

    def _spectral_radius_divergence(self, spectrum: LaplacianSpectrum) -> float:
        """
        SRD(t) = λ_max(t) / mean(λ)  (for λ > 0)

        Tracks emergence of dominant regulatory mode. Increasing SRD
        indicates symmetry breaking — one mode (hub) dominates, a
        precursor to fate commitment.
        """
        eigs = spectrum.nontrivial_eigenvalues
        if len(eigs) == 0:
            return 1.0

        mean_eig = np.mean(eigs)
        if mean_eig < 1e-12:
            return 1.0

        return spectrum.spectral_radius / mean_eig

    def _fiedler_vector_instability(self, spectrum: LaplacianSpectrum) -> float:
        """
        FVI(t) = 1 - |cos(v₂(t), v₂(t-1))|

        Measures how rapidly the Fiedler vector (network partition) is
        reorganizing. Values near 0 = stable partition; values near 1 =
        community structure is being completely reorganized.
        """
        if len(self._spectra) == 0:
            return 0.0  # No previous vector to compare

        current_fiedler = spectrum.fiedler_vector
        previous_fiedler = self._spectra[-1].fiedler_vector

        # Handle dimension mismatch (shouldn't happen in practice)
        if len(current_fiedler) != len(previous_fiedler):
            return 1.0

        # Cosine similarity (absolute value to handle sign ambiguity of eigenvectors)
        norm_curr = np.linalg.norm(current_fiedler)
        norm_prev = np.linalg.norm(previous_fiedler)

        if norm_curr < 1e-12 or norm_prev < 1e-12:
            return 1.0

        cos_sim = np.abs(
            np.dot(current_fiedler, previous_fiedler) / (norm_curr * norm_prev)
        )
        # Clamp to [0, 1] for numerical safety
        cos_sim = np.clip(cos_sim, 0.0, 1.0)

        return 1.0 - cos_sim

    def _spectral_modularity_index(self, spectrum: LaplacianSpectrum) -> float:
        """
        SMI(t) = max(gap_k) / mean(gap_k)  for k ≥ 2

        where gap_k = λ_{k+1} - λ_k (consecutive eigenvalue gaps).

        Detects emerging modular structure via spectral gaps. Large SMI
        means one gap dominates (network is partitioning into distinct
        modules) — corresponds to diverging cell fates.
        """
        eigs = spectrum.nontrivial_eigenvalues
        if len(eigs) < 3:
            return 1.0

        gaps = np.diff(eigs)
        if len(gaps) == 0:
            return 1.0

        mean_gap = np.mean(gaps)
        if mean_gap < 1e-12:
            return 1.0

        return np.max(gaps) / mean_gap

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Return all computed metrics as a trajectory dictionary.

        Returns
        -------
        dict
            Keys: 'timepoints', 'SGI', 'SE', 'SRD', 'FVI', 'SMI', etc.
            Values: np.ndarray of length T (number of timepoints computed).
        """
        if not self._results:
            return {}

        keys = self._results[0].keys()
        trajectory = {}
        for key in keys:
            trajectory[key] = np.array([r[key] for r in self._results])

        return trajectory

    def get_composite_score(
        self, weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute a composite STIGRN score by z-scoring each metric and
        taking a weighted combination.

        The composite score is designed so that *increasing values*
        indicate *approaching critical transition*.

        Parameters
        ----------
        weights : dict, optional
            Weights for each metric. Default: equal weights.
            Keys should be 'SGI', 'SE', 'SRD', 'FVI', 'SMI'.

        Returns
        -------
        np.ndarray
            Composite score at each timepoint.
        """
        if not self._results:
            return np.array([])

        if weights is None:
            weights = {"SGI": 1.0, "SE": 1.0, "SRD": 1.0, "FVI": 1.0, "SMI": 1.0}

        trajectory = self.get_trajectory()
        composite = np.zeros(len(self._results))

        for metric, weight in weights.items():
            values = trajectory[metric]

            # Z-score
            mu = np.mean(values)
            sigma = np.std(values)
            if sigma < 1e-12:
                z = np.zeros_like(values)
            else:
                z = (values - mu) / sigma

            # For SGI and SE, *decreasing* values signal transition,
            # so we negate them to make the composite uniformly "increasing = warning"
            if metric in ("SGI", "SE"):
                z = -z

            composite += weight * z

        # Normalize by total weight
        total_weight = sum(weights.values())
        if total_weight > 0:
            composite /= total_weight

        return composite

    def detect_warning(
        self,
        threshold_sigma: float = 2.0,
        min_consecutive: int = 2,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Detect early warning signal based on composite score exceeding
        a threshold.

        Parameters
        ----------
        threshold_sigma : float
            Number of standard deviations above mean for warning trigger.
        min_consecutive : int
            Minimum consecutive windows exceeding threshold to trigger warning.
        weights : dict, optional
            Metric weights for composite score.

        Returns
        -------
        dict
            'warning_triggered': bool
            'warning_timepoint': float or None (first warning timepoint)
            'composite_scores': np.ndarray
            'threshold': float
        """
        composite = self.get_composite_score(weights)
        if len(composite) < 3:
            return {
                "warning_triggered": False,
                "warning_timepoint": None,
                "composite_scores": composite,
                "threshold": np.nan,
            }

        mu = np.mean(composite)
        sigma = np.std(composite)
        threshold = mu + threshold_sigma * sigma

        above_threshold = composite > threshold

        # Find first run of min_consecutive True values
        warning_timepoint = None
        count = 0
        for i, above in enumerate(above_threshold):
            if above:
                count += 1
                if count >= min_consecutive:
                    warning_idx = i - min_consecutive + 1
                    warning_timepoint = self._timepoints[warning_idx]
                    break
            else:
                count = 0

        return {
            "warning_triggered": warning_timepoint is not None,
            "warning_timepoint": warning_timepoint,
            "composite_scores": composite,
            "threshold": threshold,
        }

    def get_fiedler_gene_ranking(
        self, timepoint_idx: int = -1, gene_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Extract gene importance ranking from the Fiedler vector at a
        given timepoint. Genes at extreme ends of the Fiedler vector
        are candidate fate-determining regulators.

        Parameters
        ----------
        timepoint_idx : int
            Index into the computed timepoints (-1 for last).
        gene_names : list of str, optional
            Gene names corresponding to adjacency matrix indices.

        Returns
        -------
        dict
            'partition_positive': indices (or names) of genes in positive partition
            'partition_negative': indices (or names) of genes in negative partition
            'ranking': indices sorted by absolute Fiedler value (most extreme first)
            'fiedler_values': the raw Fiedler vector values
        """
        spectrum = self._spectra[timepoint_idx]
        fv = spectrum.fiedler_vector

        ranking = np.argsort(-np.abs(fv))  # Descending by absolute value
        positive_mask = fv > 0
        negative_mask = fv <= 0

        result = {
            "fiedler_values": fv,
            "ranking": ranking,
            "partition_positive": np.where(positive_mask)[0],
            "partition_negative": np.where(negative_mask)[0],
        }

        if gene_names is not None:
            gene_names = np.array(gene_names)
            result["ranking_names"] = gene_names[ranking]
            result["partition_positive_names"] = gene_names[positive_mask]
            result["partition_negative_names"] = gene_names[negative_mask]

        return result

    def reset(self):
        """Clear all stored data for a fresh computation."""
        self._spectra.clear()
        self._timepoints.clear()
        self._results.clear()
