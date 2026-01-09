"""
Circulatory Fidelity: A Prior Predictive Diagnostic for Mean-Field Variational Inference

This module provides tools for computing Inference Coupling (IC), a diagnostic
that quantifies structural coupling between variables to predict MFVI failure.

PRIMARY DIAGNOSTIC:
    IC = |ρ|  (for Gaussians, equivalent to Linfoot correlation)
    
For non-Gaussian distributions, copula-based estimation is provided:
    1. Rank-transform to uniform marginals
    2. Apply normal probability integral transform
    3. Compute Pearson correlation of transformed variables
    4. Convert to Linfoot correlation: IC = sqrt(1 - exp(-2*MI))

INPUT HANDLING:
    All functions accept various array-like inputs:
    - NumPy arrays (any numeric dtype)
    - Python lists
    - PyTorch tensors (automatically converted via .detach().cpu().numpy())
    - TensorFlow tensors
    - JAX arrays
    All inputs are automatically converted to float64 for numerical stability.

COMPANION METRICS:
    - Balance Factor (B): sqrt(σ_min² / σ_max²) - architectural characterization
    - Control Coupling (CC): directed influence measure

LEGACY SUPPORT:
    The original CF = I(z;x) / min(H(z),H(x)) is retained for backwards
    compatibility but is deprecated. The Relational Invariance Theorem proves
    that IC (based on ρ alone) is sufficient for Gaussian inference diagnostics.

Reference
---------
"Circulatory Fidelity: Quantifying Structural Coupling to Diagnose 
Mean-Field Failure in Hierarchical Models" (2025)

License: MIT
"""

from __future__ import annotations
import numpy as np
from scipy.stats import rankdata, norm, pearsonr
from scipy.special import digamma
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Union
import warnings

__version__ = "1.1.1"

# =============================================================================
# INPUT HANDLING UTILITIES
# =============================================================================

def _to_numpy_float64(x) -> np.ndarray:
    """
    Convert input to numpy float64 array, handling various input types.
    
    Supports: numpy arrays, lists, PyTorch tensors, TensorFlow tensors,
    JAX arrays, and other array-like objects.
    
    Parameters
    ----------
    x : array-like
        Input data (numpy array, list, torch tensor, etc.)
    
    Returns
    -------
    np.ndarray
        Numpy array with float64 dtype
    """
    # Handle PyTorch tensors
    if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        # PyTorch tensor
        try:
            x = x.detach().cpu().numpy()
        except Exception:
            # Fallback for edge cases
            x = np.array(x.tolist())
    
    # Handle TensorFlow tensors
    elif hasattr(x, 'numpy') and hasattr(x, 'device'):
        try:
            x = x.numpy()
        except Exception:
            x = np.array(x)
    
    # Handle JAX arrays
    elif type(x).__module__.startswith('jax'):
        x = np.array(x)
    
    # Convert to numpy array
    x = np.asarray(x)
    
    # Convert to float64 for numerical stability
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    elif x.dtype != np.float64:
        x = x.astype(np.float64)
    
    return x


def _validate_inputs(x: np.ndarray, y: np.ndarray, min_samples: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and prepare inputs for IC computation.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input arrays
    min_samples : int
        Minimum required samples
    
    Returns
    -------
    x, y : np.ndarray
        Validated and flattened arrays
    
    Raises
    ------
    ValueError
        If inputs are invalid
    """
    x = _to_numpy_float64(x).flatten()
    y = _to_numpy_float64(y).flatten()
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length. Got {len(x)} and {len(y)}")
    
    if len(x) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples. Got {len(x)}")
    
    # Check for all-NaN
    if np.all(~np.isfinite(x)) or np.all(~np.isfinite(y)):
        raise ValueError("Input arrays contain only NaN/Inf values")
    
    return x, y


# =============================================================================
# PRIMARY DIAGNOSTIC: INFERENCE COUPLING (IC)
# =============================================================================

def inference_coupling(x: np.ndarray, y: np.ndarray, method: str = 'copula') -> Tuple[float, float]:
    """
    Compute Inference Coupling (IC) between two variables.
    
    IC is the primary diagnostic for predicting MFVI failure. For Gaussians,
    IC = |ρ|. For non-Gaussians, IC equals the Linfoot correlation.
    
    RECOMMENDED WORKFLOW: Use the default copula method for all applications.
    The copula estimator is exact for Gaussian data AND provides conservative
    estimates for non-Gaussian data, enabling a unified workflow without
    needing to verify distributional assumptions.
    
    Parameters
    ----------
    x : np.ndarray
        First variable, shape (n_samples,)
    y : np.ndarray
        Second variable, shape (n_samples,)
    method : str
        Estimation method:
        - 'copula' (default, recommended): Works for both Gaussian and 
          non-Gaussian. Exact for Gaussians, conservative for non-Gaussians.
        - 'pearson': Direct |ρ|. Exact for Gaussians only; biased for 
          non-Gaussian marginals. Use only when Gaussianity is verified.
        - 'ksg': k-nearest neighbor MI estimation. Use for validation or
          when non-monotonic dependence is suspected.
    
    Returns
    -------
    ic : float
        Inference Coupling in [0, 1]
    se : float
        Standard error (Fisher transform for copula/pearson, bootstrap for ksg)
    
    Notes
    -----
    The copula method applies rank transformation followed by probit transform,
    which recovers the Gaussian copula correlation. For Gaussian data, this
    equals |ρ| exactly (differences < 0.001). For non-Gaussian data with
    monotonic dependence, it provides a conservative lower bound on true IC.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = 0.8 * x + 0.6 * np.random.randn(1000)
    >>> ic, se = inference_coupling(x, y)  # copula method (recommended)
    >>> print(f"IC = {ic:.3f} ± {se:.3f}")
    IC = 0.800 ± 0.019
    """
    # Robust input conversion (handles PyTorch, TensorFlow, JAX, etc.)
    x = _to_numpy_float64(x).flatten()
    y = _to_numpy_float64(y).flatten()
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    n = len(x)
    
    if method == 'copula':
        return _ic_copula(x, y)
    elif method == 'pearson':
        rho, _ = pearsonr(x, y)
        ic = np.abs(rho)
        # Fisher transform standard error
        se = 1.0 / np.sqrt(n - 3) if n > 3 else np.nan
        return ic, se
    elif method == 'ksg':
        # KSG estimation with bootstrap SE
        mi = mutual_information_ksg(x, y)
        ic = np.sqrt(1 - np.exp(-2 * mi))
        # Bootstrap SE estimate
        se = _bootstrap_se(x, y, n_bootstrap=100)
        return ic, se
    else:
        raise ValueError(f"Unknown method: {method}. Use 'copula', 'pearson', or 'ksg'")


def _ic_copula(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Copula-based IC estimation (recommended for all applications).
    
    Algorithm:
    1. Rank-transform to uniform marginals U[0,1]
    2. Apply inverse normal CDF (probit transform)
    3. Compute Pearson correlation of transformed variables
    4. IC = |ρ| (equivalent to Linfoot correlation)
    
    Key properties:
    - EXACT for Gaussian data: Returns |ρ| with differences < 0.001 from 
      direct Pearson correlation. The rank→probit transformation preserves
      Gaussian structure.
    - CONSERVATIVE for non-Gaussian data: Provides lower bound on true IC
      for distributions with monotonic dependence structure.
    - UNIFIED WORKFLOW: No need to verify Gaussianity before estimation.
    - Closed-form standard errors via Fisher transform.
    - Returns ~0 for non-monotonic dependence (triggers Stage 2 protocol).
    """
    n = len(x)
    
    # Handle NaN/Inf by filtering
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(valid_mask):
        warnings.warn(f"Removing {np.sum(~valid_mask)} non-finite values from input")
        x = x[valid_mask]
        y = y[valid_mask]
        n = len(x)
        if n < 4:
            return np.nan, np.nan
    
    # Check for constant arrays (zero variance)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        warnings.warn("Input array is constant or near-constant; IC is undefined")
        return np.nan, np.nan
    
    # Step 1: Rank transform to uniform marginals
    # Use (rank - 0.5) / n to avoid boundary issues
    u = (rankdata(x) - 0.5) / n
    v = (rankdata(y) - 0.5) / n
    
    # Step 2: Transform to standard normal (probit transform)
    z_x = norm.ppf(u)
    z_y = norm.ppf(v)
    
    # Step 3: Compute Pearson correlation of transformed variables
    # Handle potential warnings from constant arrays
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rho, _ = pearsonr(z_x, z_y)
        except Exception:
            return np.nan, np.nan
    
    # Handle NaN from pearsonr
    if not np.isfinite(rho):
        return np.nan, np.nan
    
    # Step 4: Convert to Linfoot correlation
    # For Gaussians: MI = -0.5 * log(1 - rho^2)
    # Linfoot: r_L = sqrt(1 - exp(-2*MI)) = |rho|
    ic = np.abs(rho)
    
    # Fisher transform standard error
    se = 1.0 / np.sqrt(n - 3) if n > 3 else np.nan
    
    return float(ic), float(se)


def _bootstrap_se(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 100) -> float:
    """Bootstrap standard error for KSG-based IC."""
    n = len(x)
    ic_samples = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        mi = mutual_information_ksg(x[idx], y[idx], k=3)
        ic_samples.append(np.sqrt(1 - np.exp(-2 * max(0, mi))))
    
    return np.std(ic_samples)


# =============================================================================
# GAUSSIAN CASE (Closed-form)
# =============================================================================

def ic_gaussian(rho: float) -> float:
    """
    Compute IC for bivariate Gaussian (closed-form).
    
    For Gaussians, IC = |ρ| exactly. This is equivalent to the Linfoot
    correlation and is the primary diagnostic.
    
    Parameters
    ----------
    rho : float
        Pearson correlation coefficient in [-1, 1]
    
    Returns
    -------
    float
        IC value in [0, 1]
    """
    return np.abs(np.clip(rho, -1.0, 1.0))


def mutual_information_gaussian(rho: float) -> float:
    """
    Compute mutual information for bivariate Gaussian.
    
    I(X; Y) = -0.5 * log(1 - ρ²)
    
    Parameters
    ----------
    rho : float
        Pearson correlation coefficient in (-1, 1)
    
    Returns
    -------
    float
        Mutual information in nats
    """
    rho = np.clip(rho, -0.9999, 0.9999)
    return -0.5 * np.log(1 - rho**2)


def linfoot_correlation(rho: float) -> float:
    """
    Compute Linfoot informational correlation.
    
    r_L = sqrt(1 - exp(-2*I(X;Y)))
    
    For Gaussians, r_L = |ρ| exactly.
    
    Parameters
    ----------
    rho : float
        Pearson correlation coefficient
    
    Returns
    -------
    float
        Linfoot correlation in [0, 1]
    """
    mi = mutual_information_gaussian(rho)
    return np.sqrt(1 - np.exp(-2 * mi))


def check_nonmonotonic_dependence(x: np.ndarray, y: np.ndarray, 
                                   threshold: float = 0.15) -> Dict[str, Any]:
    """
    Check for non-monotonic dependencies that copula IC may miss.
    
    The copula estimator is invariant to monotonic transformations but returns
    IC ≈ 0 for non-monotonic relationships (e.g., Y = X², V-shapes, circles).
    This function detects such cases by comparing linear IC with quadratic IC.
    
    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray  
        Second variable
    threshold : float
        Difference threshold for flagging non-monotonicity (default: 0.15)
    
    Returns
    -------
    dict with keys:
        - 'ic_linear': Standard copula IC
        - 'ic_quadratic': IC with quadratic term (x²)
        - 'ic_interaction': IC with interaction term (x·y) if applicable
        - 'nonmonotonic_flag': True if quadratic IC >> linear IC
        - 'recommendation': Diagnostic recommendation
    
    Notes
    -----
    If nonmonotonic_flag is True, the relationship may have U-shaped, 
    V-shaped, or other non-monotonic structure. Consider:
    1. Using KSG estimator instead of copula
    2. Including quadratic/interaction terms in the model
    3. Investigating the functional form of dependence
    
    Example
    -------
    >>> x = np.random.randn(1000)
    >>> y = x**2 + 0.1 * np.random.randn(1000)  # Non-monotonic
    >>> result = check_nonmonotonic_dependence(x, y)
    >>> print(f"Linear IC: {result['ic_linear']:.3f}")
    >>> print(f"Quadratic IC: {result['ic_quadratic']:.3f}")
    >>> if result['nonmonotonic_flag']:
    ...     print("WARNING: Non-monotonic dependence detected!")
    """
    # Robust input conversion
    x = _to_numpy_float64(x).flatten()
    y = _to_numpy_float64(y).flatten()
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length. Got {len(x)} and {len(y)}")
    
    # Standard linear IC
    ic_linear, _ = _ic_copula(x, y)
    
    # IC with quadratic term
    x_sq = x**2
    ic_quad_x, _ = _ic_copula(x_sq, y)
    
    y_sq = y**2
    ic_quad_y, _ = _ic_copula(x, y_sq)
    
    ic_quadratic = max(ic_quad_x, ic_quad_y)
    
    # IC with interaction (for synergy detection)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    interaction = x_centered * y_centered
    ic_interaction, _ = _ic_copula(interaction, y)
    
    # Flag non-monotonicity if quadratic captures much more than linear
    nonmonotonic_flag = (ic_quadratic - ic_linear) > threshold
    
    if nonmonotonic_flag:
        recommendation = (
            f"Non-monotonic dependence detected (IC_quad={ic_quadratic:.3f} >> "
            f"IC_linear={ic_linear:.3f}). Consider using KSG estimator or "
            f"investigating the functional form of dependence."
        )
    elif ic_linear < 0.05 and ic_interaction > threshold:
        recommendation = (
            f"XOR-type synergy suspected (IC_linear≈0, IC_interaction={ic_interaction:.3f}). "
            f"Apply Two-Stage Protocol for synergy detection."
        )
    else:
        recommendation = "No non-monotonic or synergistic structure detected."
    
    return {
        'ic_linear': ic_linear,
        'ic_quadratic': ic_quadratic,
        'ic_interaction': ic_interaction,
        'nonmonotonic_flag': nonmonotonic_flag,
        'recommendation': recommendation
    }


# =============================================================================
# COMPANION METRICS
# =============================================================================

def balance_factor(sigma_z: float, sigma_x: float) -> float:
    """
    Compute Balance Factor (B) for architectural characterization.
    
    B = sqrt(σ_min² / σ_max²) = σ_min / σ_max
    
    B ∈ (0, 1] where:
    - B → 1: balanced system
    - B → 0: highly asymmetric system
    
    Parameters
    ----------
    sigma_z : float
        Standard deviation of z
    sigma_x : float
        Standard deviation of x
    
    Returns
    -------
    float
        Balance Factor in (0, 1]
    """
    if sigma_z <= 0 or sigma_x <= 0:
        raise ValueError("Standard deviations must be positive")
    
    sigma_min = min(sigma_z, sigma_x)
    sigma_max = max(sigma_z, sigma_x)
    
    return sigma_min / sigma_max


def control_coupling(rho: float, sigma_z: float, sigma_x: float, direction: str = 'z_to_x') -> float:
    """
    Compute Control Coupling (CC) for directed influence.
    
    CC(z → x) = IC² / B = ρ² * (σ_max / σ_min)
    CC(x → z) = IC² * B = ρ² * (σ_min / σ_max)
    
    Note: CC(z→x) * CC(x→z) = IC⁴
    
    Parameters
    ----------
    rho : float
        Correlation coefficient
    sigma_z : float
        Standard deviation of z
    sigma_x : float
        Standard deviation of x
    direction : str
        'z_to_x' or 'x_to_z'
    
    Returns
    -------
    float
        Control Coupling value
    """
    ic = ic_gaussian(rho)
    B = balance_factor(sigma_z, sigma_x)
    
    if direction == 'z_to_x':
        return ic**2 / B
    elif direction == 'x_to_z':
        return ic**2 * B
    else:
        raise ValueError("direction must be 'z_to_x' or 'x_to_z'")


# =============================================================================
# KSG ESTIMATOR (for comparison/validation)
# =============================================================================

def mutual_information_ksg(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """
    Estimate mutual information using KSG estimator.
    
    Note: Copula-based estimation is preferred for IC computation.
    KSG is retained for validation and comparison purposes.
    
    Parameters
    ----------
    X : np.ndarray
        First variable, shape (n_samples,) or (n_samples, d_x)
    Y : np.ndarray
        Second variable, shape (n_samples,) or (n_samples, d_y)
    k : int
        Number of nearest neighbors (default: 5)
    
    Returns
    -------
    float
        Estimated mutual information in nats (non-negative)
    """
    # Ensure numpy arrays with float64 dtype
    X = _to_numpy_float64(X)
    Y = _to_numpy_float64(Y)
    
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    if X.shape[0] == 1:
        X = X.T
    if Y.shape[0] == 1:
        Y = Y.T
    
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of samples")
    
    if n <= k:
        raise ValueError(f"Need more samples than k. Got n={n}, k={k}")
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(Y)):
        warnings.warn("Input contains NaN or Inf values. Results may be unreliable.")
        # Remove rows with NaN/Inf
        valid_mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
        if np.sum(valid_mask) <= k:
            return np.nan
        X = X[valid_mask]
        Y = Y[valid_mask]
        n = X.shape[0]
    
    XY = np.hstack([X, Y])
    
    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)
    
    distances, _ = tree_xy.query(XY, k=k+1, p=float('inf'))
    eps = distances[:, -1]
    eps = np.maximum(eps, 1e-10)
    
    # Compute neighbor counts directly without creating ragged arrays
    # This fixes compatibility with newer NumPy versions
    n_x = np.array([len(tree_x.query_ball_point(X[i], r=eps[i], p=float('inf'))) - 1 
                    for i in range(n)], dtype=np.float64)
    n_y = np.array([len(tree_y.query_ball_point(Y[i], r=eps[i], p=float('inf'))) - 1 
                    for i in range(n)], dtype=np.float64)
    
    # Ensure minimum count of 1 to avoid log(0)
    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)
    
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    
    return max(0, mi)


# =============================================================================
# LEGACY SUPPORT (Deprecated)
# =============================================================================

# Minimum sigma for positive differential entropy (legacy)
SIGMA_MIN = 1.0 / np.sqrt(2 * np.pi * np.e)  # ≈ 0.2420


def differential_entropy_gaussian(sigma: float) -> float:
    """
    Compute differential entropy for univariate Gaussian.
    
    H(X) = 0.5 * log(2πeσ²)
    
    DEPRECATED: This function is retained for backwards compatibility.
    The Relational Invariance Theorem proves that marginal entropies
    cancel in IC computation for Gaussian inference diagnostics.
    
    Parameters
    ----------
    sigma : float
        Standard deviation (must be positive)
    
    Returns
    -------
    float
        Differential entropy in nats
    """
    warnings.warn(
        "differential_entropy_gaussian is deprecated. "
        "Use ic_gaussian(rho) directly for inference diagnostics.",
        DeprecationWarning
    )
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)


def circulatory_fidelity_gaussian(rho: float, sigma_z: float, sigma_x: float) -> float:
    """
    Compute Circulatory Fidelity for bivariate Gaussian (DEPRECATED).
    
    CF = I(z; x) / min(H(z), H(x))
    
    DEPRECATED: This function uses the original min-entropy normalization.
    The Relational Invariance Theorem proves that IC = |ρ| is sufficient
    for Gaussian inference diagnostics. Use ic_gaussian(rho) instead.
    
    Parameters
    ----------
    rho : float
        Correlation coefficient between z and x
    sigma_z : float
        Standard deviation of z
    sigma_x : float
        Standard deviation of x
    
    Returns
    -------
    float
        CF value in [0, 1], or NaN if entropy constraint violated
    """
    warnings.warn(
        "circulatory_fidelity_gaussian is deprecated. "
        "Use ic_gaussian(rho) for inference diagnostics. "
        "The min-entropy normalization adds no predictive value for Gaussians.",
        DeprecationWarning
    )
    mi = mutual_information_gaussian(rho)
    h_z = 0.5 * np.log(2 * np.pi * np.e * sigma_z**2)
    h_x = 0.5 * np.log(2 * np.pi * np.e * sigma_x**2)
    h_min = min(h_z, h_x)
    
    if h_min <= 0:
        return np.nan
    
    return np.clip(mi / h_min, 0.0, 1.0)


# Alias for backwards compatibility
cf_gaussian = circulatory_fidelity_gaussian


# =============================================================================
# DIAGNOSTIC WORKFLOW
# =============================================================================

@dataclass
class ICDiagnostic:
    """
    Result of IC diagnostic computation.
    
    Attributes
    ----------
    ic : float
        Inference Coupling value in [0, 1]
    se : float
        Standard error of IC estimate
    method : str
        Estimation method used
    n : int
        Sample size
    recommendation : str
        Diagnostic recommendation based on thresholds
    """
    ic: float
    se: float
    method: str
    n: int
    recommendation: str
    
    def __repr__(self):
        return (f"ICDiagnostic(ic={self.ic:.3f} ± {self.se:.3f}, "
                f"method='{self.method}', recommendation='{self.recommendation}')")


def diagnose(x: np.ndarray, y: np.ndarray, 
             model_type: str = 'filtering',
             method: str = 'copula') -> ICDiagnostic:
    """
    Run IC diagnostic workflow.
    
    Parameters
    ----------
    x : np.ndarray
        First variable (e.g., latent states)
    y : np.ndarray
        Second variable (e.g., observations)
    model_type : str
        'filtering' (SVF-type) or 'pooling' (HLM-type)
    method : str
        Estimation method: 'copula', 'pearson', or 'ksg'
    
    Returns
    -------
    ICDiagnostic
        Diagnostic result with recommendation
    
    Examples
    --------
    >>> # SVF-type model
    >>> result = diagnose(latent_states, observations, model_type='filtering')
    >>> print(result)
    ICDiagnostic(ic=0.450 ± 0.032, method='copula', recommendation='Use structured VI')
    """
    # Robust input conversion
    x = _to_numpy_float64(x).flatten()
    y = _to_numpy_float64(y).flatten()
    
    ic, se = inference_coupling(x, y, method=method)
    n = len(x)
    
    # Thresholds from manuscript Table (Section 2.7 Interpretive Scale):
    # Negligible: < 0.25 (MFVI safe)
    # Weak: 0.25-0.35 (MFVI likely acceptable)
    # Moderate: 0.35-0.55 (Caution warranted)
    # Strong: 0.55-0.70 (Consider structured inference)
    # Very strong: > 0.70 (Structured inference required)
    
    if model_type == 'filtering':
        # For filtering models: high IC → MFVI fails
        if ic < 0.25:
            recommendation = "MFVI safe (negligible coupling)"
        elif ic < 0.35:
            recommendation = "MFVI likely acceptable (weak coupling)"
        elif ic < 0.55:
            recommendation = "Caution warranted (moderate coupling) - validate post-inference"
        elif ic < 0.70:
            recommendation = "Consider structured inference (strong coupling)"
        else:
            recommendation = "Structured inference required (very strong coupling)"
    elif model_type == 'pooling':
        # For pooling models: interpretation inverts (low IC → no-pooling overfits)
        if ic > 0.70:
            recommendation = "No-pooling acceptable (very strong group separation)"
        elif ic > 0.55:
            recommendation = "Partial pooling optional (strong separation)"
        elif ic > 0.35:
            recommendation = "Partial pooling recommended (moderate homogeneity)"
        else:
            recommendation = "Strong pooling required (weak separation - groups similar)"
    else:
        raise ValueError("model_type must be 'filtering' or 'pooling'")
    
    return ICDiagnostic(ic=ic, se=se, method=method, n=n, recommendation=recommendation)


# =============================================================================
# HIERARCHICAL MODELS
# =============================================================================

def ic_from_icc(icc: float) -> float:
    """
    Compute IC from Intraclass Correlation Coefficient.
    
    For hierarchical linear models: IC = sqrt(ICC)
    
    Parameters
    ----------
    icc : float
        Intraclass correlation coefficient in [0, 1]
    
    Returns
    -------
    float
        IC value in [0, 1]
    """
    if icc < 0 or icc > 1:
        raise ValueError("ICC must be in [0, 1]")
    return np.sqrt(icc)


def icc_from_variances(tau_sq: float, sigma_sq: float) -> float:
    """
    Compute ICC from variance components.
    
    ICC = τ² / (τ² + σ²)
    
    Parameters
    ----------
    tau_sq : float
        Between-group variance
    sigma_sq : float
        Within-group variance
    
    Returns
    -------
    float
        ICC value in [0, 1]
    """
    if tau_sq < 0 or sigma_sq < 0:
        raise ValueError("Variances must be non-negative")
    if tau_sq + sigma_sq == 0:
        raise ValueError("Total variance must be positive")
    return tau_sq / (tau_sq + sigma_sq)


# =============================================================================
# DIMENSIONALITY REDUCTION (Required for high-dimensional data)
# =============================================================================

def reduce_dimensions_pls(X: np.ndarray, Y: np.ndarray, n_components: int = 1,
                         cross_validate: bool = True, cv_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce dimensions using Partial Least Squares.
    
    IMPORTANT: For high-dimensional vectors, dimensionality reduction is
    MANDATORY before IC estimation. This function provides supervised
    reduction that preserves the coupling structure.
    
    WARNING: Standard PLS can overfit when N is small relative to dimension d.
    Cross-validation is enabled by default to ensure the extracted components
    represent genuine coupling rather than spurious correlation.
    
    Parameters
    ----------
    X : np.ndarray
        High-dimensional variable, shape (n_samples, d_x)
    Y : np.ndarray
        Target variable, shape (n_samples,) or (n_samples, d_y)
    n_components : int
        Number of PLS components (default: 1)
    cross_validate : bool
        If True (default), use cross-validation to verify genuine coupling.
        Raises warning if CV score is poor.
    cv_folds : int
        Number of cross-validation folds (default: 5)
    
    Returns
    -------
    X_reduced : np.ndarray
        Reduced X, shape (n_samples, n_components)
    Y_reduced : np.ndarray
        Reduced Y (or original if 1D)
    
    Notes
    -----
    The Manifold Hypothesis suggests that data from coherent generative
    processes concentrate on low-dimensional submanifolds, so this
    reduction typically preserves the relevant coupling structure.
    
    When cross_validate=True, the function verifies that the PLS projection
    captures genuine structure by computing cross-validated R² score.
    If CV-R² < 0.1, a warning is issued indicating potential spurious coupling.
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError("sklearn required for PLS. Install with: pip install scikit-learn")
    
    # Robust input conversion
    X = _to_numpy_float64(X)
    Y = _to_numpy_float64(Y)
    
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    if X.shape[0] == 1:
        X = X.T
    if Y.shape[0] == 1:
        Y = Y.T
    
    n_samples = X.shape[0]
    
    # Cross-validation check for overfitting
    if cross_validate and n_samples >= cv_folds * 2:
        pls_cv = PLSRegression(n_components=n_components)
        cv_scores = cross_val_score(pls_cv, X, Y, cv=cv_folds, scoring='r2')
        cv_r2 = np.mean(cv_scores)
        
        if cv_r2 < 0.1:
            warnings.warn(
                f"Cross-validated R² = {cv_r2:.3f} is low. "
                f"PLS may be capturing spurious correlation rather than genuine coupling. "
                f"Consider increasing sample size or reducing dimensionality.",
                UserWarning
            )
    
    pls = PLSRegression(n_components=n_components)
    X_reduced = pls.fit_transform(X, Y)[0]
    
    if Y.shape[1] == 1:
        Y_reduced = Y.flatten()
    else:
        Y_reduced = Y
    
    return X_reduced.flatten() if n_components == 1 else X_reduced, Y_reduced


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def mse_ratio_predicted(ic: float) -> float:
    """
    Predict MSE ratio from IC (filtering models).
    
    MSE_ratio ≈ 1 / (1 - IC²)
    
    This is the theoretical prediction for Gaussian SVF-type models.
    
    Parameters
    ----------
    ic : float
        Inference Coupling value
    
    Returns
    -------
    float
        Predicted MSE ratio
    """
    ic = np.clip(ic, 0, 0.999)
    return 1.0 / (1.0 - ic**2)


def ic_threshold(target_mse_ratio: float) -> float:
    """
    Compute IC threshold for target MSE ratio.
    
    IC = sqrt(1 - 1/MSE_ratio)
    
    Parameters
    ----------
    target_mse_ratio : float
        Maximum acceptable MSE degradation
    
    Returns
    -------
    float
        IC threshold
    """
    if target_mse_ratio < 1:
        raise ValueError("MSE ratio must be >= 1")
    return np.sqrt(1.0 - 1.0 / target_mse_ratio)


# =============================================================================
# WINDOWED IC FOR TIME SERIES (Maximal Coupling Rule)
# =============================================================================

def windowed_ic(
    z: np.ndarray,
    x: np.ndarray,
    window_size: int = 50,
    step_size: Optional[int] = None,
    method: str = 'copula'
) -> Dict[str, Any]:
    """
    Compute windowed Inference Coupling for non-stationary time series.
    
    The Maximal Coupling Rule: For time series with potential regime changes,
    MFVI suitability depends on IC_max, not the global average. A single
    high-IC episode can invalidate mean-field approximations.
    
    Parameters
    ----------
    z : np.ndarray
        First variable (time series), shape (T,)
    x : np.ndarray
        Second variable (time series), shape (T,)
    window_size : int
        Size of each window (default: 50)
    step_size : int or None
        Step between windows (default: window_size // 4 for 75% overlap)
    method : str
        IC estimation method ('copula' or 'ksg')
    
    Returns
    -------
    dict with keys:
        - 'ic_max': Maximum IC across windows (primary diagnostic)
        - 'ic_mean': Mean IC across windows
        - 'ic_std': Standard deviation of IC across windows
        - 'ic_series': Array of IC values for each window
        - 'window_centers': Array of window center indices
        - 'n_windows': Number of windows computed
        - 'window_size': Window size used
        - 'recommendation': Diagnostic recommendation based on IC_max
    
    Notes
    -----
    The Maximal Coupling Rule mandates using IC_max for time-series diagnostics.
    Global IC can mask transient failures during high-coupling regimes.
    
    For regime-switching models, window_size should exceed expected regime
    duration. When regime structure is unknown, compute IC_max across multiple
    window sizes for robust diagnosis.
    
    IMPORTANT: Window size must be large enough for stable correlation estimates.
    The standard error is SE ≈ 1/sqrt(W-3). For W < 30, estimates have high
    variance and may produce noise-driven false positives. We recommend W >= 50.
    
    Example
    -------
    >>> result = windowed_ic(z_series, x_series, window_size=50)
    >>> print(f"IC_max = {result['ic_max']:.3f}")
    >>> if result['ic_max'] > 0.5:
    ...     print("WARNING: High transient coupling detected")
    """
    MIN_WINDOW_SIZE = 30  # Minimum for stable correlation estimates
    RECOMMENDED_WINDOW_SIZE = 50
    
    # Robust input conversion
    z = _to_numpy_float64(z).flatten()
    x = _to_numpy_float64(x).flatten()
    T = len(z)
    
    if len(x) != T:
        raise ValueError("z and x must have same length")
    
    if window_size > T:
        raise ValueError(f"window_size ({window_size}) exceeds series length ({T})")
    
    # Warn about small window sizes
    if window_size < MIN_WINDOW_SIZE:
        warnings.warn(
            f"window_size={window_size} is below minimum recommended ({MIN_WINDOW_SIZE}). "
            f"Standard error SE ≈ {1/np.sqrt(window_size-3):.3f} is high, "
            f"which may produce noise-driven false positives. "
            f"Consider using window_size >= {RECOMMENDED_WINDOW_SIZE}.",
            UserWarning
        )
    
    if step_size is None:
        step_size = max(1, window_size // 4)  # 75% overlap default
    
    # Compute IC in each window
    ic_values = []
    window_centers = []
    
    start = 0
    while start + window_size <= T:
        z_window = z[start:start + window_size]
        x_window = x[start:start + window_size]
        
        try:
            ic, _ = inference_coupling(z_window, x_window, method=method)
            if np.isfinite(ic):
                ic_values.append(ic)
                window_centers.append(start + window_size // 2)
        except Exception:
            # Skip windows with estimation failures
            pass
        
        start += step_size
    
    if len(ic_values) == 0:
        return {
            'ic_max': np.nan,
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'ic_series': np.array([]),
            'window_centers': np.array([]),
            'n_windows': 0,
            'window_size': window_size,
            'recommendation': 'Insufficient data for windowed analysis'
        }
    
    ic_series = np.array(ic_values)
    ic_max = np.max(ic_series)
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    
    # Standard error for each window estimate
    se_per_window = 1.0 / np.sqrt(window_size - 3) if window_size > 3 else np.nan
    
    # Recommendation based on IC_max (Maximal Coupling Rule)
    # Thresholds aligned with manuscript interpretive scale (Section 2.7)
    # Note: For windowed analysis, we use IC_max which captures transient peaks
    reliability_note = ""
    if window_size < RECOMMENDED_WINDOW_SIZE:
        reliability_note = f" (Note: SE={se_per_window:.3f} with W={window_size}; consider larger windows)"
    
    if ic_max < 0.25:
        recommendation = f'MFVI safe - negligible coupling (IC_max < 0.25){reliability_note}'
    elif ic_max < 0.35:
        recommendation = f'MFVI likely acceptable - weak coupling (IC_max < 0.35){reliability_note}'
    elif ic_max < 0.55:
        recommendation = f'Caution warranted - moderate coupling (IC_max < 0.55); validate post-inference{reliability_note}'
    elif ic_max < 0.70:
        recommendation = f'Consider structured inference - strong coupling (IC_max < 0.70){reliability_note}'
    else:
        recommendation = f'Structured inference required - very strong coupling (IC_max >= 0.70){reliability_note}'
    
    return {
        'ic_max': ic_max,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_series': ic_series,
        'window_centers': np.array(window_centers),
        'n_windows': len(ic_values),
        'window_size': window_size,
        'se_per_window': se_per_window,
        'recommendation': recommendation
    }


def windowed_ic_envelope(
    z: np.ndarray,
    x: np.ndarray,
    window_sizes: Optional[List[int]] = None,
    method: str = 'copula'
) -> Dict[str, Any]:
    """
    Compute IC_max envelope across multiple window sizes.
    
    Useful when regime structure is unknown. If the envelope shows sensitivity
    to window size, this indicates regime structure warranting investigation.
    
    Parameters
    ----------
    z : np.ndarray
        First variable (time series)
    x : np.ndarray  
        Second variable (time series)
    window_sizes : list of int or None
        Window sizes to evaluate (default: [25, 50, 100, 200])
    method : str
        IC estimation method
    
    Returns
    -------
    dict with keys:
        - 'ic_max_envelope': IC_max for each window size
        - 'window_sizes': Window sizes evaluated
        - 'overall_ic_max': Maximum IC across all windows and sizes
        - 'sensitivity': Std dev of IC_max across window sizes (high = regime structure)
    """
    T = len(z)
    
    if window_sizes is None:
        # Default: range from ~5% to ~40% of series length
        min_w = max(20, T // 20)
        max_w = min(T // 2, T // 3)
        window_sizes = [w for w in [25, 50, 100, 200, 500] if min_w <= w <= max_w]
        if not window_sizes:
            window_sizes = [min(50, T // 2)]
    
    ic_max_values = []
    valid_sizes = []
    
    for w in window_sizes:
        if w > T:
            continue
        result = windowed_ic(z, x, window_size=w, method=method)
        if np.isfinite(result['ic_max']):
            ic_max_values.append(result['ic_max'])
            valid_sizes.append(w)
    
    if not ic_max_values:
        return {
            'ic_max_envelope': np.array([]),
            'window_sizes': np.array([]),
            'overall_ic_max': np.nan,
            'sensitivity': np.nan
        }
    
    return {
        'ic_max_envelope': np.array(ic_max_values),
        'window_sizes': np.array(valid_sizes),
        'overall_ic_max': np.max(ic_max_values),
        'sensitivity': np.std(ic_max_values)
    }


# =============================================================================
# MODULE INFO
# =============================================================================

def version_info():
    """Print version and method information."""
    print(f"Circulatory Fidelity v{__version__}")
    print("\nPrimary Diagnostic: IC = |ρ| (Linfoot correlation)")
    print("Recommended Estimation: Copula-based (rank + probit transform)")
    print("\nCompanion Metrics:")
    print("  - Balance Factor (B): architectural characterization")
    print("  - Control Coupling (CC): directed influence")
    print("\nNotes:")
    print("  - For high-dimensional data, use reduce_dimensions_pls() first")
    print("  - Legacy CF function deprecated (use ic_gaussian instead)")
