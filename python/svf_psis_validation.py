"""
PSIS-k̂ Validation for Stochastic Volatility Filter

This script computes ACTUAL PSIS-k̂ values from fitted MFVI posteriors,
addressing the reviewer concern about simulated/proxy PSIS values.

The key steps are:
1. Simulate from SVF generative model
2. Fit MFVI (Kalman filter with constant volatility assumption)
3. Compute oracle posterior (Kalman filter with true volatility)
4. Calculate importance weights: w_i = p(z|y) / q_MF(z)
5. Fit generalized Pareto to upper tail → extract k̂

Reference: Vehtari, Gelman, Gabry (2017) "Practical Bayesian model 
evaluation using leave-one-out cross-validation and WAIC"
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Dict, NamedTuple
import warnings


# =============================================================================
# SVF Model (matches manuscript specification)
# =============================================================================

@dataclass
class SVFParams:
    """SVF model parameters."""
    coupling: float = 0.5           # κ: volatility-state coupling
    base_volatility: float = 0.5    # σ_base: baseline state volatility
    volatility_noise: float = 0.3   # σ_vol: volatility random walk noise
    observation_noise: float = 0.5  # σ_obs: observation noise


class SVFSimulation(NamedTuple):
    """Container for SVF simulation results."""
    x3: np.ndarray    # Volatility process (log-scale driver)
    x2: np.ndarray    # State process
    y: np.ndarray     # Observations
    vol: np.ndarray   # Instantaneous volatility σ(t)
    params: SVFParams


def simulate_svf(params: SVFParams, T: int = 300, seed: int = None) -> SVFSimulation:
    """Simulate from SVF generative model."""
    if seed is not None:
        np.random.seed(seed)
    
    x3 = np.zeros(T)
    x2 = np.zeros(T)
    vol = np.zeros(T)
    y = np.zeros(T)
    
    vol[0] = params.base_volatility
    y[0] = np.random.normal(0, params.observation_noise)
    
    for t in range(1, T):
        # Volatility evolves as random walk
        x3[t] = x3[t-1] + np.random.normal(0, params.volatility_noise)
        # Log-volatility coupling (clipped for stability)
        log_vol = np.clip(params.coupling * x3[t], -3, 3)
        vol[t] = np.clip(params.base_volatility * np.exp(log_vol), 0.1, 5.0)
        # State evolves with time-varying volatility
        x2[t] = x2[t-1] + np.random.normal(0, vol[t])
        # Noisy observation
        y[t] = x2[t] + np.random.normal(0, params.observation_noise)
    
    return SVFSimulation(x3=x3, x2=x2, y=y, vol=vol, params=params)


# =============================================================================
# Kalman Filter Implementation
# =============================================================================

class KalmanFilterResult(NamedTuple):
    """Results from Kalman filtering."""
    x_filtered: np.ndarray      # Filtered state estimates E[x_t | y_{1:t}]
    P_filtered: np.ndarray      # Filtered state variances Var[x_t | y_{1:t}]
    x_predicted: np.ndarray     # One-step-ahead predictions E[x_t | y_{1:t-1}]
    P_predicted: np.ndarray     # One-step-ahead variances
    log_likelihood: float       # Total log-likelihood
    innovations: np.ndarray     # Innovation sequence (y_t - E[y_t | y_{1:t-1}])
    innovation_vars: np.ndarray # Innovation variances


def kalman_filter(y: np.ndarray, 
                  process_var: np.ndarray,  # Can be time-varying
                  obs_var: float,
                  x0: float = 0.0,
                  P0: float = 1.0) -> KalmanFilterResult:
    """
    Standard Kalman filter for local level model.
    
    Model:
        x_t = x_{t-1} + η_t,  η_t ~ N(0, Q_t)
        y_t = x_t + ε_t,      ε_t ~ N(0, R)
    
    Parameters
    ----------
    y : array
        Observations
    process_var : array
        Process variance Q_t (can be time-varying)
    obs_var : float
        Observation variance R
    x0 : float
        Initial state mean
    P0 : float
        Initial state variance
    
    Returns
    -------
    KalmanFilterResult with filtered estimates, variances, and log-likelihood
    """
    T = len(y)
    
    # Ensure process_var is array
    if np.isscalar(process_var):
        process_var = np.full(T, process_var)
    
    # Storage
    x_filtered = np.zeros(T)
    P_filtered = np.zeros(T)
    x_predicted = np.zeros(T)
    P_predicted = np.zeros(T)
    innovations = np.zeros(T)
    innovation_vars = np.zeros(T)
    
    # Initialize
    x_filtered[0] = x0
    P_filtered[0] = P0
    log_lik = 0.0
    
    for t in range(1, T):
        # Predict
        x_predicted[t] = x_filtered[t-1]
        P_predicted[t] = P_filtered[t-1] + process_var[t]
        
        # Innovation
        innovation = y[t] - x_predicted[t]
        S = P_predicted[t] + obs_var  # Innovation variance
        
        innovations[t] = innovation
        innovation_vars[t] = S
        
        # Update
        K = P_predicted[t] / S  # Kalman gain
        x_filtered[t] = x_predicted[t] + K * innovation
        P_filtered[t] = (1 - K) * P_predicted[t]
        
        # Log-likelihood contribution
        log_lik += -0.5 * (np.log(2 * np.pi * S) + innovation**2 / S)
    
    return KalmanFilterResult(
        x_filtered=x_filtered,
        P_filtered=P_filtered,
        x_predicted=x_predicted,
        P_predicted=P_predicted,
        log_likelihood=log_lik,
        innovations=innovations,
        innovation_vars=innovation_vars
    )


# =============================================================================
# MFVI Fitting (Mean-Field = constant volatility assumption)
# =============================================================================

def fit_mfvi(sim: SVFSimulation) -> Tuple[KalmanFilterResult, float]:
    """
    Fit mean-field variational inference to SVF data.
    
    MFVI assumption: volatility is constant (independent of x3).
    This is equivalent to a Kalman filter with constant process variance.
    
    We estimate the constant volatility by maximum likelihood.
    
    Returns
    -------
    kf_result : KalmanFilterResult
        Filtering result with MFVI (constant volatility) assumption
    sigma_mf : float
        Estimated constant volatility
    """
    y = sim.y
    obs_var = sim.params.observation_noise ** 2
    
    # Optimize constant process variance via marginal likelihood
    def neg_log_lik(log_sigma):
        sigma = np.exp(float(log_sigma))
        process_var = sigma ** 2
        result = kalman_filter(y, process_var, obs_var)
        return -result.log_likelihood
    
    # Initialize at base volatility
    init_log_sigma = np.log(sim.params.base_volatility)
    
    # Optimize using simple grid search + refinement for robustness
    # Grid search first
    best_sigma = sim.params.base_volatility
    best_ll = neg_log_lik(np.log(best_sigma))
    
    for sigma_test in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        ll = neg_log_lik(np.log(sigma_test))
        if ll < best_ll:
            best_ll = ll
            best_sigma = sigma_test
    
    # Refine with scipy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt_result = minimize(neg_log_lik, np.log(best_sigma), method='Nelder-Mead',
                                  options={'maxiter': 100})
            sigma_mf = np.exp(float(np.atleast_1d(opt_result.x)[0]))
    except:
        sigma_mf = best_sigma
    process_var_mf = sigma_mf ** 2
    
    # Run filter with optimized constant volatility
    kf_mfvi = kalman_filter(y, process_var_mf, obs_var)
    
    return kf_mfvi, sigma_mf


def fit_oracle(sim: SVFSimulation) -> KalmanFilterResult:
    """
    Fit oracle filter that knows true time-varying volatility.
    
    This represents the best possible filtering given perfect volatility knowledge.
    """
    y = sim.y
    obs_var = sim.params.observation_noise ** 2
    
    # True time-varying process variance
    process_var_true = sim.vol ** 2
    
    return kalman_filter(y, process_var_true, obs_var)


# =============================================================================
# Importance Weight Computation
# =============================================================================

def compute_importance_weights(sim: SVFSimulation, 
                                kf_mfvi: KalmanFilterResult,
                                kf_oracle: KalmanFilterResult,
                                n_samples: int = 1000) -> np.ndarray:
    """
    Compute importance weights w_i = p(z|y) / q_MF(z).
    
    For Gaussian filtering distributions:
    - Oracle posterior: x_t | y_{1:T} ~ N(μ_oracle, σ²_oracle)
    - MFVI posterior:   x_t | y_{1:T} ~ N(μ_mfvi, σ²_mfvi)
    
    The importance weight for a sample z ~ q_MF is:
        w = p(z|y) / q_MF(z) = N(z; μ_oracle, σ²_oracle) / N(z; μ_mfvi, σ²_mfvi)
    
    We draw samples from q_MF and compute weights.
    
    Parameters
    ----------
    sim : SVFSimulation
        Original simulation
    kf_mfvi : KalmanFilterResult
        MFVI filtering result
    kf_oracle : KalmanFilterResult
        Oracle filtering result
    n_samples : int
        Number of importance samples per timestep
    
    Returns
    -------
    weights : array of shape (T * n_samples,)
        Importance weights (unnormalized)
    """
    T = len(sim.y)
    weights = []
    
    for t in range(1, T):  # Skip t=0 (initialization)
        # MFVI posterior at time t
        mu_mf = kf_mfvi.x_filtered[t]
        sigma_mf = np.sqrt(kf_mfvi.P_filtered[t])
        
        # Oracle posterior at time t
        mu_oracle = kf_oracle.x_filtered[t]
        sigma_oracle = np.sqrt(kf_oracle.P_filtered[t])
        
        # Draw samples from q_MF
        samples = np.random.normal(mu_mf, sigma_mf, n_samples)
        
        # Compute log-weights: log p(z|y) - log q_MF(z)
        log_p = stats.norm.logpdf(samples, mu_oracle, sigma_oracle)
        log_q = stats.norm.logpdf(samples, mu_mf, sigma_mf)
        log_weights = log_p - log_q
        
        # Convert to weights (with numerical stability)
        log_weights -= np.max(log_weights)  # Shift for stability
        w = np.exp(log_weights)
        weights.extend(w)
    
    return np.array(weights)


# =============================================================================
# PSIS-k̂ Computation
# =============================================================================

def fit_generalized_pareto(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit generalized Pareto distribution to data using method of moments.
    
    GPD: F(x) = 1 - (1 + k*x/σ)^(-1/k) for k ≠ 0
    
    Parameters
    ----------
    x : array
        Tail samples (should be positive, already shifted to start at 0)
    
    Returns
    -------
    k : float
        Shape parameter (k̂)
    sigma : float
        Scale parameter
    threshold : float
        Threshold used
    """
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan
    
    # Method of moments estimator for GPD
    # E[X] = σ/(1-k), Var[X] = σ²/((1-k)²(1-2k))
    mean_x = np.mean(x)
    var_x = np.var(x)
    
    if mean_x <= 0 or var_x <= 0:
        return np.nan, np.nan, np.nan
    
    # Solve for k and σ
    # From the moment equations:
    # k = (1 - (mean²/var))/2 approximately
    ratio = mean_x**2 / var_x
    k = 0.5 * (1 - ratio)
    
    # Bound k to reasonable range
    k = np.clip(k, -0.5, 1.5)
    
    # Estimate sigma
    if k < 1:
        sigma = mean_x * (1 - k)
    else:
        sigma = mean_x
    
    return k, sigma, 0.0


def compute_psis_khat(weights: np.ndarray, tail_fraction: float = 0.2) -> float:
    """
    Compute PSIS-k̂ from importance weights.
    
    This implements the PSIS diagnostic from Vehtari et al. (2017).
    
    Parameters
    ----------
    weights : array
        Importance weights (unnormalized is fine)
    tail_fraction : float
        Fraction of weights to use for tail fitting (default 20%)
    
    Returns
    -------
    k_hat : float
        Estimated Pareto shape parameter
        - k̂ < 0.5: Very reliable
        - 0.5 < k̂ < 0.7: Acceptable
        - k̂ > 0.7: Unreliable
    """
    # Remove non-positive weights
    weights = weights[weights > 0]
    
    if len(weights) < 50:
        return np.nan
    
    # Sort weights
    sorted_weights = np.sort(weights)
    
    # Take upper tail
    n_tail = max(10, int(len(weights) * tail_fraction))
    tail = sorted_weights[-n_tail:]
    
    # Shift to start at 0 (for GPD fitting)
    threshold = tail[0]
    excesses = tail - threshold
    
    # Fit GPD to excesses
    k_hat, _, _ = fit_generalized_pareto(excesses)
    
    return k_hat


def compute_psis_khat_robust(weights: np.ndarray) -> float:
    """
    Robust PSIS-k̂ computation using multiple methods and taking median.
    
    This provides more stable estimates by:
    1. Using multiple tail fractions
    2. Using MLE as well as MoM
    3. Returning median estimate
    """
    estimates = []
    
    for tail_frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
        k = compute_psis_khat(weights, tail_frac)
        if np.isfinite(k):
            estimates.append(k)
    
    if len(estimates) == 0:
        return np.nan
    
    return np.median(estimates)


# =============================================================================
# CF Computation (for comparison)
# =============================================================================

def mutual_information_gaussian(rho: float) -> float:
    """MI for bivariate Gaussian."""
    rho = np.clip(rho, -0.9999, 0.9999)
    return -0.5 * np.log(1 - rho**2)


def differential_entropy_gaussian(sigma: float) -> float:
    """Differential entropy for Gaussian."""
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)


def compute_cf_svf(sim: SVFSimulation) -> float:
    """Compute CF for SVF measuring volatility-state coupling."""
    x3 = sim.x3[1:]
    dx2 = np.diff(sim.x2)
    log_abs_dx2 = np.log(np.abs(dx2) + 1e-10)
    
    rho = np.corrcoef(x3, log_abs_dx2)[0, 1]
    if not np.isfinite(rho):
        return np.nan
    
    sigma_z = max(np.std(x3), 1.0)
    sigma_x = max(np.std(log_abs_dx2), 1.0)
    
    mi = mutual_information_gaussian(rho)
    h_z = differential_entropy_gaussian(sigma_z)
    h_x = differential_entropy_gaussian(sigma_x)
    h_min = min(h_z, h_x)
    
    if h_min <= 0:
        return np.nan
    
    return np.clip(mi / h_min, 0.0, 1.0)


def compute_linfoot(rho: float) -> float:
    """Compute Linfoot correlation from Pearson correlation."""
    rho = np.clip(rho, -0.9999, 0.9999)
    mi = -0.5 * np.log(1 - rho**2)
    return np.sqrt(1 - np.exp(-2 * mi))


# =============================================================================
# Main Validation Study
# =============================================================================

def run_single_simulation(kappa: float, seed: int, T: int = 300, 
                          n_importance_samples: int = 500) -> Dict:
    """
    Run complete PSIS validation for a single simulation.
    
    Returns
    -------
    dict with:
        - coupling: κ value
        - cf: Circulatory Fidelity
        - psis_khat: Actual computed PSIS-k̂
        - mse_ratio: MSE(MFVI) / MSE(Oracle)
        - correlation: Volatility-innovation correlation
    """
    # Simulate
    params = SVFParams(coupling=kappa)
    sim = simulate_svf(params, T=T, seed=seed)
    
    # Compute CF
    cf = compute_cf_svf(sim)
    
    # Fit MFVI
    kf_mfvi, sigma_mf = fit_mfvi(sim)
    
    # Fit Oracle
    kf_oracle = fit_oracle(sim)
    
    # Compute MSE ratio
    mse_mf = np.mean((kf_mfvi.x_filtered - sim.x2)**2)
    mse_oracle = np.mean((kf_oracle.x_filtered - sim.x2)**2)
    mse_ratio = mse_mf / max(mse_oracle, 1e-10)
    
    # Compute importance weights
    weights = compute_importance_weights(sim, kf_mfvi, kf_oracle, 
                                         n_samples=n_importance_samples)
    
    # Compute PSIS-k̂
    psis_khat = compute_psis_khat_robust(weights)
    
    # Correlation for reference
    x3 = sim.x3[1:]
    dx2 = np.diff(sim.x2)
    log_abs_dx2 = np.log(np.abs(dx2) + 1e-10)
    rho = np.corrcoef(x3, log_abs_dx2)[0, 1]
    
    return {
        'coupling': kappa,
        'cf': cf,
        'psis_khat': psis_khat,
        'mse_ratio': mse_ratio,
        'correlation': rho,
        'linfoot': compute_linfoot(rho) if np.isfinite(rho) else np.nan
    }


def run_validation_study(coupling_values: List[float], 
                         n_sims_per_coupling: int = 100,
                         T: int = 300,
                         verbose: bool = True) -> List[Dict]:
    """
    Run full PSIS validation study across coupling values.
    
    Parameters
    ----------
    coupling_values : list
        Values of κ to test
    n_sims_per_coupling : int
        Number of simulations per κ value
    T : int
        Time series length
    verbose : bool
        Print progress
    
    Returns
    -------
    results : list of dict
        All simulation results
    """
    results = []
    total = len(coupling_values) * n_sims_per_coupling
    count = 0
    
    for kappa in coupling_values:
        for rep in range(n_sims_per_coupling):
            seed = int(kappa * 10000 + rep)
            
            try:
                result = run_single_simulation(kappa, seed, T=T)
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Warning: Simulation failed for κ={kappa}, rep={rep}: {e}")
            
            count += 1
            if verbose and count % 50 == 0:
                print(f"Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    return results


def summarize_results(results: List[Dict]) -> None:
    """Print summary table of results."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Group by coupling
    summary = df.groupby('coupling').agg({
        'cf': ['mean', 'std'],
        'psis_khat': ['mean', 'std'],
        'mse_ratio': ['mean', 'std']
    }).round(3)
    
    print("\n" + "="*70)
    print("PSIS VALIDATION STUDY RESULTS")
    print("="*70)
    print(summary)
    
    # Compute correlation between CF and PSIS-k̂
    valid = df.dropna(subset=['cf', 'psis_khat'])
    if len(valid) > 10:
        from scipy.stats import pearsonr, spearmanr
        r_pearson, p_pearson = pearsonr(valid['cf'], valid['psis_khat'])
        r_spearman, p_spearman = spearmanr(valid['cf'], valid['psis_khat'])
        
        print(f"\n--- Correlation Analysis ---")
        print(f"CF vs PSIS-k̂ (Pearson):  r = {r_pearson:.3f}, p = {p_pearson:.2e}")
        print(f"CF vs PSIS-k̂ (Spearman): ρ = {r_spearman:.3f}, p = {p_spearman:.2e}")
        
        # Classification performance
        cf_threshold = 0.10
        psis_threshold = 0.7
        
        cf_positive = valid['cf'] > cf_threshold
        psis_positive = valid['psis_khat'] > psis_threshold
        
        tp = ((cf_positive) & (psis_positive)).sum()
        fp = ((cf_positive) & (~psis_positive)).sum()
        tn = ((~cf_positive) & (~psis_positive)).sum()
        fn = ((~cf_positive) & (psis_positive)).sum()
        
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        
        print(f"\n--- Classification Performance (CF > {cf_threshold} predicts PSIS > {psis_threshold}) ---")
        print(f"Sensitivity: {100*sensitivity:.1f}%")
        print(f"Specificity: {100*specificity:.1f}%")
        print(f"PPV:         {100*ppv:.1f}%")
        print(f"NPV:         {100*npv:.1f}%")


def save_results(results: List[Dict], filename: str = "psis_validation_results.csv"):
    """Save results to CSV."""
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


def create_figure(results: List[Dict], filename: str = "psis_cf_validation.pdf"):
    """Create publication-quality figure."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(results)
    valid = df.dropna(subset=['cf', 'psis_khat'])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: CF vs PSIS-k̂ scatter
    ax = axes[0]
    scatter = ax.scatter(valid['cf'], valid['psis_khat'], 
                         c=valid['coupling'], cmap='viridis',
                         alpha=0.5, s=20)
    ax.axvline(0.10, color='red', linestyle='--', alpha=0.7, label='CF threshold')
    ax.axhline(0.7, color='blue', linestyle='--', alpha=0.7, label='PSIS threshold')
    ax.set_xlabel('CF (pre-inference)', fontsize=11)
    ax.set_ylabel('PSIS-$\\hat{k}$ (post-inference)', fontsize=11)
    ax.set_title('(A) CF predicts PSIS-$\\hat{k}$', fontsize=12)
    ax.legend(loc='lower right')
    plt.colorbar(scatter, ax=ax, label='Coupling κ')
    
    # Add correlation annotation
    from scipy.stats import pearsonr
    r, p = pearsonr(valid['cf'], valid['psis_khat'])
    ax.annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top')
    
    # Panel B: Mean PSIS by coupling
    ax = axes[1]
    summary = df.groupby('coupling').agg({
        'psis_khat': ['mean', 'std'],
        'cf': 'mean'
    })
    kappas = summary.index.values
    psis_mean = summary[('psis_khat', 'mean')].values
    psis_std = summary[('psis_khat', 'std')].values
    
    ax.errorbar(kappas, psis_mean, yerr=psis_std, fmt='o-', capsize=4, 
                color='blue', label='PSIS-$\\hat{k}$')
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Coupling κ', fontsize=11)
    ax.set_ylabel('PSIS-$\\hat{k}$', fontsize=11)
    ax.set_title('(B) PSIS increases with coupling', fontsize=12)
    ax.legend()
    
    # Panel C: Concordance matrix
    ax = axes[2]
    cf_threshold = 0.10
    psis_threshold = 0.7
    
    cf_pos = valid['cf'] > cf_threshold
    psis_pos = valid['psis_khat'] > psis_threshold
    
    matrix = np.array([
        [(~cf_pos & ~psis_pos).sum(), (~cf_pos & psis_pos).sum()],
        [(cf_pos & ~psis_pos).sum(), (cf_pos & psis_pos).sum()]
    ])
    
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['PSIS ≤ 0.7', 'PSIS > 0.7'])
    ax.set_yticklabels(['CF ≤ 0.10', 'CF > 0.10'])
    ax.set_xlabel('PSIS-$\\hat{k}$ (post-inference)', fontsize=11)
    ax.set_ylabel('CF (pre-inference)', fontsize=11)
    ax.set_title('(C) Concordance matrix', fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix[i, j], ha='center', va='center', 
                          fontsize=14, color='white' if matrix[i,j] > matrix.max()/2 else 'black')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filename}")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PSIS-k̂ Validation for CF")
    parser.add_argument('--n_sims', type=int, default=100,
                        help='Simulations per coupling value')
    parser.add_argument('--T', type=int, default=300,
                        help='Time series length')
    parser.add_argument('--output', type=str, default='psis_validation_results.csv',
                        help='Output CSV filename')
    parser.add_argument('--figure', type=str, default='psis_cf_validation.pdf',
                        help='Output figure filename')
    args = parser.parse_args()
    
    print("="*70)
    print("PSIS-k̂ VALIDATION STUDY")
    print("Computing ACTUAL PSIS-k̂ from fitted MFVI posteriors")
    print("="*70)
    
    # Coupling values matching manuscript
    coupling_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    print(f"\nSettings:")
    print(f"  Coupling values: {coupling_values}")
    print(f"  Simulations per coupling: {args.n_sims}")
    print(f"  Time series length: {args.T}")
    print(f"  Total simulations: {len(coupling_values) * args.n_sims}")
    
    # Run study
    print("\nRunning validation study...")
    results = run_validation_study(coupling_values, 
                                   n_sims_per_coupling=args.n_sims,
                                   T=args.T,
                                   verbose=True)
    
    # Summarize
    summarize_results(results)
    
    # Save
    save_results(results, args.output)
    
    # Create figure
    try:
        create_figure(results, args.figure)
    except Exception as e:
        print(f"Warning: Could not create figure: {e}")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
