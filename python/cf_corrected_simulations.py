"""
Circulatory Fidelity: Corrected Simulation Suite
================================================

Key fixes:
1. HGF: CF computed from VOLATILITY-STATE relationship properly
2. HLM: Use reliability coefficient as CF proxy (avoids negative entropy issues)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# CF COMPUTATION - CORRECTED
# =============================================================================

def mi_from_rho(rho: float) -> float:
    """MI for bivariate Gaussian."""
    rho = np.clip(rho, -0.999, 0.999)
    return -0.5 * np.log(1 - rho**2)

def cf_positive_entropy(rho: float, h_min: float) -> float:
    """CF when both entropies are positive."""
    if h_min <= 0.1:  # Threshold for numerical stability
        return np.nan
    return np.clip(mi_from_rho(rho) / h_min, 0, 1)

def reliability_to_cf(reliability: float) -> float:
    """
    Convert reliability coefficient to CF-like measure.
    Reliability = var(signal) / var(observed) ∈ [0,1]
    This IS the squared correlation, so MI = -0.5*log(1-reliability)
    We normalize to [0,1] using a reference entropy.
    """
    reliability = np.clip(reliability, 0.001, 0.999)
    # Using unit Gaussian entropy as reference
    h_ref = 0.5 * np.log(2 * np.pi * np.e)  # ≈ 1.42
    mi = -0.5 * np.log(1 - reliability)
    return np.clip(mi / h_ref, 0, 1)

# =============================================================================
# MODEL 1: HGF - CORRECTED
# =============================================================================

@dataclass
class HGFParams:
    coupling: float = 0.5
    base_vol: float = 0.3
    vol_vol: float = 0.1

def simulate_hgf(params: HGFParams, T: int = 300) -> Dict:
    """Generate HGF with tracking of volatility influence."""
    x3 = np.zeros(T)  # Log-volatility
    x2 = np.zeros(T)  # State
    vol = np.zeros(T)  # Actual volatility used
    y = np.zeros(T)
    
    x3[0] = 0
    x2[0] = 0
    
    for t in range(1, T):
        # Volatility random walk
        x3[t] = x3[t-1] + np.random.normal(0, params.vol_vol)
        
        # State volatility depends on x3
        vol[t] = params.base_vol * np.exp(params.coupling * x3[t])
        vol[t] = np.clip(vol[t], 0.01, 5.0)
        
        # State evolution
        x2[t] = x2[t-1] + np.random.normal(0, vol[t])
        
        # Observation
        y[t] = x2[t] + np.random.normal(0, 0.5)
    
    return {'x3': x3, 'x2': x2, 'vol': vol, 'y': y, 'params': params}

def hgf_cf_analytical(params: HGFParams) -> float:
    """
    Analytical CF for HGF.
    
    The key dependency: x2's variance depends on x3.
    We measure: Corr(x3, |dx2|) where dx2 = x2[t] - x2[t-1]
    
    Under the model: Var(dx2|x3) = base_vol^2 * exp(2*coupling*x3)
    So: |dx2| ~ HalfNormal(scale = base_vol * exp(coupling*x3))
    
    For high coupling, x3 strongly predicts |dx2|.
    """
    # Simulate to get empirical relationship
    T = 5000
    x3 = np.cumsum(np.random.normal(0, params.vol_vol, T))
    vol = params.base_vol * np.exp(np.clip(params.coupling * x3, -3, 3))
    dx2 = np.random.normal(0, 1, T) * vol
    
    # CF from correlation between x3 and log|dx2|
    log_abs_dx2 = np.log(np.abs(dx2) + 1e-10)
    rho = np.corrcoef(x3, log_abs_dx2)[0, 1]
    
    # Normalize by theoretical maximum
    # At coupling=0, rho≈0; at coupling→∞, rho→1 (but capped by vol_vol)
    max_rho = np.tanh(params.coupling * 2)  # Approximate ceiling
    
    return np.clip(abs(rho) / max(max_rho, 0.1), 0, 1)

def hgf_cf_empirical(sim: Dict) -> float:
    """Compute CF from simulation: how much does x3 predict state volatility?"""
    x3 = sim['x3'][1:]
    dx2 = np.diff(sim['x2'])
    
    # Correlation between x3 and |dx2|
    log_abs_dx2 = np.log(np.abs(dx2) + 1e-10)
    rho = np.corrcoef(x3, log_abs_dx2)[0, 1]
    
    if not np.isfinite(rho):
        return np.nan
    
    # Reference entropy (unit Gaussian)
    h_ref = 0.5 * np.log(2 * np.pi * np.e)
    mi = mi_from_rho(rho)
    
    return np.clip(mi / h_ref, 0, 1)

def hgf_mf_error(sim: Dict) -> float:
    """MF: ignores volatility, uses constant."""
    y = sim['y']
    params = sim['params']
    T = len(y)
    
    avg_vol = params.base_vol
    x2_est = np.zeros(T)
    var_est = np.ones(T)
    
    for t in range(1, T):
        pred_var = var_est[t-1] + avg_vol**2
        obs_var = 0.25
        K = pred_var / (pred_var + obs_var)
        x2_est[t] = x2_est[t-1] + K * (y[t] - x2_est[t-1])
        var_est[t] = (1 - K) * pred_var
    
    return np.mean((x2_est - sim['x2'])**2)

def hgf_oracle_error(sim: Dict) -> float:
    """Oracle: knows true volatility."""
    y = sim['y']
    vol = sim['vol']
    T = len(y)
    
    x2_est = np.zeros(T)
    var_est = np.ones(T)
    
    for t in range(1, T):
        pred_var = var_est[t-1] + vol[t]**2
        obs_var = 0.25
        K = pred_var / (pred_var + obs_var)
        x2_est[t] = x2_est[t-1] + K * (y[t] - x2_est[t-1])
        var_est[t] = (1 - K) * pred_var
    
    return np.mean((x2_est - sim['x2'])**2)

def hgf_sweep() -> pd.DataFrame:
    """Comprehensive HGF sweep."""
    results = []
    
    for coupling in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        print(f"  HGF coupling={coupling}")
        params = HGFParams(coupling=coupling)
        cf_anal = hgf_cf_analytical(params)
        
        for i in range(100):
            sim = simulate_hgf(params)
            cf_emp = hgf_cf_empirical(sim)
            mf_err = hgf_mf_error(sim)
            oracle_err = hgf_oracle_error(sim)
            
            results.append({
                'coupling': coupling,
                'cf_analytical': cf_anal,
                'cf_empirical': cf_emp,
                'mf_mse': mf_err,
                'oracle_mse': oracle_err,
                'mse_ratio': mf_err / max(oracle_err, 1e-6)
            })
    
    return pd.DataFrame(results)

# =============================================================================
# MODEL 2: HLM - CORRECTED
# =============================================================================

@dataclass
class HLMParams:
    n_groups: int = 30
    n_per_group: int = 10
    tau: float = 1.0
    sigma: float = 1.0
    
    @property
    def icc(self) -> float:
        return self.tau**2 / (self.tau**2 + self.sigma**2)
    
    @property
    def reliability(self) -> float:
        """Reliability of group mean as estimate of theta."""
        return self.tau**2 / (self.tau**2 + self.sigma**2 / self.n_per_group)

def simulate_hlm(params: HLMParams) -> Dict:
    theta = np.random.normal(0, params.tau, params.n_groups)
    
    y, group, theta_rep = [], [], []
    for j in range(params.n_groups):
        y_j = np.random.normal(theta[j], params.sigma, params.n_per_group)
        y.extend(y_j)
        group.extend([j] * params.n_per_group)
        theta_rep.extend([theta[j]] * params.n_per_group)
    
    return {
        'theta': theta,
        'y': np.array(y),
        'group': np.array(group),
        'params': params
    }

def hlm_cf(params: HLMParams) -> float:
    """
    CF for HLM based on reliability.
    
    Reliability = Var(theta) / Var(y_bar) = tau^2 / (tau^2 + sigma^2/n)
    
    This is Corr(theta, y_bar)^2, so Corr = sqrt(reliability)
    MI = -0.5 * log(1 - reliability)
    
    We normalize to [0,1] range.
    """
    return reliability_to_cf(params.reliability)

def hlm_no_pooling_error(sim: Dict) -> float:
    y, group = sim['y'], sim['group']
    n_groups = sim['params'].n_groups
    theta_hat = np.array([y[group == j].mean() for j in range(n_groups)])
    return np.mean((theta_hat - sim['theta'])**2)

def hlm_partial_pooling_error(sim: Dict) -> float:
    y, group = sim['y'], sim['group']
    params = sim['params']
    
    grand_mean = y.mean()
    group_means = np.array([y[group == j].mean() for j in range(params.n_groups)])
    
    shrinkage = params.reliability
    theta_hat = shrinkage * group_means + (1 - shrinkage) * grand_mean
    
    return np.mean((theta_hat - sim['theta'])**2)

def hlm_sweep() -> pd.DataFrame:
    """Comprehensive HLM sweep."""
    results = []
    
    for tau in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
        params = HLMParams(tau=tau, sigma=1.0)
        cf = hlm_cf(params)
        print(f"  HLM tau={tau} (ICC={params.icc:.2f}, CF={cf:.3f})")
        
        for i in range(100):
            sim = simulate_hlm(params)
            np_err = hlm_no_pooling_error(sim)
            pp_err = hlm_partial_pooling_error(sim)
            
            results.append({
                'tau': tau,
                'icc': params.icc,
                'reliability': params.reliability,
                'cf': cf,
                'no_pooling_mse': np_err,
                'partial_pooling_mse': pp_err,
                'mse_ratio': np_err / max(pp_err, 1e-6)
            })
    
    return pd.DataFrame(results)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CIRCULATORY FIDELITY: CORRECTED SIMULATION STUDY")
    print("="*70)
    
    print("\n[1] HGF Parameter Sweep")
    hgf_df = hgf_sweep()
    hgf_valid = hgf_df.dropna()
    
    print("\n[2] HLM Parameter Sweep")
    hlm_df = hlm_sweep()
    hlm_valid = hlm_df.dropna()
    
    # ==========================================================================
    # HGF ANALYSIS
    # ==========================================================================
    print("\n" + "="*70)
    print("HGF RESULTS")
    print("="*70)
    
    r_hgf = hgf_valid['cf_empirical'].corr(hgf_valid['mse_ratio'])
    print(f"Correlation(CF, MSE_ratio): r = {r_hgf:.3f}")
    
    print("\nBy coupling:")
    print(f"{'Coupling':>8} {'CF':>8} {'MSE Ratio':>12}")
    for c in sorted(hgf_valid['coupling'].unique()):
        sub = hgf_valid[hgf_valid['coupling'] == c]
        print(f"{c:>8.1f} {sub['cf_empirical'].mean():>8.3f} {sub['mse_ratio'].mean():>12.3f}")
    
    # T-test
    med_cf = hgf_valid['cf_empirical'].median()
    low_cf = hgf_valid[hgf_valid['cf_empirical'] < med_cf]
    high_cf = hgf_valid[hgf_valid['cf_empirical'] >= med_cf]
    t_hgf, p_hgf = stats.ttest_ind(low_cf['mse_ratio'], high_cf['mse_ratio'])
    print(f"\nLow vs High CF: t={t_hgf:.2f}, p={p_hgf:.6f}")
    print(f"  Low-CF MSE ratio:  {low_cf['mse_ratio'].mean():.3f}")
    print(f"  High-CF MSE ratio: {high_cf['mse_ratio'].mean():.3f}")
    
    # ==========================================================================
    # HLM ANALYSIS
    # ==========================================================================
    print("\n" + "="*70)
    print("HLM RESULTS")
    print("="*70)
    
    # For HLM: LOW CF (low reliability) → no-pooling bad → HIGH MSE ratio
    # So we expect NEGATIVE correlation
    r_hlm = hlm_valid['cf'].corr(hlm_valid['mse_ratio'])
    print(f"Correlation(CF, MSE_ratio): r = {r_hlm:.3f}")
    print("(Expected: negative - low CF means no-pooling overfits)")
    
    print("\nBy tau (between-group SD):")
    print(f"{'tau':>6} {'ICC':>6} {'CF':>8} {'MSE Ratio':>12}")
    for tau in sorted(hlm_valid['tau'].unique()):
        sub = hlm_valid[hlm_valid['tau'] == tau]
        print(f"{tau:>6.1f} {sub['icc'].mean():>6.2f} {sub['cf'].mean():>8.3f} {sub['mse_ratio'].mean():>12.3f}")
    
    # T-test
    med_cf_hlm = hlm_valid['cf'].median()
    low_cf_hlm = hlm_valid[hlm_valid['cf'] < med_cf_hlm]
    high_cf_hlm = hlm_valid[hlm_valid['cf'] >= med_cf_hlm]
    t_hlm, p_hlm = stats.ttest_ind(low_cf_hlm['mse_ratio'], high_cf_hlm['mse_ratio'])
    print(f"\nLow vs High CF: t={t_hlm:.2f}, p={p_hlm:.6f}")
    print(f"  Low-CF MSE ratio:  {low_cf_hlm['mse_ratio'].mean():.3f}")
    print(f"  High-CF MSE ratio: {high_cf_hlm['mse_ratio'].mean():.3f}")
    
    # ==========================================================================
    # COMBINED ANALYSIS
    # ==========================================================================
    print("\n" + "="*70)
    print("UNIFIED INTERPRETATION")
    print("="*70)
    print("""
HGF: HIGH CF → strong volatility-state coupling → MF loses information → HIGH error
     CF predicts WHEN MF discards important structure

HLM: LOW CF → weak signal (low reliability) → no-pooling overfits → HIGH error
     CF predicts WHEN pooling is necessary

Both: CF diagnoses MFVI appropriateness from prior predictive structure.
    """)
    
    # ==========================================================================
    # SAVE
    # ==========================================================================
    hgf_df.to_csv('/home/claude/simulations/hgf_corrected.csv', index=False)
    hlm_df.to_csv('/home/claude/simulations/hlm_corrected.csv', index=False)
    
    # Summary
    summary = {
        'hgf_n': len(hgf_valid),
        'hgf_r': r_hgf,
        'hgf_p': p_hgf,
        'hgf_low_cf_mse': low_cf['mse_ratio'].mean(),
        'hgf_high_cf_mse': high_cf['mse_ratio'].mean(),
        
        'hlm_n': len(hlm_valid),
        'hlm_r': r_hlm,
        'hlm_p': p_hlm,
        'hlm_low_cf_mse': low_cf_hlm['mse_ratio'].mean(),
        'hlm_high_cf_mse': high_cf_hlm['mse_ratio'].mean(),
    }
    
    import json
    with open('/home/claude/simulations/summary_corrected.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in summary.items()}, f, indent=2)
    
    print("\nResults saved.")
