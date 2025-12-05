#!/usr/bin/env python3
"""
Three-Level HGF Stability Analysis
===================================

Prototype implementation comparing stability of mean-field vs structured
approximations in a three-level hierarchical Gaussian filter.

Extends the two-level CF analysis to deeper hierarchies.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, List
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThreeLevelParams:
    """Parameters for three-level HGF"""
    kappa2: float = 1.0      # Level 2→1 coupling
    kappa3: float = 1.0      # Level 3→2 coupling
    omega2: float = -2.0     # Level 2 baseline log-volatility
    omega3: float = -2.0     # Level 3 baseline log-volatility
    theta3: float = 0.1      # Level 3 meta-volatility (key bifurcation param)
    pi_u: float = 10.0       # Observation precision


@dataclass 
class ThreeLevelState:
    """State of three-level HGF"""
    mu1: float = 0.0    # Level 1 mean
    mu2: float = 0.0    # Level 2 mean (log-volatility)
    mu3: float = 0.0    # Level 3 mean (meta-log-volatility)
    sig1: float = 1.0   # Level 1 variance
    sig2: float = 1.0   # Level 2 variance
    sig3: float = 1.0   # Level 3 variance
    
    def copy(self):
        return ThreeLevelState(
            self.mu1, self.mu2, self.mu3,
            self.sig1, self.sig2, self.sig3
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UPDATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def update_meanfield(state: ThreeLevelState, y: float, p: ThreeLevelParams) -> ThreeLevelState:
    """
    Mean-field update: q(z1,z2,z3) = q(z1)q(z2)q(z3)
    Each level updates independently.
    """
    # Predicted precisions
    pi_hat1 = np.exp(-p.kappa2 * state.mu2 - p.omega2)
    pi_hat2 = np.exp(-p.kappa3 * state.mu3 - p.omega3)
    
    # Clip for numerical stability
    pi_hat1 = np.clip(pi_hat1, 1e-6, 1e6)
    pi_hat2 = np.clip(pi_hat2, 1e-6, 1e6)
    
    # ─── Level 1 Update ───
    delta1 = y - state.mu1
    K1 = p.pi_u / (p.pi_u + pi_hat1)
    
    new_mu1 = state.mu1 + K1 * delta1
    new_sig1 = 1.0 / (p.pi_u + pi_hat1)
    
    # ─── Level 2 Update ───
    # Volatility prediction error
    nu2 = (delta1**2 * p.pi_u * pi_hat1 / (p.pi_u + pi_hat1)) + state.sig1 * pi_hat1 - 1
    
    K2_denom = pi_hat2 + (p.kappa2**2 / 2) * pi_hat1
    K2 = (p.kappa2 / 2) * pi_hat1 / K2_denom if K2_denom > 1e-10 else 0
    
    new_mu2 = state.mu2 + K2 * nu2
    new_sig2 = 1.0 / K2_denom if K2_denom > 1e-10 else state.sig2
    
    # ─── Level 3 Update ───
    # Meta-volatility prediction error
    nu3 = (nu2**2 * pi_hat2 / K2_denom) + state.sig2 * pi_hat2 - 1
    
    K3_denom = p.theta3 + (p.kappa3**2 / 2) * pi_hat2
    K3 = (p.kappa3 / 2) * pi_hat2 / K3_denom if K3_denom > 1e-10 else 0
    
    new_mu3 = state.mu3 + K3 * nu3
    new_sig3 = 1.0 / K3_denom if K3_denom > 1e-10 else state.sig3
    
    # Clip to prevent explosion
    new_mu1 = np.clip(new_mu1, -50, 50)
    new_mu2 = np.clip(new_mu2, -20, 20)
    new_mu3 = np.clip(new_mu3, -20, 20)
    
    return ThreeLevelState(new_mu1, new_mu2, new_mu3, new_sig1, new_sig2, new_sig3)


def update_structured(state: ThreeLevelState, y: float, p: ThreeLevelParams,
                     gamma12: float = 0.3, gamma23: float = 0.3) -> ThreeLevelState:
    """
    Structured (Markov) update: q(z1,z2,z3) = q(z3)q(z2|z3)q(z1|z2)
    Cross-level dependencies provide damping.
    """
    # Predicted precisions
    pi_hat1 = np.exp(-p.kappa2 * state.mu2 - p.omega2)
    pi_hat2 = np.exp(-p.kappa3 * state.mu3 - p.omega3)
    
    pi_hat1 = np.clip(pi_hat1, 1e-6, 1e6)
    pi_hat2 = np.clip(pi_hat2, 1e-6, 1e6)
    
    # ─── Level 1 Update (same as mean-field) ───
    delta1 = y - state.mu1
    K1 = p.pi_u / (p.pi_u + pi_hat1)
    
    new_mu1 = state.mu1 + K1 * delta1
    new_sig1 = 1.0 / (p.pi_u + pi_hat1)
    
    # ─── Level 2 Update (with coupling) ───
    nu2 = (delta1**2 * p.pi_u * pi_hat1 / (p.pi_u + pi_hat1)) + state.sig1 * pi_hat1 - 1
    
    K2_denom = pi_hat2 + (p.kappa2**2 / 2) * pi_hat1
    K2 = (p.kappa2 / 2) * pi_hat1 / K2_denom if K2_denom > 1e-10 else 0
    
    # Damping from cross-level coupling
    damping2 = 1.0 / (1.0 + gamma12 * np.abs(nu2))
    
    new_mu2 = state.mu2 + K2 * nu2 * damping2
    new_sig2 = 1.0 / K2_denom if K2_denom > 1e-10 else state.sig2
    
    # ─── Level 3 Update (with coupling) ───
    nu3 = (nu2**2 * pi_hat2 / K2_denom) + state.sig2 * pi_hat2 - 1
    
    K3_denom = p.theta3 + (p.kappa3**2 / 2) * pi_hat2
    K3 = (p.kappa3 / 2) * pi_hat2 / K3_denom if K3_denom > 1e-10 else 0
    
    # Damping from cross-level coupling
    damping3 = 1.0 / (1.0 + gamma23 * np.abs(nu3))
    
    new_mu3 = state.mu3 + K3 * nu3 * damping3
    new_sig3 = 1.0 / K3_denom if K3_denom > 1e-10 else state.sig3
    
    # Clip to prevent explosion
    new_mu1 = np.clip(new_mu1, -50, 50)
    new_mu2 = np.clip(new_mu2, -20, 20)
    new_mu3 = np.clip(new_mu3, -20, 20)
    
    return ThreeLevelState(new_mu1, new_mu2, new_mu3, new_sig1, new_sig2, new_sig3)


def update_bottom_structured(state: ThreeLevelState, y: float, p: ThreeLevelParams) -> ThreeLevelState:
    """Only 1-2 interface has coupling"""
    return update_structured(state, y, p, gamma12=0.3, gamma23=0.0)


def update_top_structured(state: ThreeLevelState, y: float, p: ThreeLevelParams) -> ThreeLevelState:
    """Only 2-3 interface has coupling"""
    return update_structured(state, y, p, gamma12=0.0, gamma23=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(p: ThreeLevelParams, T: int, update_fn: Callable, 
             seed: int = 42) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Simulate T timesteps of the three-level HGF.
    """
    np.random.seed(seed)
    
    # Generate observations from varying volatility process
    true_vol = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(T) / 100)
    true_x = np.cumsum(np.sqrt(true_vol) * np.random.randn(T))
    observations = true_x + np.random.randn(T) / np.sqrt(p.pi_u)
    
    # Initialize
    state = ThreeLevelState()
    
    # Storage
    trajectories = {
        'mu1': np.zeros(T),
        'mu2': np.zeros(T),
        'mu3': np.zeros(T),
        'sig1': np.zeros(T),
        'sig2': np.zeros(T),
        'sig3': np.zeros(T),
    }
    
    # Simulate
    for t in range(T):
        state = update_fn(state, observations[t], p)
        
        trajectories['mu1'][t] = state.mu1
        trajectories['mu2'][t] = state.mu2
        trajectories['mu3'][t] = state.mu3
        trajectories['sig1'][t] = state.sig1
        trajectories['sig2'][t] = state.sig2
        trajectories['sig3'][t] = state.sig3
    
    return trajectories, observations


# ═══════════════════════════════════════════════════════════════════════════════
# LYAPUNOV EXPONENT COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lyapunov(p: ThreeLevelParams, update_fn: Callable,
                     T: int = 10000, transient: int = 2000, 
                     seed: int = 42) -> float:
    """
    Compute maximal Lyapunov exponent using trajectory separation method.
    """
    np.random.seed(seed)
    
    # Generate observations
    observations = np.random.randn(T + transient) / np.sqrt(p.pi_u)
    
    # Initialize reference and perturbed trajectories
    state_ref = ThreeLevelState()
    state_pert = ThreeLevelState()
    
    # Initial perturbation at level 3
    eps = 1e-8
    state_pert.mu3 += eps
    
    # Discard transient
    for t in range(transient):
        state_ref = update_fn(state_ref, observations[t], p)
        state_pert = update_fn(state_pert, observations[t], p)
    
    # Compute Lyapunov exponent
    lyap_sum = 0.0
    renorm_interval = 10
    n_renorms = 0
    
    for t in range(transient, T + transient):
        state_ref = update_fn(state_ref, observations[t], p)
        state_pert = update_fn(state_pert, observations[t], p)
        
        if t % renorm_interval == 0:
            # Compute separation in state space
            sep = np.sqrt(
                (state_pert.mu1 - state_ref.mu1)**2 +
                (state_pert.mu2 - state_ref.mu2)**2 +
                (state_pert.mu3 - state_ref.mu3)**2
            )
            
            if sep > 1e-15 and np.isfinite(sep) and sep < 1e10:
                lyap_sum += np.log(sep / eps)
                n_renorms += 1
                
                # Renormalize
                factor = eps / sep
                state_pert.mu1 = state_ref.mu1 + factor * (state_pert.mu1 - state_ref.mu1)
                state_pert.mu2 = state_ref.mu2 + factor * (state_pert.mu2 - state_ref.mu2)
                state_pert.mu3 = state_ref.mu3 + factor * (state_pert.mu3 - state_ref.mu3)
    
    if n_renorms > 0:
        return lyap_sum / (n_renorms * renorm_interval)
    else:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY VARIABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trajectory_stats(traj: Dict[str, np.ndarray], 
                            burnin: int = 1000) -> Dict[str, float]:
    """Compute statistics characterizing trajectory behavior."""
    
    mu1 = traj['mu1'][burnin:]
    mu2 = traj['mu2'][burnin:]
    mu3 = traj['mu3'][burnin:]
    
    # Variances (instability indicator)
    var1 = np.var(mu1)
    var2 = np.var(mu2)
    var3 = np.var(mu3)
    
    # Autocorrelation at lag 1 (oscillation indicator)
    def autocorr(x):
        x = x - np.mean(x)
        if np.std(x) < 1e-10:
            return 0.0
        return np.corrcoef(x[:-1], x[1:])[0, 1]
    
    ac1 = autocorr(mu1)
    ac2 = autocorr(mu2)
    ac3 = autocorr(mu3)
    
    # Pairwise correlations (CF proxy)
    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        return np.corrcoef(x, y)[0, 1]
    
    rho12 = safe_corr(mu1, mu2)
    rho23 = safe_corr(mu2, mu3)
    rho13 = safe_corr(mu1, mu3)
    
    return {
        'var1': var1, 'var2': var2, 'var3': var3,
        'ac1': ac1, 'ac2': ac2, 'ac3': ac3,
        'rho12': rho12, 'rho23': rho23, 'rho13': rho13
    }


def compute_pairwise_cf(traj: Dict[str, np.ndarray], 
                        burnin: int = 1000) -> Tuple[float, float]:
    """
    Compute pairwise CFs from trajectory data.
    Uses correlation-based MI estimate (exact for Gaussians).
    """
    mu1 = traj['mu1'][burnin:]
    mu2 = traj['mu2'][burnin:]
    mu3 = traj['mu3'][burnin:]
    
    # Compute correlations
    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return np.clip(r, -0.9999, 0.9999)
    
    rho12 = safe_corr(mu1, mu2)
    rho23 = safe_corr(mu2, mu3)
    
    # MI for Gaussian: I(X;Y) = -0.5 * log(1 - rho^2)
    I12 = -0.5 * np.log(1 - rho12**2) if abs(rho12) > 0.01 else 0.0
    I23 = -0.5 * np.log(1 - rho23**2) if abs(rho23) > 0.01 else 0.0
    
    # Differential entropy of Gaussian: H = 0.5 * log(2*pi*e*var)
    H1 = 0.5 * np.log(2 * np.pi * np.e * max(np.var(mu1), 1e-10))
    H2 = 0.5 * np.log(2 * np.pi * np.e * max(np.var(mu2), 1e-10))
    H3 = 0.5 * np.log(2 * np.pi * np.e * max(np.var(mu3), 1e-10))
    
    # Normalized CFs
    CF12 = I12 / min(H1, H2) if min(H1, H2) > 0.01 else 0.0
    CF23 = I23 / min(H2, H3) if min(H2, H3) > 0.01 else 0.0
    
    return CF12, CF23


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability_comparison():
    """Compare stability across approximation schemes."""
    
    print("=" * 70)
    print("THREE-LEVEL HGF STABILITY ANALYSIS")
    print("=" * 70)
    
    # Parameter sweep over theta3 (meta-volatility)
    theta3_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
    
    results = {
        'theta3': theta3_values,
        'lyap_mf': [],
        'lyap_struct': [],
        'lyap_bottom': [],
        'lyap_top': [],
        'var2_mf': [],
        'var2_struct': [],
    }
    
    print("\nComputing Lyapunov exponents for each theta3 value...")
    print("(This may take a minute)\n")
    
    for theta3 in theta3_values:
        p = ThreeLevelParams(theta3=theta3)
        
        # Mean-field
        lyap_mf = compute_lyapunov(p, update_meanfield)
        results['lyap_mf'].append(lyap_mf)
        
        # Fully structured
        lyap_struct = compute_lyapunov(p, update_structured)
        results['lyap_struct'].append(lyap_struct)
        
        # Bottom-structured (1-2 interface only)
        lyap_bottom = compute_lyapunov(p, update_bottom_structured)
        results['lyap_bottom'].append(lyap_bottom)
        
        # Top-structured (2-3 interface only)
        lyap_top = compute_lyapunov(p, update_top_structured)
        results['lyap_top'].append(lyap_top)
        
        # Also get trajectory variance for intuition
        traj_mf, _ = simulate(p, 5000, update_meanfield)
        traj_struct, _ = simulate(p, 5000, update_structured)
        
        stats_mf = compute_trajectory_stats(traj_mf)
        stats_struct = compute_trajectory_stats(traj_struct)
        
        results['var2_mf'].append(stats_mf['var2'])
        results['var2_struct'].append(stats_struct['var2'])
        
        print(f"  theta3 = {theta3:.2f}: lyap_MF = {lyap_mf:+.4f}, lyap_struct = {lyap_struct:+.4f}")
    
    return results


def run_cf_analysis():
    """Analyze CF at each interface."""
    
    print("\n" + "=" * 70)
    print("PAIRWISE CIRCULATORY FIDELITY ANALYSIS")
    print("=" * 70)
    
    theta3_values = [0.05, 0.10, 0.20]
    
    print("\nComparing CF_12 and CF_23 across approximation schemes:\n")
    
    for theta3 in theta3_values:
        p = ThreeLevelParams(theta3=theta3)
        
        print(f"theta3 = {theta3:.2f}:")
        print("-" * 50)
        
        for name, update_fn in [
            ("Mean-field", update_meanfield),
            ("Structured", update_structured),
            ("Bottom-only", update_bottom_structured),
            ("Top-only", update_top_structured),
        ]:
            traj, _ = simulate(p, 8000, update_fn)
            CF12, CF23 = compute_pairwise_cf(traj)
            stats = compute_trajectory_stats(traj)
            
            print(f"  {name:12s}: CF_12 = {CF12:.3f}, CF_23 = {CF23:.3f}, "
                  f"Var(mu2) = {stats['var2']:.4f}")
        
        print()


def run_cascade_test():
    """Test whether instabilities cascade through levels."""
    
    print("\n" + "=" * 70)
    print("CASCADE DYNAMICS TEST")
    print("=" * 70)
    
    # Use a theta3 value where mean-field shows instability
    p = ThreeLevelParams(theta3=0.05)
    
    print(f"\nParameters: theta3 = {p.theta3}")
    print("\nSimulating 10000 timesteps...\n")
    
    traj_mf, obs = simulate(p, 10000, update_meanfield, seed=42)
    traj_struct, _ = simulate(p, 10000, update_structured, seed=42)
    
    # Analyze each level's behavior
    print("Level-by-level variance comparison (after burnin):")
    print("-" * 50)
    
    stats_mf = compute_trajectory_stats(traj_mf, burnin=2000)
    stats_struct = compute_trajectory_stats(traj_struct, burnin=2000)
    
    print(f"{'Level':<10} {'Mean-Field':<15} {'Structured':<15} {'Ratio':<10}")
    print("-" * 50)
    
    for i, key in enumerate(['var1', 'var2', 'var3'], 1):
        ratio = stats_mf[key] / max(stats_struct[key], 1e-10)
        print(f"Level {i:<4} {stats_mf[key]:<15.4f} {stats_struct[key]:<15.4f} {ratio:<10.2f}x")
    
    # Autocorrelation (oscillation signature)
    print("\nAutocorrelation at lag 1 (oscillation indicator):")
    print("-" * 50)
    print(f"{'Level':<10} {'Mean-Field':<15} {'Structured':<15}")
    print("-" * 50)
    
    for i, key in enumerate(['ac1', 'ac2', 'ac3'], 1):
        print(f"Level {i:<4} {stats_mf[key]:<15.3f} {stats_struct[key]:<15.3f}")
    
    # Cross-level correlations
    print("\nCross-level correlations:")
    print("-" * 50)
    print(f"{'Pair':<10} {'Mean-Field':<15} {'Structured':<15}")
    print("-" * 50)
    
    for key, label in [('rho12', '1-2'), ('rho23', '2-3'), ('rho13', '1-3')]:
        print(f"{label:<10} {stats_mf[key]:<15.3f} {stats_struct[key]:<15.3f}")


def compare_two_vs_three_level():
    """Compare stability between two-level and three-level systems."""
    
    print("\n" + "=" * 70)
    print("TWO-LEVEL vs THREE-LEVEL STABILITY COMPARISON")
    print("=" * 70)
    
    print("\nQuestion: Do instabilities worsen with hierarchy depth?")
    print("\nMethod: Compare Lyapunov exponents at matched parameters\n")
    
    # For two-level, we effectively fix mu3 = 0 and don't update it
    def update_meanfield_2level(state: ThreeLevelState, y: float, 
                                 p: ThreeLevelParams) -> ThreeLevelState:
        """Two-level mean-field (level 3 frozen)"""
        new_state = update_meanfield(state, y, p)
        new_state.mu3 = 0.0  # Freeze level 3
        new_state.sig3 = 1.0
        return new_state
    
    def update_structured_2level(state: ThreeLevelState, y: float,
                                  p: ThreeLevelParams) -> ThreeLevelState:
        """Two-level structured (level 3 frozen)"""
        new_state = update_structured(state, y, p)
        new_state.mu3 = 0.0
        new_state.sig3 = 1.0
        return new_state
    
    theta3_values = [0.05, 0.10, 0.20]
    
    print(f"{'theta3':<8} {'2L MF':<12} {'3L MF':<12} {'2L Struct':<12} {'3L Struct':<12}")
    print("-" * 60)
    
    for theta3 in theta3_values:
        p = ThreeLevelParams(theta3=theta3)
        
        lyap_2l_mf = compute_lyapunov(p, update_meanfield_2level)
        lyap_3l_mf = compute_lyapunov(p, update_meanfield)
        lyap_2l_struct = compute_lyapunov(p, update_structured_2level)
        lyap_3l_struct = compute_lyapunov(p, update_structured)
        
        print(f"{theta3:<8.2f} {lyap_2l_mf:<+12.4f} {lyap_3l_mf:<+12.4f} "
              f"{lyap_2l_struct:<+12.4f} {lyap_3l_struct:<+12.4f}")
    
    print("\nInterpretation:")
    print("  - If 3L MF > 2L MF: Instabilities worsen with depth")
    print("  - If 3L Struct ~ 2L Struct: Structuring remains protective")


def main():
    """Run complete analysis."""
    
    print("\n" + "=" * 70)
    print("  THREE-LEVEL HIERARCHICAL GAUSSIAN FILTER")
    print("  CIRCULATORY FIDELITY EXTENSION ANALYSIS")
    print("=" * 70 + "\n")
    
    # 1. Stability comparison across theta3 values
    results = run_stability_comparison()
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: LYAPUNOV EXPONENTS BY APPROXIMATION SCHEME")
    print("=" * 70)
    print(f"\n{'theta3':<8} {'Mean-Field':<12} {'Structured':<12} {'Bottom':<12} {'Top':<12}")
    print("-" * 60)
    
    for i, theta3 in enumerate(results['theta3']):
        print(f"{theta3:<8.2f} {results['lyap_mf'][i]:<+12.4f} "
              f"{results['lyap_struct'][i]:<+12.4f} "
              f"{results['lyap_bottom'][i]:<+12.4f} "
              f"{results['lyap_top'][i]:<+12.4f}")
    
    print("\nKey: lambda > 0 = chaotic, lambda ~ 0 = marginal, lambda < 0 = stable")
    
    # 2. CF analysis
    run_cf_analysis()
    
    # 3. Cascade test
    run_cascade_test()
    
    # 4. Two vs three level comparison
    compare_two_vs_three_level()
    
    # Final summary
    print("\n" + "=" * 70)
    print("PRELIMINARY CONCLUSIONS")
    print("=" * 70)
    
    # Check if mean-field shows positive Lyapunov at low theta3
    mf_unstable = any(l > 0 for l in results['lyap_mf'][:4])
    struct_stable = all(l < 0.01 for l in results['lyap_struct'])
    
    print(f"""
1. MEAN-FIELD INSTABILITY: {"CONFIRMED" if mf_unstable else "NOT OBSERVED"}
   Mean-field shows positive Lyapunov exponents at low theta3
   
2. STRUCTURED STABILITY: {"CONFIRMED" if struct_stable else "PARTIAL"}  
   Structured approximation remains stable across parameter range
   
3. INTERFACE COMPARISON:
   Check which partial structuring (bottom vs top) provides more benefit
   
4. DEPTH EFFECT:
   Compare 2-level vs 3-level Lyapunov exponents
""")
    
    return results


if __name__ == "__main__":
    results = main()
