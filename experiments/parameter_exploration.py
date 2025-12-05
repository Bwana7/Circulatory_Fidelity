#!/usr/bin/env python3
"""
Systematic Parameter Exploration for Three-Level HGF
=====================================================

Explores the full parameter space to identify stability boundaries
and interface criticality patterns.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Import from main analysis
from three_level_analysis import (
    ThreeLevelParams, ThreeLevelState,
    update_3L_meanfield, update_3L_structured,
    update_3L_bottom_structured, update_3L_top_structured,
    compute_lyapunov_3L, simulate_3L, compute_pairwise_cf
)

def systematic_theta_exploration():
    """Explore stability across fine-grained theta values."""
    print("=" * 70)
    print("SYSTEMATIC θ₃ EXPLORATION")
    print("=" * 70)
    
    theta_values = np.concatenate([
        np.linspace(0.005, 0.05, 10),
        np.linspace(0.05, 0.2, 10),
        np.linspace(0.2, 0.5, 5)
    ])
    
    results = {
        'theta': theta_values.tolist(),
        'meanfield': [],
        'structured': [],
        'bottom': [],
        'top': []
    }
    
    print("\nθ₃\t\tMF\t\tStruct\t\tBottom\t\tTop")
    print("-" * 70)
    
    for theta in theta_values:
        p = ThreeLevelParams(theta3=theta)
        
        lam_mf = np.mean([compute_lyapunov_3L(p, update_3L_meanfield, seed=s) 
                         for s in range(3)])
        lam_st = np.mean([compute_lyapunov_3L(p, update_3L_structured, seed=s) 
                         for s in range(3)])
        lam_bot = np.mean([compute_lyapunov_3L(p, update_3L_bottom_structured, seed=s) 
                          for s in range(3)])
        lam_top = np.mean([compute_lyapunov_3L(p, update_3L_top_structured, seed=s) 
                          for s in range(3)])
        
        results['meanfield'].append(lam_mf if np.isfinite(lam_mf) else None)
        results['structured'].append(lam_st if np.isfinite(lam_st) else None)
        results['bottom'].append(lam_bot if np.isfinite(lam_bot) else None)
        results['top'].append(lam_top if np.isfinite(lam_top) else None)
        
        print(f"{theta:.3f}\t\t{lam_mf:.4f}\t\t{lam_st:.4f}\t\t{lam_bot:.4f}\t\t{lam_top:.4f}")
    
    return results


def kappa_exploration():
    """Explore effect of coupling strength on stability."""
    print("\n" + "=" * 70)
    print("COUPLING STRENGTH (κ) EXPLORATION")
    print("=" * 70)
    
    kappa_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    theta_fixed = 0.05
    
    results = {'kappa': kappa_values, 'meanfield': [], 'structured': []}
    
    print("\nκ\t\tMF λ\t\tStruct λ\tDifference")
    print("-" * 60)
    
    for kappa in kappa_values:
        p = ThreeLevelParams(kappa2=kappa, kappa3=kappa, theta3=theta_fixed)
        
        lam_mf = np.mean([compute_lyapunov_3L(p, update_3L_meanfield, seed=s) 
                         for s in range(3)])
        lam_st = np.mean([compute_lyapunov_3L(p, update_3L_structured, seed=s) 
                         for s in range(3)])
        
        results['meanfield'].append(lam_mf if np.isfinite(lam_mf) else None)
        results['structured'].append(lam_st if np.isfinite(lam_st) else None)
        
        diff = lam_mf - lam_st if np.isfinite(lam_mf) and np.isfinite(lam_st) else np.nan
        print(f"{kappa:.2f}\t\t{lam_mf:.4f}\t\t{lam_st:.4f}\t\t{diff:+.4f}")
    
    return results


def omega_exploration():
    """Explore effect of baseline volatility on stability."""
    print("\n" + "=" * 70)
    print("BASELINE VOLATILITY (ω) EXPLORATION")
    print("=" * 70)
    
    omega_values = [-4.0, -3.0, -2.0, -1.0, 0.0]
    theta_fixed = 0.05
    
    results = {'omega': omega_values, 'meanfield': [], 'structured': []}
    
    print("\nω\t\tMF λ\t\tStruct λ\tDifference")
    print("-" * 60)
    
    for omega in omega_values:
        p = ThreeLevelParams(omega2=omega, omega3=omega, theta3=theta_fixed)
        
        lam_mf = np.mean([compute_lyapunov_3L(p, update_3L_meanfield, seed=s) 
                         for s in range(3)])
        lam_st = np.mean([compute_lyapunov_3L(p, update_3L_structured, seed=s) 
                         for s in range(3)])
        
        results['meanfield'].append(lam_mf if np.isfinite(lam_mf) else None)
        results['structured'].append(lam_st if np.isfinite(lam_st) else None)
        
        diff = lam_mf - lam_st if np.isfinite(lam_mf) and np.isfinite(lam_st) else np.nan
        print(f"{omega:.1f}\t\t{lam_mf:.4f}\t\t{lam_st:.4f}\t\t{diff:+.4f}")
    
    return results


def asymmetric_structuring():
    """Explore asymmetric coupling strengths at different interfaces."""
    print("\n" + "=" * 70)
    print("ASYMMETRIC STRUCTURING ANALYSIS")
    print("=" * 70)
    
    gamma_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    theta_fixed = 0.05
    p = ThreeLevelParams(theta3=theta_fixed)
    
    print("\nγ₁₂\tγ₂₃\tλ_max")
    print("-" * 40)
    
    results = []
    for g12 in gamma_values:
        for g23 in gamma_values:
            def update_fn(state, y, p):
                from three_level_analysis import update_3L_structured
                update_3L_structured(state, y, p, gamma12=g12, gamma23=g23)
            
            lam = np.mean([compute_lyapunov_3L(p, update_fn, seed=s) for s in range(3)])
            results.append({'gamma12': g12, 'gamma23': g23, 'lambda': lam})
            print(f"{g12:.2f}\t{g23:.2f}\t{lam:.4f}")
    
    return results


def cf_vs_stability():
    """Correlate CF values with Lyapunov exponents."""
    print("\n" + "=" * 70)
    print("CF vs STABILITY CORRELATION")
    print("=" * 70)
    
    theta_values = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    
    print("\nθ₃\tλ_MF\tCF₁₂\tCF₂₃\tTotal I")
    print("-" * 60)
    
    results = []
    for theta in theta_values:
        p = ThreeLevelParams(theta3=theta)
        
        # Lyapunov
        lam = np.mean([compute_lyapunov_3L(p, update_3L_meanfield, seed=s) for s in range(3)])
        
        # CF
        mu1, mu2, mu3 = simulate_3L(p, update_3L_meanfield, T=11000)
        cf = compute_pairwise_cf(mu1, mu2, mu3)
        
        total_I = cf['I12'] + cf['I23']
        results.append({
            'theta': theta, 
            'lambda': lam,
            'CF12': cf['CF12'],
            'CF23': cf['CF23'],
            'total_I': total_I
        })
        
        print(f"{theta:.2f}\t{lam:.4f}\t{cf['CF12']:.4f}\t{cf['CF23']:.4f}\t{total_I:.4f}")
    
    return results


def generate_summary_statistics():
    """Generate summary statistics for thesis."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS FOR THESIS")
    print("=" * 70)
    
    # Standard parameters
    p_standard = ThreeLevelParams(theta3=0.05)
    
    # Compute statistics
    n_trials = 10
    
    mf_lambdas = [compute_lyapunov_3L(p_standard, update_3L_meanfield, seed=s) 
                  for s in range(n_trials)]
    st_lambdas = [compute_lyapunov_3L(p_standard, update_3L_structured, seed=s) 
                  for s in range(n_trials)]
    bot_lambdas = [compute_lyapunov_3L(p_standard, update_3L_bottom_structured, seed=s) 
                   for s in range(n_trials)]
    top_lambdas = [compute_lyapunov_3L(p_standard, update_3L_top_structured, seed=s) 
                   for s in range(n_trials)]
    
    # Filter valid
    mf_lambdas = [x for x in mf_lambdas if np.isfinite(x)]
    st_lambdas = [x for x in st_lambdas if np.isfinite(x)]
    bot_lambdas = [x for x in bot_lambdas if np.isfinite(x)]
    top_lambdas = [x for x in top_lambdas if np.isfinite(x)]
    
    print(f"\nAt θ₃ = 0.05 (n = {n_trials} trials):")
    print(f"  Mean-field:     λ = {np.mean(mf_lambdas):.4f} ± {np.std(mf_lambdas):.4f}")
    print(f"  Structured:     λ = {np.mean(st_lambdas):.4f} ± {np.std(st_lambdas):.4f}")
    print(f"  Bottom-struct:  λ = {np.mean(bot_lambdas):.4f} ± {np.std(bot_lambdas):.4f}")
    print(f"  Top-struct:     λ = {np.mean(top_lambdas):.4f} ± {np.std(top_lambdas):.4f}")
    
    # CF statistics
    mu1, mu2, mu3 = simulate_3L(p_standard, update_3L_meanfield, T=20000)
    cf_mf = compute_pairwise_cf(mu1, mu2, mu3)
    
    mu1, mu2, mu3 = simulate_3L(p_standard, update_3L_structured, T=20000)
    cf_st = compute_pairwise_cf(mu1, mu2, mu3)
    
    print(f"\nCF Statistics:")
    print(f"  Mean-field:  CF₁₂ = {cf_mf['CF12']:.4f}, CF₂₃ = {cf_mf['CF23']:.4f}")
    print(f"  Structured:  CF₁₂ = {cf_st['CF12']:.4f}, CF₂₃ = {cf_st['CF23']:.4f}")
    print(f"  Correlations (MF): ρ₁₂ = {cf_mf['rho12']:.4f}, ρ₂₃ = {cf_mf['rho23']:.4f}")
    
    return {
        'mf_lambda': (np.mean(mf_lambdas), np.std(mf_lambdas)),
        'st_lambda': (np.mean(st_lambdas), np.std(st_lambdas)),
        'bot_lambda': (np.mean(bot_lambdas), np.std(bot_lambdas)),
        'top_lambda': (np.mean(top_lambdas), np.std(top_lambdas)),
        'cf_mf': cf_mf,
        'cf_st': cf_st
    }


if __name__ == "__main__":
    print("THREE-LEVEL HGF: SYSTEMATIC PARAMETER EXPLORATION")
    print("=" * 70)
    
    # Run all explorations
    theta_results = systematic_theta_exploration()
    kappa_results = kappa_exploration()
    omega_results = omega_exploration()
    cf_stability = cf_vs_stability()
    summary = generate_summary_statistics()
    
    # Save results
    all_results = {
        'theta_exploration': theta_results,
        'kappa_exploration': kappa_results,
        'omega_exploration': omega_results,
        'cf_stability': cf_stability,
        'summary': {
            'mf_lambda_mean': summary['mf_lambda'][0],
            'mf_lambda_std': summary['mf_lambda'][1],
            'st_lambda_mean': summary['st_lambda'][0],
            'st_lambda_std': summary['st_lambda'][1],
        }
    }
    
    with open('parameter_exploration_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if not np.isfinite(x) else x)
    
    print("\n" + "=" * 70)
    print("Results saved to parameter_exploration_results.json")
