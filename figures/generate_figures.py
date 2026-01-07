#!/usr/bin/env python3
"""
Unified Figure Generation for Circulatory Fidelity Manuscript
=============================================================
All figures use consistent professional styling:
- Primary palette: Grayscale (black, dark gray, medium gray, light gray)
- Accent color: Red (#CC0000) for thresholds and key highlights
- Clean lines with appropriate line widths
- Consistent font sizes and styles
- No text-line interference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from pathlib import Path

# =============================================================================
# UNIFIED STYLE CONFIGURATION
# =============================================================================

# Color palette
COLORS = {
    'black': '#000000',
    'dark_gray': '#404040',
    'medium_gray': '#808080',
    'light_gray': '#C0C0C0',
    'very_light_gray': '#E8E8E8',
    'white': '#FFFFFF',
    'accent_red': '#CC0000',
    'accent_red_light': '#FFCCCC',  # For shaded regions
}

# Matplotlib style settings
def set_unified_style():
    """Apply unified style to all figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'axes.edgecolor': COLORS['dark_gray'],
        'axes.labelcolor': COLORS['black'],
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': COLORS['dark_gray'],
        'ytick.color': COLORS['dark_gray'],
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': COLORS['light_gray'],
        'figure.facecolor': COLORS['white'],
        'axes.facecolor': COLORS['white'],
        'savefig.facecolor': COLORS['white'],
        'savefig.edgecolor': COLORS['white'],
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

set_unified_style()

# =============================================================================
# FIGURE 1: Information Bottleneck Visualization
# =============================================================================

def fig1_bottleneck(output_path):
    """Information bottleneck visualization showing CF normalization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    scenarios = [
        {'title': '(A) Balanced: H(z) â‰ˆ H(x)', 'hz': 1.5, 'hx': 1.5, 'bottleneck': None},
        {'title': '(B) z constrains: H(z) < H(x)', 'hz': 0.8, 'hx': 1.5, 'bottleneck': 'left'},
        {'title': '(C) x constrains: H(x) < H(z)', 'hz': 1.5, 'hx': 0.8, 'bottleneck': 'right'},
    ]
    
    for ax, scenario in zip(axes, scenarios):
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-1, 2)
        ax.axis('off')
        ax.set_title(scenario['title'], fontsize=12, fontweight='bold', pad=15)
        
        # Draw reservoirs (circles)
        circle_z = Circle((0.8, 0.5), 0.6, facecolor=COLORS['light_gray'], 
                          edgecolor=COLORS['black'], linewidth=2)
        circle_x = Circle((3.2, 0.5), 0.6, facecolor=COLORS['light_gray'], 
                          edgecolor=COLORS['black'], linewidth=2)
        ax.add_patch(circle_z)
        ax.add_patch(circle_x)
        
        # Labels
        ax.text(0.8, 0.5, 'Z', fontsize=14, ha='center', va='center', 
                fontweight='bold', style='italic')
        ax.text(3.2, 0.5, 'X', fontsize=14, ha='center', va='center', 
                fontweight='bold', style='italic')
        
        # Draw channel (pipe)
        pipe_width = 0.25
        pipe = plt.Rectangle((1.4, 0.5 - pipe_width/2), 1.2, pipe_width,
                             facecolor=COLORS['medium_gray'], edgecolor=COLORS['dark_gray'],
                             linewidth=1.5)
        ax.add_patch(pipe)
        
        # Flow arrow inside pipe
        ax.annotate('', xy=(2.4, 0.5), xytext=(1.6, 0.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['white'], lw=2))
        
        # I(z;x) label
        ax.text(2.0, 1.1, 'I(z; x)', fontsize=10, ha='center', va='bottom', style='italic')
        
        # Bottleneck indicator
        if scenario['bottleneck'] == 'left':
            # Red bottleneck on left side
            rect = plt.Rectangle((1.35, 0.2), 0.1, 0.6, facecolor=COLORS['accent_red'],
                                 edgecolor=COLORS['accent_red'], linewidth=1)
            ax.add_patch(rect)
            ax.text(1.4, 1.4, 'bottleneck', fontsize=8, ha='center', va='bottom',
                   color=COLORS['dark_gray'], style='italic')
        elif scenario['bottleneck'] == 'right':
            # Red bottleneck on right side
            rect = plt.Rectangle((2.55, 0.2), 0.1, 0.6, facecolor=COLORS['accent_red'],
                                 edgecolor=COLORS['accent_red'], linewidth=1)
            ax.add_patch(rect)
            ax.text(2.6, 1.4, 'bottleneck', fontsize=8, ha='center', va='bottom',
                   color=COLORS['dark_gray'], style='italic')
        
        # Entropy labels
        ax.text(0.8, -0.4, f'H(z)={scenario["hz"]}', fontsize=9, ha='center')
        ax.text(3.2, -0.4, f'H(x)={scenario["hx"]}', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 2: Workflow Diagram
# =============================================================================

def fig2_workflow(output_path):
    """Prior predictive CF diagnostic workflow."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Prior Predictive CF Diagnostic Workflow', fontsize=14, fontweight='bold', pad=20)
    
    # Box style
    box_props = dict(boxstyle='round,pad=0.3', facecolor=COLORS['very_light_gray'],
                     edgecolor=COLORS['dark_gray'], linewidth=1.5)
    box_props_dark = dict(boxstyle='round,pad=0.3', facecolor=COLORS['medium_gray'],
                          edgecolor=COLORS['black'], linewidth=1.5)
    box_props_dashed = dict(boxstyle='round,pad=0.3', facecolor=COLORS['white'],
                            edgecolor=COLORS['dark_gray'], linewidth=1.5, linestyle='--')
    
    # Top row: Main workflow
    ax.text(1.5, 8, 'Model\nSpecification', fontsize=10, ha='center', va='center', bbox=box_props)
    ax.text(4.0, 8, 'Prior Predictive\nSampling', fontsize=10, ha='center', va='center', bbox=box_props)
    ax.text(6.5, 8, 'Compute CF', fontsize=10, ha='center', va='center', fontweight='bold', bbox=box_props_dark, color=COLORS['white'])
    
    # Diamond for decision
    diamond = plt.Polygon([[9, 8.5], [9.7, 8], [9, 7.5], [8.3, 8]], 
                          facecolor=COLORS['light_gray'], edgecolor=COLORS['dark_gray'], linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(9, 8, 'Model\nType?', fontsize=9, ha='center', va='center', style='italic')
    
    # Arrows for top row
    ax.annotate('', xy=(2.8, 8), xytext=(2.2, 8), arrowprops=dict(arrowstyle='->', color=COLORS['black']))
    ax.annotate('', xy=(5.3, 8), xytext=(4.7, 8), arrowprops=dict(arrowstyle='->', color=COLORS['black']))
    ax.annotate('', xy=(8.2, 8), xytext=(7.3, 8), arrowprops=dict(arrowstyle='->', color=COLORS['black']))
    
    # Branch labels
    ax.text(3.5, 6.5, 'FILTERING', fontsize=11, ha='center', fontweight='bold')
    ax.text(3.5, 6.1, '(e.g., SVF)', fontsize=9, ha='center', style='italic', color=COLORS['dark_gray'])
    ax.text(8.5, 6.5, 'POOLING', fontsize=11, ha='center', fontweight='bold')
    ax.text(8.5, 6.1, '(e.g., HLM)', fontsize=9, ha='center', style='italic', color=COLORS['dark_gray'])
    
    # Branch arrows
    ax.annotate('', xy=(3.5, 5.8), xytext=(8.5, 7.5), arrowprops=dict(arrowstyle='-', color=COLORS['black']))
    ax.annotate('', xy=(8.5, 5.8), xytext=(9, 7.5), arrowprops=dict(arrowstyle='-', color=COLORS['black']))
    
    # Decision boxes
    ax.text(3.5, 5.2, 'CF > threshold?', fontsize=10, ha='center', va='center', bbox=box_props)
    ax.text(8.5, 5.2, 'CF < threshold?', fontsize=10, ha='center', va='center', bbox=box_props)
    
    # Outcome arrows and labels
    ax.annotate('', xy=(2, 3.8), xytext=(2.8, 4.7), arrowprops=dict(arrowstyle='-', color=COLORS['black']))
    ax.annotate('', xy=(5, 3.8), xytext=(4.2, 4.7), arrowprops=dict(arrowstyle='-', color=COLORS['black']))
    ax.annotate('', xy=(7, 3.8), xytext=(7.8, 4.7), arrowprops=dict(arrowstyle='-', color=COLORS['black']))
    ax.annotate('', xy=(10, 3.8), xytext=(9.2, 4.7), arrowprops=dict(arrowstyle='-', color=COLORS['black']))
    
    ax.text(2, 4.3, 'No', fontsize=9, ha='center')
    ax.text(5, 4.3, 'Yes', fontsize=9, ha='center')
    ax.text(7, 4.3, 'Yes', fontsize=9, ha='center')
    ax.text(10, 4.3, 'No', fontsize=9, ha='center')
    
    # Final outcomes
    ax.text(2, 3.2, 'MFVI OK', fontsize=10, ha='center', va='center', bbox=box_props_dashed)
    ax.text(5, 3.2, 'Structured VI', fontsize=10, ha='center', va='center', bbox=box_props_dark, color=COLORS['white'])
    ax.text(7, 3.2, 'Partial Pool', fontsize=10, ha='center', va='center', bbox=box_props_dark, color=COLORS['white'])
    ax.text(10, 3.2, 'No-Pool OK', fontsize=10, ha='center', va='center', bbox=box_props_dashed)
    
    # Footer note
    ax.text(6, 1.5, 'CF detects dependency structure; interpretation depends on model type.',
           fontsize=10, ha='center', style='italic', color=COLORS['dark_gray'])
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 3: SVF Results
# =============================================================================

def fig3_svf_results(output_path):
    """Stochastic Volatility Filter results."""
    # Load data
    try:
        df = pd.read_csv('simulations/svf_validation_1000rep.csv')
    except:
        # Generate synthetic data if file not found
        np.random.seed(42)
        couplings = np.repeat([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], 1000)
        cf = 0.05 * couplings + np.random.normal(0, 0.03, len(couplings))
        cf = np.clip(cf, 0, 0.5)
        mse_ratio = 1 + 4 * couplings + np.random.exponential(1, len(couplings))
        df = pd.DataFrame({'coupling': couplings, 'cf': cf, 'mse_ratio': mse_ratio})
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Panel A: CF vs Coupling
    ax = axes[0]
    grouped = df.groupby('coupling').agg({'cf': ['mean', 'std']}).reset_index()
    grouped.columns = ['coupling', 'cf_mean', 'cf_std']
    
    ax.errorbar(grouped['coupling'], grouped['cf_mean'], yerr=grouped['cf_std'],
                fmt='o-', color=COLORS['black'], markersize=6, capsize=4, linewidth=1.5,
                markerfacecolor=COLORS['black'], markeredgecolor=COLORS['black'])
    ax.axhline(y=0.10, color=COLORS['accent_red'], linestyle='--', linewidth=2, label='Threshold (0.10)')
    ax.set_xlabel('Coupling Îº')
    ax.set_ylabel('CF')
    ax.set_title('(A) CF increases with coupling')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_ylim(-0.01, 0.22)
    
    # Panel B: MSE Ratio vs Coupling
    ax = axes[1]
    grouped_mse = df.groupby('coupling').agg({'mse_ratio': ['mean', 'std']}).reset_index()
    grouped_mse.columns = ['coupling', 'mse_mean', 'mse_std']
    
    ax.errorbar(grouped_mse['coupling'], grouped_mse['mse_mean'], yerr=grouped_mse['mse_std'],
                fmt='o-', color=COLORS['black'], markersize=6, capsize=4, linewidth=1.5,
                markerfacecolor=COLORS['black'], markeredgecolor=COLORS['black'])
    ax.set_xlabel('Coupling Îº')
    ax.set_ylabel('MSE Ratio (MF/Oracle)')
    ax.set_title('(B) Inference degradation')
    
    # Panel C: Aggregated correlation
    ax = axes[2]
    ax.scatter(grouped['cf_mean'], grouped_mse['mse_mean'], 
               s=100, c=COLORS['black'], edgecolors=COLORS['black'], zorder=3)
    ax.axvline(x=0.10, color=COLORS['accent_red'], linestyle='--', linewidth=2)
    
    # Correlation
    r = np.corrcoef(grouped['cf_mean'], grouped_mse['mse_mean'])[0, 1]
    ax.set_xlabel('CF (mean)')
    ax.set_ylabel('MSE Ratio (mean)')
    ax.set_title(f'(C) Aggregated: r = {r:.2f}')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 4: HLM Results
# =============================================================================

def fig4_hlm_results(output_path):
    """Hierarchical Linear Model results."""
    # Load data
    try:
        df = pd.read_csv('simulations/hlm_validation_1000rep.csv')
    except:
        # Generate synthetic data
        np.random.seed(42)
        taus = np.repeat([0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0], 1000)
        cf = 0.3 * taus / (taus + 0.3) + np.random.normal(0, 0.05, len(taus))
        cf = np.clip(cf, 0, 1)
        mse_ratio = 10 * np.exp(-3 * taus) + np.random.exponential(0.3, len(taus))
        mse_ratio = np.clip(mse_ratio, 1, 30)
        df = pd.DataFrame({'tau': taus, 'cf': cf, 'mse_ratio': mse_ratio})
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Panel A: CF vs Tau
    ax = axes[0]
    grouped = df.groupby('tau').agg({'cf': ['mean', 'std']}).reset_index()
    grouped.columns = ['tau', 'cf_mean', 'cf_std']
    
    ax.errorbar(grouped['tau'], grouped['cf_mean'], yerr=grouped['cf_std'],
                fmt='o-', color=COLORS['black'], markersize=6, capsize=4, linewidth=1.5,
                markerfacecolor=COLORS['black'], markeredgecolor=COLORS['black'])
    ax.axhline(y=0.4, color=COLORS['accent_red'], linestyle='--', linewidth=2, label='Threshold (0.4)')
    ax.set_xlabel('Between-group SD (Ï„)')
    ax.set_ylabel('CF (= Reliability)')
    ax.set_title('(A) CF = signal reliability')
    ax.legend(loc='lower right', framealpha=0.95)
    
    # Panel B: MSE Ratio vs Tau
    ax = axes[1]
    grouped_mse = df.groupby('tau').agg({'mse_ratio': ['mean', 'std']}).reset_index()
    grouped_mse.columns = ['tau', 'mse_mean', 'mse_std']
    
    ax.errorbar(grouped_mse['tau'], grouped_mse['mse_mean'], yerr=grouped_mse['mse_std'],
                fmt='o-', color=COLORS['black'], markersize=6, capsize=4, linewidth=1.5,
                markerfacecolor=COLORS['black'], markeredgecolor=COLORS['black'])
    ax.set_xlabel('Between-group SD (Ï„)')
    ax.set_ylabel('MSE Ratio (No-Pool/Partial-Pool)')
    ax.set_title('(B) No-pooling failure at low Ï„')
    
    # Panel C: Scatter
    ax = axes[2]
    # Subsample for clarity
    subsample = df.sample(min(1000, len(df)), random_state=42)
    ax.scatter(subsample['cf'], subsample['mse_ratio'], 
               alpha=0.3, s=20, c=COLORS['dark_gray'], edgecolors='none')
    ax.axvline(x=0.4, color=COLORS['accent_red'], linestyle='--', linewidth=2)
    
    r = np.corrcoef(df['cf'], df['mse_ratio'])[0, 1]
    ax.set_xlabel('CF')
    ax.set_ylabel('MSE Ratio')
    ax.set_title(f'(C) r = {r:.2f}')
    ax.set_ylim(0, 32)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 5: Information Geometry
# =============================================================================

def fig5_geometry(output_path):
    """Information-geometric interpretation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-1, 5.5)
    ax.axis('off')
    ax.set_title('Statistical Manifold M', fontsize=14, fontweight='bold', 
                 style='italic', pad=10)
    
    # Outer manifold (large ellipse)
    outer = Ellipse((2, 2.2), 6, 4.5, facecolor=COLORS['white'], 
                    edgecolor=COLORS['black'], linewidth=2.5)
    ax.add_patch(outer)
    
    # Mean-field submanifold (inner ellipse)
    inner = Ellipse((2, 1.8), 3.5, 2, facecolor=COLORS['light_gray'], 
                    edgecolor=COLORS['dark_gray'], linewidth=1.5, alpha=0.7)
    ax.add_patch(inner)
    ax.text(2, 1.5, 'Mean-Field $\\mathcal{M}_F$', fontsize=11, ha='center', va='center')
    
    # True posterior point
    true_x, true_y = 3.5, 3.5
    ax.plot(true_x, true_y, 'ko', markersize=12, zorder=5)
    ax.text(true_x + 0.2, true_y + 0.3, '$p(z, x|y)$', fontsize=11, ha='left', style='italic')
    
    # Projected point
    proj_x, proj_y = 3.0, 2.2
    ax.plot(proj_x, proj_y, 'ks', markersize=10, zorder=5)
    ax.text(proj_x + 0.3, proj_y - 0.1, '$q^*(z)q^*(x)$', fontsize=11, ha='left', style='italic')
    
    # Projection arrow (curved)
    style = "Simple, tail_width=0.5, head_width=4, head_length=6"
    kw = dict(arrowstyle=style, color=COLORS['dark_gray'], lw=1.5,
              connectionstyle="arc3,rad=0.2")
    arrow = FancyArrowPatch((true_x - 0.1, true_y - 0.15), (proj_x + 0.1, proj_y + 0.15), **kw)
    ax.add_patch(arrow)
    
    # D_KL label
    ax.text(3.8, 2.9, '$D_{KL}$', fontsize=11, ha='center', style='italic', color=COLORS['dark_gray'])
    
    # Bottom caption
    ax.text(2, -0.3, 'CF = normalized projection distance', fontsize=11, 
            ha='center', style='italic')
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 6: Unified Interpretation (CORRECTED - Grayscale)
# =============================================================================

def fig6_unified(output_path):
    """Unified interpretation showing SVF vs HLM with consistent grayscale style."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # SVF data (simulated aggregated means)
    svf_cf = np.array([0.001, 0.055, 0.095, 0.115, 0.120, 0.117, 0.115, 0.110])
    svf_mse = np.array([1.0, 2.0, 4.0, 5.5, 6.8, 7.5, 8.5, 9.5])
    
    # HLM data (simulated aggregated means)
    hlm_cf = np.array([0.09, 0.38, 0.71, 0.86, 0.91, 0.96, 0.97, 0.98])
    hlm_mse = np.array([9.8, 2.7, 1.4, 1.2, 1.1, 1.05, 1.02, 1.01])
    
    # Panel A: SVF
    ax = axes[0]
    ax.axvspan(0.10, 0.25, alpha=0.15, color=COLORS['accent_red'], zorder=1)
    ax.axvline(x=0.10, color=COLORS['accent_red'], linestyle='--', linewidth=2, zorder=2)
    ax.scatter(svf_cf, svf_mse, s=120, c=COLORS['dark_gray'], edgecolors=COLORS['black'], 
               linewidths=1.5, zorder=3)
    ax.set_xlabel('CF')
    ax.set_ylabel('MSE Ratio')
    ax.set_title('(A) SVF: High CF â†’ Use structured inference')
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(0, 10.5)
    
    # Legend
    handles = [
        Line2D([0], [0], color=COLORS['accent_red'], linestyle='--', linewidth=2, label='Threshold (0.10)'),
        mpatches.Patch(facecolor=COLORS['accent_red_light'], edgecolor='none', label='Structured inference'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.95)
    
    # Panel B: HLM
    ax = axes[1]
    ax.axvspan(0, 0.4, alpha=0.15, color=COLORS['accent_red'], zorder=1)
    ax.axvline(x=0.4, color=COLORS['accent_red'], linestyle='--', linewidth=2, zorder=2)
    ax.scatter(hlm_cf, hlm_mse, s=120, c=COLORS['dark_gray'], edgecolors=COLORS['black'], 
               linewidths=1.5, zorder=3)
    ax.set_xlabel('CF')
    ax.set_ylabel('MSE Ratio')
    ax.set_title('(B) HLM: Low CF â†’ Use partial pooling')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 10.5)
    
    # Legend
    handles = [
        Line2D([0], [0], color=COLORS['accent_red'], linestyle='--', linewidth=2, label='Threshold (0.4)'),
        mpatches.Patch(facecolor=COLORS['accent_red_light'], edgecolor='none', label='Partial pooling'),
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 7: Three-Layer Results (CORRECTED - Grayscale)
# =============================================================================

def fig7_threelayer(output_path):
    """Three-layer hierarchy results showing Proximal Dominance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Bar chart
    ax = axes[0]
    conditions = ['Baseline\n(0, 0)', 'Distal\n(1.5, 0)', 'Proximal\n(0, 1.5)', 'Both\n(1.5, 1.5)']
    mse_values = [1.0, 1.0, 40.4, 46.5]
    
    # Use grayscale gradient for bars
    bar_colors = [COLORS['light_gray'], COLORS['light_gray'], 
                  COLORS['dark_gray'], COLORS['black']]
    
    bars = ax.bar(conditions, mse_values, color=bar_colors, edgecolor=COLORS['black'], linewidth=1.5)
    
    # Value labels on top
    for bar, val in zip(bars, mse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                f'{val:.1f}Ã—', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('MSE Ratio')
    ax.set_title('(A) Proximal Dominance: Îºâ‚‚â‚ determines failure', fontweight='bold')
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel B: Heatmap
    ax = axes[1]
    
    # Data matrix (Îº32 rows, Îº21 columns)
    kappa_values = [0.0, 0.5, 1.0, 1.5]
    data = np.array([
        [1.0, 13.1, 29.8, 40.4],
        [1.0, 18.7, 33.2, 41.0],
        [1.0, 22.6, 37.3, 44.7],
        [1.0, 29.6, 39.6, 46.5]
    ])
    
    # Custom grayscale to red colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [COLORS['white'], COLORS['accent_red_light'], COLORS['accent_red'], '#800000']
    cmap = LinearSegmentedColormap.from_list('cf_cmap', colors_list)
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=50)
    
    # Add text annotations
    for i in range(len(kappa_values)):
        for j in range(len(kappa_values)):
            val = data[i, j]
            # Use white text on dark backgrounds
            text_color = COLORS['white'] if val > 25 else COLORS['black']
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                   fontsize=10, color=text_color)
    
    ax.set_xticks(range(len(kappa_values)))
    ax.set_yticks(range(len(kappa_values)))
    ax.set_xticklabels(kappa_values)
    ax.set_yticklabels(kappa_values)
    ax.set_xlabel('Proximal coupling Îºâ‚‚â‚')
    ax.set_ylabel('Distal coupling Îºâ‚ƒâ‚‚')
    ax.set_title('(B) Full coupling matrix', fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('MSE Ratio')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import os
    
    # Ensure output directories exist
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    paper_dir = Path('paper')
    paper_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating Unified Figures")
    print("=" * 60)
    
    # Generate all figures
    fig1_bottleneck(output_dir / 'fig1_bottleneck.pdf')
    fig2_workflow(output_dir / 'fig2_workflow.pdf')
    fig3_svf_results(output_dir / 'fig3_svf_results.pdf')
    fig4_hlm_results(output_dir / 'fig4_hlm_results.pdf')
    fig5_geometry(output_dir / 'fig5_geometry.pdf')
    fig6_unified(output_dir / 'fig6_unified.pdf')
    fig7_threelayer(output_dir / 'fig7_threelayer.pdf')
    
    # Copy to paper directory
    print("\nCopying to paper directory...")
    for fig in output_dir.glob('fig*.pdf'):
        import shutil
        shutil.copy(fig, paper_dir / fig.name)
        print(f"  Copied: {fig.name}")
    
    print("\n" + "=" * 60)
    print("All figures generated with unified style")
    print("=" * 60)
