# Physiological Calibration Data

This document summarizes the empirical data used to calibrate model parameters.

## Dopamine Concentrations

| Parameter | Value | Method | Source |
|-----------|-------|--------|--------|
| Baseline tonic DA | 4–30 nM | FSCV | Robinson et al. (2002) |
| Tonic DA | 90 ± 9 nM | M-CSWV | Oh et al. (2018) |
| Phasic burst DA | 210 ± 10 nM | FSCV | Robinson et al. (2002) |
| Low DA (Parkinsonian) | < 10 nM | Post-mortem | Bergstrom & Bhardwaj (2022) |
| High DA (Amphetamine) | 150–200 nM | FSCV | Sulzer et al. (2005) |

### Notes on Measurement Methods

**FSCV (Fast-Scan Cyclic Voltammetry):** Standard method for measuring phasic dopamine release. Good temporal resolution (~100 ms) but requires subtraction of baseline, which can be variable.

**M-CSWV (Multiple Cyclic Square Wave Voltammetry):** Newer technique allowing direct measurement of tonic (baseline) dopamine without background subtraction. Suggests tonic levels are higher than previously estimated with FSCV.

### Calibration Choice

We use D₀ = 90 nM as the homeostatic setpoint, based on Oh et al. (2018). This represents "normal" tonic dopamine at rest.

## Dopamine Receptor Affinities

| Receptor | Kᵢ | Implication | Source |
|----------|-----|-------------|--------|
| D2 | 10–30 nM | Sensitive to tonic DA changes | Richfield et al. (1989) |
| D1 | 1–10 μM | Activated mainly by phasic bursts | Richfield et al. (1989) |

### Functional Interpretation

The D2 receptor affinity (Kᵢ ≈ 10–30 nM) places it squarely in the tonic DA range, making it suitable for encoding slow changes in precision (circulatory fidelity).

D1 receptors, with ~100× lower affinity, require the high concentrations achieved during phasic bursts, consistent with encoding prediction errors rather than precision.

## VTA Firing Rates

| Mode | Rate | Conditions | Source |
|------|------|------------|--------|
| Tonic | ~4.5 Hz | Baseline, awake | Grace & Bunney (1984) |
| Burst | 15–30 Hz | Reward, salient stimuli | Grace & Bunney (1984) |
| Suppressed | < 2 Hz | Aversive, negative PE | Schultz et al. (1997) |

## Metabolic Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Landauer bound | 3 × 10⁻²¹ J/bit | Landauer (1961) |
| Neural cost | ~10⁴ ATP/bit | Laughlin et al. (1998) |
| Energy per ATP | 8.3 × 10⁻²⁰ J | Standard biochemistry |
| Neural inefficiency | ~10⁸ × Landauer | Derived |

### Metabolic Budget

A typical cortical neuron firing at 4 Hz consumes approximately:
- 4.7 × 10⁸ ATP/s (Attwell & Laughlin, 2001)
- ~10⁹ ATP/s for a small cortical area (~10⁶ neurons)

This provides substantial "budget" for maintaining precision, but the cost scales with precision, creating the trade-off captured by TFE.

## Transfer Function Calibration

The sigmoid transfer function γ(D) is calibrated as follows:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| γ_max | 100 | Typical maximum precision in HGF fitting |
| D₀ | 90 nM | Homeostatic setpoint (Oh et al. 2018) |
| k_sigmoid | 4 | Steepness to match D2 receptor dynamics |

### Transfer Function Validation

At key dopamine levels:

| D (nM) | γ(D) | Interpretation |
|--------|------|----------------|
| 10 | ~1 | Parkinsonian: very low precision |
| 50 | ~27 | Below baseline |
| 90 | 50 | Homeostatic |
| 150 | ~88 | Elevated (stimulant) |
| 200 | ~97 | Near maximum |

## Bifurcation Thresholds

Based on numerical simulation with calibrated parameters:

| Threshold | Value | Conditions |
|-----------|-------|------------|
| ϑ_c (first bifurcation) | 0.045 ± 0.005 | κ=1, ω=-2, π_u=10, γ=50 |
| ϑ_chaos | 0.12 ± 0.01 | Same conditions |
| λ_max at ϑ=0.20 | > 1.0 bits/step | Mean-field |
| λ_max at ϑ=0.20 | < 0 | Structured |

## Clinical Populations

| Condition | Estimated DA | Predicted γ | Expected Dynamics |
|-----------|--------------|-------------|-------------------|
| Healthy | 90 nM | 50 | Stable |
| Schizophrenia | 130–180 nM | 75–92 | At risk for chaos |
| Parkinson's | < 20 nM | < 5 | Over-damped |
| ADHD | Variable | Variable | Increased variability |
| Stimulant use | 150–300 nM | 88–99 | Potentially chaotic |

## References

Attwell, D., & Laughlin, S. B. (2001). An energy budget for signaling in the grey matter of the brain. *Journal of Cerebral Blood Flow & Metabolism*, 21(10), 1133-1145.

Bergstrom, K., & Bhardwaj, A. (2022). Dopamine dynamics in Parkinson's disease. *Neuropharmacology*, 209, 108981.

Grace, A. A., & Bunney, B. S. (1984). The control of firing pattern in nigral dopamine neurons: Single spike firing. *Journal of Neuroscience*, 4(11), 2866-2876.

Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM Journal of Research and Development*, 5(3), 183-191.

Laughlin, S. B., et al. (1998). The metabolic cost of neural information. *Nature Neuroscience*, 1(1), 36-41.

Oh, Y., et al. (2018). Tracking tonic dopamine levels in vivo using multiple cyclic square wave voltammetry. *Biosensors and Bioelectronics*, 121, 174-182.

Richfield, E. K., et al. (1989). Anatomical and affinity state comparisons between dopamine D1 and D2 receptors in the rat central nervous system. *Neuroscience*, 30(3), 767-777.

Robinson, D. L., et al. (2002). Detecting subsecond dopamine release with fast-scan cyclic voltammetry in vivo. *Clinical Chemistry*, 49(10), 1763-1773.

Schultz, W., et al. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.

Sulzer, D., et al. (2005). Mechanisms of neurotransmitter release by amphetamines. *Progress in Neurobiology*, 75(6), 406-433.
