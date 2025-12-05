# Derivation of the Dopamine-Precision Transfer Function from Receptor Binding Kinetics

**Supplementary Material for Circulatory Fidelity Thesis**

---

## 1. Motivation

The thesis proposes that tonic dopamine concentration modulates the precision parameter γ in hierarchical inference. Previous versions posited a sigmoid transfer function with parameters chosen for convenience. This supplement derives the functional form from receptor binding kinetics, clarifying which aspects are principled and which remain modeling choices.

---

## 2. Receptor-Ligand Binding: First Principles

### 2.1 Mass Action Kinetics

Consider a receptor R binding a ligand L (dopamine) to form a complex RL:

$$R + L \underset{k_{\text{off}}}{\stackrel{k_{\text{on}}}{\rightleftharpoons}} RL$$

At equilibrium, the rate of complex formation equals the rate of dissociation:

$$k_{\text{on}} [R][L] = k_{\text{off}} [RL]$$

### 2.2 Dissociation Constant

The dissociation constant K_d is defined as:

$$K_d = \frac{k_{\text{off}}}{k_{\text{on}}} = \frac{[R][L]}{[RL]}$$

K_d has units of concentration and represents the ligand concentration at which half of receptors are occupied at equilibrium.

### 2.3 Fractional Occupancy

Let R_T = [R] + [RL] be the total receptor concentration. The fractional occupancy θ (proportion of receptors bound to ligand) is:

$$\theta = \frac{[RL]}{R_T} = \frac{[RL]}{[R] + [RL]}$$

Substituting from the equilibrium condition:

$$\theta = \frac{[L]}{K_d + [L]}$$

This is the **Langmuir isotherm** or **Hill equation with n=1**.

### 2.4 Cooperative Binding (Hill Equation)

When receptor binding exhibits cooperativity (multiple binding sites influencing each other), the relationship becomes:

$$\theta = \frac{[L]^n}{K_d^n + [L]^n}$$

where n is the **Hill coefficient**:
- n = 1: No cooperativity (standard Langmuir)
- n > 1: Positive cooperativity (binding facilitates further binding)
- n < 1: Negative cooperativity (binding inhibits further binding)

For G-protein coupled receptors like dopamine receptors, n typically ranges from 0.8 to 1.5, depending on the receptor subtype and cellular context.

---

## 3. Application to Dopamine Receptors

### 3.1 D2 Receptor Parameters

D2 receptors are the primary candidates for encoding tonic dopamine levels because their binding affinity matches the tonic concentration range.

**Empirical values from Richfield et al. (1989) and Seeman et al. (2006):**

| Parameter | Value | Source |
|-----------|-------|--------|
| K_d (D2 high-affinity state) | 2-10 nM | Seeman et al. (2006) |
| K_d (D2 low-affinity state) | 10-30 nM | Richfield et al. (1989) |
| K_d (D2 population average) | ~20 nM | Consensus |
| Hill coefficient n | 0.8-1.2 | Typical GPCR range |

**Note on affinity states:** D2 receptors exist in high-affinity and low-affinity states. The population-average K_d (~20 nM) is appropriate for modeling bulk effects on neural gain.

### 3.2 Comparison with Tonic Dopamine Concentrations

| Condition | [DA] (nM) | θ at K_d = 20 nM | Source |
|-----------|-----------|------------------|--------|
| Parkinsonian | < 10 | < 0.33 | Bergstrom & Bhardwaj (2022) |
| Low normal | 20-40 | 0.50-0.67 | Robinson et al. (2002) |
| Baseline (M-CSWV) | 90 ± 9 | 0.82 | Oh et al. (2018) |
| Elevated | 150-200 | 0.88-0.91 | Sulzer et al. (2005) |

The D2 receptor K_d places it squarely within the dynamic range of tonic dopamine fluctuations, making it suitable for encoding precision.

---

## 4. From Occupancy to Precision

### 4.1 The Linking Assumption

We assume that D2 receptor occupancy modulates neural gain, and that this gain corresponds to precision in the inference framework. Specifically:

**Assumption 1 (Gain Modulation):** The effective gain g of a neural population is a monotonic function of D2 receptor occupancy θ.

This assumption is supported by:
- D2 receptors modulate excitability of striatal medium spiny neurons (Surmeier et al., 2007)
- Dopamine modulates signal-to-noise ratio in prefrontal cortex (Seamans & Yang, 2004)
- Computational models linking dopamine to gain control (Servan-Schreiber et al., 1990)

### 4.2 Linear vs. Nonlinear Mapping

The simplest assumption is that precision is proportional to occupancy:

**Assumption 2 (Linear Mapping):** γ = γ_max · θ

This gives:

$$\gamma(D) = \gamma_{\max} \cdot \frac{D^n}{K_d^n + D^n}$$

where we have substituted [L] = D (dopamine concentration).

**Alternative (Threshold Mapping):** One could instead assume a threshold effect where precision only increases above some occupancy level. We do not pursue this here but note it as an alternative.

### 4.3 The γ_max Parameter

The parameter γ_max represents the maximum achievable precision when all receptors are saturated (θ → 1). This is **not** constrained by receptor kinetics—it depends on downstream neural circuitry.

**Possible constraints on γ_max:**
- Must be finite (biological systems have bounded gain)
- Should be calibrated to behavioral data (e.g., from HGF fits)
- Typical values in HGF modeling: γ_max ∈ [10, 1000]

We retain γ_max as a free parameter to be determined empirically.

---

## 5. The Derived Transfer Function

### 5.1 Final Form

Combining the Hill equation with the linear mapping assumption:

$$\boxed{\gamma(D) = \gamma_{\max} \cdot \frac{D^n}{K_d^n + D^n}}$$

**Parameters:**
- D: Dopamine concentration (nM) — **measurable**
- K_d ≈ 20 nM: D2 receptor dissociation constant — **constrained by pharmacology**
- n ≈ 1: Hill coefficient — **constrained by receptor biophysics**
- γ_max: Maximum precision — **free parameter, to be fit**

### 5.2 Properties

1. **γ(0) = 0**: Zero dopamine → zero precision (complete reliance on priors)

2. **γ(K_d) = γ_max/2**: At the dissociation constant, precision is half-maximal

3. **γ(D) → γ_max as D → ∞**: Precision saturates at high dopamine

4. **Sensitivity:** The derivative dγ/dD is maximal at D = K_d, meaning the system is most sensitive to dopamine changes around the dissociation constant.

### 5.3 Comparison with Previous Formulation

The previous (ad hoc) formulation was:

$$\gamma(D) = \frac{\gamma_{\max}}{1 + \exp\left(-k \cdot \frac{D - D_0}{D_0}\right)}$$

This sigmoid has a different functional form but similar qualitative behavior. The key differences:

| Aspect | Hill (Derived) | Sigmoid (Ad Hoc) |
|--------|----------------|------------------|
| Functional form | Rational | Exponential |
| Inflection point | D = K_d (pharmacological) | D = D_0 (arbitrary) |
| Low-D behavior | γ ∝ D^n | γ → γ_max/(1+e^k) > 0 |
| Parameter origin | Receptor binding | Curve fitting |

The Hill form has the advantage that K_d is directly measurable via radioligand binding assays, whereas D_0 in the sigmoid was an unconstrained fitting parameter.

---

## 6. Incorporating D1 Receptors (Optional Extension)

### 6.1 D1 Receptor Parameters

D1 receptors have lower affinity for dopamine:

| Parameter | Value | Source |
|-----------|-------|--------|
| K_d (D1) | 1-10 μM | Richfield et al. (1989) |
| Hill coefficient | ~1 | Standard |

This is 100-1000× higher than D2, meaning D1 receptors respond primarily to phasic dopamine bursts (which reach 200+ nM), not tonic levels.

### 6.2 Dual-Receptor Model

If both receptor types contribute to precision, the total effect might be:

$$\gamma(D) = \gamma_{\text{tonic}} \cdot \theta_{D2}(D) + \gamma_{\text{phasic}} \cdot \theta_{D1}(D)$$

where:
- θ_D2(D) = D / (K_d^{D2} + D) with K_d^{D2} ≈ 20 nM
- θ_D1(D) = D / (K_d^{D1} + D) with K_d^{D1} ≈ 2000 nM

At tonic levels (D ≈ 50-100 nM):
- θ_D2 ≈ 0.7-0.8 (substantial occupancy)
- θ_D1 ≈ 0.02-0.05 (minimal occupancy)

This supports the interpretation that D2 receptors encode tonic precision while D1 receptors respond to phasic signals.

**We do not incorporate this extension into the main thesis** as it adds complexity without additional empirical constraints. It is noted here as a direction for future work.

---

## 7. Limitations of the Derivation

### 7.1 What Is Derived vs. Assumed

**Derived from first principles:**
- The Hill equation form (from mass-action kinetics)
- The K_d value being the relevant concentration scale (from equilibrium thermodynamics)

**Assumed (not derived):**
- That D2 occupancy maps to precision (plausible but not proven)
- That the mapping is linear (simplest assumption; could be nonlinear)
- That γ_max is constant (could depend on other factors)

### 7.2 Biological Complications

Real dopamine signaling involves complexities we ignore:

1. **Spatial heterogeneity:** Dopamine concentrations vary across brain regions
2. **Temporal dynamics:** Receptor binding is not instantaneous; there are kinetic delays
3. **Receptor trafficking:** The number of available receptors changes over time
4. **Downstream cascades:** D2 activation triggers intracellular cascades with their own dynamics
5. **Other receptors:** D3, D4, D5 receptors also bind dopamine with varying affinities

The derived transfer function should be understood as a **reduced model** that captures the essential relationship, not a complete biophysical description.

### 7.3 Empirical Testability

The derivation makes specific predictions that could be tested:

1. **K_d correspondence:** The inflection point of the γ(D) curve should occur near the D2 K_d (~20 nM)

2. **D2 antagonist effects:** D2 blockers should right-shift the curve (effectively increasing K_d)

3. **Occupancy-behavior correlation:** PET measures of D2 occupancy should correlate with computational estimates of precision

These predictions distinguish the receptor-kinetic model from arbitrary sigmoid fits.

---

## 8. Summary

The dopamine-precision transfer function:

$$\gamma(D) = \gamma_{\max} \cdot \frac{D^n}{K_d^n + D^n}$$

is **not arbitrary**. Its form derives from equilibrium receptor binding kinetics (Hill equation), and its key parameter K_d is constrained by measured D2 receptor affinity (~20 nM).

What remains unconstrained is:
- γ_max (the scaling parameter, to be fit to behavioral data)
- The assumption that occupancy maps linearly to precision

This derivation resolves the concern that the transfer function was "ad hoc curve-fitting" while being honest about what aspects remain modeling choices.

---

## References

Bergstrom, K., & Bhardwaj, A. (2022). Dopamine dynamics in Parkinson's disease. *Neuropharmacology*, 209, 108981.

Oh, Y., et al. (2018). Tracking tonic dopamine levels in vivo using multiple cyclic square wave voltammetry. *Biosensors and Bioelectronics*, 121, 174-182.

Richfield, E. K., et al. (1989). Anatomical and affinity state comparisons between dopamine D1 and D2 receptors in the rat central nervous system. *Neuroscience*, 30(3), 767-777.

Robinson, D. L., et al. (2002). Detecting subsecond dopamine release with fast-scan cyclic voltammetry in vivo. *Clinical Chemistry*, 49(10), 1763-1773.

Seamans, J. K., & Yang, C. R. (2004). The principal features and mechanisms of dopamine modulation in the prefrontal cortex. *Progress in Neurobiology*, 74(1), 1-58.

Seeman, P., et al. (2006). Dopamine supersensitivity correlates with D2High states, implying many paths to psychosis. *PNAS*, 103(9), 3440-3445.

Servan-Schreiber, D., et al. (1990). A network model of catecholamine effects: Gain, signal-to-noise ratio, and behavior. *Science*, 249(4971), 892-895.

Sulzer, D., et al. (2005). Mechanisms of neurotransmitter release by amphetamines. *Progress in Neurobiology*, 75(6), 406-433.

Surmeier, D. J., et al. (2007). D1 and D2 dopamine-receptor modulation of striatal glutamatergic signaling in striatal medium spiny neurons. *Trends in Neurosciences*, 30(5), 228-235.
