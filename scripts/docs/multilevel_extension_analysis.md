# Extending Circulatory Fidelity to Deeper Hierarchies

## Exploratory Analysis

---

## 1. The Three-Level Generative Model

### 1.1 Structure

The natural extension of the two-level HGF is the three-level HGF, which Mathys et al. (2014) actually describe as the "standard" case:

```
Level 3 (meta-volatility):     z₃ₜ | z₃ₜ₋₁ ~ N(z₃ₜ₋₁, ϑ₃⁻¹)
Level 2 (volatility):          z₂ₜ | z₂ₜ₋₁, z₃ₜ ~ N(z₂ₜ₋₁, exp(κ₃z₃ₜ + ω₃))
Level 1 (hidden state):        z₁ₜ | z₁ₜ₋₁, z₂ₜ ~ N(z₁ₜ₋₁, exp(κ₂z₂ₜ + ω₂))
Observations:                  yₜ | z₁ₜ ~ N(z₁ₜ, πᵤ⁻¹)
```

This captures a richer structure:
- Level 1: The hidden state we're tracking
- Level 2: How volatile level 1 is (log-volatility)
- Level 3: How volatile the volatility itself is (meta-volatility)

### 1.2 Interpretation

In learning contexts:
- Level 1: Current value of the tracked quantity
- Level 2: How quickly the environment is changing
- Level 3: How stable the rate of change is (e.g., distinguishing "noisy but stable" from "undergoing regime change")

---

## 2. Approximation Schemes for Three Levels

### 2.1 Full Mean-Field

```
q(z₁, z₂, z₃) = q(z₁)q(z₂)q(z₃)
```

**Properties:**
- All levels update independently
- No cross-level information preserved
- Simplest computationally
- **Prediction:** Should exhibit instabilities similar to (or worse than) two-level case

### 2.2 Fully Structured (Markov)

```
q(z₁, z₂, z₃) = q(z₃)q(z₂|z₃)q(z₁|z₂)
```

**Properties:**
- Respects the generative model's conditional independence structure
- z₁ ⊥ z₃ | z₂ (conditionally independent given the middle level)
- Most expensive computationally
- **Prediction:** Should be most stable

### 2.3 Fully Structured (Non-Markov)

```
q(z₁, z₂, z₃) = q(z₃)q(z₂|z₃)q(z₁|z₂, z₃)
```

**Properties:**
- Even richer dependency structure
- z₁ depends directly on z₃ as well as z₂
- May be appropriate if the true posterior has such dependencies
- Most expensive

### 2.4 Partially Structured Options

**Bottom-structured only:**
```
q(z₁, z₂, z₃) = q(z₃)q(z₂)q(z₁|z₂)
```
Maintains dependency between levels 1-2, but 2-3 are independent.

**Top-structured only:**
```
q(z₁, z₂, z₃) = q(z₃)q(z₂|z₃)q(z₁)
```
Maintains dependency between levels 2-3, but 1-2 are independent.

**These partial structures allow us to test which cross-level dependencies are most important for stability.**

---

## 3. Extending CF to Three Levels

### 3.1 Option A: Pairwise CFs

Track CF at each interface:

```
CF₁₂ = I(z₁; z₂) / min(H(z₁), H(z₂))
CF₂₃ = I(z₂; z₃) / min(H(z₂), H(z₃))
```

**Advantages:**
- Direct extension of two-level analysis
- Can identify which interface is critical for stability
- Easy to interpret

**Disadvantages:**
- Misses higher-order dependencies (e.g., I(z₁; z₃) that isn't mediated by z₂)
- Two numbers instead of one

### 3.2 Option B: Total Correlation

The total correlation (multi-information) generalizes mutual information:

```
TC(z₁, z₂, z₃) = H(z₁) + H(z₂) + H(z₃) - H(z₁, z₂, z₃)
              = I(z₁; z₂) + I(z₁,z₂; z₃)
              = sum of all redundancies
```

Normalized version:
```
CF_total = TC(z₁, z₂, z₃) / min(H(z₁), H(z₂), H(z₃))
```

**Advantages:**
- Single number capturing all dependency
- Well-studied in information theory
- Zero if and only if all variables are mutually independent

**Disadvantages:**
- Doesn't distinguish which pairs are dependent
- May be dominated by one strong pairwise relationship

### 3.3 Option C: Hierarchical Decomposition

Decompose the total correlation by level:

```
TC = I(z₂; z₃) + I(z₁; z₂, z₃)
   = [upper interface] + [lower interface given upper]
```

This respects the hierarchical structure and allows us to track:
- How much information flows between levels 2-3
- How much additional information flows to level 1

### 3.4 Recommendation

For the thesis extension, I recommend **Option A (pairwise CFs)** for these reasons:
1. Direct continuity with two-level analysis
2. Allows testing which interface matters for stability
3. Simpler to compute and interpret
4. Can always compute TC post-hoc if needed

---

## 4. Fisher Information Matrix Structure

### 4.1 For Three Levels Under Mean-Field

The FIM becomes 3×3 block-diagonal:

```
G_MF = | G₃₃  0    0   |
       | 0    G₂₂  0   |
       | 0    0    G₁₁ |
```

Each level's parameters are geometrically isolated.

### 4.2 For Three Levels Under Markov-Structured

```
G_struct = | G₃₃  G₃₂  0   |
           | G₂₃  G₂₂  G₂₁ |
           | 0    G₁₂  G₁₁ |
```

The FIM is tri-diagonal (in blocks), reflecting the Markov structure where only adjacent levels are directly coupled.

### 4.3 Key Observation

Under mean-field, the FIM has **three isolated blocks** instead of two. This means:
- No coordinated updates between ANY pair of levels
- Potentially worse instability than two-level case
- The "damping" effect of structured approximation is lost at multiple interfaces

---

## 5. Stability Analysis: Theoretical Predictions

### 5.1 Cascade Hypothesis

**Hypothesis:** In a three-level mean-field system, instabilities may cascade:
- Level 3 volatility estimate oscillates
- This drives oscillations in level 2's precision estimate
- Which in turn destabilizes level 1

The cascade could either:
- **Amplify:** Each level adds to the instability
- **Attenuate:** Higher levels are more stable, buffering lower levels

### 5.2 Interface Criticality Hypothesis

**Hypothesis:** The stability of the system depends primarily on the "weakest" interface—the one with lowest CF.

If CF₂₃ is high but CF₁₂ is low (or vice versa), the system may still be unstable.

### 5.3 Bifurcation Structure

For three levels, we expect:
- More parameters → higher-dimensional bifurcation structure
- Possibly new types of bifurcations (Neimark-Sacker, etc.)
- Potentially chaotic dynamics at lower ϑ values

---

## 6. Computational Feasibility

### 6.1 Update Equations

The three-level HGF updates are well-documented (Mathys et al., 2014). For each level i ∈ {1, 2, 3}:

```
μᵢ(t) = μᵢ(t-1) + Kᵢ · δᵢ(t)
```

where Kᵢ is the Kalman-like gain and δᵢ is the prediction error.

The key additions for level 3:
- Level 3 gain: K₃ = (κ₃/2) · π̂₂ / (ϑ₃ + (κ₃²/2) · π̂₂)
- Level 3 PE: δ₃ = (δ₂)² · π̂₂ - 1

### 6.2 Simulation Cost

Three-level simulations are approximately:
- 1.5× the computational cost of two-level (one more level to update)
- Parameter space increases from 4D to ~7D (κ₂, κ₃, ω₂, ω₃, ϑ₂, ϑ₃, πᵤ)

This is tractable but requires more systematic parameter sweeps.

### 6.3 Lyapunov Exponent Computation

The Jacobian for the three-level system is larger but still computable:
- 6×6 for means only (μ₁, μ₂, μ₃ plus their perturbations)
- 12×12 if including variances

The Benettin algorithm extends straightforwardly.

---

## 7. Implementation Plan

### Phase 1: Three-Level HGF Implementation

```julia
struct ThreeLevelHGF
    κ₂::Float64  # Level 2-1 coupling
    κ₃::Float64  # Level 3-2 coupling
    ω₂::Float64  # Level 2 baseline
    ω₃::Float64  # Level 3 baseline
    ϑ₃::Float64  # Level 3 volatility (meta-meta-volatility)
    πᵤ::Float64  # Observation precision
end
```

### Phase 2: Approximation Variants

Implement:
1. Full mean-field
2. Full structured (Markov)
3. Bottom-structured only
4. Top-structured only

### Phase 3: Stability Analysis

For each approximation:
1. Bifurcation diagrams (varying ϑ₃, with ϑ₂ implicit in dynamics)
2. Lyapunov exponents
3. Pairwise CF tracking (CF₁₂, CF₂₃)

### Phase 4: Comparison

Key questions:
1. Does mean-field become unstable at lower ϑ compared to two-level?
2. Which partial structuring provides most stability benefit?
3. Is there a "critical interface"?

---

## 8. Expected Results and Their Implications

### 8.1 If instabilities worsen with depth

This would support the claim that CF becomes MORE important in deeper hierarchies—exactly where biological inference operates.

**Implication:** The stability-preserving role of cross-level dependencies is amplified in realistic (deep) hierarchies.

### 8.2 If one interface dominates

E.g., if CF₂₃ matters more than CF₁₂ for stability.

**Implication:** Neuromodulatory systems might target specific hierarchical interfaces preferentially.

### 8.3 If new dynamical phenomena emerge

E.g., quasi-periodic behavior, intermittency, different routes to chaos.

**Implication:** Richer phenomenology that might map onto different psychiatric presentations.

---

## 9. Challenges and Limitations

### 9.1 Parameter Space Explosion

With 7 parameters instead of 4, systematic exploration becomes harder. May need:
- Latin hypercube sampling
- Focus on physiologically relevant parameter regimes
- Dimensional reduction (fix some parameters at standard values)

### 9.2 Analytical Intractability

The two-level case was already at the edge of analytical tractability. Three levels will likely be entirely numerical.

### 9.3 Biological Mapping

The dopamine-precision hypothesis becomes more complex:
- Does dopamine modulate all levels equally?
- Are different neuromodulators responsible for different levels?
- How do we test this experimentally?

---

## 10. Viability Assessment

### 10.1 Is this extension feasible?

**YES**, with caveats:

| Aspect | Feasibility | Notes |
|--------|-------------|-------|
| Theory | High | CF extends naturally; multiple options available |
| Computation | High | Standard HGF code exists; moderate additional cost |
| Analysis | Medium | Larger parameter space; less analytical insight |
| Interpretation | Medium | More complex; harder to map to biology |

### 10.2 Recommended Scope

For thesis inclusion, I recommend:

**Minimum viable extension:**
1. Implement three-level HGF with mean-field and Markov-structured approximations
2. Compare stability (Lyapunov exponents) for fixed "standard" parameters
3. Track CF₁₂ and CF₂₃ separately
4. Report whether instabilities worsen with depth

**Full extension (if time permits):**
1. Add partial structuring variants
2. Systematic parameter sweeps
3. Bifurcation analysis
4. New experimental predictions

### 10.3 Timeline Estimate

- Phase 1 (implementation): 1-2 days
- Phase 2 (basic analysis): 2-3 days  
- Phase 3 (full analysis): 1-2 weeks

---

## 11. Conclusion

Extending CF to three or more levels is **viable and valuable**. The extension:

1. **Addresses a stated limitation** of the current thesis
2. **Tests the robustness** of our findings
3. **Generates new predictions** about hierarchical depth
4. **Moves closer** to realistic biological hierarchies

The main question is scope: a minimal extension demonstrating that the core findings hold (or strengthen) in deeper hierarchies would significantly strengthen the thesis. A full exploration of the richer dynamical landscape would be a substantial research contribution in its own right.

**Recommendation:** Proceed with the minimal viable extension, structured to allow expansion if initial results are promising.
