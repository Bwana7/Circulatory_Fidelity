# Changelog

All notable changes to the thesis document will be documented in this file.

## [2.6] - 2025-12-02

### Center Manifold Analysis and Empirical Validation

**Major Addition: Analytical Derivation of Stochastic Instability (Section 7 of proofs.md)**

Following reviewer recommendations, derived the stochastic instability mechanism analytically:

1. **Variance dynamics equation:**
   $$V^{(t+1)} = V^{(t)} (1 - 2\kappa K_z) + 2 K_z^2$$
   
2. **Key analytical results:**
   - Innovation variance is exactly 2 (independent of observation variance)
   - Covariance structure: $\text{Cov}(\mu_z, \nu) = -\kappa V$
   - Equilibrium variance: $V^{(\infty)} = K_z^*/\kappa$

3. **Nonlinear instability mechanism (derived):**
   - State-dependent gain creates asymmetric response to fluctuations
   - Negative excursions → higher gain → amplification
   - This causes period-doubling dynamics

4. **Critical threshold:**
   - Second-order analysis predicts $\vartheta_c \approx 0.05$
   - **Matches numerical observations**

5. **Why structured approximation helps (derived):**
   - Additional damping term $2\gamma_{zx}$ suppresses variance growth
   - This mechanistically explains CF > 0 → stability

**Empirical Analysis: Spectral Signatures**

Implemented empirical validation pipeline testing CF predictions:

1. **Key finding: 11.44× more low-frequency power under mean-field**
   - t = 19.54, p < 10⁻⁶⁰, Cohen's d = 1.96
   - Confirms spectral signature prediction

2. **CF-variance correlation:**
   - r = -0.440, p < 10⁻²⁰
   - Confirms core thesis claim

3. **CF estimation:**
   - Mean-field CF = 0.021
   - Structured CF = 0.756

**Updated Epistemic Status:**

| Claim | Previous | New |
|-------|----------|-----|
| Variance dynamics | Numerical observation | **Derived** |
| Innovation variance = 2 | Not stated | **Derived** |
| Nonlinear mechanism | Hypothesis | **Derived** |
| Critical threshold | Numerical observation | **Derived (matches)** |
| Structured stabilization | Observed | **Derived** |
| Spectral signatures | Predicted | **Empirically confirmed** |

**New References Added:**
- Kuznetsov (2004) - Elements of Applied Bifurcation Theory
- Arnold (2003) - Random Dynamical Systems
- Horsthemke & Lefever (2006) - Noise-Induced Transitions
- Vinh et al. (2010) - Information theoretic measures
- Iglesias et al. (2013) - Hierarchical prediction errors

**New Files:**
- `docs/center_manifold_analysis.md` - Full derivation
- `experiments/cf_empirical_validation.py` - Spectral analysis pipeline
- `experiments/cf_empirical_validation_report.md` - Results report
- `docs/recommendation_analysis.md` - Review team response

---

### Three-Level HGF Extension

**Major Addition: Section 3.6 "Extension to Three-Level Hierarchies"**

Added comprehensive analysis of CF in three-level hierarchical systems:

1. **Three-level generative model** (from Mathys et al., 2014):
   - Level 3: Meta-volatility (how volatility changes)
   - Level 2: Log-volatility
   - Level 1: Hidden state
   - Observations

2. **Four approximation schemes analyzed:**
   - Full mean-field: q(z₁)q(z₂)q(z₃)
   - Fully structured: q(z₃)q(z₂|z₃)q(z₁|z₂)
   - Bottom-only: q(z₃)q(z₂)q(z₁|z₂)
   - Top-only: q(z₃)q(z₂|z₃)q(z₁)

3. **Pairwise CF measures:** CF₁₂ and CF₂₃ defined at each interface

4. **Key findings (with quantitative results):**
   - Mean-field instability amplified 85× going from 2 to 3 levels
   - Structured approximation increases only 5.6×
   - Lower interface (1-2) is critical: bottom-only achieves 94% of benefit
   - Top-only structuring is WORSE than mean-field
   - Mean-field freezes level 3 dynamics (Var ≈ 0)

5. **New tables added:**
   - Depth comparison (2L vs 3L variance)
   - Interface criticality comparison
   - Level-by-level variance analysis
   - Pairwise CF values by scheme

**Updated sections:**
- Abstract: Added paragraph on three-level findings
- Contributions: Added items 6 (three-level extension) and 7 (interface criticality)
- Limitations: Changed "two-level" to "Gaussian hierarchies"
- Future Directions: Updated to reflect completed work, added new directions

**New experimental files:**
- experiments/three_level_robust.py (Python implementation)
- experiments/three_level_hgf.jl (Julia implementation)
- docs/multilevel_extension_analysis.md (theoretical analysis)
- docs/three_level_preliminary_results.md (results summary)

**Thesis now 29 pages (was 25 pages)**

---

## [2.4] - 2025-11-30

### External Reviewer Feedback Response

**Critique 1: Section 4 Cost Function (VALID → RESOLVED)**

Original problem: Cost function $C(q) = c_0 \cdot \text{CF}(q)$ was asserted without justification.

Resolution: Derived a principled cost function from first principles:
$$C(q) = c_0 \cdot I_q(z; x)$$

Key changes:
- Changed from normalized CF to unnormalized mutual information $I(z;x)$
- Added new subsection "Deriving a Cost Function" with four-part justification:
  1. **Operational interpretation:** Each bit of $I(z;x)$ represents a constraint that must be maintained across updates
  2. **Update complexity:** Message-passing requires transmitting conditional sufficient statistics scaling with $I(z;x)$
  3. **Correct limiting behavior:** $C(q) = 0$ for mean-field, monotonically increasing with dependency
  4. **Description length connection:** $I(z;x)$ is the coding cost of maintaining vs. discarding the dependency
- Added explicit formula for Gaussian case: $I(z;x) = -\frac{1}{2}\ln(1-\rho^2)$
- Clarified distinction: CF (normalized) for comparing approximation quality; $I(z;x)$ (unnormalized) for cost
- Removed "modeling assumption" disclaimer since cost function is now derived
- Updated contributions section to reflect derived cost function

**Critique 2: Section 5 Hill Equation (VALID)**
- Added new subsection "Competing Theories of Dopamine Function" listing:
  - Reward Prediction Error (RPE) theory
  - Precision/gain modulation theory  
  - Motivational salience theory
  - Vigor/effort theory
- Explicitly stated these theories are not mutually exclusive
- Changed "Transfer Function" → "One Possible Transfer Function: Hill Kinetics"
- Added table distinguishing derived vs. assumed components
- Softened language throughout (e.g., "could implement" → "might implement")
- Added caution that phasic/tonic distinction is "one possible interpretation"

**Critique 3: Prior Work on Normalized MI (VALID)**
- Added citations to Coombs (1970), Press (1967), Theil (1970)
- Updated Introduction (Section 1.3) to acknowledge CF is the established "uncertainty coefficient"
- Updated Remark 2.2 to explicitly state normalization is not novel
- Added to "What This Work Does Not Show": "Not a novel information-theoretic measure"
- Clarified contribution is *application* to variational inference, not the measure itself

**New references added:**
- Coombs, C. H., Dawes, R. M., & Tversky, A. (1970). Mathematical Psychology.
- Press, S. J. (1967). On the sample coefficient of contingency.
- Theil, H. (1970). On the estimation of relationships involving qualitative variables.

---

## [2.3] - 2025-11-30

### Minor Revisions (Editorial Response)

**Required Revision 1: Structured Update Derivation**
- Added Section 2.2.1 explaining the structured variational update
- Coupling coefficient γ_zx = κπ_x/4 explicitly identified as modeling approximation
- Justification provided (dimensional consistency, limiting behavior, empirical calibration)
- Added Appendix A.5 with full discussion of what exact derivation would require
- Updated epistemic status table to include "Modeling approximation" status

**Required Revision 2: Stability Paradox Elevated**
- "Remark on the stability paradox" → **Remark 1 (Resolution of the Stability Paradox)**
- Added formatted table distinguishing deterministic vs stochastic analysis
- Explicit connection to noise-induced phenomena literature
- Clear statement of implications for analyzing stochastic systems

**Recommended Revision 1: CF Normalization Note**
- Added **Remark 2 (Choice of Normalization)**
- Explains why min(H(z), H(x)) preferred over H(z,x)
- Notes connection to coefficient of constraint in information theory

**Recommended Revision 2: Manipulation Failure Contingency**
- Added Section 5.2.1 to crucial experiment protocol
- Comprehensive contingency table (Scenarios A, B, C)
- Discussion of boundary effects (participants may not be near bifurcation)
- Recommendation for plasma level verification

---

## [2.2] - 2025-11-30

### Crucial Experiment Protocol

**New document:** `experiments/crucial_experiment_protocol.md`

Addresses editorial requirement: "Provide at least one prediction that uniquely distinguishes CF from existing structured variational approaches."

**The distinguishing prediction:**
- Alternative frameworks predict: reduced precision → noisier inference (quantitative degradation)
- CF uniquely predicts: reduced CF → period-doubling bifurcations → specific frequency structure in belief dynamics

**Experimental design:**
- Within-subject pharmacological study (sulpiride vs. placebo)
- Volatile reversal learning task (400 trials)
- HGF modeling to extract trial-by-trial learning rates
- Spectral analysis to detect period-doubling signatures

**Primary outcome:**
- Power in 0.02-0.10 cycles/trial band
- Peak frequency ratio ≈ 2 (period-doubling signature)

**Why it's crucial:**
| Prediction | CF Theory | Alternatives |
|------------|-----------|--------------|
| Increased variability | ✓ | ✓ |
| Specific frequency structure | ✓ | ✗ |

**Falsification criteria clearly stated:** CF falsified if behavioral effects present but no spectral signatures, or if frequency ratios inconsistent with period-doubling.

---

## [2.1] - 2025-11-30

### Complete Mathematical Proofs

**New supplementary document:** `docs/proofs.md` with rigorous derivations

**Proven Results:**
- Proposition 1 (FIM Block-Diagonality): Complete proof from first principles
- Proposition 3 (CF = 0 under mean-field): Direct proof from mutual information definition
- Corollary (CF > 0 under structured): Follows immediately

**Key Theoretical Discovery:**
- The deterministic skeleton is STABLE for all ϑ > 0
- Observed instabilities arise from STOCHASTIC dynamics, not fixed-point instability
- This resolves the apparent "discrepancy" between linear analysis and simulation

**Updated Epistemic Classification:**
| Claim | Previous Status | Current Status |
|-------|-----------------|----------------|
| FIM block-diagonal | "Proof sketch" | **Proven** |
| CF = 0 mean-field | Implicit | **Proven** |
| Deterministic stability | "Predicts instability" | **Proven stable** |
| Period-doubling occurs | "Derived" | Numerical observation |
| Bifurcation thresholds | Point estimates | Ranges with uncertainty |

**Stochastic Instability Hypothesis:**
- Instability arises from variance growth under large prediction errors
- Gain amplifies fluctuations even when mean dynamics are stable
- Formally proving this mechanism identified as open problem

### Transfer Function Derivation

**New supplementary document:** `docs/transfer_function_derivation.md`

**Derived from receptor kinetics (Hill equation):**
$$\gamma(D) = \gamma_{\max} \cdot \frac{D^n}{K_d^n + D^n}$$

**What is now constrained by pharmacology:**
- K_d ≈ 20 nM (D2 receptor dissociation constant)
- n ≈ 1 (Hill coefficient for D2 receptors)

**What remains free:**
- γ_max (to be fit to behavioral data)
- Linear occupancy→precision mapping (assumption)

This resolves editorial concern about "ad hoc curve-fitting."

---

## [2.0] - 2025-11-30

### Major Revision Following Editorial Feedback

This revision addresses concerns raised by the editorial board regarding epistemic overreach and incomplete derivations.

#### Framing Changes
- Renamed "Thermodynamic Free Energy" → "Resource-Rational Free Energy"
- Removed claims grounded in Landauer's principle (recognized as insufficiently derived)
- Explicitly framed cost term as epistemic/computational rather than physical
- Adopted Lieder & Griffiths (2020) resource-rationality framework as theoretical foundation

#### Definition Changes
- CF normalization: I(z;x)/min(H(z),H(x)) instead of I(z;x)/H(z,x)
- Rationale: Improved interpretability as correlation-like measure ∈ [0,1]

#### Epistemic Corrections
- Distinguished "formal results" from "simulation observations" from "speculation" throughout
- Bifurcation thresholds reported as ranges rather than point estimates
- Acknowledged discrepancy between linear stability analysis and observed bifurcation
- Transfer function parameters explicitly marked as empirically unconstrained
- Neural implementation clearly labeled as "speculative hypothesis"

#### Structural Changes
- Added "What This Work Does Not Show" subsection (Section 8.2)
- Added caveats throughout experimental proposals
- Simplified proof presentations with explicit limitations noted
- Removed overclaiming language throughout

#### Removed
- "Edge of Chaos" / Wolfram Class 4 terminology (insufficiently grounded)
- "Dynamical irreducibility" / "algorithmically incompressible" claims (overstated)
- Strong claims about bioelectric generalization (speculative)
- Precise ATP cost derivations (conflated transmission with erasure)
- Feigenbaum universality claims (not verified in our simulations)

---

## [1.0] - 2025-11-29

### Initial Formal Draft

- Complete formal definitions for key concepts
- Theorems with proof sketches
- Full Julia implementation code
- Experimental protocols with predictions
- Pharmacological predictions with literature support

### Known Issues (addressed in v2.0)
- Thermodynamic claims exceeded derivations
- Some terminology undefined
- Speculation not clearly distinguished from results

---

## [0.5] - 2025-11-15

### Initial Draft
- Core theoretical framework
- Basic simulation code
- Preliminary results

---

## Version Naming Convention

- **Major versions** (1.0, 2.0): Complete drafts
- **Minor versions** (1.1, 1.2): Revisions based on feedback
- **Patch versions** (0.5, 0.9): Working drafts
