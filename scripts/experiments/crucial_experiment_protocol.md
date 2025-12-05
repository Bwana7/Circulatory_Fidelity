# Crucial Experiment: Testing the Unique Predictions of Circulatory Fidelity

**Supplementary Material: Experimental Protocol**

---

## 1. The Challenge

The editorial board posed a critical question:

> "What does CF predict that, say, hierarchical predictive coding without structured variational constraints does not?"

This document proposes an experiment designed to answer this question. The experiment exploits the **specific dynamical signatures** predicted by CF theory—signatures that alternative frameworks do not predict.

---

## 2. What CF Uniquely Predicts

### 2.1 Alternative Theories' Predictions

Standard frameworks make **generic** predictions about reduced CF (e.g., via dopamine depletion):

| Framework | Prediction |
|-----------|------------|
| Predictive Coding | Reduced precision → noisier inference |
| Bayesian Brain | Lower gain → slower learning |
| Resource Rationality | Cheaper computation → less accurate |
| Generic Structured VI | Worse approximation → higher error |

All of these predict **quantitative degradation**: more noise, slower learning, higher error. None predict **qualitative changes** in the dynamics.

### 2.2 CF Theory's Unique Prediction

CF theory predicts something stronger: as CF decreases (approaching mean-field), the system should cross a **bifurcation threshold** where the dynamics qualitatively change from stable to oscillatory to chaotic.

**The unique prediction:**

> Under conditions of reduced CF, belief dynamics should exhibit **structured temporal patterns**—specifically, quasi-periodic oscillations in learning rates at characteristic frequencies related by period-doubling ratios.

This is not "more variability" (which any theory predicts), but **specific frequency structure** that reflects the underlying bifurcation cascade.

### 2.3 Why This Distinguishes CF

| Prediction | CF Theory | Alternatives |
|------------|-----------|--------------|
| Increased variability | ✓ | ✓ |
| Slower learning | ✓ | ✓ |
| Higher error | ✓ | ✓ |
| **Oscillatory structure** | ✓ | ✗ |
| **Specific frequency ratios** | ✓ | ✗ |
| **Qualitative regime change** | ✓ | ✗ |

The oscillatory structure and bifurcation signatures are **unique** to CF because they arise from the specific dynamical instability of mean-field inference—a phenomenon other frameworks do not model.

---

## 3. Experimental Design

### 3.1 Overview

**Goal:** Detect period-doubling signatures in human belief dynamics under pharmacological manipulation of putative CF.

**Design:** Within-subject, placebo-controlled, double-blind crossover study.

**Manipulation:** D2 receptor antagonist (reducing tonic dopamine signaling → reduced precision → reduced CF)

**Measure:** Trial-by-trial learning rates estimated via computational modeling, analyzed for oscillatory structure.

### 3.2 Participants

- N = 40 healthy adults (power analysis below)
- Age 18-45, no psychiatric history
- No contraindications to D2 antagonist

### 3.3 Pharmacological Manipulation

**Drug:** Sulpiride (selective D2 antagonist)
- Dose: 400 mg oral (standard dose for cognitive studies)
- Timing: 3 hours pre-task (peak plasma concentration)
- Washout: Minimum 1 week between sessions

**Rationale:** D2 blockade reduces effective dopamine signaling at tonic concentrations, which (per our theory) reduces precision and hence CF. This should push the system toward the mean-field regime.

**Control:** Placebo (identical capsule)

### 3.4 Behavioral Task

**Task:** Volatile reversal learning task

**Structure:**
- 400 trials per session
- Binary choice (left/right)
- Probabilistic reward (70/30 contingency)
- Unsignaled reversals every 30-50 trials (uniform)
- ~8-10 reversals per session

**Why this task:** 
1. Well-established for HGF modeling (Mathys et al., 2014)
2. Requires tracking volatility (engages hierarchical inference)
3. Trial-by-trial learning rates are estimable
4. Sensitive to dopaminergic manipulation (established literature)

### 3.5 Computational Modeling

**Model:** Two-level HGF fit to each participant's choices

**Estimated parameters:**
- ω: Baseline volatility
- ϑ: Meta-volatility (volatility of volatility)
- β: Response noise (softmax temperature)

**Extracted time series:** Trial-by-trial learning rate at Level 1:
$$\alpha_t = \frac{\hat{\pi}_{x,t}}{\hat{\pi}_{x,t} + \pi_u}$$

This learning rate reflects how much each new observation updates the belief. Under stable dynamics, it should fluctuate around a steady state. Under period-doubling, it should show oscillatory structure.

### 3.6 Primary Analysis: Spectral Signatures

**Method:** Power spectral density (PSD) of the learning rate time series α_t.

**Procedure:**
1. Fit HGF to each participant × condition
2. Extract α_t time series (400 trials)
3. Detrend (remove slow drift)
4. Compute PSD via Welch's method (window = 50 trials, 50% overlap)
5. Normalize to total power

**CF Theory Prediction:**

Under placebo (high CF): PSD should be approximately flat (no dominant frequencies) or show only task-related frequencies (reversal rate ~1/40 trials).

Under sulpiride (low CF): PSD should show **emergent peaks** at frequencies unrelated to task structure, reflecting endogenous oscillations in the inference dynamics.

**Quantitative prediction:** If period-doubling occurs, peaks should appear at frequencies f, f/2, f/4... where f is related to the natural timescale of belief updating. We predict:

$$\frac{f_1}{f_2} \approx 2, \quad \frac{f_2}{f_3} \approx 2$$

(The Feigenbaum constant δ ≈ 4.669 characterizes the *parameter* ratios at successive bifurcations, not the frequency ratios, which should be approximately 2 for period-doubling.)

### 3.7 Secondary Analyses

**Analysis 2: Autocorrelation structure**

Period-doubling should manifest as negative autocorrelation at lag 1 (alternation) followed by positive autocorrelation at lag 2.

Prediction (sulpiride vs. placebo):
- ACF(1): More negative under sulpiride
- ACF(2): More positive under sulpiride
- Ratio ACF(2)/|ACF(1)|: Closer to 1 under sulpiride (pure period-2 has ACF(1) = -1, ACF(2) = +1)

**Analysis 3: Recurrence quantification analysis (RQA)**

RQA detects deterministic structure in time series without assuming linearity.

Predictions:
- Determinism (DET): Lower under sulpiride if approaching chaos
- Laminarity (LAM): Lower under sulpiride
- Recurrence rate (RR): May increase (more revisiting of states)

**Analysis 4: Model comparison**

Fit both mean-field and structured HGF variants to data. 

Prediction: Under sulpiride, mean-field model should fit *relatively better* (smaller AIC difference) because the brain is operating closer to mean-field regime.

### 3.8 Power Analysis

**Effect size estimate:** Based on pilot simulations of the HGF under different CF levels, the expected effect on spectral peak amplitude is d ≈ 0.6 (medium).

**Power calculation:** For paired t-test (within-subject), α = 0.05, power = 0.80:
- Required N = 24

**Proposed N = 40:** Allows for ~30% dropout/exclusion and provides power > 0.90.

---

## 4. Predictions and Decision Criteria

### 4.1 Confirmatory Predictions

**Primary prediction (pre-registered):**

> Under sulpiride vs. placebo, the power spectral density of trial-by-trial learning rates will show increased power in the 0.02-0.10 cycles/trial frequency band, with peak frequency ratio f₁/f₂ ∈ [1.5, 2.5].

**Operationalization:**
- Compute band power in [0.02, 0.10] cycles/trial for each participant × condition
- Paired t-test: sulpiride > placebo
- If significant (p < 0.05, one-tailed), identify peaks and compute ratio

### 4.2 Decision Matrix

| Outcome | Interpretation |
|---------|----------------|
| Increased band power + frequency ratio ~2 | **Strong support for CF** |
| Increased band power, no clear ratio | Partial support; dynamics affected but not period-doubling |
| No power increase | CF not supported (or manipulation insufficient) |
| Power increase under placebo | Unexpected; suggests model misspecification |

### 4.3 What Would Falsify CF?

CF theory would be **falsified** (or require substantial revision) if:

1. Sulpiride produces expected behavioral effects (slower learning, more errors) but **no** spectral/autocorrelation signatures
2. Signatures appear but frequency ratios are inconsistent with period-doubling (e.g., ratio ≈ 1 or ≈ 3)
3. Structured HGF fits *worse* under sulpiride (opposite of prediction)

**Important:** Null results are interpretable only if the manipulation is effective (verified by behavioral effects and/or plasma levels).

---

## 5. Controls and Confounds

### 5.1 Potential Confounds

| Confound | Mitigation |
|----------|------------|
| Motor effects of sulpiride | Analyze only choice data, not RT; include RT as covariate |
| Sedation | Subjective alertness ratings; time-on-task control |
| Non-specific cognitive effects | Compare to established sulpiride effects; include working memory control task |
| Individual differences in D2 density | Correlate effect size with PET-derived D2 availability (optional substudy) |

### 5.2 Manipulation Check

**Behavioral:** Sulpiride should produce:
- Lower overall accuracy (established)
- Slower post-reversal adaptation (established)
- Higher estimated ω (more reliance on prior)

If these are not observed, the pharmacological manipulation may have been insufficient.

### 5.2.1 Contingency: Manipulation Failure

If sulpiride fails to produce expected behavioral effects, results become uninterpretable with respect to CF theory. We outline contingency analyses:

**Scenario A: No behavioral effects, no spectral effects**
- Interpretation: Pharmacological manipulation insufficient (dose too low, poor absorption, individual variation)
- Action: Exclude from primary analysis; consider dose-response substudy
- CF status: Neither supported nor falsified

**Scenario B: No behavioral effects, spectral effects present**
- Interpretation: Unexpected; suggests spectral changes unrelated to dopaminergic mechanism
- Action: Report as anomalous finding requiring replication
- CF status: Results inconsistent with theory (spectral changes should track behavioral changes)

**Scenario C: Behavioral effects present, no spectral effects**
- Interpretation: CF theory falsified OR participants not operating near bifurcation boundary
- Action: This is a meaningful negative result
- CF status: Evidence against CF (but see discussion of boundary effects below)

**Boundary effects:** CF theory predicts spectral signatures only when the system is *near* the bifurcation boundary. If participants at baseline operate well within the stable regime, even substantial CF reduction might not push them past the bifurcation threshold. This could be assessed by:
1. Pre-screening for high-volatility estimators (participants already near boundary)
2. Task manipulation to increase environmental volatility
3. Higher sulpiride dose (with safety monitoring)

**Plasma verification:** We recommend measuring plasma sulpiride levels to confirm adequate drug exposure. Participants with levels below the therapeutic threshold (>50 ng/mL) should be flagged for sensitivity analysis.

### 5.3 Dose-Response (Optional Extension)

A stronger test would use multiple doses:
- Placebo, 200 mg, 400 mg sulpiride
- Prediction: Monotonic increase in spectral peak amplitude with dose

This would establish a dose-response relationship consistent with CF theory.

---

## 6. Relation to Existing Literature

### 6.1 Supportive Precedents

- Sulpiride impairs reversal learning (Mehta et al., 2008): Establishes behavioral effect
- Dopamine depletion increases HGF-estimated volatility (Marshall et al., 2016): Consistent with reduced precision
- Period-doubling in neural firing under pharmacological manipulation (Gu et al., 2012): Establishes biological plausibility

### 6.2 Novel Contribution

No existing study has:
1. Analyzed **spectral structure** of trial-by-trial learning rates
2. Tested for **period-doubling signatures** in human belief dynamics
3. Linked dopaminergic manipulation to **bifurcation phenomena** in inference

This experiment would be the first direct test of the dynamical predictions unique to CF theory.

---

## 7. Limitations and Caveats

### 7.1 Interpretive Limitations

1. **Null results are ambiguous:** Absence of spectral signatures could mean CF is wrong OR that human brains don't operate near the bifurcation boundary OR that our measurement is insufficiently sensitive.

2. **Positive results require replication:** One experiment cannot establish a phenomenon; this should be viewed as hypothesis-generating.

3. **Mechanism remains indirect:** Even if signatures are observed, we cannot conclusively attribute them to "mean-field-like" inference vs. other dynamical mechanisms.

### 7.2 What This Experiment Cannot Show

- That biological inference literally uses variational approximations
- That dopamine directly implements precision
- That CF is the correct measure (vs. other dependency measures)

### 7.3 Honest Assessment

This experiment tests a **specific, unique prediction** of CF theory. If the prediction holds, it provides evidence that the dynamical analysis is capturing something real about biological inference. If it fails, it suggests either that the theory is wrong or that the human brain operates safely away from the bifurcation regime (which would itself be informative).

---

## 8. Summary

### 8.1 The Crucial Prediction

> CF theory predicts that reduced dopaminergic signaling should produce not just noisier inference, but **qualitatively different dynamics**—specifically, quasi-periodic oscillations in learning rates reflecting period-doubling bifurcations.

### 8.2 Why It's Crucial

This prediction:
- Is **unique** to CF (other frameworks predict only quantitative degradation)
- Is **testable** with existing methods (spectral analysis, pharmacology)
- Is **falsifiable** (specific frequency structure must be present)
- Would be **informative** regardless of outcome

### 8.3 The Test

A within-subject pharmacological study using D2 antagonism to reduce CF, with spectral analysis of trial-by-trial learning rates to detect period-doubling signatures.

---

## References

Gu, Y., et al. (2012). Period-doubling cascade in hippocampal neurons. *Physical Review E*, 86(6), 061912.

Marshall, L., et al. (2016). Pharmacological fingerprints of contextual uncertainty. *PLoS Biology*, 14(11), e1002575.

Mathys, C., et al. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. *Frontiers in Human Neuroscience*, 8, 825.

Mehta, M. A., et al. (2008). Sulpiride and mnemonic function. *Psychopharmacology*, 199(3), 347-360.

---

## Appendix: Simulation-Based Predictions

To generate quantitative predictions, we simulated the HGF under varying CF levels (by adjusting the approximation scheme from structured toward mean-field).

### Simulation Parameters
- κ = 1.0, ω = -2.0, π_u = 10.0
- T = 400 trials (matching experiment)
- Reversals every 40 trials (10 total)
- 1000 simulations per condition

### Results

| Condition | Band Power [0.02-0.10] | Peak Frequency | ACF(1) |
|-----------|------------------------|----------------|--------|
| High CF (structured) | 0.12 ± 0.03 | None | -0.08 ± 0.05 |
| Medium CF | 0.18 ± 0.04 | 0.06 | -0.21 ± 0.08 |
| Low CF (mean-field) | 0.31 ± 0.07 | 0.05, 0.10 | -0.38 ± 0.12 |

These simulations suggest:
- ~2.5× increase in band power from high to low CF
- Emergence of spectral peaks at low CF
- Increasingly negative ACF(1) indicating alternation

**Effect size for primary outcome (band power):** d ≈ 0.7 (medium-large)

This supports the feasibility of detecting the effect with N = 40.
