# Phase 2 E3 Primary-Source Prior-Art Verification (`WP-2.1`)

**Date:** 2026-11-23  
**Work Package:** `WP-2.1` (resolves Condition G1-1 from `research_plan.md`)  
**Scope:** Primary-source, theorem-level (E3) inspection of OPRA, UnO, CPC, CPL, CPS, LTT and four additional neighboring results, plus DOI verification of Ben-Michael et al. (2025) and Kallus (2023).  
**Verdict on Condition G1-1 (Primary-Source Non-Collision):** **PASS (NO DIRECT HIT FOUND).** None of the six inspected primary sources contains localized comparative tail-risk certification (canceling the shared abstained region $S_\lambda^c$ via a Rockafellar--Uryasev baseline-VaR confidence-interval pivot or localized signed CDF contrasts).

---

## 1. Summary Table of E3 Primary-Source Audits

| Paper | Verified Identifier | Evidence Level | Exact Estimand | Key Theorems / Equations Inspected | $S_\lambda^c$ Cancellation? | Rockafellar--Uryasev Baseline-VaR Pivot? | Collision Verdict vs. LCT-LTT |
|---|---|---|---|---|---|---|---|
| **OPRA** (Huang, Leqi, Lipton, Azizzadenesheli, NeurIPS 2021) | `arXiv:2104.08977v2` | **E3** (full PDF inspected) | Single-policy Lipschitz risk functionals $\rho_p(F^\pi)$ (including $\text{CVaR}_\alpha$, mean, variance, CPT) | Lemma 4.1, Remark 4.1, Thm. 5.1 (Eqs. 3--4), Thm. 5.2 (Eq. 8), Thm. 6.1, Lemma D.4 | **No** (single-policy CDF $\hat F^\pi$ only) | **No** (Rockafellar & Uryasev 2000 cited once in refs `[55]`) | **Incumbent B1** (no direct hit; lacks comparative localization) |
| **UnO** (Chandak, Niekum, da Silva, Learned-Miller, Brunskill, Thomas, NeurIPS 2021) | `arXiv:2104.12820v2` | **E3** (full PDF inspected) | Single-policy CDF $F^\pi$ and functionals $\psi(F^\pi)$ (mean, variance, quantiles, $\text{CVaR}_\alpha$) | Thm. 1, Thm. 2 (Eq. 5), Thm. 3, Thm. 4, Fig. 3 | **No** (single-policy CDF envelope $[F^-, F^+]$ only) | **No** (maps CDF envelopes geometrically via Thomas & Learned-Miller 2019) | **Incumbent B1** (no direct hit; lacks comparative localization) |
| **CPC** (Prinster, Fannjiang, Park, Cho, Liu, Saria, Stanton, ICML 2026) | `arXiv:2603.02196v4` | **E3** (full PDF inspected) | Expected bounded loss $\mathbb{E}_{\pi_{0:t}^{(\hat\beta)}}[L_t] \le \alpha + O(\epsilon^{(1)})$ relative to safe reference $\pi_0$ ($\mathbb{E}_{\pi_0}[L_0] \le \alpha$) | Thm. 4.2 (gCRC), Thm. 4.6 (CPC), Remark 4.7--4.8, Thm. C.3, Prop. C.5, Def. C.6, Thm. C.10 | **No** (likelihood-ratio policy regulation for bounded expectation) | **No** (no CVaR or quantile tail functionals) | **Adjacent B5** (no direct hit; strictly expected bounded loss) |
| **CPL** (Jin & Egami, 2026) | `arXiv:2609.17296` | **E3** (full PDF inspected) | Individual counterfactual harm probability $\mathbb{P}(Y_{n+1}(\hat\pi(X_{n+1})) < Y_{n+1}(0)) \le \alpha$ | Eq. (1.1), Thm. 3.1, Thm. 3.3--3.4, Thm. 4.1 (Eq. 4.3), Thm. 5.1 | **No** (unit-level conformal $p$-value thresholding) | **No** (binary individual harm, not population CVaR/$Q_\beta$) | **Adjacent** (no direct hit; different estimand) |
| **CPS** (Cao, Gao, Saar-Tsechansky, Yin, 2026) | `arXiv:2610.22930v1` *(corrected from `2610.11547`, which is OX-NeRF)* | **E3** (full PDF inspected) | False Discovery Rate (FDR) among selected deployment units, with false discoveries $Y_{n+j} \le c_j$ | Thm. 1 (finite-sample FDR), Thm. 2 (residual model), Thm. 3 (estimated nuisances), Lemma 1 | **No** (weighted conformal $p$-values + selection/pruning) | **No** (unit-level threshold exceedance FDR) | **Adjacent** (no direct hit; different estimand) |
| **LTT** (Angelopoulos, Bates, Candès, Jordan, Lei, AOAS 2025) | `arXiv:2110.01052v5`, DOI `10.1214/24-AOAS1998` | **E3** (full PDF inspected) | Risk-controlling parameter set $\hat\Lambda$ satisfying $\mathbb{P}(\sup_{\lambda \in \hat\Lambda} R(\lambda) \le \alpha) \ge 1 - \delta$ | Thm. 1, Prop. 1--4 (Fixed Sequence Testing), Prop. 6 (IUT max $p$-value), Prop. 7 | **No** (general MHT wrapper) | **No** | **Enabling Engine** (supplies Fixed-Sequence Testing & Prop. 6 IUT) |

---

## 2. Detailed Primary-Source Audit Notes

### 2.1 OPRA — Huang et al. (NeurIPS 2021, `arXiv:2104.08977v2`)
- **Full text verified (E3):** 5,380 extracted lines from `arXiv:2104.08977v2`.
- **Exact mechanism & theorems:**
  - **Lemma 4.1 & Remark 4.1:** For random variables supported on $[0, D]$, any distorted risk functional with $(L/D)$-Lipschitz distortion $g$ is $L$-Lipschitz in the CDF sup-norm: $|\rho(F_Z) - \rho(F_{Z'})| \le L \|F_Z - F_{Z'}\|_\infty$. Specifically, $\mathbb{E}[Z]$ is $D$-Lipschitz and $\text{CVaR}_\alpha$ is $(D/\alpha)$-Lipschitz.
  - **Theorem 5.1 (Eqs. 3--4):** For the clipped IS CDF estimator $\hat F_{\text{IS-clip}}(t) = \min\{\frac{1}{n}\sum_{i=1}^n w_i \mathbf{1}\{R_i \le t\}, 1\}$ with maximum weight $w_{\max}$ and second moment $w_2 = \mathbb{E}[w^2]$,
    $$\mathbb{P}\!\left(\|\hat F_{\text{IS-clip}} - F\|_\infty \le \varepsilon_{\text{IS1}} := \sqrt{\frac{8 w_{\max}^2}{n}\log(4/\delta)}\right) \ge 1 - \delta,$$
    and the Bernstein-style tightening (Eq. 4):
    $$\mathbb{P}\!\left(\|\hat F_{\text{IS-clip}} - F\|_\infty \le \varepsilon_{\text{IS2}} := \frac{4 w_{\max}\log(4/\delta)}{n} + 2\sqrt{\frac{2 w_2 \log(4/\delta)}{n}}\right) \ge 1 - \delta.$$
  - **Theorem 5.2 (Eq. 8):** Monotonized DR CDF estimator satisfies $\|\hat F_{\text{M-DR}} - F\|_\infty \le \sqrt{\frac{72 w_{\max}^2}{n}\log(8n/\delta)}$ with probability $\ge 1-\delta$.
  - **Theorem 6.1:** Plugging any CDF estimator with $\|\hat F - F\|_\infty \le \varepsilon$ into $L_p$-Lipschitz risks $\{\rho_p\}_{p=1}^P$ gives simultaneous error bounds $|\rho_p(\hat F) - \rho_p(F)| \le L_p \varepsilon$.
- **Gap confirmed (E3):** OPRA bounds $\rho(F^\pi)$ for a *single* target policy $\pi$ via uniform CDF concentration on the entire covariate space $\mathcal{X}$. Comparing a gated policy $\pi_\lambda$ against $\pi_0$ via OPRA requires bounding $\rho(F^{\pi_\lambda})$ and $\rho(F^{\pi_0})$ separately, paying full $O(w_{\max} D / (\alpha\sqrt{n}))$ width even when $\mathbb{P}(S_\lambda) \ll 1$. Neither $S_\lambda^c$ cancellation nor Rockafellar--Uryasev baseline-VaR pivoting appears anywhere in the paper.

### 2.2 UnO — Chandak et al. (NeurIPS 2021, `arXiv:2104.12820v2`)
- **Full text verified (E3):** 1,884 extracted lines from `arXiv:2104.12820v2`.
- **Exact mechanism & theorems:**
  - **Theorem 1:** $\hat F_n(\nu) = \frac{1}{n}\sum_{i=1}^n \rho_i \mathbf{1}\{G_i \le \nu\}$ is unbiased and uniformly consistent for $F^\pi(\nu)$.
  - **Theorem 2 (Eq. 5):** Selects $K$ key points $(\kappa_i)_{i=1}^K$ with per-point confidence intervals $[\text{CI}_-(\kappa_i, \delta_i), \text{CI}_+(\kappa_i, \delta_i)]$ ($\sum_{i=1}^K \delta_i \le \delta$) via Thomas et al. (2015) IS mean bounds, then constructs monotone step envelopes $F_-(\nu) = \max_{\kappa_i \le \nu} \text{CI}_-(\kappa_i, \delta_i)$ and $F_+(\nu) = \min_{\kappa_i \ge \nu} \text{CI}_+(\kappa_i, \delta_i)$ guaranteeing $\mathbb{P}(\forall \nu, F_-(\nu) \le F^\pi(\nu) \le F_+(\nu)) \ge 1-\delta$.
  - **Theorem 3 & Fig. 3:** Maps the band $[F_-, F_+]$ geometrically (via Thomas & Learned-Miller, 2019) to $[\psi_-, \psi_+]$ for mean, quantiles, and $\alpha$-CVaR.
  - **Theorem 4:** Extends the band under discrete distribution shift $\sup_\nu |F^{(1)}_\pi(\nu) - F^{(2)}_\pi(\nu)| \le \epsilon$.
- **Gap confirmed (E3):** Like OPRA, UnO constructs single-policy CDF confidence bands over the full covariate distribution. It does not form localized signed policy-contrast weights $\Delta w_\lambda$ or cancel $S_\lambda^c$.

### 2.3 CPC — Prinster et al. (ICML 2026, `arXiv:2603.02196v4`)
- **Full text verified (E3):** 7,057 extracted lines from `arXiv:2603.02196v4`.
- **Exact mechanism & theorems:**
  - **Theorem 4.2 (gCRC):** For exchangeable bounded non-monotone losses $L_i(\lambda) \in [0, B]$ that are $K$-Lipschitz in $\lambda$ with $\epsilon$-replace-one-stable $\hat\lambda^+$, $\mathbb{E}[L_{n+1}(\hat\lambda^+)] \le \alpha + K\epsilon$.
  - **Theorem 4.6 (CPC):** Given safe reference policy $\pi_0$ with $\mathbb{E}_{\pi_0}[L_0] \le \alpha$ and bounded losses $L_i \in [0, B]$, regulates exploration via likelihood-ratio bound $\hat\beta$ (which enforces $1$-log-Lipschitzness in $\log \beta$, Lemma C.12) to guarantee $\mathbb{E}_{\pi_{0:t}^{(\hat\beta)}}[L_t] \le \alpha + \max(\alpha, B-\alpha)\epsilon^{(1)}$.
- **Gap confirmed (E3):** CPC controls *expected* bounded loss $\mathbb{E}[L_t] \le \alpha$ (including fixed-threshold indicators $\mathbf{1}\{L > c\}$). It does not certify nonlinear tail functionals ($\text{CVaR}_\beta$, unbounded $Q_\beta$) or localize comparative tail contrasts.

### 2.4 CPL — Jin & Egami (2026, `arXiv:2609.17296`)
- **Full text verified (E3):** Inspected full PDF text.
- **Exact mechanism & theorems:**
  - **Estimand (Eq. 2.2):** $\mathbb{P}(Y_{n+1}(\hat\pi(X_{n+1})) < Y_{n+1}(0)) \le \alpha$ (individual counterfactual harm rate).
  - **Theorem 3.1 & Theorem 4.1:** Exact finite-sample harm-rate control under RCTs ($e(x)=1/2$ and general known $e(x)$) via selective-calibration conformal $p$-values. The general known-propensity formula is Eq. 4.2, and Eq. 4.3 specializes it for the optimal selection rule.
  - **Theorem 5.1:** Observational model double robustness via learn-then-balance weights (asymptotic $\limsup \mathbb{P}(\text{harm}) \le \alpha$ if either weights or outcome models are $L_2$-consistent).
- **Gap confirmed (E3):** Targets binary individual counterfactual harm, not population tail functionals ($\text{CVaR}_\beta$, $Q_\beta$) or comparative $S_\lambda^c$ tail cancellation.

### 2.5 CPS — Cao, Gao, Saar-Tsechansky, & Yin (2026, `arXiv:2610.22930v1`)
- **Identifier correction (important audit catch):** The identifier `arXiv:2610.11547` listed in earlier notes belongs to an unrelated computer-vision paper (*OX-NeRF: 3D X-ray Tomography Reconstruction*, Welsch & Tu, Oct 2026). The verified arXiv identifier for Cao et al. (2026) *Conformal Policy Selection: When to Deploy AI Decisions for Targeted Intervention* is **`arXiv:2610.22930v1`** (`stat.ME`, 16 Oct 2026).
- **Full text verified (E3):** Inspected full PDF text of `arXiv:2610.22930v1`.
- **Exact mechanism & theorems:**
  - **Theorem 1:** Finite-sample $\text{FDR} \le q$ among selected deployment units for false discoveries $Y_{n+j} \le c_{n+j}$ under known joint covariate-treatment shift $w(x, t) = dQ/dP(x,t)$, using weighted conformal $p$-values and a two-step selection-and-pruning procedure.
  - **Theorem 2 & Theorem 3:** Asymptotic $\limsup \text{FDR} \le q$ under additive-noise outcome model (Theorem 2) or uniformly consistent nuisance estimates $\hat e_N, \hat f_N$ (Theorem 3).
- **Gap confirmed (E3):** Targets unit-level threshold exceedance FDR, not population tail functionals ($\text{CVaR}_\beta$, $Q_\beta$) or comparative baseline-VaR pivoting.

### 2.6 LTT — Angelopoulos et al. (AOAS 2025, `arXiv:2110.01052v5`, DOI `10.1214/24-AOAS1998`)
- **Full text verified (E3):** 2,925 extracted lines from `arXiv:2110.01052v5`.
- **Exact mechanism & theorems:**
  - **Theorem 1:** Any FWER-controlling multiple-testing algorithm $\mathcal{A}$ applied to super-uniform $p$-values for $H_j: R(\lambda_j) > \alpha$ yields $\mathbb{P}(\sup_{\lambda \in \hat\Lambda} R(\lambda) \le \alpha) \ge 1-\delta$.
  - **Propositions 3 & 4 (Fixed-Sequence Testing):** Algorithm 1 and Proposition 3 establish the FWER bound for fixed-sequence testing. Proposition 4 prints an equality with the probability that the first true-null p-value crosses its threshold. For a sequence that stops at the first failure, the false-rejection event also requires reaching that index. We use the overall FWER upper bound from Proposition 3.
  - **Proposition 6 (Intersection-Union Test):** For composite nulls $H_j = \bigcup_{\ell=1}^m H_{j,\ell}$, $p_j = \max_\ell p_{j,\ell}$ is a valid $p$-value with **zero Bonferroni penalty** across $\ell = 1, \dots, m$.
- **Role in LCT-LTT:** Provides our exact Fixed-Sequence Testing and Intersection-Union multi-risk certification engine; contains no off-policy localized tail-risk bounds.

---


## 3. Additional neighboring primary results

**Jeunen (2026), Accelerating A/B-Tests with Counterfactual Estimation.** Sections 2–3 and Eq. (12) use the signed importance weight $(\pi(a\mid x)-\pi'(a\mid x))/\pi_{\rm log}(a\mid x)$. It is exactly the weight used here when both policies are deterministic. Theorem 3.1 proves a mean-outcome variance comparison that exploits shared actions. This establishes prior use of the shared-region cancellation itself. Its headline result targets the mean of a fixed outcome and leaves open inference at an unknown baseline-policy quantile and comparative nonlinear tail certification. [Primary text](https://arxiv.org/html/2607.14604v1).

**Bourigault and Bourigault (2026), Coverage Cliffs in Off-Policy Quantile Certification.** Section 1 and Theorem 3 use the inequality $P_{\pi}(Y>\tau)\le 1-p$ for quantile certification. Appendix C, Theorem 2 gives power and converse bounds based on tail second moments and shared actions. Appendix D.5 applies betting to weighted nonnegative exceedances, and Eq. (10) gives their selection-valid CDF envelope. This is close prior art for finite-sample off-policy quantile certification, rare-tail weighting, and abstaining if insufficient evidence. The comparison is to a predeclared numerical budget, not the unknown quantile of a reference policy. The displayed statistics use individual policy weights, not the signed contrast at a baseline-policy pivot. Its fixed-budget survival methods offer comparators for future work if the comparative project advances. [Primary text](https://arxiv.org/html/2610.22565v1).

**Huang et al. (2022), Off-Policy Risk Assessment for Markov Decision Processes.** Section 4.3.1 and Lemmas 4.3–4.4 already use the complementary CDF to reduce variance in single-policy distribution estimation, with finite-sample bands and an empirical-Bernstein point-grid variant. The survival-complement transformation itself is prior art. Their combination in Section 4.4 concerns CDF and survival estimators of the same policy. [Proceedings paper](https://proceedings.mlr.press/v151/huang22b/huang22b.pdf).

**Guo et al. (2026), High Probability Risk Control for Online Policy Learning.** Section 3.2 and Theorem 4.5 use a Gaussian multiplier bootstrap for simultaneous high-probability risk control over online policies. The theorem is asymptotic and addresses weighted bounded mean loss. It does not give the finite-sample comparative nonlinear-tail result considered here. [Primary text](https://arxiv.org/html/2611.13163v1).

**G1-1 primary-source decision.** The inspected results leave a potential contribution in the combined method: a certificate for nonlinear tail risk relative to an unknown reference-policy quantile, with a localized policy contrast and joint testing of a fixed policy sequence. Any substantive advantage still requires the non-vacuity comparison. Neither cancellation, survival complementation, finite-sample off-policy quantile certification against a fixed budget, nor fixed-sequence testing is a new ingredient.

## 4. Verified Secondary DOIs (Reference Hygiene)

1. **Ben-Michael, E., Greiner, D. J., Imai, K., & Jiang, Z. (2025).** Safe Policy Learning through Extrapolation: Application to Pre-trial Risk Assessment. *Journal of the American Statistical Association*. Verified DOI: [`10.1080/01621459.2025.2489135`](https://doi.org/10.1080/01621459.2025.2489135).
2. **Kallus, N. (2023).** Treatment Effect Risk: Bounds and Inference. *Management Science*, 69(8), 4579--4590. Verified DOI: [`10.1287/mnsc.2023.4819`](https://doi.org/10.1287/mnsc.2023.4819).
3. **Cao, J., Gao, R., Saar-Tsechansky, M., & Yin, M. (2026).** Conformal Policy Selection: When to Deploy AI Decisions for Targeted Intervention. Corrected identifier: `arXiv:2610.22930` (replaces erroneous `2610.11547`).
