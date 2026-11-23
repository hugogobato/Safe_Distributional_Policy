# Research Idea Dossier: Safe Distributional Policy Learning

**Working Title:** Safe Distributional Policy Learning: Conformalized Finite-Sample Tail-Risk (CVaR) Guarantees for Semicontinuous and Distributional Outcomes  
**Origin:** Idea 7 from `/home/hugo/Stuff/Research/Research_Ideas.md`  
**Internal Stack Referenced:** `Wasserstein_Causal_Forests` (`WCF`), `zicbcf` (`ZIC-BCF-Smear`), `CP_TATE` (`tcda_uq`)  
**Target Venues (Original Seed):** ICML, NeurIPS, JASA, Management Science  
**Dossier Status:** Initial formalization & critical risk audit (pre-Phase 0 gate)

---

## 1. Executive Summary

Standard individualized treatment rule (ITR) and contextual-bandit policy learners optimize expected outcomes (mean welfare). In healthcare expenditure management, e-commerce spend/returns, and enterprise AI service-level agreements (SLAs), outcomes are heavily right-skewed, zero-inflated (semicontinuous), or observed as unit-level distributions, and decision-makers face asymmetric downside constraints: a policy that improves average welfare is unacceptable if it worsens extreme tail risk (e.g., 95th-percentile latency, upper-tail Conditional Value-at-Risk [CVaR] of medical costs, or extensive-margin zero-conversion probability).

Recent distribution-free safe policy frameworks—notably **Conformal Policy Learning** (Jin & Egami, 2026), **Conformal Policy Control** (Prinster et al., ICML 2026), and **Conformal Policy Selection** (Cao et al., 2026)—enforce finite-sample safety guarantees, but only for **linear expectations of bounded per-observation losses** or **individual binary harm indicators** on scalar outcomes. Conversely, risk-sensitive ITRs targeting CVaR or quantiles (Wang et al., 2018; Qi et al., 2023) rely on asymptotic plug-in estimation without finite-sample safety certification or an abstention/fallback gate.

This dossier formalizes the proposed framework: layering a finite-sample, propensity-adjusted tail-risk certification and abstention/fallback gate on top of nonparametric distributional causal learners (`ZIC-BCF-Smear`, `WCF`). Critically, this dossier exposes two major structural distinctions that are conflated in the original seed and must be separated cleanly before any implementation or theory begins:

1. **Regime A vs. Regime B (Unit of Analysis):** Scalar semicontinuous outcomes $Y_i \in \{0\} \cup (0,\infty)$ per unit $i$ (where CVaR and upper quantiles are *population-level* or *conditional-distribution* nonlinear functionals) versus unit-level distributional outcomes $\mu_i$ per unit $i$ (where $\text{CVaR}(\mu_i)$ and $Q_\tau(\mu_i)$ are *observable unit-level scalar functionals* of $\mu_i$).
2. **Conformal Risk Control (CRC) vs. Nonlinear Tail-Risk Certification:** Standard weighted CRC (Angelopoulos et al., 2024; Zecchin et al., 2025) certifies expectations of *bounded* per-unit monotone losses $\mathbb{E}[L(Y_{\text{test}}, \pi_\lambda(X_{\text{test}}))] \le \alpha$. Population CVaR is neither linear in a single observation nor finite-sample upper-boundable without a support/tail restriction (Bahadur & Savage, 1956). Valid finite-sample certification in Regime A therefore requires either (i) **localized comparative Rockafellar–Uryasev / weighted CDF concentration** coupled with **Learn-then-Test (LTT)** (Angelopoulos et al., 2025) or (ii) **finite-sample quantile/distortion risk control** (Snell et al., 2023; Chen et al., 2025) adapted to off-policy weights—and must distinguish **population-marginal safety of a gated policy** from **post-selection conditional safety** and **pointwise covariate-conditional safety**.

---

## 2. Scientific Problem and Why It Matters

Decision-makers deploying personalized policies from randomized experiments (A/B tests, policy lotteries) or observational logs face a recurrent failure mode: **mean-tail divergence**.

1. **Zero-inflated healthcare expenditures (extensive vs. intensive margin conflict):** Expanding insurance coverage or care-management interventions (e.g., Oregon Health Insurance Experiment, MEPS dental/medical expenditure) often increases utilization on the extensive margin ($\mathbb{P}(Y > 0)$) while reducing catastrophic right-tail out-of-pocket spending or shifting positive expenditures heterogeneously across age, chronic-condition, and income strata. A mean-optimal policy can inadvertently assign an intervention to subgroups where catastrophic upper-tail expenditure ($\text{CVaR}_{0.90}$ or $Q_{0.95}$) worsens.
2. **E-commerce spend, returns, and conversion:** Promotional or routing policies may lift average gross spend by increasing high-basket purchases while simultaneously raising zero-conversion rates or tail return losses.
3. **Enterprise AI and cloud SLAs:** Routing prompts or workloads across heterogeneous models/backends (e.g., small vs. frontier LLMs, cached vs. agentic workflows) often lowers *average* latency or token cost while violating p95/p99 tail-latency SLAs on complex query strata.

In all three settings, practitioners already operate a **trusted status-quo / fallback policy** $\pi_0$ (e.g., standard care, uniform routing, control arm). They do not want an unconstrained black-box ITR that maximizes plug-in welfare everywhere; they want a **gated personalized policy** $\pi_\lambda$ that deploys a learned personalized action $\hat\pi(x)$ only where counterfactual downside safety is statistically certified, and falls back to $\pi_0(x)$ everywhere else.

---

## 3. Unit of Analysis and Observable Data Regimes (Critical Disambiguation)

The original seed combines "semicontinuous/zero-inflated" outcomes with "unit-level distributions." Mathematically, these define **two fundamentally different observation regimes** with different estimands, identification structures, and collision profiles against prior art. They must NOT be conflated.

| Dimension | Regime A: Scalar Semicontinuous Outcomes | Regime B: Unit-Level Distributional Outcomes |
|---|---|---|
| **Unit of analysis $i$** | Individual person, patient, user, or request ($i = 1, \dots, n$) | Cluster, classroom, hospital, village, or microservice ($i = 1, \dots, n$) |
| **Observed data** | $(X_i, A_i, Y_i)$ with $A_i \in \{0, 1\}$, $Y_i \in \{0\} \cup (0, \infty)$ (or capped $[0, B]$) | $(X_i, A_i, \hat\mu_i)$ (or quantile curve $\hat Q_i \in \mathcal{Q}_K$ estimated from $m_i$ sub-observations) |
| **Internal learner fit** | `zicbcf` (`ZIC-BCF-Smear`: Probit hurdle + log-Gaussian active stage + Duan smearing) or two-part distributional/quantile forests | `Wasserstein_Causal_Forests` (`DRCalibratedCWDB` + `ZIPTAdapter` on $K$-point quantile grids) + `CP_TATE` functional conformal machinery |
| **Where CVaR / $Q_\beta$ lives** | **Population-level** functional $\rho(P_{Y(\pi)})$ (or conditional functional $\rho(P_{Y(\pi)\mid X=x})$); NEVER observed for a single unit $i$ | **Unit-level** observable scalar functional $R_i = \rho(\mu_i)$ (up to finite-$m_i$ subsampling error in $\hat\mu_i$), OR full curve $Q_i \in \mathcal{Q}_K$ |
| **Core mathematical obstacle** | Nonlinear population functional $\text{CVaR}_\beta(P_{Y(\pi)})$ is NOT an expectation of a single-unit loss; unbounded right tail triggers Bahadur–Savage impossibility unless capped at $B$ or shifted to quantiles $Q_\beta$ | Single-functional reduction $R_i = \rho(\mu_i)$ **collapses to scalar conformal policy learning** (vacuity risk!) unless certifying *simultaneous tail-curve dominance* and/or correcting finite-$m_i$ tail bias |
| **Primary public data** | **OHIE** (randomized Medicaid lottery; known $e(x)$ for ITT), **MEPS 2023 HC-251** (observational dental/medical spend; locally present in `zicbcf/applied_study/h251.csv`) | **Project STAR** (RCT classrooms), **Egger et al. Kenya** (RCT villages), **CPS MORG** (state-year wage distributions)—all already wired into `WCF`; public randomized latency/SLA A/B datasets are `UNVERIFIED` / unavailable |

### Sign Convention
To avoid ambiguity across cost/latency (lower is better) and reward/spend (higher is better), define $L_i(a) \in \mathbb{R}$ as a **loss** (e.g., medical expenditure, tail latency, negative net profit, or zero-conversion indicator $\mathbf{1}\{Y_i(a)=0\}$) where **smaller is safer**, and let $U_i(a)$ denote the primary **utility/welfare** outcome (which may equal $-L_i(a)$ or be a separate reward metric, e.g., accuracy vs. latency SLA).

---

## 4. Mathematical Estimands and Finite-Sample Safety Guarantees

Let $\pi_0 : \mathcal{X} \to \{0,1\}$ be a known safe baseline policy (commonly $\pi_0(x) \equiv 0$, the status quo) and let $\hat\pi : \mathcal{X} \to \{0,1\}$ be a candidate personalized policy trained on a split $\mathcal{D}_{\text{tr}}$ independent of the calibration split $\mathcal{D}_{\text{cal}}$ of size $n_{\text{cal}}$. Let $s : \mathcal{X} \to \mathbb{R}$ be a trained safety/benefit score function (larger $s(x)$ indicates stronger predicted tail-safety margin for deploying $\hat\pi(x)$). Parameterize a nested family of **gated (abstaining/fallback) policies** by $\lambda \in \Lambda \subset \mathbb{R} \cup \{+\infty\}$:
$$
\pi_\lambda(x) = \begin{cases} \hat\pi(x) & \text{if } s(x) > \lambda \text{ (deploy personalized action; } x \in S_\lambda\text{)}, \\ \pi_0(x) & \text{if } s(x) \le \lambda \text{ (abstain / fall back to } \pi_0\text{; } x \notin S_\lambda\text{)}. \end{cases}
$$
At $\lambda = +\infty$, $S_\infty = \emptyset$ and $\pi_\infty \equiv \pi_0$ (trivially safe). Note that the deployment region $S_\lambda = \{x \in \mathcal{X} : s(x) > \lambda\}$ is monotone decreasing in $\lambda$ ($S_{\lambda_2} \subseteq S_{\lambda_1}$ for $\lambda_2 > \lambda_1$), though the population risk $\rho(P_{L(\pi_\lambda)})$ is generally **not** guaranteed to be monotone in $\lambda$ (because a misspecified score $s(x)$ can rank unsafe units ahead of safe units).

### 4.1 Distributional Risk Functionals

For a scalar loss random variable $L$ with CDF $F_L(t) = \mathbb{P}(L \le t)$:

1. **Upper Quantile (Value-at-Risk) at level $\beta \in (0,1)$:**
   $$
   \text{VaR}_\beta(L) = Q_\beta(L) = \inf\{t \in \mathbb{R} : F_L(t) \ge \beta\}.
   $$
2. **Upper-Tail Conditional Value-at-Risk (Expected Shortfall) at level $\beta \in (0,1)$:**
   By Rockafellar & Uryasev (2000),
   $$
   \text{CVaR}_\beta(L) = \inf_{\eta \in \mathbb{R}} \left\{ \eta + \frac{1}{1-\beta}\mathbb{E}\left[(L - \eta)_+\right] \right\},
   $$
   with the infimum attained at any $\eta^* \in [Q_\beta^-(L), Q_\beta^+(L)]$.
3. **Extensive-Margin / Threshold-Exceedance Risk:**
   For semicontinuous outcomes $Y \in \{0\} \cup (0,\infty)$, the zero-conversion risk is $R_{\text{ext}}(\pi) = \mathbb{P}(Y(\pi) = 0)$ and fixed-SLA exceedance risk is $R_c(\pi) = \mathbb{P}(L(\pi) > c)$. Both are bounded linear expectations of binary indicators $\in \{0,1\}$.

### 4.2 Four Distinct Safety Guarantees (Explicitly Disambiguated)

A rigorous formulation must state *which* guarantee is targeted, because they differ sharply in achievability:

1. **G-Marg-RegimeA (Population-Marginal Comparative Tail Safety of the Gated Policy — PAC / High-Probability over $\mathcal{D}_{\text{cal}}$):**
   Choose $\hat\lambda = \hat\lambda(\mathcal{D}_{\text{cal}})$ such that, for a vector of risk functionals $\rho_1, \dots, \rho_J$ (e.g., $\text{CVaR}_{0.90}$, $Q_{0.95}$, and extensive-margin zero probability) and tolerances $\varepsilon_1, \dots, \varepsilon_J \ge 0$,
   $$
   \mathbb{P}_{\mathcal{D}_{\text{cal}}}\!\left( \forall j \in \{1,\dots,J\},\; \rho_j\!\left(P_{L(\pi_{\hat\lambda})}\right) \le \rho_j\!\left(P_{L(\pi_0)}\right) + \varepsilon_j \right) \ge 1 - \delta.
   $$
   *Achievability:* **Strictly achievable in finite samples** under known propensities (RCT/A/B test) via Learn-then-Test / multiple-testing-calibrated finite-sample off-policy bounds (with no boundedness assumption for $Q_\beta$ and $R_{\text{ext}}$; requiring a known upper bound $L \le B$ or tail assumption for $\text{CVaR}_\beta$).
2. **G-Subgroup (Finite-Partition Subgroup-Conditional Tail Safety):**
   For a prespecified finite partition $\mathcal{X} = \bigsqcup_{g=1}^G \mathcal{X}_g$ (e.g., age $\times$ comorbidity strata or SLA service tiers), certify simultaneously across $g = 1, \dots, G$:
   $$
   \mathbb{P}_{\mathcal{D}_{\text{cal}}}\!\left( \forall g \in \{1,\dots,G\},\; \rho\!\left(P_{L(\pi_{\hat\lambda}) \mid X \in \mathcal{X}_g}\right) \le \rho\!\left(P_{L(\pi_0) \mid X \in \mathcal{X}_g}\right) + \varepsilon_g \right) \ge 1 - \delta.
   $$
   *Achievability:* Achievable in finite samples by applying G-Marg-RegimeA per stratum with multiplicity adjustment across $G$ strata (provided sufficient per-stratum $n_{\text{cal}, g}$).
3. **G-PostSel (Post-Selection Safety Conditional on Deployed Region $S_{\hat\lambda}$):**
   Certify $\rho(P_{L(\hat\pi) \mid X \in S_{\hat\lambda}}) \le \rho(P_{L(\pi_0) \mid X \in S_{\hat\lambda}}) + \varepsilon$.
   *Achievability:* Harder than G-Marg-RegimeA because $S_{\hat\lambda}$ is data-dependent and denominator $\mathbb{P}(X \in S_{\hat\lambda})$ can be small; requires selective calibration (akin to Jin & Egami, 2026; Cao et al., 2026) or uniform bounds over $\lambda$ with minimum deployment-mass floor $\mathbb{P}(X \in S_\lambda) \ge p_{\min}$.
4. **G-Pointwise (Pointwise Covariate-Conditional Tail Safety for Every $x \in S_{\hat\lambda}$):**
   $\forall x \in S_{\hat\lambda},\; \rho(P_{L(\hat\pi) \mid X = x}) \le \rho(P_{L(\pi_0) \mid X = x})$.
   *Achievability:* **Provably impossible** in finite samples without distributional/smoothness assumptions for continuous covariates $X$ (analogous to Barber et al., 2021 for conditional conformal coverage). The original seed's phrase *"deploys only on covariate regions where counterfactual tail safety is certified"* must NOT be overclaimed as G-Pointwise.
5. **G-RegimeB (Unit-Level Distributional Safety — Regime B):**
   When each unit $i$ carries a distribution $\mu_i$ (quantile curve $Q_i \in \mathcal{Q}_K$), one may target marginal expectation of a unit-level bounded risk loss $\mathbb{E}[h(\mu(\pi_{\hat\lambda}), \mu(\pi_0))] \le \alpha$ (which collapses to scalar CPC/W-CRC unless $h$ certifies **simultaneous tail-curve dominance** $\sup_{\tau \in [\tau_0, 1]} (Q_i(\pi_{\hat\lambda})(\tau) - c(\tau))_+$, incorporating finite-$m_i$ within-unit subsampling error).

### 4.3 Propensity-Weighted Shift vs. Policy-Induced Covariate Shift

Under observational or randomized logging with behavior propensity $e(a \mid x) = \mathbb{P}(A = a \mid X = x)$, evaluating a target policy $\pi$ induces the importance-weight (likelihood-ratio) Radon–Nikodym derivative:
$$
w_\pi(x, a) = \frac{\mathbf{1}\{a = \pi(x)\}}{e(a \mid x)}, \qquad \mathbb{E}\left[w_\pi(X, A) f(X, Y)\right] = \mathbb{E}\left[f(X, Y(\pi(X)))\right].
$$
Crucially:
- **Propensity-weighted counterfactual shift** maps observed arm-$a$ data $(X_i, Y_i) \mid A_i = a$ to the interventional distribution $(X, Y(a))$ via weights $1/e(a\mid X)$.
- **Policy-induced shift** compares the logging joint law $P(X, A, Y)$ to the policy joint law $P_\pi(X, A, Y) = P(X)\mathbf{1}\{A = \pi(X)\}P(Y \mid X, A)$ via $w_\pi(X, A)$.
- For a **gated policy** $\pi_\lambda$, the signed policy-contrast weight
  $$
  \Delta w_\lambda(X, A) = w_{\pi_\lambda}(X, A) - w_{\pi_0}(X, A) = \mathbf{1}\{X \in S_\lambda\} \left( \frac{\mathbf{1}\{A = \hat\pi(X)\}}{e(A \mid X)} - \frac{\mathbf{1}\{A = \pi_0(X)\}}{e(A \mid X)} \right)
  $$
  is **identically zero outside the deployed region $S_\lambda$**. This localization is the key variance-reduction and tightness mechanism for comparative tail-risk certification.

---

## 5. Proposed Method

### 5.1 Original Seed Conception (Preserved As Written)
The original seed proposes:
1. Fitting a nonparametric distributional causal learner (`ZIC-BCF-Smear` for semicontinuous outcomes or `WCF` for unit-level distributions) on training data to estimate counterfactual distributions and risk contrasts.
2. Constructing an abstention/fallback gate $\pi_\lambda$ controlled by a threshold $\lambda$ that defaults to $\pi_0$ outside certified covariate regions.
3. Applying "weighted conformal risk control over monotone distributional risk functionals (CVaR, upper quantiles, extensive-margin probability)" to calibrate $\lambda$ with finite-sample $\ge 1-\alpha$ tail safety under propensity-weighted shift.

### 5.2 Mathematically Valid Mechanism (Required Repair for Regime A)
Because naive weighted CRC fails on nonlinear, non-monotone-in-$\lambda$ population CVaR/quantiles (see Section 10), a valid finite-sample certification engine for **Regime A** replaces single-point CRC with **Localized Comparative Off-Policy Tail-Risk Learn-then-Test (LCT-LTT)**:

1. **Stage 1 — Distributional Policy & Safety-Score Learning (on $\mathcal{D}_{\text{tr}}$):**
   Fit a two-part semicontinuous distributional learner (e.g., `ZIC-BCF-Smear` or cross-fitted two-part distributional quantile/GBDT learner) to estimate:
   - conditional extensive-margin probabilities $\hat p_a(x) = \widehat{\mathbb{P}}(Y(a) > 0 \mid X = x)$,
   - conditional positive-part tail distributions $\hat F_{a,+}(y \mid x)$ and local tail-excess functions $\hat m_a(x; \hat\eta_0) = \widehat{\mathbb{E}}[(L(a) - \hat\eta_0)_+ \mid X = x]$,
   - candidate personalized policy $\hat\pi(x)$ (maximizing expected welfare subject to plug-in tail-safety preference, or minimizing plug-in CVaR), and
   - scalar safety-ranking score $s(x)$ ordering covariate points from safest/most-beneficial deployment to riskiest.
2. **Stage 2 — Localized Comparative Tail-Risk Certification (on $\mathcal{D}_{\text{cal}}$):**
   Over a finite candidate grid $\Lambda = \{\lambda_1 > \lambda_2 > \dots > \lambda_M\}$ (from most conservative $\lambda_1 = +\infty$ to most aggressive), test the null hypothesis of tail-risk violation $H_0(\lambda): \rho(P_{L(\pi_\lambda)}) - \rho(P_{L(\pi_0)}) > \varepsilon$ using exact finite-sample valid $p$-values (or upper confidence bounds) that **cancel the shared abstained region $S_\lambda^c$**:
   - **For CVaR (under known support cap $L \in [0, B]$):** First compute a distribution-free $(1 - \delta_1)$ finite-sample confidence interval $[\underline\eta_0, \overline\eta_0]$ for the baseline Value-at-Risk $\eta_0^* = Q_\beta(L(\pi_0))$ using weighted CDF concentration (requiring no bound $B$). Then, exploiting Rockafellar–Uryasev optimality of $\eta_0^*$ for $\pi_0$ and suboptimality of $\eta_0^*$ for $\pi_\lambda$,
     $$
     \text{CVaR}_\beta(L(\pi_\lambda)) - \text{CVaR}_\beta(L(\pi_0)) \le \sup_{\eta \in [\underline\eta_0, \overline\eta_0]} \frac{1}{1-\beta} \mathbb{E}\left[ \Delta w_\lambda(X, A) (L - \eta)_+ \right].
     $$
     Because $\Delta w_\lambda(X, A) = 0$ on $X \notin S_\lambda$, the empirical variance of the influence-localized excess term $\Delta w_\lambda(X_i, A_i)(L_i - \eta)_+$ scales with $\mathbb{P}(X \in S_\lambda)$ rather than $1$, enabling tight empirical-Bernstein / Hoeffding / betting-martingale (Waudby-Smith & Ramdas, 2024) finite-sample upper confidence bounds even when $S_\lambda$ covers a modest fraction of the population.
   - **For Upper Quantiles $Q_\beta$ (no upper bound $B$ needed!):** Certify $Q_\beta(L(\pi_\lambda)) \le Q_\beta(L(\pi_0)) + \varepsilon$ by verifying $F_{L(\pi_\lambda)}(\eta_0^* + \varepsilon) \ge \beta$ across $\eta \in [\underline\eta_0, \overline\eta_0]$ via the localized CDF contrast $\mathbb{E}[\Delta w_\lambda(X, A) \mathbf{1}\{L \le \eta + \varepsilon\}]$, whose boundedness in $[-1/\kappa, 1/\kappa]$ requires **zero** upper bound on $L$.
   - **For Extensive-Margin / Fixed-SLA Risk:** Directly bound $\mathbb{E}[\Delta w_\lambda(X, A) \mathbf{1}\{Y = 0\}]$ (bounded, localized to $S_\lambda$).
   - **Stage 3 — Sequential / Fixed-Sequence LTT Gate Selection:** Test $\lambda_1, \lambda_2, \dots, \lambda_M$ in decreasing order (most conservative to most aggressive) via fixed-sequence testing (which controls family-wise error rate at $\delta$ with **zero** Bonferroni loss over $M$ until the first failure to reject $H_0(\lambda_k)$!) and output $\hat\lambda = \lambda_{k^*-1}$ (or fall back to $\pi_0$ if even $\lambda_2$ fails).

*(Remark: If extended to Regime B with unit-level distributions $\mu_i$, Stage 2 instead uses weighted functional conformal bands / simultaneous tail-curve certification from `CP_TATE` + `WCF`, with finite-$m_i$ within-unit subsampling inflation.)*

---

## 6. Explicit Assumptions

1. **A1 (SUTVA / Consistency & No Interference):** $Y_i = Y_i(A_i)$; unit $i$'s outcome depends only on its own assigned action.
2. **A2 (Unconfoundedness / Ignorability):** $(Y_i(0), Y_i(1)) \perp\!\!\!\perp A_i \mid X_i$.
3. **A3 (Strict Overlap / Positivity):** There exists known $\kappa > 0$ such that $\kappa \le e(1 \mid x) \le 1 - \kappa$ for all $x \in \mathcal{X}$ (or at least on $S_\lambda$). Weight variance and finite-sample bound widths scale directly with $1/\kappa$.
4. **A4 (Exchangeability / i.i.d. Sampling):** Calibration observations $\{(X_i, A_i, Y_i)\}_{i \in \mathcal{D}_{\text{cal}}}$ are i.i.d. from the target deployment distribution $P$ (for observational survey data like MEPS, complex survey clustering/weights violate i.i.d. and require explicit treatment or scope restriction to the analytic sample).
5. **A5 (Propensity Regime — Exact vs. Approximate):**
   - **A5a (Known Propensity — RCT / A/B test / Lottery):** $e(a \mid x)$ is known exactly $\implies$ **exact finite-sample** $(1-\delta)$ certification.
   - **A5b (Estimated Propensity — Observational Data):** $e(a \mid x)$ is estimated by $\hat e(a \mid x)$ on $\mathcal{D}_{\text{tr}}$ $\implies$ finite-sample bound incurs an additive propensity-error slack proportional to $\mathbb{E}[|\Delta \hat w_\lambda - \Delta w_\lambda|]$; exact finite-sample guarantees hold only under an explicit sensitivity bound $\Gamma$ on propensity misspecification, or asymptotically under doubly robust estimation.
6. **A6 (Support / Tail Regularity — Functional-Specific):**
   - For $Q_\beta$ and $R_{\text{ext}}$: **None** (works for arbitrary heavy-tailed / unbounded outcomes).
   - For $\text{CVaR}_\beta$: Known upper bound $L \in [0, B]$ (e.g., policy stop-loss cap, SLA request timeout) OR known upper bound on a higher moment / sub-exponential tail parameter (necessary by Bahadur & Savage, 1956).

---

## 7. Claimed Contributions vs. Prior Art

### Original Seed Claims
- **C1:** Finite-sample distribution-free tail-risk (CVaR, upper quantile, extensive-margin) safety certification for personalized policies under propensity-weighted shift.
- **C2:** Abstention/fallback gate that deploys personalized actions only where counterfactual tail safety is certified.
- **C3:** Modular integration with nonparametric distributional causal learners (`WCF` and `ZIC-BCF`) for semicontinuous and unit-level distributional outcomes.
- **C4:** Empirical superiority over plug-in CVaR ITRs and mean-optimal causal forest policies on healthcare costs (MEPS, OHIE) and latency/SLA A/B datasets.

### Critical Positioning Against Closest Prior Art (Verified)
- **vs. Conformal Policy Learning (Jin & Egami, 2026, arXiv:2609.17296):** CPL certifies finite-sample "do no harm" on *individual counterfactual harm* $\mathbb{P}(Y(1) < Y(0), \pi(X)=1) \le \alpha$ for scalar outcomes via conformal $p$-values. It does not address nonlinear population tail functionals (CVaR, $Q_\beta$), joint extensive/intensive margins in semicontinuous outcomes, or unit-level distributions.
- **vs. Conformal Policy Control (Prinster et al., ICML 2026, arXiv:2603.02196):** CPC regulates an optimized policy against a safe reference policy $\pi_0$ via likelihood-ratio bounds calibrated with conformal risk control for **expected bounded (possibly non-monotonic) per-observation losses**. It does not certify nonlinear tail functionals like CVaR or quantiles in Regime A, nor comparative CVaR localization.
- **vs. Conformal Policy Selection (Cao et al., 2026, arXiv:2610.11547):** CPS builds a selective deployment gate via weighted conformal $p$-values controlling finite-sample FDR for per-unit threshold exceedances $\mathbf{1}\{Y_i(\pi) > c_i\}$. It targets scalar unit-level exceedance FDR, not population CVaR / upper-quantile safety.
- **vs. High-Confidence Off-Policy Evaluation & Risk Assessment (Thomas et al., 2015; Chandak et al., NeurIPS 2021 *UnO*; Huang et al., NeurIPS 2021 *OPRA*):** **THIS IS THE MOST DANGEROUS COLLISION.** OPRA and UnO already establish finite-sample high-confidence off-policy bounds for CVaR and quantiles in contextual bandits/RL. Pairing OPRA/UnO bounds with Thomas-style High-Confidence Policy Improvement (HCPI) and LTT creates a strong incumbent baseline. The **only defensible methodological contribution** beyond OPRA/UnO + LTT is **localized comparative certification** (canceling $S_\lambda^c$ via the Rockafellar–Uryasev baseline-VaR pivot and localized CDF contrasts, preventing the $O(1/\sqrt{n})$ dead-zone that causes naive two-arm OPRA/UnO to abstain 100% on targeted personalized gates) paired with **two-part semicontinuous tail-risk scoring**.
- **vs. Conformal Tail / Quantile Risk Control (Snell et al., ICLR 2023; Chen et al., 2025 arXiv:2502.20285; Yeh et al., 2025 arXiv:2510.08748; Chen et al., 2026 arXiv:2606.00320):** These target predictive or online CVaR/distortion risk control, not off-policy causal policy gating with counterfactual propensity weights and comparative baseline cancellation.
- **vs. Risk-Sensitive ITRs (Wang et al., JASA 2018; Qi et al., JASA 2023; Kallus, Management Science 2023):** These estimate CVaR-optimal or quantile-optimal ITRs (or ITE-CVaR bounds) asymptotically without a finite-sample certified abstention gate.

---

## 8. Strongest Simple Baselines

Any credible empirical evaluation must benchmark against all five of the following comparators under equal sample splits, tuning budgets, and information sets:

1. **B1 — Two-Arm OPRA / UnO + LTT Fallback Gate (Strongest Incumbent with Same Guarantee):** Apply existing finite-sample off-policy CVaR/quantile upper bounds (Huang et al., 2021 OPRA / Chandak et al., 2021 UnO) separately to $\pi_\lambda$ and $\pi_0$, and select $\hat\lambda$ via fixed-sequence LTT. Tests whether localized comparative cancellation actually produces a practically meaningful gain in certified deployment rate and welfare.
2. **B2 — Mean-Optimal Causal Forest Policy + Localized LTT Gate (Ablation of Distributional Learner):** Replace `ZIC-BCF-Smear` / distributional tail-risk scoring with standard mean-CATE causal forest (`grf` / `econml`) for $\hat\pi$ and $s(x)$, while keeping the *same* localized tail-risk certification gate. Isolates whether nonparametric two-part / distributional modeling is genuinely needed or merely decorative.
3. **B3 — Plug-in CVaR / Quantile-Optimal ITR (Uncertified Risk-Sensitive Incumbent):** Estimate CVaR-optimal (Qi et al., 2023) or quantile-optimal (Wang et al., 2018) ITR without conformal/LTT gating. Demonstrates empirical safety-violation rates under finite samples, heavy tails, and model misspecification.
4. **B4 — Mean-Optimal Causal Forest ITR (Uncertified Standard Incumbent):** Standard CATE $> 0$ deployment rule without safety gating.
5. **B5 — Scalar Conformal Policy Control / Indicator-Loss Weighted CRC (Linear-Loss Conformal Incumbent):** Apply weighted CRC / CPC (Prinster et al., 2026) or CPL (Jin & Egami, 2026) to either a fixed-threshold exceedance indicator $\mathbf{1}\{L > c\}$ (Regime A) or unit-level scalarized $\text{CVaR}(\mu_i)$ (Regime B). Directly tests whether tail-functional machinery beats fixed-threshold / scalar conformal baselines.

---

## 9. Candidate Simulation Designs and Public Datasets

### 9.1 Mechanistic Simulation Scenarios (Regime A Primary)
1. **Mean-Tail Conflict under Zero-Inflation:** Treatment lowers mean cost via extensive-margin reduction or moderate-spender savings, yet increases upper-tail ($Q_{0.95}$, $\text{CVaR}_{0.90}$) costs in a latent high-risk covariate region (e.g., $X_1 > 0.5$). Mean-optimal ITRs violate tail safety severely; plug-in CVaR ITRs over-deploy in finite samples; localized LCT-LTT certifies deployment strictly on the safe region.
2. **Global Null / Globally Unsafe Policy:** Treatment has zero benefit or strictly worsens tail risk everywhere; certified method must abstain ($\hat\lambda = +\infty$) with probability $\ge 1 - \delta$.
3. **Targeted Small-Region Deployment ($P(S^*) \in \{0.10, 0.25, 0.50\}$):** Tests the exact regime where naive two-arm OPRA/UnO suffers full $O(1/\sqrt{n})$ non-canceling width and abstains completely, whereas localized comparative certification retains high power.
4. **Heavy-Tailed Semicontinuous Regimes (Tweedie, Log-Normal Hurdle, Pareto-Tail Hurdle) with Known Cap $B$ vs. Uncapped Quantiles:** Stress-tests CVaR sensitivity to cap $B$ versus boundedness-free quantile certification.
5. **Known RCT Propensity vs. Estimated Observational Propensity (with Weak Overlap $\kappa \in \{0.05, 0.1, 0.25\}$):** Quantifies exact finite-sample validity under RCTs and degradation/sensitivity bounds under observational propensity estimation error.

### 9.2 Dataset Audit (Verified vs. Unverified)
1. **Oregon Health Insurance Experiment (OHIE) — VERIFIED PUBLIC RCT (Primary Regime A Candidate):**
   - *Source:* NBER public-use files (`https://www.nber.org/programs-projects/projects/oregon-health-insurance-experiment`).
   - *Structure:* 2008 Medicaid lottery among low-income uninsured adults; randomized lottery assignment gives **known propensity** $e(x)$ (stratified by household size), making exact finite-sample ITT safety certification strictly valid without observational confounding. Outcomes include zero-inflated, heavy-tailed medical/out-of-pocket expenditure and ED/hospital utilization. (Caveat: actual Medicaid enrollment has imperfect compliance, so ITT lottery policy vs. IV/LATE must be carefully distinguished.)
2. **MEPS 2023 Full-Year Consolidated File (HC-251) — VERIFIED LOCALLY (Secondary Observational Regime A Candidate):**
   - *Source:* AHRQ HC-251, already downloaded and cleaned locally at `/home/hugo/Stuff/Submitted_Research/zicbcf/applied_study/h251.csv` ($n = 18{,}763$ analytic records, dental insurance treatment, heavily zero-inflated right-skewed dental expenditure).
   - *Caveats:* **Observational** (unknown propensity $e(x)$, so exact finite-sample guarantees do NOT hold without sensitivity bounds) AND complex survey design (`PERWT23F`, `VARSTR`, `VARPSU`) violating naive i.i.d. exchangeability. Best used as an observational stress test / sensitivity study, NOT as the flagship proof of exact finite-sample validity.
3. **Public Latency / SLA A/B Test Datasets — `UNVERIFIED` / UNAVAILABLE:**
   - Web verification confirms no standard open-access randomized A/B test dataset with covariates and raw request-level tail latency exists (public A/B logs like Criteo/Open Bandit track clicks/conversions, while public latency traces like M-Lab/NetLatency lack randomized treatments). Do NOT anchor the paper on an unverified latency A/B dataset; if a systems/AI SLA study is desired, it requires generating a controlled real LLM-routing benchmark log (e.g., routing open-weight/API models on a public prompt benchmark with logged latency and accuracy) or pivoting to verified RCT datasets.
4. **WCF Randomized Distributional Datasets (If Pivoting to Regime B) — VERIFIED LOCALLY:**
   - Project STAR classrooms and Egger et al. (2022) Kenyan cash-transfer villages are already audited in `/home/hugo/Stuff/Submitted_Research/Wasserstein_Causal_Forests/results/applied_study_exploration/` with known randomization and unit-level quantile curves $Q_i$.

---

## 10. Key Mathematical and Empirical Risks

1. **Non-Linearity & Unboundedness of CVaR vs. Standard CRC (HIGH / FATAL IF UNADDRESSED):**
   Standard weighted CRC certifies $\mathbb{E}[L(Y, \pi_\lambda(X))] \le \alpha$ for bounded monotone $L \in [0, B]$. Population $\text{CVaR}_\beta(L(\pi_\lambda))$ is nonlinear and cannot be plugged into single-point CRC. Moreover, by Bahadur & Savage (1956), no nontrivial finite-sample distribution-free upper confidence bound exists for the mean (hence CVaR) of an unbounded positive random variable without a known cap $B$ or tail/moment bound. *Mitigation:* Use localized Rockafellar–Uryasev + LTT for bounded/capped CVaR, and highlight boundedness-free localized upper-quantile ($Q_\beta$) certification for uncapped heavy-tailed outcomes.
2. **Marginal vs. Conditional Safety Overclaiming (HIGH):**
   Weighted conformal / off-policy LTT certifies population-marginal safety of $\pi_{\hat\lambda}$ (or finite-subgroup safety), NOT pointwise conditional safety at every $x \in S_{\hat\lambda}$. Overclaiming covariate-region conditional certification will trigger immediate desk/referee rejection.
3. **Observational Propensity Error Breaking Finite-Sample Guarantees (HIGH):**
   Under estimated propensities $\hat e(x)$ (as in MEPS), exact finite-sample guarantees fail by an unobservable weight-estimation bias. Anchoring finite-sample claims on an observational dataset without known propensities or a formal propensity-sensitivity model ($\Gamma$-bounds) is scientifically invalid.
4. **Conservatism / Zero-Deployment Collapse (MEDIUM-HIGH):**
   Finite-sample tail bounds (especially for $\beta = 0.95$ CVaR with propensity weights $1/\kappa$ and cap $B$) can be so wide at realistic $n_{\text{cal}} \in [1000, 10000]$ that $\hat\lambda = +\infty$ (100% abstention), yielding a vacuous "always do nothing" policy. Betting-martingale / empirical-Bernstein localized bounds and moderate tail levels ($\beta \in \{0.80, 0.90\}$) must be tested early in Phase 3–4 before investing in theory.
5. **Direct Prior-Art Collapse into Incremental Combination (HIGH):**
   With Jin & Egami (2026 CPL), Prinster et al. (ICML 2026 CPC), Cao et al. (2026 CPS), and Huang/Chandak (2021 OPRA/UnO) already in the literature, simply plugging `ZIC-BCF` or `WCF` into an existing off-policy bound will be judged `INCREMENTAL-ONLY` at ICML/NeurIPS/JASA/Management Science. The paper lives or dies on whether **localized comparative tail-risk certification + semicontinuous two-part tail scoring** delivers a large, reproducible empirical power/deployment advantage where incumbents fail or abstain completely.
