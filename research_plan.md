# Research Plan: Safe Distributional Policy Learning (Pivot A — Localized Comparative Tail-Risk Policy Certification for Semicontinuous Outcomes)

**Input Dossier:** [research_idea.md](file:///home/hugo/Stuff/Research/Safe_Distributional_Policy/research_idea.md)  
**Working Directory:** `/home/hugo/Stuff/Research/Safe_Distributional_Policy/`  
**Internal Stack Referenced:** [Wasserstein_Causal_Forests](file:///home/hugo/Stuff/Submitted_Research/Wasserstein_Causal_Forests/README.md), [zicbcf](file:///home/hugo/Stuff/Submitted_Research/zicbcf/README.md), [CP_TATE](file:///home/hugo/Stuff/Submitted_Research/CP_TATE/README.md)  
**Plan Standard:** Source-grounded, simulation-first research program with hard `GO` / `CONDITIONAL GO` / `PIVOT` / `INCREMENTAL-ONLY` / `KILL` gates.

**Execution status.** Phases 0 and 1 are recorded in Sections 3–4. Gate G2 advances to Phase 3 with a quantile-first pivot. The 5%-first pilot missed the deployment threshold. A pre-run sensitivity beginning at 15% cleared it for the uncapped quantile at the focal sample size. Use joint upper-quantile and extensive-margin safety as the primary target and retain capped CVaR as a secondary diagnostic. The gate memo and linked results provide the execution record for changes to the original plan below. [Gate G2 memo](docs/gate_g2_memo.md), [independent validity check](docs/phase2_independent_check.md), [read-only data feasibility](docs/phase5_data_feasibility.md)

Phase 3 passed its implementation and correctness checks. Phase 4 uses the frozen quantile-first protocol and the original confirmatory seeds. [Phase 3 readiness](docs/phase3_readiness.md), [Phase 4 preregistration](docs/phase4_preregistration.md)

**Terminal Gate G3 decision.** `KILL` for the current research program. Across 200 paired replicates in the four decision-critical cells, the fixed-cutoff indicator baseline with the same extensive-margin guard matched LCT-LTT deployment 188 times and exceeded it 12 times. Both methods had zero observed unsafe prefixes in those cells. The decision used 300 core and 150 focal replications. The remaining stress grid and applied study stop at this gate. [Gate G3 memo](docs/gate_g3_memo.md), [core results](results/phase4_sim/core_dgp1_5_r50.csv), [focal paired results](results/phase4_sim/focal/paired_comparisons.csv)

---

## 1. Executive Verdict

1. **Current Classification:** **`Promising but unproven` (ONLY under mandatory Gate G0 Pivot A)**; the **original unpivoted seed formulation is `DEAD IN THE PRESENT FORM`** (fails Phase 0 due to three demonstrated mathematical defects and collides directly with 2025–2026 conformal policy papers).
2. **Confidence and Evidence Level:** Moderate-High confidence in Phase 0 and Phase 1 verdicts (supported by E3/E4 inspection of primary papers, internal repositories, and symbolic counterexamples); **UNTESTED** (E0) on empirical power, non-vacuity of finite-sample CVaR/quantile certification at realistic $n_{\text{cal}}$, and applied value.
3. **One-Sentence Proposed Contribution (Post-Pivot A):** A finite-sample off-policy tail-risk certification and abstention-gate framework (**Localized Comparative Tail-Risk Learn-then-Test, LCT-LTT**) that cancels the shared abstained covariate region via a Rockafellar–Uryasev baseline-VaR pivot and localized CDF contrasts, enabling non-vacuous finite-sample upper-quantile ($Q_\beta$), capped-CVaR, and extensive-margin safety certification for personalized policies on zero-inflated semicontinuous outcomes where standard two-arm off-policy bounds abstain completely and linear-loss conformal policy methods fail to guard tail risk.
4. **Real Contribution vs. Engine vs. Application vs. Decoration:**
   - **Real Contribution:** Localized comparative off-policy tail-risk certification for gated policies (the Rockafellar–Uryasev baseline-VaR pivot + localized CDF contrast + fixed-sequence LTT gate over nonlinear tail functionals and extensive-margin risk simultaneously).
   - **Engine:** Two-part semicontinuous distributional learners (`ZIC-BCF-Smear` and fast cross-fitted two-part quantile/GBDT learners) generating the candidate policy $\hat\pi(x)$ and tail-safety ranking score $s(x)$, plus betting-martingale / empirical-Bernstein finite-sample bounds.
   - **Application:** Randomized healthcare expenditure policy gating on the **Oregon Health Insurance Experiment (OHIE)** (primary exact finite-sample ITT study with known lottery propensities), paired with an observational propensity-sensitivity stress test on **MEPS 2023 HC-251** (already locally cached).
   - **Decoration (Cut from Primary Paper):** Bundling Regime B (`WCF` unit-level distributions) into the same paper; unverified claims about public randomized latency/SLA A/B datasets; claiming pointwise covariate-conditional safety.
5. **Strongest Reason It Could Become a Strong Field Paper:** Recent 2026 conformal policy papers (CPL, CPC, CPS) only protect linear expectations of bounded losses or binary harm indicators, while existing high-confidence off-policy CVaR bounds (OPRA, UnO) evaluate target and baseline policies separately without canceling the abstained region $S_\lambda^c$—causing near-100% abstention when certifying targeted personalized improvements at small tolerances $\varepsilon \approx 0$. If localized comparative certification + two-part semicontinuous scoring unlocks high certified deployment rates with zero empirical tail-safety violations on OHIE and realistic heavy-tailed DGPs, it closes a clean, consequential methodological gap for ICML/NeurIPS/JASA/Management Science.
6. **Strongest Reason It Could Fail or Become Incremental:** Two concrete kill modes: (a) finite-sample tail bounds (especially for $\beta = 0.90\text{--}0.95$ CVaR under cap $B$ and propensity weights $1/\kappa$) may still be too conservative at $n_{\text{cal}} \in [2000, 10000]$ to certify nontrivial deployment, collapsing to "always abstain"; OR (b) a simple baseline—mean-CATE causal forest + fixed-threshold CPC/CPL or two-arm OPRA + LTT—may achieve nearly identical certified deployment and welfare, reducing the remaining delta to `INCREMENTAL-ONLY`.
7. **Next Unresolved Gate:** **Gate G2 (Minimum Enabling Formalization)** (preceded by formal adoption of Gate G0 Pivot A and bounded Condition G1-1/G1-2 verification).
8. **Single Cheapest Decisive Next Action:** Complete **WP-2.1** (exact mathematical formalization of localized comparative CVaR/quantile/extensive-margin finite-sample bounds and proof of non-vacuous variance scaling $O(\mathbb{P}(S_\lambda)/n)$) and **WP-2.2** (1D synthetic analytical/numerical tightness check verifying that localized LCT-LTT certifies $\ge 30\%$ deployment where two-arm OPRA+LTT certifies $\le 5\%$ at $n_{\text{cal}} = 5{,}000$), alongside **WP-5.0** (read-only OHIE/MEPS schema & sample-size feasibility audit).

---

## 2. Idea Reconstruction and Claim Decomposition

### 2.1 Faithful Reconstruction of the Original Proposal (As Written in Seed)
- **Scientific problem:** Deploying personalized treatment policies on semicontinuous/zero-inflated or unit-level distributional outcomes (healthcare costs, e-commerce spend/returns, enterprise AI SLAs) without worsening downside tail risk (95th-percentile latency, CVaR loss, zero-conversion risk).
- **Unit of analysis & observable data:** Bundled two regimes—(A) individual units with scalar semicontinuous outcomes $Y_i \in \{0\} \cup (0, \infty)$ and (B) cluster/service units with unit-level distributions $\mu_i$ (quantile curves $Q_i \in \mathcal{Q}_K$).
- **Decision problem:** Learn a personalized policy $\hat\pi(x)$ with an abstention/fallback gate returning to baseline $\pi_0(x)$ outside certified covariate regions.
- **Target / Estimand:** Finite-sample probability $\ge 1 - \alpha$ that the personalized policy does not worsen monotone distributional risk functionals (CVaR, upper quantiles, extensive-margin probability) under propensity-weighted shift.
- **Proposed mechanism:** Weighted Conformal Risk Control (CRC) layered on top of nonparametric distributional learners (`WCF`, `ZIC-BCF`) with an abstention/fallback gate.
- **Assumptions (implicit in seed):** Unconfoundedness, overlap, exchangeability, known/estimated propensities, boundedness/monotonicity for CRC.
- **Intended evidence:** Synthetic comparisons against plug-in CVaR ITRs and mean-optimal causal forests, plus empirical studies on MEPS, Oregon Health Insurance Experiment (OHIE), and latency/SLA A/B datasets.
- **Target audience / venue tier:** ICML, NeurIPS, JASA, Management Science.

### 2.2 Claim Decomposition Table

| Claim | Type | Feasibility | Novelty | Importance | Evidence Status | Keep, Cut, or Pivot |
|---|---|---|---|---|---|---|
| **C1:** Finite-sample distribution-free tail-risk (CVaR, $Q_\beta$, extensive-margin) safety guarantee for gated personalized policies via weighted CRC | `contribution` | **Broken as stated** (CRC fails on nonlinear/unbounded CVaR) | High *only* after localized comparative repair | High (3/3) | E4 defect found (Phase 0) | **PIVOT** to Localized Comparative Off-Policy Tail-Risk LTT (LCT-LTT) |
| **C2:** Abstention/fallback gate certifying safety "on covariate regions" | `contribution` | Pointwise impossible; population-marginal & finite-subgroup feasible | Medium-High (must beat CPS/CPL/CPC + OPRA) | High (3/3) | E3 prior-art check | **PIVOT** scope to population-marginal & finite-subgroup safety (plus optional post-selection floor) |
| **C3a:** Layering on `ZIC-BCF-Smear` / two-part semicontinuous distributional learner (Regime A) | `engine` | High (`zicbcf` built & tested locally) | Low alone (score generator) | Medium-High if mean-tail conflict holds | E3 local repo (`zicbcf`) | **KEEP** as primary scoring engine (must pass ablation vs. mean-CATE forest) |
| **C3b:** Simultaneous inclusion of `WCF` unit-level distributional outcomes (Regime B) | `decoration` (dilutes Regime A paper) | High (`WCF` built locally), but scalarized CVaR collapses to scalar CPC/CPL | Low unless adding simultaneous tail-curve + finite-$m_i$ correction | Medium | E3 local repo (`WCF`, `CP_TATE`) | **CUT** from primary paper (preserved as separate Pivot B option) |
| **C4a:** Applied study on OHIE (RCT) + MEPS (observational sensitivity) | `application` | High (MEPS local; OHIE public NBER) | Medium-High if novel safe policy regions discovered | High (3/3) | E3 MEPS local; E2 OHIE public | **KEEP** (OHIE primary exact-propensity RCT; MEPS secondary sensitivity) |
| **C4b:** Applied study on public latency/SLA A/B dataset | `application` | **Low / Blocked** (no verified public randomized latency A/B dataset) | Medium | Medium | E2 search confirms none publicly available (`UNVERIFIED`) | **CUT** (do not anchor paper on unavailable data; optional controlled LLM-routing log only if needed later) |

### 2.3 Seven Load-Bearing Strategic Identifiers
1. **Load-bearing contribution:** Localized comparative finite-sample off-policy certification for nonlinear tail functionals (upper quantiles $Q_\beta$ without boundedness, capped $\text{CVaR}_\beta$ via the Rockafellar–Uryasev baseline-VaR pivot, and joint extensive-margin risk), canceling the shared abstained region $S_\lambda^c$.
2. **Load-bearing assumption:** Known propensity scores $e(a \mid x)$ with strict overlap $\kappa \le e(1 \mid x) \le 1 - \kappa$ (for exact finite-sample guarantees; in observational data, an explicit propensity-sensitivity bound $\Gamma$ or asymptotic DR validity), plus known support cap $B$ specifically for $\text{CVaR}_\beta$ (not needed for $Q_\beta$).
3. **Most dangerous prior-art collision:** **OPRA** (Huang et al., NeurIPS 2021) + **UnO** (Chandak et al., NeurIPS 2021) + **HCPI** (Thomas et al., ICML 2015) + **LTT** (Angelopoulos et al., AOAS 2025) for off-policy CVaR/quantile finite-sample bounds, alongside **CPL** (Jin & Egami, 2026), **CPC** (Prinster et al., ICML 2026), and **CPS** (Cao et al., 2026) for conformal safe policy gating.
4. **Strongest simple baseline:** **B1 (Two-Arm OPRA/UnO + Fixed-Sequence LTT)** for the certification layer, and **B2 (Mean-Optimal Causal Forest + Localized LCT-LTT)** for the scoring layer.
5. **Cheapest decisive experiment:** A 1-day Python simulation comparing certified deployment fraction, realized welfare, and empirical tail-risk violation rate of **LCT-LTT** vs. **B1 (Two-Arm OPRA+LTT)**, **B2 (Mean-Forest + LCT-LTT)**, and **B5 (Indicator-Loss CPC)** across targeted safe-region sizes $\mathbb{P}(S^*) \in \{0.15, 0.35, 0.60\}$ and mean-tail conflict DGPs at $n_{\text{cal}} \in \{2000, 5000, 10000\}$.
6. **Hardest credible referee objection:** *"You combined existing off-policy CVaR/quantile bounds (Huang et al. 2021; Chandak et al. 2021) with Learn-then-Test (Angelopoulos et al. 2025) and your own semicontinuous estimator (`ZIC-BCF`), and either your CVaR bounds still abstain almost everywhere under heavy tails or a simple indicator-loss CPC / mean-CATE forest achieves the same deployed policy."*
7. **Attractive component to cut:** **Regime B (`WCF` unit-level distributions)** and **latency/SLA A/B dataset promises**. Including both scalar semicontinuous outcomes and unit-level distributions makes the paper mathematically incoherent, dilutes the core localized tail-risk mechanism, and exposes the paper to a trivial scalar-collapse critique on Regime B.

---

## 3. Fatal-Flaw Certificate (Phase 0 & Gate G0)

### 3.1 Phase 0 Audit Checks (1–8)

| Check | Audit Finding | Classification | Witness / Mathematical Consequence |
|---|---|---|---|
| **1. Definitions & mathematical typing** | Seed conflates Regime A (scalar $Y_i \in \{0\} \cup (0,\infty)$, where CVaR/$Q_\beta$ is a nonlinear functional of $P_{Y(\pi)}$) and Regime B (unit-level distribution $\mu_i$, where $\rho(\mu_i)$ is an observable scalar functional of unit $i$). | **Demonstrated defect (D1)** | In Regime A, $\text{CVaR}_\beta(Y_i)$ is undefined (not a random variable); in Regime B, $\rho(\mu_i)$ is a random scalar per unit, reducing single-functional safety to standard scalar CPC/CPL (`CP_TATE` Lemma 4.1). |
| **2. Assumption consistency & non-vacuity** | Seed claims distribution-free finite-sample CVaR guarantees for unbounded heavy-tailed healthcare costs / latency via CRC without stating a support bound $B$ or known propensity $e(x)$. | **Demonstrated defect (D2a)** | **Witness (Bahadur & Savage, 1956):** For any unbounded nonnegative family $\mathcal{P}$ containing all two-point distributions $(1-p)\delta_0 + p\delta_M$ ($M \to \infty$), no nontrivial finite-sample test can upper-bound $\mathbb{E}[Y]$ or $\text{CVaR}_\beta(Y)$ uniformly over $\mathcal{P}$ at level $1-\delta$ with size $n$. Any valid finite-sample CVaR certificate strictly requires known $Y \le B$ or a tail/moment bound. (By contrast, upper quantiles $Q_\beta$ need NO upper bound $B$!) |
| **3. Identification under observation scheme** | Comparative marginal distributional safety $\rho(P_{L(\pi_\lambda)}) \le \rho(P_{L(\pi_0)}) + \varepsilon$ IS point-identified under A1–A3 (depends only on marginal potential laws $P_{L(0)}, P_{L(1)}$, not the unidentifiable joint law of $(L(0), L(1))$). Individual counterfactual harm IS NOT point-identified. Observational MEPS lacks known $e(x)$ and has survey clustering. | **Nonblocking caveat (RCT) / Demonstrated defect (observational MEPS if claiming exact finite-sample)** | Under estimated $\hat e(x)$ in MEPS, weighted finite-sample guarantees incur an unobservable TV slack $\frac{1}{2}\mathbb{E}[|\Delta \hat w_\lambda - \Delta w_\lambda|]$ (Lei & Candès, 2021, Thm. 3). Exact finite-sample claims must be restricted to known-propensity designs (RCT/OHIE) or stated with explicit $\Gamma$-sensitivity bounds. |
| **4. Existence & well-definedness of target & method** | Population-marginal gated safety $\pi_\lambda$ is well-defined and always feasible ($\lambda = +\infty \implies \pi_\infty = \pi_0$ is trivially safe). Pointwise covariate-region conditional safety is **unachievable** distribution-free. | **Demonstrated defect (D3)** | **Witness (Barber et al., 2021; Vovk, 2012):** For continuous $X$ with atomless distribution, any distribution-free finite-sample pointwise conditional guarantee must be trivial (100% abstention everywhere). Moreover, post-selection conditional safety on $S_{\hat\lambda}$ requires selective calibration, not naive weighted CRC. |
| **5. Information leakage / unavailable variables** | Splitting $\mathcal{D}_{\text{tr}}$ (learner $\hat\pi, s, \hat e$) from $\mathcal{D}_{\text{cal}}$ (certification of $\hat\lambda$) avoids leakage. However, evaluating $\hat\eta_0 = \hat Q_\beta(L(\pi_0))$ on $\mathcal{D}_{\text{cal}}$ without a valid confidence interval $[\underline\eta_0, \overline\eta_0]$ leaks calibration data into the Rockafellar–Uryasev threshold! | **Unresolved gap (repaired in Phase 2)** | Plugging a point estimate $\hat\eta_0$ into $\text{CVaR}_\beta(\pi_\lambda) - \text{CVaR}_\beta(\pi_0)$ fails to lower-bound $\text{CVaR}_\beta(\pi_0)$ by an unknown gap $\frac{1}{1-\beta}\int_{\eta_0^*}^{\hat\eta_0}(F_0(u)-\beta)\,du \ge 0$. Taking $\sup_{\eta \in [\underline\eta_0, \overline\eta_0]}$ over a $(1-\delta_1)$ finite-sample CI for $\eta_0^*$ eliminates leakage and guarantees validity. |
| **6. Mismatch between proposed method & target** | Weighted Conformal Risk Control (Angelopoulos et al., 2024; Zecchin et al., 2025) certifies $\mathbb{E}[L(Y_{\text{test}}, \pi_{\hat\lambda}(X_{\text{test}}))] \le \alpha$ for a *monotone, bounded, per-observation* loss. | **Demonstrated defect (D2b)** | **Counterexample:** Let $L(\pi_0) \sim \text{Bernoulli}(0.04) \cdot 100$ ($\text{CVaR}_{0.90} = 40$, $\mathbb{E}[L] = 4$) and $L(\hat\pi) \sim \text{Bernoulli}(0.08) \cdot 50$ ($\text{CVaR}_{0.90} = 40$, $\mathbb{E}[L] = 4$), or mixture with $\pi_\lambda$: (i) $\text{CVaR}_\beta(L(\pi_\lambda))$ is nonlinear in $P_{L(\pi_\lambda)}$ (not $\mathbb{E}[\ell(L_i)]$), and (ii) when $s(x)$ is misspecified, $\lambda \mapsto \text{CVaR}_\beta(L(\pi_\lambda))$ is **non-monotone in $\lambda$**, violating CRC's core monotonicity requirement! |
| **7. Smallest counterexamples & degenerate cases** | (a) Degenerate case $\mathbb{P}(S_\lambda) = 0$: $\pi_\lambda = \pi_0$ so $\Delta \text{CVaR} = 0$, yet two-arm OPRA/UnO outputs positive upper bound $O(B/(\kappa(1-\beta)\sqrt{n})) > 0$, failing to certify even near-zero change! (b) Point mass at zero with $\mathbb{P}(Y=0) > \beta$: $Q_\beta(Y) = 0$ and $\text{CVaR}_\beta(Y) = \frac{1}{1-\beta}\mathbb{E}[Y]$, collapsing CVaR to rescaled mean! | **Nonblocking insight (sharpens mechanism & DGP constraints)** | Degenerate case (b) warns that simulations and applications MUST ensure the upper tail lies strictly in the positive continuous part ($\mathbb{P}(Y=0) < \beta$), otherwise CVaR collapses trivially to a scaled expectation! |
| **8. Known equivalence / vacuity check** | In Regime B, certifying a single scalar functional $R_i = \text{CVaR}_\beta(\mu_i)$ reduces verbatim to scalar CPC (Prinster et al., 2026) or CPL (Jin & Egami, 2026) applied to $R_i$. In Regime A, certifying only a fixed-threshold indicator $\mathbf{1}\{L > c\}$ reduces verbatim to scalar CPC/CPS. | **Demonstrated vacuity risk if unpivoted** | Forces cutting single-functional Regime B and focusing strictly on **nonlinear comparative tail functionals** ($Q_\beta$, capped $\text{CVaR}_\beta$, joint with extensive-margin risk) where scalar CPC/CPL does NOT apply. |

### 3.2 Gate G0 Verdict
- **Gate G0 Decision:** **`PIVOT`** (Original bundled seed **FAILS** Phase 0 due to D1, D2a, D2b, and D3).
- **Mandatory Repair — Pivot A (Primary Scope):**
  - *What changes:* (1) Restrict strictly to **Regime A** (scalar semicontinuous outcomes $Y_i \in \{0\} \cup (0, B]$ for CVaR and unbounded $Y_i \in \{0\} \cup (0, \infty)$ for $Q_\beta$ and extensive-margin risk); (2) replace naive weighted CRC with **Localized Comparative Tail-Risk Learn-then-Test (LCT-LTT)** using the Rockafellar–Uryasev baseline-VaR pivot $[\underline\eta_0, \overline\eta_0]$ and localized CDF contrasts over $\Delta w_\lambda$; (3) state exact finite-sample guarantees strictly for **population-marginal and finite-subgroup comparative safety** under **known propensities** (RCTs like OHIE), with explicit $\Gamma$-sensitivity / asymptotic DR bounds for observational data (MEPS).
  - *What is lost:* Unit-level distributional `WCF` outcomes (Regime B), unbounded CVaR claims without a cap $B$, pointwise covariate-conditional safety claims, and public latency/SLA A/B datasets.
  - *What becomes new & valid:* Exact, non-vacuous finite-sample comparative tail-risk certification that cancels the shared abstained region $S_\lambda^c$ and jointly controls upper quantiles, capped CVaR, and extensive-margin risk.
  - *Alternative Pivot (Pivot B — Preserved Only as Backup):* If the user specifically prioritizes `WCF` unit-level distributions (Project STAR, Egger Kenya), pivot instead to **Simultaneous Tail-Curve Safe Policy Learning with Finite-$m_i$ Subsampling Correction** (certifying $\sup_{\tau \in [\tau_0, 1]} (Q_i(\pi_\lambda)(\tau) - Q_i(\pi_0)(\tau))$ using `CP_TATE` + `WCF`). **Do NOT combine Pivot A and Pivot B in one paper.**
  - *Gates to rerun:* Gate G1 below is evaluated directly on **Pivot A** (while recording the collision verdict for the original seed).

---

## 4. Verified Prior Art and Nearest-Neighbor Map (Phase 1 & Gate G1)

### 4.1 Search Strings Across All Relevant Vocabularies
1. Conformal policy / causal decision vocabulary: `"Conformal Policy Learning" OR "conformalized policy" "do no harm" OR "off-policy" 2024 2025 2026`; `"Conformal Policy Control" Prinster`; `"Conformal Policy Selection" 2610.11547`; `"conformal" "individualized treatment" abstention OR selective OR "safe policy" Lei Candes Jin Ren Taufiq`.
2. RL / contextual-bandit off-policy tail-risk vocabulary: `"high-confidence off-policy" CVaR OR quantile Chandak OR Thomas OR Huang OR Keramati`; `"High-Confidence Policy Improvement" Thomas 2015`; `"Off-Policy Risk Assessment" Huang 2021 CVaR`.
3. Conformal risk control & tail-risk vocabulary: `"Conformal Risk Control" Angelopoulos CVaR OR quantile OR "weighted conformal" OR "Learn then Test"`; `"Quantile Risk Control" Snell`; `"Conformal Tail Risk Control" Chen`; `"Rockafellar-Uryasev" conformal OR "Conformal Risk Training" "OCE" CVaR`; `"weighted conformal risk control" covariate shift`.
4. Biostatistics / econometrics ITR vocabulary: `"individualized treatment" CVaR OR "conditional value at risk" OR "quantile-optimal" Qi Wang Kallus Zhou`; `"Treatment Effect Risk: Bounds and Inference" Kallus`; `"Safe Policy Learning" Ben-Michael Imai Jiang`.
5. Data & software vocabulary: `"Oregon Health Insurance Experiment" public data NBER`; `public dataset "latency" "A/B test" SLA tail latency`; local inspection of `Wasserstein_Causal_Forests`, `zicbcf`, `CP_TATE`.

### 4.2 Compact Evidence Ledger

| ID | Claim or Need | Source | Exact Supporting Location | Evidence Level | Implication |
|---|---|---|---|---|---|
| **EV-1** | 2026 Conformal Policy Learning ("do no harm") | Jin & Egami (2026), arXiv:2609.17296 | Abstract & framework summary (conformal $p$-values, selective calibration, learn-then-balance weights) | E2 (verified preprint record; full proof text to inspect in WP-2.1) | Direct collision on "do no harm" conformal policy gating; targets scalar individual harm, NOT population CVaR/quantiles. |
| **EV-2** | 2026 Conformal Policy Control (safe reference policy regulation) | Prinster et al. (ICML 2026), arXiv:2603.02196 | Abstract, Theorem 4.6 summary (likelihood-ratio regulation, non-monotonic bounded losses) | E2 | Direct collision on gated policy safety vs. $\pi_0$; targets *expected* bounded loss, NOT nonlinear CVaR/quantiles. |
| **EV-3** | 2026 Conformal Policy Selection (deployment gate with FDR control) | Cao, Gao, Saar-Tsechansky, Yin (2026), arXiv:2610.11547 | Abstract & method summary (weighted conformal $p$-values + BH pruning, finite-sample & DR FDR) | E2 | Direct collision on selective deployment gate; targets unit-level threshold exceedance FDR, NOT CVaR/quantiles. |
| **EV-4** | Off-policy finite-sample CVaR/quantile bounds (OPRA) | Huang, Leqi, Lipton, Azizzadenesheli (NeurIPS 2021), arXiv:2104.08977 | Abstract & NeurIPS paper summary (IS/DR CDF uniform concentration $\to$ Lipschitz risk / CVaR bounds) | E2 | **#1 strongest incumbent** for off-policy CVaR/quantile certification; evaluates single policies without $S_\lambda^c$ cancellation. |
| **EV-5** | Universal off-policy high-confidence CVaR/quantile bounds (UnO) | Chandak et al. (NeurIPS 2021), arXiv:2104.12820 | Abstract & NeurIPS summary (UnO high-confidence CDF/CVaR/quantile bounds) | E2 | Co-equal #1 incumbent with OPRA; forms baseline B1 when paired with HCPI/LTT. |
| **EV-6** | High-Confidence Policy Improvement (HCPI) | Thomas, Theocharous, Ghavamzadeh (ICML/AAAI 2015) | PMLR / AAAI summary (finite-sample high-confidence safety vs. behavior policy) | E2 | Establishes the safe policy-improvement paradigm with fallback to $\pi_0$ (for expected returns). |
| **EV-7** | Learn then Test (LTT) & Conformal Risk Control (CRC) | Angelopoulos et al. (AOAS 2025, DOI:10.1214/24-AOAS1998; ICLR 2024, arXiv:2208.02814) | Published AOAS/ICLR records; CRC bounded monotone expectation vs. LTT multiple hypothesis testing | E3 | Confirms CRC requires bounded monotone expectation; LTT fixed-sequence testing is the valid engine for non-monotone/nonlinear risks. |
| **EV-8** | Predictive Conformal Tail / Quantile / CVaR Risk Control | Snell et al. (ICLR 2023, arXiv:2212.13629); Chen et al. (2025, arXiv:2502.20285); Yeh et al. (2025, arXiv:2510.08748); Chen et al. (2026, arXiv:2606.00320) | arXiv records (distortion risks, L-statistics/Berk-Jones, OCE/CVaR, Rockafellar–Uryasev online CVaR) | E2 | Predictive/online CVaR conformal control exists, but none handles off-policy counterfactual policy gating or comparative $S_\lambda^c$ cancellation. |
| **EV-9** | Weighted CRC under covariate shift & Counterfactual CP | Zecchin et al. (2025, arXiv:2501.11413); Lei & Candès (JRSS-B 2021, DOI:10.1111/rssb.12445); Taufiq et al. (NeurIPS 2022, arXiv:2206.04405) | Primary records + `CP_TATE/docs/phase4_conformal.md` | E3 | Confirms weighted CP/CRC gives marginal guarantees under known weights and incurs TV slack under estimated $\hat e(x)$. |
| **EV-10** | Asymptotic CVaR / Quantile ITRs & Treatment Effect Risk | Wang et al. (JASA 2018, DOI:10.1080/01621459.2017.1330204); Qi, Pang, Liu (JASA 2023, DOI:10.1080/01621459.2022.2038180); Kallus (Manag. Sci. 2023); Ben-Michael et al. (JASA 2025) | Journal DOI records & abstracts | E2 | Plug-in/asymptotic CVaR & quantile ITRs lack finite-sample safety gates; Kallus bounds ITE-CVaR asymptotically; Ben-Michael does maximin extrapolation under deterministic baseline. |
| **EV-11** | Local `zicbcf`, `WCF`, `CP_TATE` codebases & MEPS HC-251 data | `/home/hugo/Stuff/Submitted_Research/{zicbcf,Wasserstein_Causal_Forests,CP_TATE}` | Inspected `README.md`, `zicbcf/applied_study/h251.csv` ($n=18{,}763$), `CP_TATE/docs/phase4_conformal.md` | E3 | Confirms `ZIC-BCF-Smear` and MEPS 2023 are ready locally; confirms MEPS is observational + survey-weighted (`PERWT23F`, `VARSTR`, `VARPSU`). |
| **EV-12** | OHIE public RCT data vs. Latency/SLA A/B datasets | NBER OHIE portal; web search for open latency A/B data | NBER public-use documentation; open latency A/B search | E2 (OHIE) / E1 (`UNVERIFIED` for latency A/B) | OHIE provides known lottery propensity $e(x)$ for exact finite-sample ITT validation; no verified public randomized latency A/B dataset exists. |

### 4.3 Nearest-Neighbor Comparison Table (Evaluated Against Pivot A)

| Source | Target | Observation Regime | Method | Guarantee or Inference | Evidence | Remaining Gap (vs. Pivot A) | Direct-Hit Risk |
|---|---|---|---|---|---|---|---|
| **Huang et al. (NeurIPS 2021, OPRA)** & **Chandak et al. (NeurIPS 2021, UnO)** | Policy CVaR, quantiles, Lipschitz risks of $P_{L(\pi)}$ | Contextual bandits / RL with known behavior policy $e(a\mid x)$ | IS / DR CDF uniform concentration $\to$ plug-in risk bounds | Finite-sample high-confidence upper/lower bounds on $\rho(P_{L(\pi)})$ | E2 | Evaluates single policies; **does NOT localize comparative difference $\rho(\pi_\lambda)-\rho(\pi_0)$ to $S_\lambda$**, causing severe conservatism ($O(1/\sqrt{n})$ dead-zone) for small-region personalized gates; no two-part semicontinuous scoring or LTT gate optimization | **HIGH** (Strongest incumbent; forms B1) |
| **Prinster et al. (ICML 2026, CPC)** | Expected bounded loss $\mathbb{E}_{\pi_\lambda}[L] \le \alpha$ vs. safe reference $\pi_0$ | Off-policy / safe exploration with known reference $\pi_0$ | Likelihood-ratio regulation + conformal risk control | Finite-sample expected risk control for non-monotonic bounded losses | E2 | Targets **expected** bounded loss, NOT nonlinear tail functionals ($\text{CVaR}_\beta$, $Q_\beta$); cannot certify upper quantiles without boundedness or comparative CVaR | **HIGH** (Direct collision on expected/indicator loss) |
| **Jin & Egami (2026, CPL)** | Individual counterfactual harm $\mathbb{P}(Y(1) < Y(0), \pi(X)=1) \le \alpha$ | RCT (exact) & observational (learn-then-balance) scalar outcomes | Conformal $p$-values + selective calibration | Finite-sample (RCT) & doubly robust (obs.) individual harm rate | E2 | Targets binary individual harm, NOT population tail risk ($\text{CVaR}_\beta$, $Q_\beta$) or joint extensive/intensive margins in semicontinuous outcomes | **MEDIUM-HIGH** |
| **Cao et al. (2026, CPS)** | Unit-level exceedance FDR among deployed units | Offline logged data $\to$ target AI policy | Weighted conformal $p$-values + BH pruning | Finite-sample & DR asymptotic FDR | E2 | Targets scalar per-unit threshold exceedance FDR, NOT nonlinear distributional tail functionals | **MEDIUM-HIGH** |
| **Snell et al. (ICLR 2023)** & **Chen et al. (2025, CTRC)** | Quantile / distortion / CVaR risk of predictive losses | i.i.d. supervised / LLM alignment (no actions/propensities) | Order-statistic / L-statistic / Berk–Jones bounds | Finite-sample high-probability tail-risk control | E2 | No off-policy counterfactual action selection, propensity weighting, or comparative baseline cancellation | **MEDIUM** |
| **Angelopoulos et al. (AOAS 2025, LTT)** | Arbitrary non-monotone population risk control over $\lambda \in \Lambda$ | i.i.d. calibration set | Multiple hypothesis testing (fixed-sequence / FWER) | Finite-sample $(1-\delta)$ PAC risk control | E3 | General meta-framework; requires domain-specific valid finite-sample $p$-values / localized off-policy tail-risk bounds | **LOW-MEDIUM** (Enabling engine) |
| **Qi, Pang, Liu (JASA 2023)** & **Wang et al. (JASA 2018)** | CVaR-optimal / quantile-optimal ITRs | RCT / observational scalar outcomes | Empirical risk minimization / IPW / DR optimization | Asymptotic consistency & value convergence | E2 | No finite-sample safety certification or fallback abstention gate | **LOW** (Uncertified baseline B3) |
| **Kallus (Management Science 2023)** | CVaR of individual treatment effect (ITE-CVaR) | Observational / RCT scalar outcomes | Fréchet–Hoeffding bounds + debiased ML | Asymptotic confidence intervals for partial-ID bounds | E2 | Passive evaluation of ITE-CVaR, not finite-sample gated policy certification of $P_{L(\pi_\lambda)}$ | **LOW** |

### 4.4 What Is and Is Not Novel
- **NOT Novel:** Using conformal/LTT calibration for safe policy gating in general (Jin & Egami 2026; Prinster et al. 2026; Cao et al. 2026); finite-sample off-policy CVaR/quantile bounds for a *single* fixed policy (Huang et al. 2021; Chandak et al. 2021); estimating semicontinuous CATEs (`zicbcf`) or unit-level distributional effects (`WCF`).
- **Potentially Novel Layer (Must Be Proven Empirically at Gate G3):**
  1. **Localized Comparative Tail-Risk Certification:** Exploiting the Rockafellar–Uryasev baseline-VaR pivot $[\underline\eta_0, \overline\eta_0]$ and localized policy-contrast weights $\Delta w_\lambda(X, A) = 0$ on $S_\lambda^c$ to produce variance-localized finite-sample bounds on $\text{CVaR}_\beta(\pi_\lambda) - \text{CVaR}_\beta(\pi_0)$ and $Q_\beta(\pi_\lambda) - Q_\beta(\pi_0)$ (plus joint extensive-margin risk), breaking the $O(1/\sqrt{n})$ abstention dead-zone of two-arm OPRA/UnO.
  2. **Semicontinuous Two-Part Tail-Safety Scoring Integration:** Demonstrating empirically when two-part semicontinuous distributional scoring (`ZIC-BCF-Smear` / two-part quantile-CVaR learners) beats mean-CATE forests specifically under extensive/intensive margin conflict and heavy right tails.

### 4.5 Strongest Incumbent & Strongest Simple Baseline
- **Strongest Incumbent (Same Guarantee):** **B1 — Two-Arm OPRA/UnO + Fixed-Sequence LTT Gate** (using identical scoring function $s(x)$ and $\hat\pi(x)$, but bounding $\rho(\pi_\lambda)$ and $\rho(\pi_0)$ separately via Huang et al. 2021 / Chandak et al. 2021).
- **Strongest Simple Baseline (Scoring Ablation & Linear-Loss Incumbent):** **B2 — Mean-Optimal Causal Forest + Localized LCT-LTT Gate**, and **B5 — Fixed-Threshold Indicator-Loss CPC/LTT** (Prinster et al., 2026).

### 4.6 Gate G1 Verdict
- **Gate G1 Decision:** **`CONDITIONAL GO` (on Pivot A)** — while **original unpivoted idea = `KILL` / `INCREMENTAL-ONLY`**.
- **Two Bounded Conditions (Must Clear at Gates G2 and G3, Otherwise Terminate as `INCREMENTAL-ONLY` or `KILL`):**
  1. **Condition G1-1 (Primary Source Verification & Non-Collapse vs. B1):** Inspect exact theorems in `arXiv:2104.08977` (OPRA), `arXiv:2104.12820` (UnO), `arXiv:2603.02196` (CPC), and `arXiv:2609.17296` (CPL) during WP-2.1 to confirm none contains a localized comparative Rockafellar–Uryasev / localized CDF cancellation bound, AND verify at Gate G2/G3 that localized LCT-LTT yields a large certified-deployment advantage over B1.
  2. **Condition G1-2 (Non-Collapse vs. B2 & B5):** Verify at Gate G3 that semicontinuous distributional scoring beats B2 (mean-CATE forest) and tail-functional certification beats B5 (fixed-threshold indicator CPC) in substantively relevant mean-tail conflict regimes.

---

## 5. Impact Thesis and Skeptical-Referee Test

1. **Why the problem matters:** High-stakes interventions in healthcare financing, pricing/promotions, and enterprise AI routing routinely exhibit mean-tail divergence and massive zero-inflation. Deploying a personalized policy without finite-sample downside tail certification risks catastrophic expenditure spikes or SLA breaches; naive conservative certification abstains everywhere and forfeits safe personalized gains.
2. **What changes if the method works:** Decision-makers gain a drop-in post-hoc certification gate that turns any semicontinuous or distributional ITR learner into a finite-sample tail-safe policy capable of certifying targeted covariate deployments where existing two-arm off-policy bounds are vacuous.
3. **Who would use or cite it:** Causal inference & policy-learning researchers (JASA, Management Science, Biometrika), safe RL / conformal decision-making researchers (ICML, NeurIPS), and health-economics / applied experimentation teams evaluating heavy-tailed cost or utilization policies.
4. **Why a simpler incumbent is insufficient:** Plug-in CVaR/quantile ITRs (B3) violate tail safety in finite samples under heavy tails and model misspecification; mean-optimal forests (B4) ignore tail risk completely; scalar CPL/CPC (B5) cannot certify nonlinear CVaR or unbounded upper quantiles; and two-arm OPRA/UnO + LTT (B1) pays full $O(1/\sqrt{n})$ noise on the shared abstained region $S_\lambda^c$, collapsing to 100% abstention on targeted deployments.
5. **Why the contribution is not merely a combination:** If the method were merely "run `ZIC-BCF` then run OPRA," it *would* be an incremental combination and would be killed at Gate G3. Its claim to non-incremental methodology rests specifically on **localized comparative tail-risk certification** (the baseline-VaR Rockafellar–Uryasev pivot + localized CDF contrast with variance scaling $O(\mathbb{P}(S_\lambda)/n)$) combined with **joint extensive-margin + tail certification** for semicontinuous outcomes.
6. **Most damaging plausible referee paragraph:**
   > *"The authors combine well-known ingredients: off-policy CDF/CVaR concentration (Huang et al., 2021; Chandak et al., 2021), Learn-then-Test (Angelopoulos et al., 2025), and two-part causal forests (`ZIC-BCF`). Their main technical trick—canceling the abstained region via a baseline-VaR confidence interval—is a straightforward application of Rockafellar–Uryasev, still requires an artificial upper cap $B$ for CVaR, and degrades to asymptotic validity on observational data like MEPS. Meanwhile, in finite samples with $n_{\text{cal}} \le 5{,}000$ and $\beta = 0.95$, the localized bound either remains too wide to deploy meaningfully or produces nearly the same deployed region as a simple mean-CATE forest paired with the same gate."*
7. **Evidence needed to answer it:**
   - **Analytical + empirical proof of non-vacuity:** Show sharp certified deployment gains (e.g., $\ge +25$ percentage points certified deployment rate and $\ge 50\%$ of oracle safe welfare captured at $n_{\text{cal}} = 5{,}000$) where B1 (two-arm OPRA+LTT) certifies $< 5\%$.
   - **Boundedness-free upper-quantile track:** Highlight $Q_\beta$ certification alongside capped $\text{CVaR}_\beta$ so the framework does NOT depend solely on artificial truncation $B$.
   - **Clean scoring ablation (vs. B2):** Demonstrate quantitatively where two-part semicontinuous scoring beats mean-CATE forests by a large margin (and honestly report where mean-CATE suffices).
   - **Exact RCT flagship application (OHIE):** Anchor exact finite-sample certification on known lottery propensities (OHIE), using observational MEPS strictly as an honest propensity-sensitivity stress test.

### Impact-Dimension Rubric (Initial Triage — Pre-Simulation)

| Dimension | Score (0–3) | Justification / Status |
|---|---|---|
| Problem importance | **3** | Consequential policy/medical/operational problem (zero-inflated heavy-tailed downside risk under personalized assignment). |
| Novelty after prior art | **2 (Conditional)** | Defensible new layer (localized comparative tail-risk certification + joint extensive/tail semicontinuous gating), contingent on Conditions G1-1 & G1-2. |
| Mechanism or insight | **2** | Sharp, falsifiable variance-localization mechanism ($O(\mathbb{P}(S_\lambda)/n)$ vs. $O(1/n)$) + Rockafellar–Uryasev baseline-VaR pivot. |
| Empirical advantage | **UNTESTED** | Hypothesis to be tested at Gate G3 against B1–B5. |
| Applied value | **UNTESTED** | Hypothesis to be tested at Gate G4 on OHIE (primary) and MEPS (sensitivity). |
| Generality | **2** | Reusable post-hoc certification gate for any binary/multi-arm policy learner on semicontinuous or heavy-tailed outcomes. |
| Credibility | **2** | Strong after Gate G0 Pivot A repairs (RCT exactness, explicit cap $B$ for CVaR, boundedness-free $Q_\beta$, marginal/subgroup scope). |
| Paper coherence | **2** | Coherent once Regime B (`WCF`) and unavailable latency datasets are cut. |

---

## 6. Dependency Graph and Gate Map

```text
[CLEARED: PIVOT A] G0 Fatal Viability
   └─> [CLEARED: CONDITIONAL GO] G1 Prior Art & Impact Triage
          ├─> Phase 2: Minimum Enabling Formalization ──> [NEXT GATE] G2 Formalization Gate
          │                                                    │
          │                                                    ▼ (DORMANT UNTIL GATE G2)
          │                                            Phase 3: Prototype & Correctness Ladder
          │                                                    │
          │                                                    ▼ (DORMANT UNTIL GATE G2 + PHASE 3 PASS)
          │                                            Phase 4: Simulation-First Falsification ──> G3 Simulation Gate
          │                                                                                             │
          └─> [PARALLEL READ-ONLY ONLY] WP-5.0 OHIE/MEPS Feasibility Audit                              ▼ (DORMANT UNTIL GATE G3)
                                                                                                Phase 5: Applied Study (OHIE + MEPS) ──> G4 Applied Gate
                                                                                                                                             │
                                                                                                                                             ▼ (DORMANT UNTIL GATE G4)
                                                                                                                                     Phase 6: Reassess Impact ──> G5 Theory Gate
                                                                                                                                                                      │
                                                                                                                                                                      ▼ (DORMANT UNTIL GATE G5)
                                                                                                                                                              Phase 7: Deferred Theory
                                                                                                                                                                      │
                                                                                                                                                                      ▼ (DORMANT UNTIL GATE G5)
                                                                                                                                                              Phase 8: Paper & Venue Stress Test ──> G6 Submission Gate
```

### Project-Specific Gate Register

| Gate | Question | Evidence Required | `GO` Condition | `PIVOT` Condition | `INCREMENTAL-ONLY` Condition | `KILL` Condition |
|---|---|---|---|---|---|---|
| **G0 Fatal Viability** | Is the object coherent and identifiable enough to test? | Formal spec, counterexamples, typing audit | *(Failed for original seed)* | **PASSED VIA PIVOT A:** Regime A localized comparative LCT-LTT; explicit cap $B$ for CVaR; boundedness-free $Q_\beta$; known-propensity exactness | N/A | Contradiction or vacuity unresolved |
| **G1 Prior Art & Impact** | Is a consequential gap left after verified prior art? | E2/E3 nearest-neighbor table + repo checks | **CONDITIONAL GO (Pivot A):** Conditions G1-1 (E3 primary-paper check + non-collapse vs. B1) & G1-2 (non-collapse vs. B2/B5) bounded | Pivot B (Regime B simultaneous tail-curve) if Pivot A collapses | Only routine `ZIC-BCF` + OPRA wrapper remains | Direct hit found in E3 inspection of OPRA/CPC/CPL |
| **G2 Enabling Formalization** | Is LCT-LTT mathematically valid, implementable, and non-vacuous? | Exact algorithm, enabling validity lemma, E3 confirmation of G1-1, 1D analytic tightness pilot | Validity proof verified + toy numerical check shows empirical FWER $\le \delta + 0.01$ AND localized LCT-LTT certifies $\ge 25\%$ deployment where B1 certifies $\le 5\%$ at $n_{\text{cal}}=5000$ | Localized CVaR too conservative under heavy cap $B$, pivot primary target to boundedness-free $Q_\beta$ + extensive-margin safety | Localization improves deployment by $< 10$ percentage points over B1 | Validity lemma fails numerically or collapses to B1 |
| **G3 Simulation Engine** | Does LCT-LTT beat fair strong baselines (B1–B5) for the claimed reason? | Preregistered 10-DGP suite, 50 seeds, ablations, compute profile | Empirical tail-safety violation $\le \delta + 0.015$ across all known-propensity DGPs; $\ge +20$ pp certified deployment (or $\ge +25\%$ safe welfare gain) over B1 in targeted regimes; $\ge +15\%$ safe welfare gain over B2 in mean-tail conflict regimes; beats B5 on tail safety/welfare | Wins cleanly on $Q_\beta$ + extensive margin only (drop CVaR), or wins only in moderate-tail $\beta \in [0.80, 0.90]$ | Gains $< 5\text{--}10$ pp over B1 or B2, or fragile to tuning/cap $B$ | Violation exceeds $\delta + 0.03$, zero deployment across realistic $n_{\text{cal}}$, or B1/B2/B5 matches LCT-LTT |
| **G4 Applied Value** | Does it yield a credible, nontrivial result on OHIE (primary) & MEPS (sensitivity)? | OHIE ITT replication (Finkelstein et al. 2012), overlap/placebo checks, sensitivity, novel certified policy subgroup | Reproduces published OHIE ITT benchmark; placebo treatment-permutation test abstains $\ge 95\%$ of runs; certifies nontrivial personalized policy ($\ge 15\%$ population deployed) with credible tail-safety & welfare improvement over B1/B3/B4 | OHIE has insufficient tail heterogeneity; pivot primary application to another verified RCT with semicontinuous heavy-tailed outcomes | Purely descriptive showcase with identical policy to B2/B5 | Fails OHIE benchmark replication, fails placebo, or certifies $0\%$ deployment |
| **G5 Theory Investment** | Will substantial theory secure a surviving load-bearing claim? | Updated 0–3 impact table, proof-target & source map | `GO-THEORY`: Simulation & application both $\ge 2$ and finite-sample efficiency/power rate or DR observational bound secures top-venue claim; OR `GO-WRITE` if empirical engine suffices | Narrower theorem (e.g., quantile-only power bound) | Theory would only decorate an incremental empirical result | Core story failed G3/G4 |
| **G6 Submission Case** | Is the surviving project coherent at ICML/NeurIPS/JASA/Manag. Sci.? | Skeptical-referee pass, reproducibility audit, venue-fit map | Strong field-paper candidate (all core dimensions $\ge 2$, zero blockers) | Reframe venue (e.g., Biometrics / AOAS / AISTATS) | Technically valid but below user's top-venue bar | Central story indefensible |

---

## 7. Phase-by-Phase Execution Program

*(Phases 0 and 1 are completed in Sections 3 and 4 above. Execution begins at Phase 2 and read-only WP-5.0.)*

### Phase 2: Minimum Enabling Formalization — `STATUS: ACTIVE`

- **Purpose & scientific question:** Establish the exact enabling mathematics and unambiguous algorithm for Localized Comparative Tail-Risk Learn-then-Test (LCT-LTT), confirm E3 primary-source non-collision (Condition G1-1), and verify via a tiny numerical pilot that localization breaks the two-arm OPRA/UnO abstention dead-zone before building the full simulation harness.
- **Prerequisites:** Adoption of Gate G0 Pivot A.
- **Deliverables:** Formal enabling memo (`docs/phase2_formalization.md`), E3 primary-paper audit log (`docs/phase2_e3_prior_art_check.md`), and 1D tightness pilot report (`results/phase2_pilot/tightness_check.json`).
- **Gate evidence:** Enabling validity proposition + E3 non-collision verification + 1D numerical tightness pilot.
- **Gate G2 Criteria:** See Gate Register (G2).
- **Estimated compute & data:** Synthetic only; $< 5$ CPU minutes, $< 1$ GB RAM.
- **Risks & recovery:** If Hoeffding/empirical-Bernstein localized CVaR bounds remain too loose under heavy cap $B$, test Waudby-Smith & Ramdas (2024) betting-martingale bounds in WP-2.2; if still vacuous for CVaR while $Q_\beta$ succeeds, pivot primary target to boundedness-free $Q_\beta$ + extensive-margin safety at Gate G2.

#### Atomic Work Packages (Phase 2)

```text
WP ID and title: WP-2.1 — E3 Primary-Source Inspection & Enabling Validity Formalization
Status: ACTIVE
Gate served: G2 (and resolves Condition G1-1)
Objective: Inspect exact theorem statements in OPRA (arXiv:2104.08977), UnO (arXiv:2104.12820), CPC (arXiv:2603.02196), and CPL (arXiv:2609.17296), then formalize the exact LCT-LTT algorithm and prove the enabling finite-sample validity lemma for localized comparative CVaR (capped), upper-quantile (uncapped), and extensive-margin risk.
Why this changes a decision: Confirms whether localized comparative tail-risk certification is genuinely absent from the closest 2021–2026 primary sources and proves the algorithm outputs valid finite-sample (1 - delta) certificates without calibration leakage.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/research_idea.md
  - Primary PDFs/arXiv sources: 2104.08977, 2104.12820, 2603.02196, 2609.17296, 2610.11547, 2110.01052
Preconditions: Gate G0 Pivot A accepted.
Actions:
  1. Inspect theorem statements and bound formulas of OPRA (Huang et al. 2021), UnO (Chandak et al. 2021), CPC (Prinster et al. 2026), and CPL (Jin & Egami 2026); log exact estimand, bound formula, and whether S_lambda^c cancellation appears in docs/phase2_e3_prior_art_check.md.
  2. Formalize the exact two-stage calibration split: (a) finite-sample (1 - delta_1) CI [eta_lo, eta_hi] for baseline VaR eta_0* = Q_beta(L(pi_0)) via weighted CDF concentration (DKW/Anderson/betting); (b) localized comparative CVaR upper bound sup_{eta in [eta_lo, eta_hi]} UCB_{delta_2}(E[Delta w_lambda (L - eta)_+] / (1 - beta)) for L in [0, B]; (c) boundedness-free localized upper-quantile test on E[Delta w_lambda 1{L <= eta + eps}]; (d) localized extensive-margin test on E[Delta w_lambda 1{Y = 0}].
  3. Prove the enabling proposition (FWER <= delta under fixed-sequence testing over grid Lambda = {lambda_1 > ... > lambda_M} with exact known e(a|x), and explicit additive TV slack / Gamma-sensitivity adjustment for observational hat{e}(a|x)).
  4. Prove variance localization: Var(Delta w_lambda (L - eta)_+) = O(P(X in S_lambda) * B^2 / kappa), contrasting with O(B^2 / kappa) for separate two-arm bounds.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase2_e3_prior_art_check.md
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase2_formalization.md
Verification:
  - Mechanical: Both markdown documents exist, cite exact theorem numbers/equations from inspected primary papers, and contain complete step-by-step proofs with explicit quantifier scopes.
  - Scientific: Independent line-by-line algebra check confirms zero calibration leakage and exact finite-sample FWER <= delta under A1–A6.
Scientific pass rule: No direct hit for localized comparative Rockafellar–Uryasev / CDF cancellation in inspected primary theorems AND enabling validity lemma holds without hidden asymptotic approximations.
Scientific fail rule: Direct hit found in OPRA/UnO/CPC/CPL OR proof reveals an irreparable finite-sample leakage/validity flaw.
Gate consequence: Pass unlocks WP-2.2; Fail triggers immediate KILL (direct hit) or PIVOT (re-scope).
Dependencies: Gate G0, Gate G1.
Can run in parallel with: WP-5.0 (read-only OHIE/MEPS data feasibility audit).
Compute, RAM, and expected runtime: Minimal (local reading & symbolic checks; ~2–4 hours human/agent effort).
Likely trap: Accidentally taking a point estimate hat{eta}_0 on D_cal instead of a uniform supremum over a valid CI [eta_lo, eta_hi], or neglecting the supremum discretization error over eta in [eta_lo, eta_hi].
Recovery action: Exploit piecewise-linearity / monotonicity in eta (evaluating sup over the finite breakpoints {L_i} intersect [eta_lo, eta_hi] plus endpoints) to compute the exact supremum with zero discretization error.
```

```text
WP ID and title: WP-2.2 — Analytical & Toy 1D Non-Vacuity Tightness Pilot (LCT-LTT vs. Two-Arm OPRA+LTT)
Status: ACTIVE (immediately after WP-2.1)
Gate served: G2
Objective: Run a minimal, oracle-score 1D numerical pilot comparing finite-sample certified deployment rates of localized LCT-LTT (empirical-Bernstein and betting-martingale bounds) against two-arm OPRA/UnO + LTT across n_cal in {2000, 5000, 10000} and beta in {0.80, 0.90, 0.95}.
Why this changes a decision: Cheapest possible falsifier for the #1 empirical risk (zero-deployment conservatism) and Condition G1-1 (non-collapse vs. B1). If localized bounds still abstain everywhere or match B1 even with an oracle score s*(x), kill/pivot before building full learners or simulations.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase2_formalization.md
Preconditions: WP-2.1 passed.
Actions:
  1. Write a self-contained Python script scripts/phase2_tightness_pilot.py implementing: (i) localized comparative CVaR (capped) & upper-quantile LCT-LTT using both empirical-Bernstein and betting-martingale (Waudby-Smith & Ramdas, 2024) UCBs; (ii) two-arm OPRA/UnO + LTT baseline (B1).
  2. Evaluate on a clean semicontinuous log-normal hurdle DGP (40% zeros, known cap B = 50 for CVaR; uncapped for Q_beta) with true safe region P(S*) in {0.15, 0.35, 0.60}, overlap kappa = 0.25, tolerance eps = 0, delta = 0.05, across 200 Monte Carlo draws for n_cal in {2000, 5000, 10000} and beta in {0.80, 0.90, 0.95}.
  3. Record empirical FWER (violation rate), mean certified deployment fraction P(S_{hat_lambda}), and wall time per run.
  4. Write Gate G2 decision memo docs/gate_g2_memo.md.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/scripts/phase2_tightness_pilot.py
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase2_pilot/tightness_check.csv
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/gate_g2_memo.md
Verification:
  - Mechanical: Script runs deterministically from clean environment with fixed seed list (0..199); CSV contains all 2 x 3 x 3 x 3 grid cells with zero NaNs.
  - Scientific: Empirical violation rate <= 0.06 (nominal delta = 0.05 + MC SE tolerance) for both LCT-LTT and B1; localized LCT-LTT beats B1 in certified deployment.
Scientific pass rule: At n_cal = 5000 and beta = 0.90 (P(S*) = 0.35), localized LCT-LTT achieves empirical FWER <= 0.06 AND certifies >= 25% deployment while two-arm OPRA+LTT certifies <= 5% (or beats B1 by >= +20 percentage points).
Scientific fail rule: Both CVaR and Q_beta localized LCT-LTT certify < 10% deployment at n_cal = 10000, OR gain over B1 is < 10 percentage points.
Gate consequence: Pass clears Gate G2 and unlocks Phase 3; Partial pass (Q_beta passes, CVaR fails) triggers Pivot to Q_beta + extensive-margin primary target; Full fail triggers KILL/INCREMENTAL-ONLY.
Dependencies: WP-2.1.
Can run in parallel with: WP-5.0.
Compute, RAM, and expected runtime: <= 4 CPU cores, < 2 GB RAM, ~2–5 minutes total.
Likely trap: Choosing an overly dense candidate grid Lambda without fixed-sequence ordering (causing unnecessary Bonferroni penalty), or setting cap B orders of magnitude above the 99.9th percentile.
Recovery action: Use fixed-sequence testing (zero multiplicity penalty over Lambda) and evaluate sensitivity to realistic stop-loss caps B in {Q_0.99, 2*Q_0.99, 5*Q_0.99}.
```

---

### Phase 3: Prototype and Correctness Ladder — `STATUS: DORMANT UNTIL GATE G2`

- **Purpose & scientific question:** Build a clean, modular, deterministic Python/R-bridged prototype of LCT-LTT and all five baselines (B1–B5), verified through unit tests, special-case reductions, and a reproducible smoke run on a toy known-truth DGP.
- **Prerequisites:** Gate G2 `GO` (or `PIVOT` to quantile-primary scope).
- **Deliverables:** Installable package `src/safe_dist_policy/`, pytest suite `tests/`, smoke run report `results/phase3_smoke/smoke_summary.csv`.
- **Gate evidence:** 100% deterministic unit tests passing; special-case reductions verified; smoke run cleanly reproduces known oracle ordering.
- **Go / Pivot / Kill Criteria:** Must pass all mechanical and special-case correctness checks before launching Phase 4 sweeps (no separate numbered gate; internal Phase 3 readiness signoff).
- **Estimated compute & data:** $< 15$ minutes on 4 CPU cores, $< 4$ GB RAM.
- **Risks & recovery:** `zicbcf` runs in R (MCMC ~1–3 min per fit); for 50-seed multi-DGP sweeps in Phase 4, calling full MCMC `zicbcf` 500+ times is slow. *Recovery:* Implement a fast Python two-part semicontinuous distributional learner (`TwoPartQuantileGBDT` / `TwoPartLogNormalSmear`) as the fast primary sweep engine, and benchmark `ZIC-BCF-Smear` via `rpy2`/`Rscript` on a dedicated 20-seed comparative subset + OHIE/MEPS applications!

#### Atomic Work Packages (Phase 3)

```text
WP ID and title: WP-3.1 — Core Certification Package & Baseline Suite Implementation
Status: DORMANT UNTIL GATE G2
Gate served: G3 (readiness prerequisite)
Objective: Implement the modular safe_dist_policy package containing LCT-LTT (localized CVaR, upper-quantile, extensive-margin, joint Bonferroni/fixed-sequence testing, subgroup extension), fast two-part semicontinuous distributional learners, R bridge to zicbcf, and all five baselines (B1–B5).
Why this changes a decision: Guarantees equal data splits, equal propensity access, and fair apples-to-apples implementations across proposed method and all incumbents before any benchmark runs.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase2_formalization.md
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/scripts/phase2_tightness_pilot.py
  - /home/hugo/Stuff/Submitted_Research/zicbcf/R/ (interface reference)
Preconditions: Gate G2 cleared.
Actions:
  1. Create package layout under src/safe_dist_policy/: bounds.py (empirical-Bernstein, betting-martingale, weighted CDF CI), certification.py (LCT-LTT, subgroup LCT-LTT, B1 two-arm OPRA/UnO+LTT, B5 indicator CPC/LTT), learners.py (TwoPartDistributionalLearner, ZICBCFBridge, MeanCausalForest B2/B4, PlugInCVaRITR B3), dgps.py, and eval.py.
  2. Enforce a shared ExperimentSplit contract (D_tr, D_cal, D_eval) so every method receives identical splits, features, and propensity information.
  3. Pin environment dependencies in pyproject.toml and lockfile.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/pyproject.toml
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/src/safe_dist_policy/{bounds,certification,learners,dgps,eval}.py
Verification:
  - Mechanical: Package installs cleanly via pip/uv in isolated .venv; static type/lint checks pass with zero errors.
  - Scientific: All 5 baselines (B1–B5) and LCT-LTT run end-to-end on synthetic arrays and return schema-valid PolicyCertificate objects.
Scientific pass rule: Code matches Phase 2 equations exactly with zero data leakage between D_tr and D_cal.
Scientific fail rule: Any comparator uses different sample splits, uncalibrated hyperparameters, or leaks D_cal into learner fitting.
Gate consequence: Unlocks WP-3.2.
Dependencies: Gate G2.
Can run in parallel with: None (core codebase foundation).
Compute, RAM, and expected runtime: Dev/test only (< 5 min runtime, < 2 GB RAM).
Likely trap: Mixing up loss vs. utility sign conventions across healthcare cost (minimize loss) vs. spend/reward (maximize utility) learners.
Recovery action: Enforce strict internal canonicalization where all certification modules operate exclusively on non-negative loss L >= 0 (smaller is safer) with explicit utility U passed separately.
```

```text
WP ID and title: WP-3.2 — Deterministic Unit Tests, Special-Case Reductions & Smoke Run
Status: DORMANT UNTIL GATE G2
Gate served: G3 (readiness prerequisite)
Objective: Verify mathematical correctness via automated unit tests, exact special-case reductions, and a 5-seed smoke test comparing LCT-LTT against Oracle and B1–B5.
Why this changes a decision: Prevents silent bugs (e.g., off-by-one quantile indices, wrong propensity weights, leaked calibration thresholds) from corrupting expensive Phase 4 simulations.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/src/safe_dist_policy/
Preconditions: WP-3.1 complete.
Actions:
  1. Write tests/test_certification_correctness.py testing: (a) lambda = +inf always returns pi_0 with zero risk difference; (b) when P(Y=0) > beta, CVaR reduces numerically to E[Y]/(1 - beta); (c) constant-outcome noiseless DGP recovers exact oracle safe set; (d) exact invariance to row permutation of D_cal; (e) supremum over eta in [eta_lo, eta_hi] matches brute-force dense grid to < 1e-10; (f) null DGP (A independent of Y) respects nominal FWER <= delta across 500 fast synthetic draws.
  2. Run a 5-seed end-to-end smoke benchmark on DGP-1 (clean mean-tail conflict) and log runtime and peak RSS memory per seed.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/tests/test_certification_correctness.py
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase3_smoke/smoke_summary.csv
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase3_smoke/resource_profile.json
Verification:
  - Mechanical: pytest passes 100% of tests deterministically; resource_profile.json reports per-seed wall time and peak RAM.
  - Scientific: Null FWER in [0, delta + 0.015]; noiseless oracle recovery error = 0; runtime estimate informs Phase 4 worker caps.
Scientific pass rule: 100% unit tests pass and smoke benchmark produces valid non-NaN metrics for all methods.
Scientific fail rule: Any validity violation on null DGP or failed special-case reduction.
Gate consequence: Unlocks Phase 4 simulation suite.
Dependencies: WP-3.1.
Can run in parallel with: None.
Compute, RAM, and expected runtime: <= 4 CPU cores, < 4 GB RAM, ~5–10 minutes.
Likely trap: Ignoring Monte Carlo precision in null-FWER unit test (e.g., running only 20 draws).
Recovery action: Run 500 fast null draws with analytic oracle scores so MC standard error is < 0.01.
```

---

### Phase 4: Simulation-First Falsification — `STATUS: DORMANT UNTIL GATE G2 (AND PHASE 3 PASS)`

- **Purpose & scientific question:** Stress-test LCT-LTT against B1–B5 across a preregistered 10-scenario mechanistic DGP ladder (see Section 8) to establish whether LCT-LTT has a reproducible, practically important advantage driven by the claimed mechanisms (variance localization + two-part distributional scoring), and map its exact failure regions.
- **Prerequisites:** Gate G2 cleared + WP-3.2 passed.
- **Deliverables:** Preregistered simulation protocol (`docs/phase4_preregistration.md`), audited multi-seed simulation parquet/CSV tables (`results/phase4_sim/`), ablation figures, and standalone Gate G3 memo (`docs/gate_g3_memo.md`).
- **Gate evidence:** 50-seed results across all 10 DGPs with 95% MC CIs, worst-regime tables, ablations, and scaling profile.
- **Gate G3 Criteria:** See Gate Register (G3).
- **Estimated compute & data:** Pilot-calibrated; estimated 1–3 hours total on 6–8 physical cores (capped safely below 10 physical cores on i9-13900H, `OMP_NUM_THREADS=1`, peak RAM < 16 GB), with per-DGP checkpointing; `ZIC-BCF-Smear` 20-seed comparative slice routed to Colab shards if > 30 min.
- **Risks & recovery:** Weak-baseline illusion (mitigated by equal tuning budgets and strong B1/B2/B5); heavy-tail cap sensitivity (tested explicitly in DGP-8).

#### Atomic Work Packages (Phase 4)

```text
WP ID and title: WP-4.1 — Preregistration & Core Mechanism Sweep (DGPs 1–5: Recovery, Null, Baseline-Favorable, Mechanism-Favorable, Crossover)
Status: DORMANT UNTIL GATE G2
Gate served: G3
Objective: Freeze the simulation protocol (metrics, thresholds, seed policy, tuning budgets) and execute the 50-seed comparative sweep across DGPs 1–5 comparing LCT-LTT against Oracle and B1–B5.
Why this changes a decision: Tests directly whether localized certification beats B1 (two-arm OPRA+LTT) and whether distributional scoring beats B2 (mean-CATE forest) and B5 (indicator CPC) without post-hoc tuning.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/src/safe_dist_policy/
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase3_smoke/resource_profile.json
Preconditions: WP-3.2 passed.
Actions:
  1. Freeze preregistration document docs/phase4_preregistration.md specifying exact DGP parameters, primary/secondary metrics, kill thresholds, seed list (1000..1049), and shared tuning budgets.
  2. Run scripts/run_phase4_core.py across DGPs 1–5 (50 seeds each) using process-level parallelism capped at <= 6 workers (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1) with per-seed atomic JSON/Parquet checkpointing and resume support.
  3. Aggregate results with mean, median, IQR, 95% MC CIs, and empirical FWER violation rates.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase4_preregistration.md
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/scripts/run_phase4_core.py
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase4_sim/core_dgp1_5_r50.parquet
Verification:
  - Mechanical: All 5 DGPs x 50 seeds x 7 methods (Oracle + LCT-LTT + B1–B5) present with zero missing rows; checksum verified.
  - Scientific: Null DGP-3 empirical FWER <= 0.065; clean recovery DGP-1 matches theoretical scaling.
Scientific pass rule: Meets Gate G3 primary thresholds on DGPs 1–5 (empirical FWER <= 0.065; >= +20 pp certified deployment over B1 on targeted regimes; >= +15% safe welfare over B2 on mean-tail conflict DGP-4).
Scientific fail rule: Empirical FWER > 0.08 on any known-propensity DGP OR fails to beat B1/B2/B5 by predeclared thresholds.
Gate consequence: Feeds directly into WP-4.3 Gate G3 memo; severe validity failure halts WP-4.2 immediately for debugging/kill.
Dependencies: WP-3.2.
Can run in parallel with: WP-4.2 (only after confirming WP-4.1 pilot seed passes).
Compute, RAM, and expected runtime: <= 6 physical cores, < 12 GB RAM, ~45–90 minutes with checkpointing.
Likely trap: Nested BLAS/OpenMP threading inside scikit-learn/grf workers causing CPU thrashing and RAM spikes.
Recovery action: Hard-set OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 in worker entrypoints and verify htop/psutil memory before scaling workers.
```

```text
WP ID and title: WP-4.2 — Stress-Test, Misspecification, ZIC-BCF Subset & Scaling Sweep (DGPs 6–10)
Status: DORMANT UNTIL GATE G2
Gate served: G3
Objective: Run DGPs 6–10 covering crossover grid (sample size x safe-region size), nuisance/propensity misspecification, heavy-tail cap sensitivity & OHIE/MEPS-mimicking defects, ZIC-BCF-Smear comparative subset (20 seeds), and runtime/memory scaling.
Why this changes a decision: Maps exact failure boundaries (weak overlap, estimated observational propensities, extreme Pareto tails) and verifies whether ZIC-BCF-Smear adds value over faster two-part GBDT learners.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase4_preregistration.md
  - /home/hugo/Stuff/Submitted_Research/zicbcf/
Preconditions: WP-3.2 passed; WP-4.1 1-seed check valid.
Actions:
  1. Execute scripts/run_phase4_stress.py on DGPs 6–9 (50 seeds for Python learners; 20 seeds for R ZIC-BCF-Smear via checkpointed subprocess or Colab shard notebook notebooks/phase4_zicbcf_colab.ipynb with mandatory google.colab.files.download fallback).
  2. Execute DGP-10 runtime and peak-RAM scaling over n in {1000, 5000, 20000, 100000}.
  3. Record empirical degradation under estimated propensities (with and without Gamma-sensitivity inflation) and heavy-tail cap B sweeps.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/scripts/run_phase4_stress.py
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/notebooks/phase4_zicbcf_colab.ipynb
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase4_sim/stress_dgp6_10.parquet
Verification:
  - Mechanical: All shards/seeds complete; schema matches WP-4.1; Colab notebook includes download fallback.
  - Scientific: Quantifies exact crossover sample size n* and overlap threshold kappa* where LCT-LTT beats B1/B2 and where it degrades gracefully to abstention.
Scientific pass rule: Under known propensities and realistic caps, LCT-LTT maintains FWER <= 0.065 and scales sub-linearly/linearly in memory; under estimated propensities, Gamma-adjusted LCT-LTT restores validity.
Scientific fail rule: Uncontrolled FWER breakdown under mild nuisance misspecification or prohibitive O(n^2) memory blowup.
Gate consequence: Feeds into WP-4.3 Gate G3 memo.
Dependencies: WP-3.2.
Can run in parallel with: WP-4.1 (respecting total machine core cap <= 6–8 cores).
Compute, RAM, and expected runtime: Python stress sweep ~45–60 min on 6 cores (< 12 GB RAM); ZIC-BCF 20-seed subset ~2–4 hours on Colab shards.
Likely trap: Comparing untuned baselines against a heavily tuned LCT-LTT learner.
Recovery action: Apply identical 3-fold CV hyperparameter grids across LCT-LTT, B1, B2, B3, B4, and B5.
```

```text
WP ID and title: WP-4.3 — Adversarial Analysis & Standalone Gate G3 Memo
Status: DORMANT UNTIL GATE G2
Gate served: G3
Objective: Perform an adversarial synthesis of all Phase 4 outputs (identifying B1/B2/B5's best cases, LCT-LTT's failure regions, and ablation evidence) and issue one hard Gate G3 verdict.
Why this changes a decision: Hard kill/pivot gate preventing investment in an applied study or theory if fair baselines erase the claimed advantage.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase4_preregistration.md
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase4_sim/core_dgp1_5_r50.parquet
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase4_sim/stress_dgp6_10.parquet
Preconditions: WP-4.1 and WP-4.2 complete.
Actions:
  1. Generate diagnostic tables and crossover plots under results/phase4_sim/figures/.
  2. Write standalone docs/gate_g3_memo.md containing all 8 required items from plan-contract Section 4.4 (preregistered expectations, tables/plots, deviations, strongest baseline's best case, proposed method's failure region, ablation evidence, practical effect sizes, single Gate G3 decision).
  3. Explicitly check whether ZIC-BCF-Smear justifies its runtime over fast two-part GBDT; drop ZIC-BCF if it adds < 3% welfare gain.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/gate_g3_memo.md
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase4_sim/figures/crossover_deployment_welfare.pdf
Verification:
  - Mechanical: Memo contains all 8 required sections and links to exact parquet tables and figures.
  - Scientific: Gate G3 decision strictly obeys preregistered thresholds without post-hoc moving of goalposts.
Scientific pass rule: Gate G3 = GO (or scientifically meaningful PIVOT, e.g., quantile-only).
Scientific fail rule: Gate G3 = INCREMENTAL-ONLY or KILL -> terminate execution and write diagnostic archive.
Gate consequence: GO/PIVOT unlocks Phase 5 confirmatory applied study; INCREMENTAL-ONLY or KILL stops the project.
Dependencies: WP-4.1, WP-4.2.
Can run in parallel with: None.
Compute, RAM, and expected runtime: < 5 minutes, < 4 GB RAM.
Likely trap: Reporting statistical significance (p < 0.05 across 50 seeds) for a trivial 2 percentage-point deployment difference.
Recovery action: Evaluate strictly against preregistered practical effect thresholds (+20 pp deployment over B1, +15% safe welfare over B2).
```

---

### Phase 5: Applied-Study Gate — `STATUS: WP-5.0 ACTIVE (READ-ONLY); WP-5.1–5.3 DORMANT UNTIL GATE G3`

- **Purpose & scientific question:** Test whether LCT-LTT yields a credible, nontrivial policy finding on real semicontinuous healthcare expenditure data—anchored by the **Oregon Health Insurance Experiment (OHIE)** randomized Medicaid lottery (known propensity $e(x)$ for exact finite-sample ITT safety) and stress-tested on observational **MEPS 2023 HC-251** (propensity-sensitivity study).
- **Prerequisites:** WP-5.0 (read-only feasibility check) may run immediately; confirmatory WP-5.1–5.3 are strictly `DORMANT UNTIL GATE G3`.
- **Deliverables:** Read-only feasibility report (`docs/phase5_data_feasibility.md`), frozen preprocessing & replication scripts (`scripts/applied_ohie_*.py`, `scripts/applied_meps_*.py`), and Gate G4 memo (`docs/gate_g4_memo.md`).
- **Gate evidence:** Benchmark replication (Finkelstein et al., 2012 QJE for OHIE; `zicbcf` MEPS benchmark), placebo permutation test, overlap/sensitivity diagnostics, and comparison against B1–B5.
- **Gate G4 Criteria:** See Gate Register (G4).
- **Estimated compute & data:** OHIE public-use files (~50–200 MB from NBER); MEPS HC-251 already local (60 MB); < 1 hour CPU runtime on 4–6 cores, < 8 GB RAM.
- **Risks & recovery:** OHIE has lottery noncompliance (distinguish ITT lottery-offer policy from IV Medicaid enrollment); if OHIE shows too little covariate heterogeneity in spending tails, inspect alternative verified public RCTs with semicontinuous heavy-tailed outcomes at Gate G4 rather than forcing MEPS observational data as an exact finite-sample flagship.

#### Atomic Work Packages (Phase 5)

```text
WP ID and title: WP-5.0 — Early Read-Only Data Feasibility Audit (OHIE & MEPS)
Status: ACTIVE (READ-ONLY FEASIBILITY ONLY — NO MODEL TUNING OR OUTCOME PEEKING)
Gate served: G4 (feasibility prerequisite; informs Phase 4 DGP-9 realistic parameters)
Objective: Audit access conditions, exact schemas, sample sizes, zero-inflation rates, propensity stratification (household size in OHIE), and missingness of OHIE public-use files and local MEPS 2023 HC-251 without fitting policy learners or peeking at subgroup treatment effects.
Why this changes a decision: Prevents building an entire simulation suite around data regimes (e.g., sample size, zero-mass, tail support) that do not exist in the target public datasets.
Inputs:
  - NBER OHIE public-use data dictionary / download terms
  - /home/hugo/Stuff/Submitted_Research/zicbcf/applied_study/README.md
  - /home/hugo/Stuff/Submitted_Research/zicbcf/applied_study/meps_2023_dental_hurdle_model_data.csv
Preconditions: None (read-only audit permitted prior to Gate G3 per plan-contract Section 5.1).
Actions:
  1. Verify OHIE public-use downloadability, license/use terms, sample size of administrative/survey expenditure modules, exact randomization stratification variable (numhh_list), zero-inflation rate, and right-tail quantiles (unstratified by treatment arm to avoid peeking!).
  2. Audit local MEPS 2023 HC-251 schema (n = 18,763), zero-inflation rate, positive expenditure quantiles, propensity overlap diagnostics (covariates vs. dental insurance only, no outcome regressions), and survey-weight design variables (PERWT23F, VARSTR, VARPSU).
  3. Record realistic summary statistics (zero mass, skewness, sample size, propensity range kappa) to parameterize DGP-9 in Phase 4.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase5_data_feasibility.md
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase5_feasibility/data_audit_summary.json
Verification:
  - Mechanical: JSON contains verified n, zero_fraction, covariate counts, propensity kappa range, and license terms for both OHIE and MEPS.
  - Scientific: Confirms P(Y = 0) < beta (so CVaR/Q_beta target the positive continuous tail!) and n_cal >= 3000 is available for OHIE.
Scientific pass rule: OHIE public-use expenditure/utilization data is accessible with n >= 6000 (allowing >= 3000 calibration units) and known randomization weights; P(Y = 0) < 0.85.
Scientific fail rule: OHIE expenditure module is restricted-access or too small (n < 2000) -> immediately flag need to identify an alternative public RCT before Gate G3.
Gate consequence: Informs DGP-9 parameters; data-access failure triggers early dataset pivot.
Dependencies: None.
Can run in parallel with: WP-2.1, WP-2.2.
Compute, RAM, and expected runtime: < 10 minutes, < 2 GB RAM.
Likely trap: Accidentally inspecting treatment-by-subgroup outcome contrasts during feasibility audit (undermining confirmatory integrity).
Recovery action: Strictly prohibit joining treatment indicator with outcomes in WP-5.0 scripts; inspect marginal outcome distribution and propensity distribution separately.
```

```text
WP ID and title: WP-5.1 — OHIE Primary Confirmatory Study: Benchmark Replication, Safe Policy Gating & Placebos
Status: DORMANT UNTIL GATE G3
Gate served: G4
Objective: Freeze OHIE preprocessing, reproduce the published Finkelstein et al. (2012, QJE) ITT lottery benchmark, run LCT-LTT vs. B1–B5 for ITT catastrophic out-of-pocket spending / ED-cost tail safety, and execute permutation placebos.
Why this changes a decision: Provides the flagship real-world test under EXACT known randomization propensities e(x) (household-size stratified lottery), validating both finite-sample safety and practical policy insight.
Inputs:
  - Verified OHIE public-use files
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/src/safe_dist_policy/
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase5_data_feasibility.md
Preconditions: Gate G3 = GO/PIVOT; WP-5.0 passed.
Actions:
  1. Freeze preprocessing rules, inclusion/exclusion criteria, train/cal/eval stratified splits (preserving household-size lottery strata and household clustering!), and cap B (for CVaR; uncapped for Q_0.90/Q_0.95) in docs/phase5_ohie_protocol.md.
  2. Reproduce Finkelstein et al. (2012) ITT mean & extensive-margin utilization/catastrophic-spending effects within published SE tolerance.
  3. Fit LCT-LTT and baselines B1–B5; evaluate certified deployment rate, held-out evaluation-split tail risk (CVaR, Q_0.90, Q_0.95, extensive margin), and mean welfare across 20 repeated random train/cal/eval splits (cluster-level splitting by household_id).
  4. Run 100 treatment-permutation placebo tests (permuting lottery indicator within household-size strata) and verify LCT-LTT abstains (>= 95% of runs).
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/phase5_ohie_protocol.md
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/scripts/applied_ohie_run.py
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase5_applied/ohie_results.parquet
Verification:
  - Mechanical: All 20 splits + 100 placebos complete; household_id cluster isolation verified by assertion (zero household leakage across splits).
  - Scientific: Benchmark ITT matches published Finkelstein et al. estimate; placebo abstention rate >= 95%.
Scientific pass rule: Reproduces OHIE benchmark, passes placebo, and certifies a nontrivial safe personalized policy (>= 15% deployed) with credible improvement over B1/B3/B4.
Scientific fail rule: Fails benchmark replication, violates placebo, or abstains 100% across splits.
Gate consequence: Feeds directly into WP-5.3 Gate G4 memo.
Dependencies: Gate G3, WP-5.0.
Can run in parallel with: WP-5.2.
Compute, RAM, and expected runtime: <= 4 cores, < 8 GB RAM, ~20–40 minutes.
Likely trap: Ignoring household-level clustering in OHIE (multiple household members share lottery assignment!), which violates unit-level i.i.d. exchangeability.
Recovery action: Operate at the **household unit** (or randomly sample one adult per household / household-clustered weighting) so calibration units are strictly independent!
```

```text
WP ID and title: WP-5.2 — MEPS 2023 Secondary Observational Sensitivity Study
Status: DORMANT UNTIL GATE G3
Gate served: G4
Objective: Apply LCT-LTT with Gamma-sensitivity propensity bounds to MEPS 2023 dental expenditure (reproducing zicbcf's analytic-sample benchmark) to evaluate observational robustness under estimated propensities and survey-weight heterogeneity.
Why this changes a decision: Tests whether the framework gracefully handles observational confounding and propensity estimation error via Gamma-sensitivity curves rather than making brittle exact finite-sample claims.
Inputs:
  - /home/hugo/Stuff/Submitted_Research/zicbcf/applied_study/meps_2023_dental_hurdle_model_data.csv
  - /home/hugo/Stuff/Submitted_Research/zicbcf/applied_study/ate_results.csv
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/src/safe_dist_policy/
Preconditions: Gate G3 = GO/PIVOT; WP-5.0 passed.
Actions:
  1. Reproduce the unweighted analytic-sample ATE and hurdle-margin effect from zicbcf/applied_study/ate_results.csv.
  2. Run LCT-LTT and B1–B5 across propensity-sensitivity levels Gamma in {1.0, 1.05, 1.10, 1.25, 1.50}, explicitly reporting the critical sensitivity threshold Gamma* at which certified deployment collapses to 0%.
  3. Conduct a survey-design sensitivity check comparing unweighted analytic-sample policies against PERWT23F-weighted target evaluation (explicitly documenting why survey clustering breaks i.i.d. exchangeability).
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/scripts/applied_meps_sensitivity.py
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase5_applied/meps_sensitivity.parquet
Verification:
  - Mechanical: Reproduces zicbcf ATE within tolerance; all Gamma levels logged.
  - Scientific: Monotone shrinkage of certified deployment region as Gamma increases; clear interpretation boundaries stated.
Scientific pass rule: Clean reproduction + informative Gamma-sensitivity frontier showing policy stability under plausible observational propensity error.
Scientific fail rule: Severe overlap failure (kappa < 0.02) or immediate collapse at Gamma = 1.05 indicating non-viable observational transport.
Gate consequence: Secondary robustness evidence for WP-5.3 Gate G4 memo.
Dependencies: Gate G3, WP-5.0.
Can run in parallel with: WP-5.1.
Compute, RAM, and expected runtime: <= 4 cores, < 8 GB RAM, ~15–30 minutes.
Likely trap: Claiming exact finite-sample validity on observational MEPS data or ignoring survey design weights.
Recovery action: Label MEPS explicitly as an observational sensitivity benchmark (analytic sample + survey-weighted comparison), never as proof of exact finite-sample guarantees.
```

```text
WP ID and title: WP-5.3 — Applied-Study Synthesis & Gate G4 Memo
Status: DORMANT UNTIL GATE G3
Gate served: G4
Objective: Synthesize OHIE and MEPS applied evidence, audit all anomalies before calling them discoveries, and issue a single Gate G4 verdict.
Why this changes a decision: Determines whether empirical evidence supports a strong field paper and justifies Phase 6 impact reassessment.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase5_applied/{ohie_results,meps_sensitivity}.parquet
Preconditions: WP-5.1 and WP-5.2 complete.
Actions:
  1. Compile all 6 required items of the Application Gate Memo (plan-contract Section 5.3) in docs/gate_g4_memo.md.
  2. Verify whether the certified OHIE policy reveals a substantively interpretable subgroup pattern (e.g., chronic-condition / baseline-utilization strata) unavailable to mean-optimal policies.
  3. Issue one Gate G4 decision (GO / PIVOT / INCREMENTAL-ONLY / KILL).
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/gate_g4_memo.md
Verification:
  - Mechanical: Memo links all tables, placebo checks, and benchmark comparisons.
  - Scientific: Decision strictly matches Gate G4 criteria.
Scientific pass rule: Gate G4 = GO (or clean PIVOT).
Scientific fail rule: Gate G4 = INCREMENTAL-ONLY or KILL -> stop project.
Gate consequence: Unlocks Phase 6.
Dependencies: WP-5.1, WP-5.2.
Can run in parallel with: None.
Compute, RAM, and expected runtime: < 5 minutes.
Likely trap: Treating a noisy subgroup contrast as a scientific discovery without checking stability across repeated splits.
Recovery action: Require >= 80% split-stability agreement on subgroup deployment characterization.
```

---

### Phase 6: Reassess Impact Before Theory — `STATUS: DORMANT UNTIL GATE G4`

- **Purpose & scientific question:** Re-score all 8 impact dimensions using actual Gate G3 and Gate G4 evidence, answer *"Why not use B1 (OPRA+LTT), B2 (mean-CATE forest + LCT-LTT), or B5 (indicator CPC)?"* strictly from data, and decide whether substantial theory earns its cost (`GO-THEORY`, `GO-WRITE`, `PIVOT`, `INCREMENTAL-ONLY`, or `KILL`).
- **Prerequisites:** Gate G4 cleared.
- **Deliverables:** `docs/gate_g5_impact_reassessment.md`.
- **Gate G5 Criteria:** See Gate Register (G5).

#### Atomic Work Package (Phase 6)

```text
WP ID and title: WP-6.1 — Post-Evidence Impact Audit & Theory Investment Decision (Gate G5)
Status: DORMANT UNTIL GATE G4
Gate served: G5
Objective: Update the 0–3 impact rubric with actual simulation and applied scores, draft the evidence-backed one-sentence contribution, and choose exactly one Gate G5 decision (GO-THEORY, GO-WRITE, PIVOT, INCREMENTAL-ONLY, KILL).
Why this changes a decision: Prevents sinking weeks into decorative theorems if empirical results already suffice (GO-WRITE) or failed to clear the strong-paper bar (INCREMENTAL-ONLY/KILL).
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/{gate_g3_memo,gate_g4_memo}.md
Preconditions: Gate G4 cleared.
Actions:
  1. Re-score all 8 impact dimensions from gate-rubric.md using observed G3/G4 evidence.
  2. Write a 1-paragraph direct answer to "Why not use B1, B2, or B5?" backed exclusively by observed metrics.
  3. Rank candidate theory targets (Section 10) by decision value; cut any target that does not explain an observed phenomenon or protect reported inference.
  4. Issue one Gate G5 verdict.
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/gate_g5_impact_reassessment.md
Verification:
  - Mechanical: Updated 8-dimension table + explicit Gate G5 decision present.
  - Scientific: No score relies on unverified promises; every retained theory target cites an observed G3/G4 phenomenon.
Scientific pass rule: GO-THEORY (unlocks Phase 7) or GO-WRITE (skips Phase 7, proceeds to Phase 8).
Scientific fail rule: INCREMENTAL-ONLY or KILL -> terminate.
Gate consequence: Controls entry into Phase 7 (theory) vs. Phase 8 (paper stress test).
Dependencies: WP-5.3.
Can run in parallel with: None.
Compute, RAM, and expected runtime: Minimal (~1 hour analysis).
Likely trap: Choosing GO-THEORY by habit when the empirical engine is already complete or weak.
Recovery action: Apply the strict test: "Does this theorem explain the observed B1 vs. LCT-LTT power gap or validate the MEPS Gamma-sensitivity inference?" If no, cut it.
```

---

### Phase 7: Evidence-Earned Theory — `STATUS: DORMANT UNTIL GATE G5 (ONLY IF GO-THEORY)`

- **Purpose & scientific question:** Prove only the high-value theoretical results earned by Gates G3–G4 (see Section 10 for exact source-mapped targets), invoking `math-theory-dev-plan` and numerical falsification scripts before formal proof writing.
- **Prerequisites:** Gate G5 = `GO-THEORY`.
- **Deliverables:** Verified theorem proofs (`theory/proofs.tex` + `theory/numerical_falsifiers.py`).
- **Estimated compute & data:** Symbolic + numerical falsification scripts (< 30 min CPU, < 4 GB RAM).

#### Atomic Work Packages (Phase 7)

```text
WP ID and title: WP-7.1 — Finite-Sample Validity & Power-Localization Rate Theorem (Targets T1 & T2)
Status: DORMANT UNTIL GATE G5
Gate served: G6
Objective: Prove exact finite-sample FWER <= delta for joint (CVaR, Q_beta, extensive-margin) localized LCT-LTT (T1) and derive the sharp power / certified-deployment rate showing O(sqrt(P(S_lambda)/n)) localized width vs. O(sqrt(1/n)) two-arm width (T2), with numerical falsification scripts.
Why this changes a decision: Formalizes the exact mathematical explanation for why LCT-LTT breaks the two-arm OPRA/UnO abstention dead-zone observed in Phase 4.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/{phase2_formalization,gate_g5_impact_reassessment}.md
Preconditions: Gate G5 = GO-THEORY.
Actions:
  1. Run math-theory-dev-plan validity audit on Targets T1 and T2.
  2. Map every assumption line-by-line to Rockafellar & Uryasev (2000), Angelopoulos et al. (AOAS 2025, LTT), and Waudby-Smith & Ramdas (JRSS-B 2024) / Maurer & Pontil (2009) empirical-Bernstein inequalities.
  3. Write numerical falsification tests checking tightness of theoretical constants and power-rate scaling across P(S_lambda) in {0.05, 0.10, 0.25, 0.50} and n in {10^3, 10^4, 10^5}.
  4. Write complete proofs (clean LaTeX, no em dashes, no bullet lists per style rules).
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/theory/proof_T1_T2.tex
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/theory/falsify_T1_T2.py
Verification:
  - Mechanical: Falsification script passes with zero counterexamples; LaTeX compiles cleanly.
  - Scientific: All constants explicit; scaling exponents match empirical Phase 4 slopes within 95% CI.
Scientific pass rule: Complete, independently audited proofs with zero counterexamples.
Scientific fail rule: Counterexample found -> tighten assumption or downgrade theorem tag honestly.
Gate consequence: Feeds into Phase 8 submission case.
Dependencies: Gate G5.
Can run in parallel with: WP-7.2.
Compute, RAM, and expected runtime: < 20 minutes CPU, < 4 GB RAM.
Likely trap: Claiming uniform rate over uncountable Lambda without covering-number / bracketing control when using non-fixed-sequence selection.
Recovery action: Prove T1–T2 for finite pre-specified grid Lambda with fixed-sequence testing (exact, no covering number needed), and state any continuous-Lambda extension strictly as a clearly tagged corollary.
```

```text
WP ID and title: WP-7.2 — Observational Propensity-Error Sensitivity & Doubly Robust Asymptotic Safety (Target T3)
Status: DORMANT UNTIL GATE G5
Gate served: G6
Objective: Prove finite-sample validity under bounded multiplicative propensity misspecification (Gamma-sensitivity model) and doubly robust asymptotic safety when either propensity or conditional tail-excess nuisance is consistently estimated.
Why this changes a decision: Protects the interpretation of the MEPS observational study and answers referee questions about observational deployment.
Inputs:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/results/phase5_applied/meps_sensitivity.parquet
Preconditions: Gate G5 = GO-THEORY (and MEPS sensitivity retained in paper).
Actions:
  1. Adapt Jin, Ren, & Candès (2023) / Lei & Candès (2021) sensitivity bounds to localized signed contrast weights Delta w_lambda.
  2. Derive doubly robust orthogonalized localized tail-excess score and prove asymptotic validity under standard product-rate nuisance error o_p(n^{-1/2}).
  3. Verify numerically on DGP-7 (propensity misspecification).
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/theory/proof_T3_sensitivity_dr.tex
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/theory/falsify_T3.py
Verification:
  - Mechanical: Script passes; LaTeX compiles cleanly.
  - Scientific: Theoretical Gamma-slack upper-bounds observed empirical miscoverage on DGP-7.
Scientific pass rule: Proof verified and numerically tight.
Scientific fail rule: DR orthogonality breaks due to non-differentiability of (L - eta)_+ at point masses -> restrict to continuous positive part with explicit atom condition.
Gate consequence: Feeds into Phase 8 submission case.
Dependencies: Gate G5.
Can run in parallel with: WP-7.1.
Compute, RAM, and expected runtime: < 20 minutes CPU, < 4 GB RAM.
Likely trap: Assuming naive plug-in AIPW preserves exact finite-sample guarantees (it does not; only asymptotic DR!).
Recovery action: Keep exact finite-sample Gamma-sensitivity theorem strictly separate from asymptotic DR theorem.
```

---

### Phase 8: Paper and Venue Stress Test — `STATUS: DORMANT UNTIL GATE G5`

- **Purpose & scientific question:** Assemble the smallest coherent paper around the surviving contribution, cut all decorative modules, run a skeptical-referee simulation and reproducibility audit, and assess realistic venue fit (ICML / NeurIPS / JASA / Management Science) without predicting acceptance.
- **Prerequisites:** Gate G5 = `GO-WRITE` or Phase 7 complete.
- **Deliverables:** `docs/gate_g6_submission_stress_test.md`, reproducibility manifest.

#### Atomic Work Package (Phase 8)

```text
WP ID and title: WP-8.1 — Coherent Paper Blueprint, Skeptical-Referee Pass & Reproducibility Audit (Gate G6)
Status: DORMANT UNTIL GATE G5
Gate served: G6
Objective: Lock the final contribution statement, verify clean-environment end-to-end reproducibility of every figure/table, simulate 3 hostile referees (conformal specialist, off-policy RL/OPRA specialist, health-econometrics specialist), and issue the final Gate G6 classification.
Why this changes a decision: Ensures submission-readiness and prevents submitting a diffuse or under-defended manuscript.
Inputs:
  - All surviving outputs from Phases 2–7.
Preconditions: Gate G5 cleared (and Phase 7 complete if GO-THEORY).
Actions:
  1. Run full clean-checkout reproducibility check verifying every reported table/figure regenerates from pinned scripts and seeds.
  2. Draft 3 domain-specific skeptical referee reports and verify every major objection has a concrete table, figure, or theorem response.
  3. Compare surviving scope against recent ICML, NeurIPS, JASA, and Management Science papers to select the primary target venue based on methodological vs. applied balance.
  4. Issue final classification (strong field-paper candidate / promising but unproven / incremental-only / dead).
Outputs and exact paths:
  - /home/hugo/Stuff/Research/Safe_Distributional_Policy/docs/gate_g6_submission_stress_test.md
Verification:
  - Mechanical: Reproducibility script exits 0; all artifact checksums match.
  - Scientific: Zero unresolved high-severity referee objections.
Scientific pass rule: Final classification = strong field-paper candidate.
Scientific fail rule: Unresolved fatal objection -> pivot venue or kill.
Gate consequence: Authorizes full manuscript drafting.
Dependencies: Gate G5 (and WP-7.1/7.2 if GO-THEORY).
Can run in parallel with: None.
Compute, RAM, and expected runtime: Reproducibility check ~1–2 hours; audit ~2 hours.
Likely trap: Re-introducing cut decorative modules (Regime B WCF or synthetic "latency" stories) during paper framing.
Recovery action: Enforce strict scope discipline: one clean Regime A contribution, one flagship RCT (OHIE), one observational sensitivity benchmark (MEPS).
```

---

## 8. Simulation Study Specification

*(Executable ONLY after Gate G2 clears; Phase 4 `DORMANT UNTIL GATE G2`.)*

### 8.1 Claims-to-Experiments Matrix

| Claim | Mechanism | DGP | Primary Metric | Baseline | Ablation | Pass Threshold | Falsifier | Output |
|---|---|---|---|---|---|---|---|---|
| **M1:** Localized comparative certification breaks two-arm abstention dead-zone | Signed contrast $\Delta w_\lambda = 0$ on $S_\lambda^c$ $\implies$ variance scales as $O(\mathbb{P}(S_\lambda)/n)$ | DGP-4, DGP-5, DGP-6 | Certified deployment rate $\mathbb{P}(S_{\hat\lambda})$ & realized safe welfare at FWER $\le \delta$ | **B1** (Two-arm OPRA/UnO + LTT) | LCT-LTT with betting-martingale vs. empirical-Bernstein UCB | $\ge +20$ pp deployment (or $\ge +25\%$ safe welfare) over B1 at $n_{\text{cal}}=5000$, $\mathbb{P}(S^*) \in [0.15, 0.40]$ | B1 matches LCT-LTT within 10 pp or both certify $< 10\%$ | `core_dgp1_5_r50.parquet` |
| **M2:** Two-part semicontinuous distributional scoring beats mean-CATE scoring under mean-tail conflict | Modeling extensive margin + positive-tail dispersion separates tail-safe from tail-unsafe subgroups | DGP-4, DGP-9 | Realized welfare & tail-safe precision among deployed units | **B2** (Mean causal forest + LCT-LTT) | `ZIC-BCF-Smear` vs. fast `TwoPartQuantileGBDT` vs. mean forest | $\ge +15\%$ realized safe welfare over B2 at fixed FWER $\le \delta$ | B2 achieves $\ge 90\%$ of LCT-LTT welfare across mean-tail conflict DGPs | `core_dgp1_5_r50.parquet`, `stress_dgp6_10.parquet` |
| **M3:** Tail-functional certification beats linear-loss / uncertified incumbents | Nonlinear $\text{CVaR}_\beta$ & $Q_\beta$ certification guards upper tail when mean or fixed-threshold losses miss tail shift | DGP-1, DGP-4, DGP-8 | Empirical tail-risk violation rate (FWER) & realized $\Delta \text{CVaR}_\beta, \Delta Q_\beta$ | **B3** (Plug-in CVaR ITR), **B4** (Mean forest), **B5** (Fixed-threshold CPC) | Single-risk ($Q_\beta$ only) vs. joint ($Q_\beta$ + CVaR + extensive margin) | LCT-LTT FWER $\le \delta + 0.015$ while B3/B4 exceed $2\delta$ and B5 either violates CVaR/$Q_\beta$ or loses $\ge 15\%$ welfare | B5 matches LCT-LTT safety & welfare without tail-functional bounds | `core_dgp1_5_r50.parquet` |
| **M4:** Exact finite-sample validity under known propensities & controlled degradation under estimated propensities | Exact importance weights $e(a\mid x)$ vs. $\Gamma$-sensitivity inflation for $\hat e(a\mid x)$ | DGP-3 (Null), DGP-7 (Misspec.) | Empirical FWER across 50 seeds (and 500 draws in WP-3.2) | Unadjusted LCT-LTT vs. $\Gamma$-adjusted LCT-LTT | Known $e(x)$ vs. misspecified $\hat e(x)$ at $\kappa \in \{0.05, 0.10, 0.25\}$ | FWER $\le 0.065$ (known $e$); $\Gamma$-adjusted restores FWER $\le 0.065$ under bounded propensity misspecification | FWER $> 0.08$ under known $e(x)$ | `core_dgp1_5_r50.parquet`, `stress_dgp6_10.parquet` |

### 8.2 Ten-Step DGP Ladder
1. **DGP-1 (Clean / Near-Noiseless Known-Truth Recovery):** Low-noise semicontinuous hurdle with sharp known safe region $S^* = \{X_1 < 0\}$; verifies exact oracle recovery.
2. **DGP-2 (Correctly Specified Stochastic Semicontinuous):** Log-normal hurdle ($40\%$ zero mass, $\beta = 0.90$, known cap $B$) with smooth covariate-dependent location and scale; verifies calibration and deployment efficiency.
3. **DGP-3 (Global Null / Globally Unsafe Case):** Two sub-cases: (3a) pure null $Y(1) \stackrel{d}{=} Y(0)$ and (3b) globally tail-unsafe $Y(1)$ heavier-tailed everywhere; verifies FWER $\le \delta$ and $\ge 95\%$ abstention on (3b).
4. **DGP-4 (Baseline-Favorable Case — Pure Location Shift, No Mean-Tail Conflict):** Treatment shifts location uniformly with constant variance and no zero-inflation conflict; expected region where **B2 (Mean-Forest + LCT-LTT)** matches distributional scoring (honest baseline-win benchmark).
5. **DGP-5 (Proposed-Mechanism-Favorable Case — Targeted Mean-Tail & Extensive/Intensive Conflict):** Treatment improves mean and extensive margin on $60\%$ of population, reduces tail risk on $S^*$ ($\mathbb{P}(S^*) = 0.30$), and *worsens* upper-tail variance/Pareto tail on the remaining $30\%$ (invisible to mean CATE!).
6. **DGP-6 (Crossover Grid):** Factorial sweep over $n_{\text{cal}} \in \{1000, 2500, 5000, 10000\}$, safe-region size $\mathbb{P}(S^*) \in \{0.10, 0.25, 0.50, 0.80\}$, and tail level $\beta \in \{0.80, 0.90, 0.95\}$; maps exact crossover frontier between LCT-LTT and B1/B2.
7. **DGP-7 (Nuisance & Propensity Misspecification):** Nonlinear confounding, weak overlap $\kappa \in \{0.05, 0.10, 0.25\}$, and misspecified logistic $\hat e(x)$; tests unadjusted vs. $\Gamma$-sensitivity LCT-LTT.
8. **DGP-8 (Heavy-Tail & Support-Cap Stress Test):** Pareto / Tweedie heavy right tails testing capped-CVaR sensitivity across $B \in \{Q_{0.99}, 2Q_{0.99}, 5Q_{0.99}\}$ vs. boundedness-free $Q_\beta$ certification.
9. **DGP-9 (Application-Mimicking Defects):** Parameterized directly from WP-5.0 OHIE/MEPS feasibility statistics (exact zero-mass, skewness, discrete/skewed covariates, stratified household propensities).
10. **DGP-10 (Runtime & Memory Scaling):** $n \in \{1000, 5000, 20000, 100000\}$, $p \in \{10, 50\}$; measures wall time and peak RSS memory for every learner and certification engine.

### 8.3 Fair Comparison Protocol
- **Shared Splits:** Every replication $r \in \{1000, \dots, 1049\}$ generates a shared $(\mathcal{D}_{\text{tr}}, \mathcal{D}_{\text{cal}}, \mathcal{D}_{\text{eval}})$ split (e.g., 40%/60% train/cal on sample $n$, evaluated on independent oracle evaluation sample $n_{\text{eval}} = 100{,}000$ for near-exact ground-truth $\text{CVaR}_\beta(\pi_{\hat\lambda})$ and $Q_\beta(\pi_{\hat\lambda})$).
- **Equal Information & Tuning Budget:** All learners (LCT-LTT, B1–B5) receive identical covariates $X$, identical propensity access, and identical 3-fold CV hyperparameter budgets on $\mathcal{D}_{\text{tr}}$. Oracle policy/score variants are labeled `ORACLE` and reported strictly as upper bounds.
- **Metric Hierarchy:**
  - *Primary Safety Metric:* Empirical family-wise tail-risk violation rate (FWER) across replications vs. nominal $\delta = 0.05$.
  - *Primary Utility Metrics (conditional on valid FWER):* Certified deployment rate $\mathbb{P}(S_{\hat\lambda})$ and realized welfare gain $\mathbb{E}[U(\pi_{\hat\lambda}) - U(\pi_0)]$.
  - *Secondary Metrics:* Realized tail-risk change $\Delta \text{CVaR}_\beta, \Delta Q_\beta, \Delta R_{\text{ext}}$, worst-regime performance, runtime, and peak RAM.
- **Failed-Run Logging:** All runs (including 100% abstention or numerical exceptions) are preserved in parquet logs with explicit status flags.

### 8.4 Simulation Gate Memo
Standalone document `docs/gate_g3_memo.md` containing: (1) preregistered expectations, (2) result tables & crossover plots with 95% MC CIs, (3) protocol deviations, (4) strongest baseline's best case (DGP-4), (5) proposed method's failure regions (e.g., extreme $B$, $\kappa=0.05$, small $n_{\text{cal}}$), (6) ablation evidence isolating localization (vs. B1) and two-part scoring (vs. B2), (7) practical effect sizes, and (8) one Gate G3 verdict.

---

## 9. Applied-Study Specification

*(WP-5.0 is read-only and `ACTIVE` now; confirmatory protocol WP-5.1–5.3 is strictly `DORMANT UNTIL GATE G3`.)*

### 9.1 Early Read-Only Feasibility Audit (WP-5.0)
Strictly limited to access verification, license checks, marginal outcome zero-fraction and right-tail quantiles (unstratified by treatment), covariate schemas, propensity/stratification balance, and cluster sizes for OHIE and MEPS 2023. Zero policy tuning or treatment-outcome inspection.

### 9.2 Confirmatory Application Protocol (14 Items — Primary: OHIE RCT; Secondary: MEPS Observational)
1. **Substantive question & target population:** Among low-income uninsured adults eligible for Oregon's 2008 Medicaid lottery (OHIE), can a data-driven personalized eligibility/outreach policy deploy lottery-offer expansion to subgroups that maximize health/financial-protection utility while guaranteeing with finite-sample confidence $\ge 95\%$ that upper-tail ($Q_{0.90}, Q_{0.95}$ and capped $\text{CVaR}_{0.90}$) medical/out-of-pocket expenditure or ED-cost risk does not worsen relative to status quo?
2. **Unit, treatment, timing, outcome, covariates, estimand:** Unit = **independent household** (handling OHIE household-cluster randomization rigorously); Treatment $A \in \{0,1\}$ = randomized lottery selection (ITT); Outcome $Y \in \{0\} \cup (0, \infty)$ = semicontinuous medical/out-of-pocket expenditure / catastrophic-spending loss and utilization; Covariates $X$ = baseline demographics, household size (`numhh_list`), pre-lottery indicators; Estimand = population-marginal and finite-subgroup comparative tail safety ($\Delta Q_\beta \le \varepsilon$, $\Delta \text{CVaR}_\beta \le \varepsilon$, $\Delta R_{\text{ext}} \le \varepsilon_{\text{ext}}$).
3. **Identification assumptions & design:** OHIE satisfies A1–A5a **by experimental design** (known randomization probability $e(1 \mid X)$ determined exactly by household-size strata `numhh_list`; zero unmeasured confounding for ITT). MEPS (secondary) is observational (requires unconfoundedness A2, evaluated via $\Gamma$-sensitivity bounds A5b).
4. **Inclusion & exclusion rules:** Frozen in `docs/phase5_ohie_protocol.md` prior to comparative estimation (matching Finkelstein et al., 2012 analytic sample and household-cluster deduplication).
5. **Power & precision assessment:** Computed from WP-5.0 sample size and DGP-9 crossover curves before confirmatory runs.
6. **Frozen preprocessing:** All covariate encodings, missingness indicators, and CVaR stop-loss caps $B$ frozen before running any comparator.
7. **Known benchmark to reproduce:** Finkelstein et al. (2012, *QJE*) ITT lottery effects on healthcare utilization, out-of-pocket spending, and catastrophic medical expenditure (OHIE); `zicbcf/applied_study/ate_results.csv` analytic-sample dental expenditure ATE & hurdle effect (MEPS).
8. **Strongest incumbent analysis:** B1 (Two-arm OPRA/UnO + LTT), B2 (Mean-CATE forest + LCT-LTT), B3 (Plug-in CVaR ITR), B4 (Mean-CATE ITR), and B5 (Fixed-threshold catastrophic-spend indicator CPC).
9. **Negative controls & placebos:** 100 within-stratum treatment-permutation placebos (must abstain $\ge 95\%$ of runs) plus pre-lottery baseline covariate negative-control outcome if available in OHIE.
10. **Sensitivity analyses:** Cap sensitivity $B$ (for CVaR vs. uncapped $Q_\beta$), calibration-split ratio sensitivity, and observational propensity-sensitivity $\Gamma \in [1.0, 1.5]$ (MEPS).
11. **Predeclared primary novel finding:** Identify covariate strata where personalized expansion is finite-sample certified to improve financial protection / welfare without worsening upper-tail spending risk, contrasting with regions where mean-optimal or plug-in CVaR rules over-deploy.
12. **Subgroup policy:** Finite-partition subgroup certification (G-Subgroup) across prespecified age / household-size / baseline-risk strata.
13. **Robustness to measurement & composition changes:** Sensitivity to household vs. individual clustering (OHIE) and unweighted analytic vs. `PERWT23F` survey-weighted evaluation (MEPS).
14. **Interpretation boundaries:** OHIE certifies ITT lottery-offer policies (not IV/LATE treatment-on-the-treated without compliance assumptions); MEPS provides observational sensitivity bounds, not exact finite-sample guarantees.

### 9.3 Application Gate Memo
Standalone `docs/gate_g4_memo.md` verifying all 6 items of plan-contract Section 5.3 before opening Phase 6.

---

## 10. Deferred Theory Program — `STATUS: DORMANT UNTIL GATE G5`

Substantial mathematical theory is strictly deferred until Gates G3 and G4 confirm a live empirical engine and credible applied finding. Every target below is narrow, decision-driven, and mapped to verified primary sources:

| Target | Why Evidence Says It Matters | Exact Statement Sketch | Tag | Source Result | Adaptation Gap | Numerical Falsifier | Stop Rule |
|---|---|---|---|---|---|---|---|
| **T1 (Joint Finite-Sample Validity of LCT-LTT)** | Protects interpretation of reported finite-sample $(1-\delta)$ tail-safety certificates across CVaR, $Q_\beta$, and extensive margin | Under A1–A5a (known $e$) and A6 ($L \le B$ for CVaR; none for $Q_\beta$), fixed-sequence LCT-LTT over grid $\Lambda$ with baseline-VaR CI $[\underline\eta_0, \overline\eta_0]$ satisfies $\mathbb{P}_{\mathcal{D}_{\text{cal}}}(\forall j, \rho_j(\pi_{\hat\lambda}) \le \rho_j(\pi_0)+\varepsilon_j) \ge 1 - \delta$. | `adaptation` | Rockafellar & Uryasev (2000, Thm. 1); Angelopoulos et al. (AOAS 2025, Thm. 1 LTT); Waudby-Smith & Ramdas (JRSS-B 2024) | Combining exact baseline-VaR CI supremum over $\eta$ with localized signed off-policy contrast weights $\Delta w_\lambda$ and fixed-sequence FWER control | 500-draw null & boundary DGP check (`falsify_T1_T2.py`): empirical FWER must stay $\le \delta + 0.015$ | Drop if enabling Phase 2 proof already suffices without extension |
| **T2 (Variance-Localized Power & Certified-Deployment Rate)** | Explains the observed empirical advantage over B1 (two-arm OPRA/UnO + LTT) | For a safe region $S_\lambda$ with mass $p_\lambda = \mathbb{P}(X \in S_\lambda)$ and tail safety margin $\gamma_\lambda > 0$, localized LCT-LTT certifies $\lambda$ with probability $\ge 1-\delta'$ whenever $n_{\text{cal}} \gtrsim \frac{p_\lambda B^2 \log(1/\delta)}{\kappa (1-\beta)^2 \gamma_\lambda^2}$, strictly dominating two-arm OPRA/UnO's $O(1/(\kappa \gamma_\lambda^2))$ sample complexity by factor $1/p_\lambda$. | `adaptation` | Maurer & Pontil (COLT 2009) empirical Bernstein; Huang et al. (NeurIPS 2021, Thm. 4.2 OPRA) | Accounting for the extra baseline-VaR CI width $|\overline\eta_0 - \underline\eta_0| = O(n^{-1/2})$ inside the supremum over $\eta$ (showing this adds only a higher-order $O(p_\lambda n^{-1/2})$ or controllable term!) | Regress minimum certifiable $n_{\text{cal}}$ against $p_\lambda \in \{0.05, 0.1, 0.25, 0.5\}$; slope must match predicted $p_\lambda$ scaling | Cut if baseline-VaR CI width dominates localized variance in practice (making $1/p_\lambda$ gain disappear!) |
| **T3 (Observational $\Gamma$-Sensitivity & Asymptotic DR Safety)** | Validates MEPS observational sensitivity curves and answers observational-deployment objections | (a) Under multiplicative propensity error $e/\hat e \in [1/\Gamma, \Gamma]$, inflated localized bounds preserve exact finite-sample $(1-\delta)$ safety; (b) With cross-fitted orthogonalized localized excess scores, asymptotic validity holds if $\|\hat e - e\|_2 \|\hat m - m\|_2 = o_p(n^{-1/2})$. | `adaptation` / `conjecture` (for non-smooth quantile DR) | Lei & Candès (JRSS-B 2021, Thm. 3); Jin, Ren, Candès (2023); Kallus (Manag. Sci. 2023) | Extending $\Gamma$-sensitivity to signed contrast weights $\Delta w_\lambda$ and verifying Hadamard/gateaux differentiability of localized CVaR excess under semicontinuous point mass at 0 | DGP-7 misspecified propensity sweep (`falsify_T3.py`): $\Gamma$-bound must cover empirical FWER | Drop part (b) if non-smooth point-mass edge cases complicate proof without helping MEPS |

---

## 11. Risk Register

| Risk | Probability | Damage | Earliest Detector | Prevention | Recovery | Terminal? | Owner Package |
|---|---|---|---|---|---|---|---|
| **R1: Zero-deployment conservatism** (finite-sample tail bounds abstain ~100% at realistic $n_{\text{cal}}$) | Medium-High | High | WP-2.2 (1D tightness pilot) | Fixed-sequence testing (zero Bonferroni over $\Lambda$), betting-martingale UCBs, localized $S_\lambda^c$ cancellation | Pivot primary target to boundedness-free $Q_\beta$ ($\beta \in \{0.80, 0.90\}$) + extensive margin, or tighten score $s(x)$ | Yes (if both CVaR & $Q_\beta$ abstain) | WP-2.2, WP-4.1 |
| **R2: Prior-art direct hit / collapse to B1 (OPRA+LTT)** | Medium | High | WP-2.1 (E3 audit) & WP-2.2 | Verify exact theorems of OPRA/UnO/CPC/CPL; require $\ge +20$ pp deployment gain over B1 | Terminate as `INCREMENTAL-ONLY` or pivot to Pivot B (Regime B simultaneous tail-curve) | Yes | WP-2.1, WP-2.2, WP-4.3 |
| **R3: Weak-baseline illusion vs. B2 (mean forest) or B5 (indicator CPC)** | Medium | High | WP-4.1 & WP-4.3 | Equal tuning budgets, shared splits, explicit DGP-4 baseline-favorable case | If B2 matches everywhere, reframe contribution purely around localized certification gate (cutting `ZIC-BCF` claims); if B5 matches, KILL | Conditional | WP-4.1, WP-4.3 |
| **R4: Baseline-VaR CI width dominates localized variance** | Medium | Medium-High | WP-2.1 & WP-2.2 | Check exact empirical contribution of $\sup_{\eta \in [\underline\eta_0, \overline\eta_0]}$ vs. pointwise $\eta_0^*$ in WP-2.2 | Use asymmetric betting-martingale CDF inversion for tighter $[\underline\eta_0, \overline\eta_0]$; evaluate exact impact on T2 | Conditional | WP-2.1, WP-2.2, WP-7.1 |
| **R5: OHIE access / clustering / low tail heterogeneity** | Low-Medium | High | WP-5.0 (read-only audit) | Audit OHIE early; enforce household-level clustering | Identify alternative public RCT with semicontinuous heavy-tailed outcomes before Gate G3 | Conditional | WP-5.0, WP-5.1 |
| **R6: Observational MEPS overclaiming** (claiming exact finite-sample guarantees without known $e(x)$ or ignoring survey weights) | Medium | High | Phase 0 / WP-5.2 | Keep OHIE as primary exact-propensity RCT; position MEPS strictly as $\Gamma$-sensitivity stress test | Enforce explicit interpretation boundaries in WP-5.2 and WP-8.1 | No (preventable) | WP-5.2, WP-8.1 |
| **R7: Calibration data leakage into $\hat\eta_0$ or $\hat\pi$** | Low-Medium | Fatal | WP-3.2 (null FWER test) | Strict `ExperimentSplit` contract; 500-draw null FWER automated test | Fix split isolation before Phase 4 | No (preventable) | WP-3.1, WP-3.2 |
| **R8: Compute bottleneck from R `ZIC-BCF-Smear` MCMC** | High | Medium | WP-3.2 resource profile | Use fast Python two-part learner for 50-seed sweeps; cap workers $\le 6$ physical cores; shard `ZIC-BCF` 20-seed slice to Colab | Route long MCMC shards to Google Colab with checkpointing + download fallback | No | WP-3.2, WP-4.2 |
| **R9: Diffuse paper story** (re-bundling Regime B `WCF` or unavailable latency datasets) | Medium | Medium-High | WP-4.3 & WP-8.1 | Explicitly cut Regime B and latency/SLA A/B datasets from primary scope | Enforce one-sentence contribution discipline at Gate G5 & G6 | No | WP-6.1, WP-8.1 |

---

## 12. Reproducibility and Artifact Map

### 12.1 Intended Directory Structure
```text
/home/hugo/Stuff/Research/Safe_Distributional_Policy/
├── research_idea.md                  # Input dossier (created)
├── research_plan.md                  # This master plan (created)
├── pyproject.toml                    # Pinned Python environment & dependencies (WP-3.1)
├── docs/
│   ├── phase2_e3_prior_art_check.md  # WP-2.1 E3 theorem verification
│   ├── phase2_formalization.md       # WP-2.1 enabling math & LCT-LTT algorithm
│   ├── gate_g2_memo.md               # WP-2.2 Gate G2 decision
│   ├── phase4_preregistration.md     # WP-4.1 frozen simulation protocol
│   ├── gate_g3_memo.md               # WP-4.3 Gate G3 decision
│   ├── phase5_data_feasibility.md    # WP-5.0 read-only OHIE/MEPS audit
│   ├── phase5_ohie_protocol.md       # WP-5.1 frozen OHIE protocol
│   ├── gate_g4_memo.md               # WP-5.3 Gate G4 decision
│   ├── gate_g5_impact_reassessment.md# WP-6.1 Gate G5 decision
│   └── gate_g6_submission_stress_test.md # WP-8.1 Gate G6 audit
├── src/safe_dist_policy/             # Core package (WP-3.1)
├── tests/                            # Deterministic unit tests (WP-3.2)
├── scripts/                          # Reproducible experiment drivers
├── notebooks/                        # Colab shard notebooks (with download fallback)
├── data/                             # Read-only raw vs. processed boundaries (gitignored)
│   ├── raw/                          # Immutable downloads (OHIE; symlink/ref to MEPS)
│   └── processed/                    # Frozen preprocessed analytic tables
├── results/                          # Checkpointed outputs, parquet summaries, figures
└── theory/                           # Deferred proofs & numerical falsifiers (DORMANT)
```

### 12.2 Compute, Parallelism, Seeding & Colab Policy
- **Hardware Cap (13th Gen Intel Core i9-13900H — 10 physical / 20 logical cores):** Inspect active CPU/RAM load (`psutil` / `nproc` / `free -h`) prior to every sweep; cap process-level workers to **$\le 6$ physical cores** (leaving headroom for concurrent research jobs) and hard-set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` inside each worker to eliminate nested parallelism.
- **Seed Policy:** Deterministic `np.random.SeedSequence` per `(dgp_id, seed_id)`; Phase 2 pilot seeds `0..199`, Phase 3 smoke seeds `0..4`, Phase 4 confirmatory seeds `1000..1049`.
- **Checkpointing:** Every replication writes an atomic per-seed record before merging to `.parquet` with SHA-256 manifest checksums, enabling zero-loss resume.
- **Google Colab Download Fallback (Mandatory for any generated Colab notebook such as `notebooks/phase4_zicbcf_colab.ipynb`):**
```python
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
```

---

## 13. Immediate Actions (Strictly Through Next Unresolved Gate: Gate G2)

Do **not** launch Phase 3–8 codebases, full simulations, confirmatory applied analyses, substantial theory, or manuscript writing until Gate G2 clears. Execute ONLY the following bounded tasks now:

1. **`WP-2.1` (E3 Primary-Source Inspection & Enabling Formalization):** Inspect the exact theorems of OPRA (`arXiv:2104.08977`), UnO (`arXiv:2104.12820`), CPC (`arXiv:2603.02196`), and CPL (`arXiv:2609.17296`), formalize LCT-LTT, and prove the enabling finite-sample validity and variance-localization lemma.
   - *Named outputs:* `docs/phase2_e3_prior_art_check.md`, `docs/phase2_formalization.md`.
2. **`WP-2.2` (1D Non-Vacuity Tightness Pilot & Gate G2 Decision):** Run the 200-draw synthetic 1D tightness check comparing localized LCT-LTT (empirical-Bernstein & betting-martingale) vs. two-arm OPRA/UnO + LTT (B1) across $n_{\text{cal}} \in \{2000, 5000, 10000\}$ and $\beta \in \{0.80, 0.90, 0.95\}$, and write the Gate G2 verdict.
   - *Named outputs:* `scripts/phase2_tightness_pilot.py`, `results/phase2_pilot/tightness_check.csv`, `docs/gate_g2_memo.md`.
3. **`WP-5.0` (Parallel Read-Only OHIE/MEPS Data Feasibility Audit):** Verify OHIE public-use downloadability, household clustering, sample size, and marginal zero-inflation/tail quantiles alongside local MEPS 2023 HC-251 schema (strictly without outcome-by-treatment peeking).
   - *Named outputs:* `docs/phase5_data_feasibility.md`, `results/phase5_feasibility/data_audit_summary.json`.

**Stop at Gate G2.** Review `docs/gate_g2_memo.md` and `docs/phase5_data_feasibility.md` before unlocking Phase 3.

---

## 14. References

1. **Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2025).** Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. *The Annals of Applied Statistics*, 19(2), 1641–1662. DOI: [10.1214/24-AOAS1998](https://doi.org/10.1214/24-AOAS1998) | arXiv: [2110.01052](https://arxiv.org/abs/2110.01052). *(Maps to: Enabling LTT fixed-sequence FWER control, Target T1.)*
2. **Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024).** Conformal Risk Control. *ICLR 2024*. arXiv: [2208.02814](https://arxiv.org/abs/2208.02814). *(Maps to: Phase 0 defect D2b—bounded monotone expectation scope of CRC.)*
3. **Bahadur, R. R., & Savage, L. J. (1956).** The Nonexistence of Certain Statistical Procedures in Nonparametric Problems. *The Annals of Mathematical Statistics*, 27(4), 1115–1122. DOI: [10.1214/aoms/1177728077](https://doi.org/10.1214/aoms/1177728077). *(Maps to: Phase 0 defect D2a—impossibility of finite-sample distribution-free upper bounds on unbounded CVaR.)*
4. **Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021).** The Limits of Distribution-Free Conditional Predictive Inference. *Information and Inference*, 10(2), 455–482. DOI: [10.1093/imaiai/iaaa017](https://doi.org/10.1093/imaiai/iaaa017). *(Maps to: Phase 0 defect D3—impossibility of pointwise conditional finite-sample safety for continuous $X$.)*
5. **Ben-Michael, E., Greiner, D. J., Imai, K., & Jiang, Z. (2025).** Safe Policy Learning through Extrapolation: Application to Pre-trial Risk Assessment. *Journal of the American Statistical Association*. *(DOI: `UNVERIFIED` — verify published JASA DOI in WP-2.1; arXiv preprint verified via web search.)*
6. **Cao, J., Gao, R., Saar-Tsechansky, M., & Yin, M. (2026).** Conformal Policy Selection: When to Deploy AI Decisions for Targeted Intervention. arXiv: [2610.11547](https://arxiv.org/abs/2610.11547). *(Maps to: Phase 1 nearest-neighbor collision on selective deployment gates.)*
7. **Chandak, Y., Niekum, S., da Silva, B. C., Learned-Miller, E., Brunskill, E., & Thomas, P. S. (2021).** Universal Off-Policy Evaluation. *NeurIPS 2021*. arXiv: [2104.12820](https://arxiv.org/abs/2104.12820). *(Maps to: Strongest incumbent B1—UnO high-confidence CVaR/quantile bounds.)*
8. **Chen, C. Y.-C., Shen, J., Deng, Z., & Lei, L. (2025).** Conformal Tail Risk Control for Large Language Model Alignment. arXiv: [2502.20285](https://arxiv.org/abs/2502.20285). *(Maps to: Phase 1 nearest-neighbor predictive distortion/CVaR risk control.)*
9. **Chen, C. et al. (2026).** Adversarially Robust Control of Conditional Value-at-Risk via Rockafellar–Uryasev Conformal Inference. arXiv: [2606.00320](https://arxiv.org/abs/2606.00320). *(Maps to: Phase 1 nearest-neighbor online Rockafellar–Uryasev CVaR conformal inference.)*
10. **Finkelstein, A., Taubman, S., Wright, B., Bernstein, M., Gruber, J., Newhouse, J. P., Allen, H., Baicker, K., & Oregon Health Study Group (2012).** The Oregon Health Insurance Experiment: Evidence from the First Year. *The Quarterly Journal of Economics*, 127(3), 1057–1106. DOI: [10.1093/qje/qjs020](https://doi.org/10.1093/qje/qjs020). *(Maps to: Phase 5 OHIE benchmark replication.)*
11. **Huang, A., Leqi, L., Lipton, Z. C., & Azizzadenesheli, K. (2021).** Off-Policy Risk Assessment in Contextual Bandits. *NeurIPS 2021*. arXiv: [2104.08977](https://arxiv.org/abs/2104.08977). *(Maps to: Strongest incumbent B1—OPRA finite-sample off-policy CVaR/quantile bounds; Target T2 comparison.)*
12. **Jin, Y., & Egami, N. (2026).** Conformal Policy Learning with Distribution-Free Safety Guarantees. arXiv: [2609.17296](https://arxiv.org/abs/2609.17296). *(Maps to: Phase 1 nearest-neighbor collision on "do no harm" CPL.)*
13. **Jin, Y., Ren, Z., & Candès, E. J. (2023).** Sensitivity Analysis of Individual Treatment Effects: A Robust Conformal Inference Approach. *PNAS*, 120(6), e2214889120. DOI: [10.1073/pnas.2214889120](https://doi.org/10.1073/pnas.2214889120). *(Maps to: Target T3 observational $\Gamma$-sensitivity bounds.)*
14. **Kallus, N. (2023).** Treatment Effect Risk: Bounds and Inference. *Management Science*, 69(8), 4579–4590. *(DOI: `UNVERIFIED` — arXiv:2201.05893; verify exact Management Science DOI in WP-2.1. Maps to: Phase 1 comparison & Target T3 DR inference.)*
15. **Lei, L., & Candès, E. J. (2021).** Conformal Inference of Counterfactuals and Individual Treatment Effects. *JRSS-B*, 83(5), 911–938. DOI: [10.1111/rssb.12445](https://doi.org/10.1111/rssb.12445) | arXiv: [2006.06138](https://arxiv.org/abs/2006.06138). *(Maps to: Phase 0 propensity-error TV slack; Target T3.)*
16. **Maurer, A., & Pontil, M. (2009).** Empirical Bernstein Bounds and Sample Variance Penalization. *COLT 2009*. arXiv: [0907.3740](https://arxiv.org/abs/0907.3740). *(Maps to: Target T2 variance-localized empirical-Bernstein rate.)*
17. **Prinster, D., Fannjiang, C., Park, J. W., Cho, K., Liu, A., Saria, S., & Stanton, S. (2026).** Conformal Policy Control. *ICML 2026*. arXiv: [2603.02196](https://arxiv.org/abs/2603.02196). *(Maps to: Phase 1 nearest-neighbor collision on safe reference policy gating; Baseline B5.)*
18. **Qi, Z., Pang, J.-S., & Liu, Y. (2023).** On Robustness of Individualized Decision Rules. *Journal of the American Statistical Association*, 118(544), 2143–2157. DOI: [10.1080/01621459.2022.2038180](https://doi.org/10.1080/01621459.2022.2038180) | arXiv: [1903.04367](https://arxiv.org/abs/1903.04367). *(Maps to: Uncertified CVaR ITR baseline B3.)*
19. **Rockafellar, R. T., & Uryasev, S. (2000).** Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21–41. DOI: [10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038). *(Maps to: Core comparative baseline-VaR CVaR pivot; Targets T1 & T2.)*
20. **Snell, J. C., Zollo, T. P., Deng, Z., Pitassi, T., & Zemel, R. (2023).** Quantile Risk Control: A Flexible Framework for Bounding the Probability of High-Loss Predictions. *ICLR 2023*. arXiv: [2212.13629](https://arxiv.org/abs/2212.13629). *(Maps to: Phase 1 predictive quantile/distortion risk control.)*
21. **Taufiq, M. F., Ton, J.-F., Cornish, R., Teh, Y. W., & Doucet, A. (2022).** Conformal Off-Policy Prediction in Contextual Bandits. *NeurIPS 2022*. arXiv: [2206.04405](https://arxiv.org/abs/2206.04405). *(Maps to: Phase 1 off-policy conformal predictive intervals.)*
22. **Thomas, P. S., Theocharous, G., & Ghavamzadeh, M. (2015).** High Confidence Policy Improvement. *ICML 2015*, PMLR 37:2380–2388. URL: [https://proceedings.mlr.press/v37/thomas15.html](https://proceedings.mlr.press/v37/thomas15.html). *(Maps to: Foundational HCPI safe-fallback paradigm.)*
23. **Wang, L., Zhou, Y., Song, R., & Sherwood, B. (2018).** Quantile-Optimal Treatment Regimes. *Journal of the American Statistical Association*, 113(523), 1243–1254. DOI: [10.1080/01621459.2017.1330204](https://doi.org/10.1080/01621459.2017.1330204). *(Maps to: Uncertified quantile-optimal ITR baseline B3.)*
24. **Waudby-Smith, I., & Ramdas, A. (2024).** Estimating Means of Bounded Random Variables by Betting. *JRSS-B*, 86(1), 1–27. DOI: [10.1093/jrsssb/qkad009](https://doi.org/10.1093/jrsssb/qkad009). *(Maps to: Tight finite-sample betting-martingale UCBs for WP-2.2 and Target T1.)*
25. **Yeh, C., Christianson, N., Wierman, A., & Yue, Y. (2025).** Conformal Risk Training: End-to-End Optimization of Conformal Risk Control. *NeurIPS 2025*. arXiv: [2510.08748](https://arxiv.org/abs/2510.08748). *(Maps to: Phase 1 OCE/CVaR conformal risk training.)*
26. **Zecchin, M., et al. (2025).** Generalization and Informativeness of Weighted Conformal Risk Control Under Covariate Shift. arXiv: [2501.11413](https://arxiv.org/abs/2501.11413). *(Maps to: Phase 0/1 weighted CRC under covariate shift.)*
27. **Internal Codebases & Datasets (Verified Locally):** `Wasserstein_Causal_Forests` (`/home/hugo/Stuff/Submitted_Research/Wasserstein_Causal_Forests`), `zicbcf` (`/home/hugo/Stuff/Submitted_Research/zicbcf`, including MEPS 2023 HC-251 at `applied_study/h251.csv`), `CP_TATE` (`/home/hugo/Stuff/Submitted_Research/CP_TATE`), and NBER Oregon Health Insurance Experiment public-use portal ([https://www.nber.org/programs-projects/projects/oregon-health-insurance-experiment](https://www.nber.org/programs-projects/projects/oregon-health-insurance-experiment)).
