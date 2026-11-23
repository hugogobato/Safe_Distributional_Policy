# Phase 2 Enabling Formalization: Localized Comparative Tail-Risk Learn-then-Test (LCT-LTT)

**Date:** 2026-11-23  
**Work Package:** `WP-2.1` (Gate G2 Enabling Formalization)  
**Scope:** Canonical notation, exact finite-sample validity proofs (zero calibration leakage, zero discretization error), variance localization proof, baseline-VaR CI width analysis, observational $\Gamma$-sensitivity and TV-slack extensions, and exact algorithm specification.

---

## 1. Canonical Setup and Assumptions

Let $\mathcal{D}_{\text{tr}}$ and $\mathcal{D}_{\text{cal}} = \{(X_i, A_i, L_i)\}_{i=1}^n$ be independent training and calibration samples. All proofs below condition on $\mathcal{D}_{\text{tr}}$, treating the candidate policy $\hat\pi: \mathcal{X} \to \mathcal{A}$ (with $\mathcal{A} = \{0, 1\}$, extending immediately to finite multi-arm $\mathcal{A}$), baseline policy $\pi_0: \mathcal{X} \to \mathcal{A}$, priority score $s: \mathcal{X} \to \mathbb{R}$, and (in observational settings) estimated propensity $\hat e(a \mid x)$ as **deterministic** functions.

- **Canonical Loss Convention:** $L_i \ge 0$ is a non-negative scalar loss (smaller is safer). For semicontinuous healthcare expenditure $Y_i \in \{0\} \cup (0, \infty)$, $L_i = Y_i$ (or stop-loss capped loss $L_i = \min(Y_i, B) \in [0, B]$ for CVaR). Utility/welfare $U_i$ (if distinct from $-L_i$) is tracked separately.
- **Potential Losses:** Each unit has potential losses $\{L_i(a)\}_{a \in \mathcal{A}}$ and observed loss $L_i = L_i(A_i)$. For any deterministic policy $\pi: \mathcal{X} \to \mathcal{A}$, $L(\pi) := L(\pi(X))$ has CDF $F_\pi(t) := \mathbb{P}(L(\pi) \le t)$ and survival function $S_\pi(t) := \mathbb{P}(L(\pi) > t) = 1 - F_\pi(t)$.
- **Gated Policy Family:** Fix a finite, pre-specified, strictly decreasing threshold grid $\Lambda = \{\lambda_1 > \lambda_2 > \dots > \lambda_M\}$ (with structural never-switch fallback $\lambda_0 := +\infty$). For each $\lambda \in \Lambda \cup \{+\infty\}$, define the deployment set $S_\lambda := \{x \in \mathcal{X} : s(x) \ge \lambda\}$ (so $S_{+\infty} = \emptyset$ and $S_{\lambda_1} \subseteq S_{\lambda_2} \subseteq \dots \subseteq S_{\lambda_M}$), the effective switch set $\widetilde S_\lambda := \{x \in S_\lambda : \hat\pi(x) \neq \pi_0(x)\}$ with mass $\tilde p_\lambda := \mathbb{P}(X \in \widetilde S_\lambda) \le p_\lambda := \mathbb{P}(X \in S_\lambda)$, and the gated policy
  $$\pi_\lambda(x) := \mathbf{1}\{x \in S_\lambda\}\,\hat\pi(x) + \mathbf{1}\{x \notin S_\lambda\}\,\pi_0(x).$$
- **Propensity & Localized Contrast Weights:** Let $e(a \mid x) := \mathbb{P}(A = a \mid X = x)$ and define
  $$w_0(X, A) := \frac{\mathbf{1}\{A = \pi_0(X)\}}{e(A \mid X)}, \qquad w_\lambda(X, A) := \frac{\mathbf{1}\{A = \pi_\lambda(X)\}}{e(A \mid X)},$$
  $$\Delta w_\lambda(X, A) := w_\lambda(X, A) - w_0(X, A) = \mathbf{1}\{X \in \widetilde S_\lambda\}\,\frac{\mathbf{1}\{A = \hat\pi(X)\} - \mathbf{1}\{A = \pi_0(X)\}}{e(A \mid X)}.$$
  Notice that $\Delta w_\lambda(X, A) \equiv 0$ identically on $\widetilde S_\lambda^c$, and $|\Delta w_\lambda(X, A)| \le 1/\kappa$ everywhere.

### Core Assumptions
- **(A1) SUTVA / Consistency:** $L_i = L_i(A_i)$.
- **(A2) Ignorability:** $\{L_i(a)\}_{a \in \mathcal{A}} \perp\!\!\!\perp A_i \mid X_i$.
- **(A3) Strict Overlap:** There exists known $\kappa \in (0, 1/2]$ (or arm-specific $\kappa_a > 0$) such that $e(a \mid x) \ge \kappa$ for all $x \in \mathcal{X}$ and $a \in \{\pi_0(x), \hat\pi(x)\}$.
- **(A4) Independent Calibration Sample:** $\mathcal{D}_{\text{cal}} = \{(X_i, A_i, L_i)\}_{i=1}^n$ are i.i.d. draws from $P$, independent of $\mathcal{D}_{\text{tr}}$.
- **(A5a) Known Propensity (Exact RCT Regime):** $e(a \mid x)$ is known exactly. *(Replaced by **(A5b)** $\Gamma$-sensitivity $\Gamma^{-1} \le \hat e(a \mid x)/e(a \mid x) \le \Gamma$ in Section 5 for observational studies.)*
- **(A6) Bounded Support (CVaR Only):** $L(a) \in [0, B]$ a.s. for a known cap $B > 0$. **Not required** for upper-quantile $Q_\beta$ or extensive-margin risk $R_{\text{ext}}$.

Under (A1)--(A3), for any measurable function $g: [0, \infty) \to \mathbb{R}$ with $\mathbb{E}[|g(L(a))|] < \infty$,
$$\mathbb{E}\big[w_0(X, A)\,g(L)\big] = \mathbb{E}\big[g(L(\pi_0))\big], \qquad \mathbb{E}\big[\Delta w_\lambda(X, A)\,g(L)\big] = \mathbb{E}\big[g(L(\pi_\lambda))\big] - \mathbb{E}\big[g(L(\pi_0))\big].$$

---

## 2. Exact Finite-Sample Baseline-VaR Confidence Interval $[\underline\eta_0, \overline\eta_0]$

Let $F_0(t) := \mathbb{P}(L(\pi_0) \le t)$, $S_0(t) := 1 - F_0(t) = \mathbb{P}(L(\pi_0) > t)$, and fix $\beta \in (0, 1)$. Define the deterministic baseline quantile (Value-at-Risk):
$$\eta_0^* := Q_\beta(L(\pi_0)) := \inf\{t \in \mathbb{R} : F_0(t) \ge \beta\}.$$
Because $F_0$ is right-continuous and non-decreasing, $F_0(\eta_0^*) \ge \beta$ and $F_0(t) < \beta$ for all $t < \eta_0^*$.

### Survival-Complement Reformulation (Variance Reduction for Upper Tails)
Since $\mathbb{E}[w_0(X, A)] = 1$ under (A1)--(A3) and (A5a), we have the exact identity
$$S_0(t) = 1 - F_0(t) = \mathbb{E}\big[w_0(X, A)\,\mathbf{1}\{L > t\}\big], \qquad w_0(X, A)\,\mathbf{1}\{L > t\} \in [0, 1/\kappa].$$
For upper quantiles $\text{Var}(w_0\mathbf{1}\{L > \eta_0^*\}) \le (1-\beta)/\kappa$, since $S_0(\eta_0^*) \le 1-\beta$. The corresponding generic second-moment bound for the CDF score is $F_0(\eta_0^*)/\kappa$. The actual ratio of variances depends on the design and conditional outcome distribution.

### Lemma 1 (Pointwise-Inversion Finite-Sample CI for $\eta_0^*$ Without Uniform CDF Bands)
*Suppose (A1)--(A5a) hold and $\beta \in (0, 1)$. For each fixed $t \in \mathbb{R}$, let $\underline S_0(t; \delta_1/2)$ and $\overline S_0(t; \delta_1/2)$ be pointwise finite-sample lower and upper confidence bounds for $S_0(t) = \mathbb{E}[w_0(X, A)\mathbf{1}\{L > t\}]$ satisfying, for every fixed deterministic $t \in \mathbb{R}$,*
$$\mathbb{P}\big(\overline S_0(t; \delta_1/2) \ge S_0(t)\big) \ge 1 - \frac{\delta_1}{2}, \qquad \mathbb{P}\big(\underline S_0(t; \delta_1/2) \le S_0(t)\big) \ge 1 - \frac{\delta_1}{2}.$$
*Define $\underline F_0(t) := 1 - \overline S_0(t; \delta_1/2)$ and $\overline F_0(t) := 1 - \underline S_0(t; \delta_1/2)$ (neither is required to be monotone in $t$!). Set:*
$$\underline\eta_0 := \inf\big\{t \in \mathcal{T} : \overline F_0(t) \ge \beta\big\} = \inf\big\{t \in \mathcal{T} : \underline S_0(t; \delta_1/2) \le 1 - \beta\big\},$$
$$\overline\eta_0 := \sup\big\{t \in \mathcal{T} : \underline F_0(t) < \beta\big\} = \sup\big\{t \in \mathcal{T} : \overline S_0(t; \delta_1/2) > 1 - \beta\big\},$$
*where $\mathcal{T} = [0, B]$ under (A6) and $\mathcal{T} = [0, \infty)$ without (A6) (with $\inf \emptyset := \sup \mathcal{T}$ and $\sup \emptyset := 0$). Then, even in the presence of arbitrary ties or point masses (including a point mass at $0$),*
$$\mathbb{P}\big(\underline\eta_0 \le \eta_0^* \le \overline\eta_0\big) \ge 1 - \delta_1.$$

**Proof.**
1. *Lower endpoint ($\mathbb{P}(\underline\eta_0 > \eta_0^*) \le \delta_1/2$):* Since $\eta_0^* \in \mathcal{T}$, if $\underline\eta_0 > \eta_0^*$, then $\eta_0^* \notin \{t \in \mathcal{T} : \overline F_0(t) \ge \beta\}$, which implies $\overline F_0(\eta_0^*) < \beta$. Because $F_0$ is right-continuous, $F_0(\eta_0^*) \ge \beta$, so $S_0(\eta_0^*) \le 1 - \beta$. Hence
   $$\{\underline\eta_0 > \eta_0^*\} \subseteq \{\overline F_0(\eta_0^*) < \beta\} = \{\underline S_0(\eta_0^*; \delta_1/2) > 1 - \beta\} \subseteq \{\underline S_0(\eta_0^*; \delta_1/2) > S_0(\eta_0^*)\}.$$
   Since $\eta_0^*$ is deterministic, pointwise validity at $t = \eta_0^*$ gives $\mathbb{P}(\underline\eta_0 > \eta_0^*) \le \delta_1/2$.
2. *Upper endpoint ($\mathbb{P}(\overline\eta_0 < \eta_0^*) \le \delta_1/2$):* If $\eta_0^* = 0$, then $\overline\eta_0 \ge 0 = \eta_0^*$ deterministically, so $\mathbb{P}(\overline\eta_0 < \eta_0^*) = 0 \le \delta_1/2$. Assume $\eta_0^* > 0$, and define the deterministic sequence $t_m := \max(0, \eta_0^* - 1/m)$ for $m \ge 1$; for all $m > 1/\eta_0^*$, $t_m \in [0, \eta_0^*)$. Define the events $E_m := \{\underline F_0(t_m) \ge \beta\} = \{\overline S_0(t_m; \delta_1/2) \le 1 - \beta\}$.
   If $\omega \in \{\overline\eta_0 < \eta_0^*\}$, then there exists $M(\omega) \in \mathbb{N}$ such that $\overline\eta_0(\omega) < t_m < \eta_0^*$ for all $m \ge M(\omega)$. By definition of $\overline\eta_0(\omega) = \sup\{t \in \mathcal{T} : \underline F_0(t)(\omega) < \beta\}$, any $t \in \mathcal{T}$ with $t > \overline\eta_0(\omega)$ must satisfy $\underline F_0(t)(\omega) \ge \beta$. Hence $\omega \in E_m$ for all $m \ge M(\omega)$, meaning
   $$\{\overline\eta_0 < \eta_0^*\} \subseteq \liminf_{m \to \infty} E_m.$$
   By Fatou's Lemma applied to $\mathbf{1}_{E_m} \ge 0$,
   $$\mathbb{P}(\overline\eta_0 < \eta_0^*) \le \mathbb{E}\!\left[\liminf_{m \to \infty} \mathbf{1}_{E_m}\right] \le \liminf_{m \to \infty} \mathbb{P}(E_m).$$
   For every $m > 1/\eta_0^*$, $t_m < \eta_0^*$ implies $F_0(t_m) < \beta \iff S_0(t_m) > 1 - \beta$, so $E_m \subseteq \{\overline S_0(t_m; \delta_1/2) < S_0(t_m)\}$. Because $t_m$ is deterministic, pointwise validity yields $\mathbb{P}(E_m) \le \delta_1/2$ for all $m > 1/\eta_0^*$, and therefore $\mathbb{P}(\overline\eta_0 < \eta_0^*) \le \delta_1/2$.
3. *Union bound:* $\mathbb{P}(\underline\eta_0 \le \eta_0^* \le \overline\eta_0) \ge 1 - \mathbb{P}(\underline\eta_0 > \eta_0^*) - \mathbb{P}(\overline\eta_0 < \eta_0^*) \ge 1 - \delta_1$. $\square$

**Exact Finite Computation (Zero Discretization Error):** As $t$ varies over $\mathcal{T}$, the vector $(\mathbf{1}\{L_i > t\})_{i=1}^n$ changes values only at the distinct observed baseline-arm losses $0 < v_1 < \dots < v_K$ (those with $w_{0, i} > 0$). Thus $\underline S_0(t)$ and $\overline S_0(t)$ are piecewise-constant right-continuous step functions on $\mathcal{T}$, constant on $[0, v_1)$, $[v_1, v_2)$, $\dots$, $[v_K, \sup\mathcal{T}]$. Setting $v_0 := 0$ and $v_{K+1} := \sup\mathcal{T}$:
- $\underline\eta_0 = \min\{v_k : k \in \{0, \dots, K\},\; \underline S_0(v_k; \delta_1/2) \le 1 - \beta\}$ (or $v_{K+1}$ if empty);
- $\overline\eta_0 = v_{k^*}$ where $k^* := \min\{k \in \{0, \dots, K+1\} : \forall j \in \{k, \dots, K\},\; \overline S_0(v_j; \delta_1/2) \le 1 - \beta\}$ (note: at $t = v_K$, $\mathbf{1}\{L_i > v_K\} = 0$ for all baseline units; if $\overline S_0(v_K) > 1-\beta$, then $k^* = K+1$ and $\overline\eta_0 = \sup\mathcal{T}$).
Crucially, this requires **no uniform DKW band** and **no Bonferroni penalty** over $K$, and works identically with empirical-Bernstein or one-sided betting tests at the survival threshold $1-\beta$, with unit-interval normalization as described below.

---

## 3. Exact Localized Comparative Bounds and Joint Fixed-Sequence LTT Validity

Let $\text{UCB}_{\delta'}(\{Z_i\}_{i=1}^n; [a, b])$ denote any valid finite-sample $(1-\delta')$ upper confidence bound for $\mathbb{E}[Z]$ given i.i.d. draws $Z_1, \dots, Z_n \in [a, b]$. We instantiate two concrete UCBs in WP-2.2:
1. **Maurer--Pontil (2009) Empirical-Bernstein UCB:** With range $R = b - a$, sample mean $\hat\mu$, and unbiased sample variance $\hat V = \frac{1}{n-1}\sum_{i=1}^n (Z_i - \hat\mu)^2$,
   $$\text{UCB}_{\text{EB}}(\delta') := \hat\mu + \sqrt{\frac{2 \hat V \log(2/\delta')}{n}} + \frac{7 R \log(2/\delta')}{3(n-1)}.$$
2. **Waudby-Smith & Ramdas (2024, JRSS-B) Betting-Martingale Test (aGRAPA / PrPl-EB):** Rescale $U_i = (Z_i - a)/(b - a) \in [0, 1]$ and $m_0(\tau) = (\tau - a)/(b - a) \in (0, 1)$. Define the predictable ($\mathcal{F}_{i-1}$-measurable) plug-in mean and variance $\hat\mu_{i-1} = \frac{m_0 + \sum_{k<i} U_k}{i}$, $\hat\sigma_{i-1}^2 = \frac{0.0625 + \sum_{k<i}(U_k - \hat\mu_{i-1})^2}{i}$, and predictable betting fraction
   $$\lambda_i := \text{clip}\!\left(\frac{m_0 - \hat\mu_{i-1}}{\hat\sigma_{i-1}^2 + (m_0 - \hat\mu_{i-1})^2},\; 0,\; \frac{c}{1 - m_0}\right), \qquad c = 0.75.$$
   Under $H_0: \mathbb{E}[Z] \ge \tau \iff \mathbb{E}[U] \ge m_0$, $K_t^-(m_0) := \prod_{i=1}^t (1 + \lambda_i(m_0 - U_i))$ is a nonnegative supermartingale with $K_0^- = 1$. By Ville's inequality, $\mathbb{P}(\max_{1 \le t \le n} K_t^-(m_0) \ge 1/\delta') \le \delta'$, so rejecting $H_0$ iff $\max_{t \le n} K_t^-(m_0) \ge 1/\delta'$ is a finite-sample test of size at most $\delta'$.


The pilot inverts a monotone fixed-time betting mixture in Lemma 1 and in the separate-policy betting baseline. For $Z_i \in [a,b]$ and $a<\tau<b$, each fixed $c\in(0,1)$ yields a product of factors $1+c(\tau-Z_i)/(b-\tau)$. The factors are positive and have conditional expectation at most one under $\mathbb{E}[Z_i\mid\mathcal F_{i-1}]\ge\tau$. The equally weighted mean of the terminal products over the nine predeclared fractions in the pilot is an e-value, so a threshold of $1/\delta'$ tests this null at level $\delta'$. Each factor increases with $\tau$ and decreases with $Z_i$. Thus bisection in the hypothesized mean produces a valid outward-rounded upper bound, and the survival tests can be inverted by binary search over the survival breakpoints. Reflection gives lower bounds. The direct localized aGRAPA tests require only predictability at each fixed null and do not use a bisection assumption. [Waudby-Smith and Ramdas, Section 4 and Appendix E.9](https://arxiv.org/abs/2010.09686).

### 3.1 Capped $\text{CVaR}_\beta$ Localized Comparative Bound (Under A6: $L \in [0, B]$)
By [Rockafellar & Uryasev (2002, Theorem 10)](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf), which covers atoms, $\text{CVaR}_\beta(L) = \inf_{\eta \in \mathbb{R}}\{\eta + \frac{1}{1-\beta}\mathbb{E}[(L-\eta)_+]\}$ with infimum attained at $\eta = Q_\beta(L)$. Evaluating the infimum for $L(\pi_\lambda)$ at the deterministic baseline quantile $\eta_0^* = Q_\beta(L(\pi_0))$ gives
$$\text{CVaR}_\beta(L(\pi_\lambda)) - \text{CVaR}_\beta(L(\pi_0)) \le \frac{1}{1-\beta}\Big(\mathbb{E}[(L(\pi_\lambda) - \eta_0^*)_+] - \mathbb{E}[(L(\pi_0) - \eta_0^*)_+]\Big) = \frac{1}{1-\beta}\mathbb{E}\big[Z_{\lambda, \eta_0^*}^{\text{CVaR}}\big],$$
where $Z_{\lambda, \eta}^{\text{CVaR}} := \Delta w_\lambda(X, A)\,(L - \eta)_+ \in \left[-\frac{B-\eta}{\kappa}, +\frac{B-\eta}{\kappa}\right]$ vanishes identically on $\widetilde S_\lambda^c$.

**Exact Supremum Over $\eta \in [\underline\eta_0, \overline\eta_0]$ (Zero Discretization Error):**
- *For Empirical Bernstein (`EB`):* Let $\mathcal{B}_\lambda := \{\underline\eta_0, \overline\eta_0\} \cup (\{L_i : \Delta w_{\lambda, i} \neq 0\} \cap (\underline\eta_0, \overline\eta_0))$, sorted as $\tau_0 < \tau_1 < \dots < \tau_r$. On every sub-interval $[\tau_j, \tau_{j+1}]$, each $Z_{\lambda, \eta, i}^{\text{CVaR}}$ is affine in $\eta$, so $\hat\mu(\eta)$ and $R(\eta) = 2(B-\eta)/\kappa$ are affine in $\eta$, while $\sqrt{\hat V(\eta)} = \frac{1}{\sqrt{n-1}}\|Z(\eta) - \bar Z(\eta)\|_2$ is **convex** in $\eta$ by the triangle inequality. Hence $\eta \mapsto \text{UCB}_{\text{EB}}(\lambda, \eta)$ is convex on every $[\tau_j, \tau_{j+1}]$ and attains its exact maximum at an endpoint:
  $$\widehat U_{\text{CVaR}}^{\text{EB}}(\lambda; \delta_2) := \frac{1}{1-\beta} \max_{\eta \in \mathcal{B}_\lambda} \text{UCB}_{\text{EB}}\!\left(\{Z_{\lambda, \eta, i}^{\text{CVaR}}\}_{i=1}^n; \delta_2\right) = \frac{1}{1-\beta}\sup_{\eta \in [\underline\eta_0, \overline\eta_0]} \text{UCB}_{\text{EB}}(\lambda, \eta).$$
- *For Betting Martingales (`Betting`) via Deterministic Cell Envelope:* Fix a deterministic partition $0 = u_0 < u_1 < \dots < u_G = B$ before inspecting $\mathcal{D}_{\text{cal}}$ (e.g., $G = 250$). Use half-open cells $[u_{g-1},u_g)$ and close the last cell at $B$. For each cell, define the pointwise envelope
  $$\overline Z_{\lambda, g}^{\text{CVaR}} := (\Delta w_\lambda)_+(L - u_{g-1})_+ - (\Delta w_\lambda)_-(L - u_g)_+ \in \left[-\frac{B - u_g}{\kappa},\; +\frac{B - u_{g-1}}{\kappa}\right].$$
  For every $\eta \in [u_{g-1}, u_g]$, $Z_{\lambda, \eta}^{\text{CVaR}} \le \overline Z_{\lambda, g}^{\text{CVaR}}$ almost surely. Because $\eta_0^*$ is deterministic, choose a deterministic cell $g^*$ containing $\eta_0^*$, with a fixed convention at partition boundaries; on $\mathcal{E}_1 = \{\eta_0^* \in [\underline\eta_0, \overline\eta_0]\}$, this cell necessarily intersects $[\underline\eta_0, \overline\eta_0]$. Testing $H_{0, \text{CVaR}}(\lambda)$ by requiring that **every** cell $g$ intersecting $[\underline\eta_0, \overline\eta_0]$ rejects $\mathbb{E}[\overline Z_{\lambda, g}^{\text{CVaR}}] > (1-\beta)\varepsilon_{\text{CVaR}}$ at level $\delta_2$ requires **no union bound over $G$**, incurs **zero discretization error**, and incurs mean-envelope slack $\le p_\lambda (B/G)/(1-\beta)$.

### 3.2 Uncapped Upper-Quantile $Q_\beta$ Localized Comparative Bound (No Support Bound $B$!)
Fix $\varepsilon_Q \ge 0$. Since $F_0(\eta_0^*) \ge \beta$ and $Q_\beta(L(\pi_\lambda)) \le \eta_0^* + \varepsilon_Q \iff F_\lambda(\eta_0^* + \varepsilon_Q) \ge \beta$, a sufficient condition for comparative quantile safety (valid under arbitrary ties and point masses!) is $F_0(\eta_0^*) - F_\lambda(\eta_0^* + \varepsilon_Q) \le 0$. Writing in survival form:
$$F_0(\eta) - F_\lambda(\eta + \varepsilon_Q) = \mathbb{P}(L(\pi_\lambda) > \eta + \varepsilon_Q) - \mathbb{P}(L(\pi_0) > \eta) = \mathbb{E}\big[Z_{\lambda, \eta}^{Q, \varepsilon_Q}\big],$$
where
$$Z_{\lambda, \eta}^{Q, \varepsilon_Q} := \Delta w_\lambda(X, A)\,\mathbf{1}\{L > \eta + \varepsilon_Q\} - w_0(X, A)\,\mathbf{1}\{\eta < L \le \eta + \varepsilon_Q\} \in \left[-\frac{1}{\kappa}, +\frac{1}{\kappa}\right].$$
- When $\varepsilon_Q = 0$, the second term vanishes identically: $Z_{\lambda, \eta}^{Q, 0} = \Delta w_\lambda(X, A)\,\mathbf{1}\{L > \eta\}$, which vanishes on $\widetilde S_\lambda^c$. Proposition 1 gives its variance bound in terms of the conditional survival probabilities within the switch region.
- **Exact breakpoint checks for empirical Bernstein and betting.** Write $Z_{\lambda,\eta}^{Q,\varepsilon_Q}=w_\lambda\mathbf{1}\{L>\eta+\varepsilon_Q\}-w_0\mathbf{1}\{L>\eta\}$. Its sample vector is right-continuous and piecewise constant. Check the lower endpoint, any finite upper endpoint, and all points from $\{L_i-\varepsilon_Q:w_{\lambda,i}>0\}\cup\{L_i:w_{0,i}>0\}$ strictly inside the interval. For $\varepsilon_Q=0$ only losses with $\Delta w_{\lambda,i}\neq 0$ can change the vector. When the upper endpoint is infinite, the final breakpoint also represents the half-line above it. Requiring rejection at every representative is equivalent to checking every $\eta$ in the interval, with a level-$\delta_2$ test at the deterministic true baseline quantile.

### 3.3 Extensive-Margin Risk $R_{\text{ext}}(\pi)$
For harm indicator $H(Y) = \mathbf{1}\{Y > 0\}$ (cost incurrence) or $H(Y) = \mathbf{1}\{Y = 0\}$ (zero conversion), $R_{\text{ext}}(\pi_\lambda) - R_{\text{ext}}(\pi_0) = \mathbb{E}[Z_\lambda^{\text{ext}}]$ where $Z_\lambda^{\text{ext}} := \Delta w_\lambda(X, A)\,H(Y) \in [-1/\kappa, 1/\kappa]$. No baseline-VaR CI is needed; test $\mathbb{E}[Z_\lambda^{\text{ext}}] > \varepsilon_{\text{ext}}$ directly at level $\delta_2$.

### 3.4 Theorem 1 (Exact Finite-Sample Joint Validity of Fixed-Sequence LCT-LTT)
*Assume (A1)--(A5a) (and (A6) if capped $\text{CVaR}_\beta$ is included). Consider any subset of risk functionals $\mathcal{J} \subseteq \{\text{CVaR}_\beta, Q_\beta, R_{\text{ext}}\}$ at shared quantile level $\beta \in (0, 1)$ with nonnegative tolerances $\{\varepsilon_j\}_{j \in \mathcal{J}}$ and overall error budget $\delta \in (0, 1)$. Set $\delta_1 \in (0, \delta)$ (or $\delta_1 = 0$ if $\mathcal{J} = \{R_{\text{ext}}\}$) and $\delta_2 := \delta - \delta_1$. Construct $[\underline\eta_0, \overline\eta_0]$ at level $1 - \delta_1$ via Lemma 1. Over the pre-specified decreasing grid $\Lambda = \{\lambda_1 > \dots > \lambda_M\}$, run Fixed-Sequence Testing:*
- *For $m = 1, \dots, M$, certify $\lambda_m$ iff EVERY risk $j \in \mathcal{J}$ rejects its localized null at level $\delta_2$ (with **no Bonferroni penalty** over $j \in \mathcal{J}$ and **no multiplicity penalty** over $m \in \{1, \dots, M\}$).*
- *Stop at the first $m^*$ that fails certification and output $\hat\lambda := \lambda_{m^*-1}$ (with $\lambda_0 = +\infty \implies \pi_{\hat\lambda} = \pi_0$ if $m^* = 1$).*

*Then, conditional on $\mathcal{D}_{\text{tr}}$,*
$$\mathbb{P}_{\mathcal{D}_{\text{cal}}}\!\Big(\forall j \in \mathcal{J},\;\; \rho_j(L(\pi_{\hat\lambda})) \le \rho_j(L(\pi_0)) + \varepsilon_j\Big) \ge 1 - \delta.$$

**Proof.** Let $\mathcal{N} := \{m \in \{1, \dots, M\} : \exists j \in \mathcal{J},\; \rho_j(L(\pi_{\lambda_m})) > \rho_j(L(\pi_0)) + \varepsilon_j\}$ be the deterministic set of unsafe thresholds. If $\mathcal{N} = \emptyset$, the violation probability is $0 \le \delta$. Otherwise, let $m_0 := \min \mathcal{N}$ (deterministic). Because fixed-sequence testing stops at the first failure, selecting any unsafe threshold implies that the sequence reached and certified $\lambda_{m_0}$. The same implication holds for certifying any unsafe threshold in the whole prefix. If every test passes, the procedure outputs $\lambda_M$. Since $m_0 \in \mathcal{N}$, pick one deterministic $j_0 \in \mathcal{J}$ such that $\rho_{j_0}(L(\pi_{\lambda_{m_0}})) > \rho_{j_0}(L(\pi_0)) + \varepsilon_{j_0}$. Certifying $\lambda_{m_0}$ requires in particular rejecting the localized test for $j_0$ at $\lambda_{m_0}$ (Berger's Intersection-Union Test; Angelopoulos et al., AOAS 2025, Prop. 6):
- If $j_0 = R_{\text{ext}}$, $\mathbb{E}[Z_{\lambda_{m_0}}^{\text{ext}}] > \varepsilon_{\text{ext}}$, so rejection happens with probability $\le \delta_2 \le \delta$.
- If $j_0 = Q_\beta$, $F_\lambda(\eta_0^* + \varepsilon_Q) < \beta \le F_0(\eta_0^*)$, so $\mathbb{E}[Z_{\lambda_{m_0}, \eta_0^*}^{Q, \varepsilon_Q}] > 0$. On $\mathcal{E}_1 = \{\eta_0^* \in [\underline\eta_0, \overline\eta_0]\}$, $\eta_0^*$ falls in a breakpoint interval represented in $\mathcal{B}_{\lambda_{m_0}}^Q$, so rejection implies the single deterministic test at $\eta_0^*$ falsely rejected (probability $\le \delta_2$). By union bound with Lemma 1, $\mathbb{P}(\text{reject } j_0) \le \mathbb{P}(\mathcal{E}_1^c) + \delta_2 \le \delta_1 + \delta_2 = \delta$.
- If $j_0 = \text{CVaR}_\beta$, $\frac{1}{1-\beta}\mathbb{E}[Z_{\lambda_{m_0}, \eta_0^*}^{\text{CVaR}}] \ge \text{CVaR}_\beta(\pi_{\lambda_{m_0}}) - \text{CVaR}_\beta(\pi_0) > \varepsilon_{\text{CVaR}}$. On $\mathcal{E}_1$, either $\text{UCB}_{\text{EB}}(\lambda_{m_0}, \eta_0^*) \le (1-\beta)\varepsilon_{\text{CVaR}}$ (for EB) or the deterministic cell $g^*$ containing $\eta_0^*$ falsely rejected $\mathbb{E}[\overline Z_{\lambda_{m_0}, g^*}^{\text{CVaR}}] > (1-\beta)\varepsilon_{\text{CVaR}}$ (for Betting). In both cases, $\mathbb{P}(\text{reject } j_0) \le \mathbb{P}(\mathcal{E}_1^c) + \delta_2 \le \delta_1 + \delta_2 = \delta$.

Hence $\mathbb{P}(\exists j \in \mathcal{J} : \rho_j(\pi_{\hat\lambda}) > \rho_j(\pi_0) + \varepsilon_j) \le \delta$. $\square$

If both uncapped quantiles and capped CVaR are certified, construct the uncapped baseline interval and map both endpoints through $x\mapsto\min(x,B)$ for the capped CVaR tests. This map preserves the confidence event because $Q_\beta(\min(Y,B))=\min(Q_\beta(Y),B)$. The identity follows from equality of the capped and uncapped CDFs below $B$ and the jump to 1 at $B$. Alternatively allocate $\delta_1$ across the separate intervals.

*(Remark on multiple quantile levels or subgroups: If certifying $K$ distinct quantile levels $\{\beta_1, \dots, \beta_K\}$ simultaneously, split $\delta_1$ as $\delta_1/K$ across the $K$ baseline-VaR CIs so $\mathbb{P}(\bigcap_k \mathcal{E}_{1,k}) \ge 1 - \delta_1$, while still paying ZERO Bonferroni on $\delta_2$ via IUT. If certifying a single global $\hat\lambda$ safe across all $G$ finite subgroups simultaneously, the joint null is again a union over subgroups and requires zero Bonferroni on $\delta_2$; if selecting separate thresholds $\{\hat\lambda_g\}_{g=1}^G$ per subgroup, allocate $\delta/G$ per subgroup via Bonferroni. The shared CI budget also covers all subgroup baseline quantiles. For a fixed subgroup of positive probability, condition on the subgroup inclusion indicators to apply the same i.i.d. calculation to its observations. When a selected test is unavailable for lack of calibration observations, retain the structural no-switch fallback. This argument concerns independent units and does not automatically extend to household dependence.)*

---

## 4. Variance Localization and Effect of Baseline-VaR CI Width

### Proposition 1 (Variance Localization & Baseline-VaR Perturbation Scaling)
*Under (A1)--(A6), for any fixed $\lambda \in \Lambda$ with effective switch mass $\tilde p_\lambda = \mathbb{P}(X \in \widetilde S_\lambda)$ and any $\eta \in [0, B]$. At $\tilde p_\lambda=0$ the contrast vanishes almost surely, and both the variance and pivot perturbation are zero. In the conditional-survival formulas below assume $\tilde p_\lambda>0$.*
1. **CVaR Variance Localization:**
   $$\text{Var}\big(Z_{\lambda, \eta}^{\text{CVaR}}\big) \le \mathbb{E}\!\left[\mathbf{1}\{X \in \widetilde S_\lambda\}\left(\frac{\mathbb{E}[(L(\hat\pi)-\eta)_+^2 \mid X]}{e(\hat\pi(X)\mid X)} + \frac{\mathbb{E}[(L(\pi_0)-\eta)_+^2 \mid X]}{e(\pi_0(X)\mid X)}\right)\right] \le \frac{2(B - \eta)^2}{\kappa}\,\tilde p_\lambda.$$
   Separate inverse-propensity stop-loss scores each have a second-moment bound of $(B-\eta)^2/\kappa$, which contains no factor of $\tilde p_\lambda$. A $\sqrt{\tilde p_\lambda}$ improvement in standard-error scale is possible when these separate variances remain bounded away from zero as $\tilde p_\lambda$ decreases and the range term is negligible. An improvement of that size is not a uniform lower bound on comparative power.
2. **Quantile variance localization at zero tolerance.** Conditioning on $X$ and using the disjoint action indicators on $\widetilde S_\lambda$ gives
   $$\operatorname{Var}(Z_{\lambda,\eta_0^*}^{Q,0})\le\frac{\tilde p_\lambda}{\kappa}\left[S_{\hat\pi\mid\widetilde S_\lambda}(\eta_0^*)+S_{\pi_0\mid\widetilde S_\lambda}(\eta_0^*)\right].$$
   This always yields $O(\tilde p_\lambda/\kappa)$. An additional factor of $1-\beta$ requires conditional tail probabilities within the switch region to be $O(1-\beta)$. The marginal baseline quantile does not impose this condition. For example, let $X$ be uniform on $[0,1]$ and $L(0)=X$. The top $1-\beta$ covariate region has conditional baseline survival equal to 1 at $\eta_0^*=\beta$.
3. **Baseline-VaR CI Width Impact (Risk R4 Analysis):** *On the coverage event, for any $\eta \in [\underline\eta_0, \overline\eta_0]$,*
   $$\big|\mathbb{E}[Z_{\lambda, \eta}^{\text{CVaR}}] - \mathbb{E}[Z_{\lambda, \eta_0^*}^{\text{CVaR}}]\big| = \left|\int_{\eta_0^*}^\eta \big(S_\lambda(u) - S_0(u)\big)\,du\right| \le \tilde p_\lambda \cdot |\eta - \eta_0^*| \cdot \sup_{u \in [\underline\eta_0, \overline\eta_0]}\big|S_{\hat\pi \mid \widetilde S_\lambda}(u) - S_{\pi_0 \mid \widetilde S_\lambda}(u)\big|.$$
   This is a deterministic Lipschitz bound. A positive continuous baseline density near $\eta_0^*$ converts small survival error into small quantile error. The second-moment bound for the survival score gives a local standard-error scale $\sqrt{(1-\beta)/(\kappa n)}$ at a fixed level. A corresponding uniform rate for the implemented confidence interval requires a local inversion argument and is deferred.

---

## 5. Observational Extension: $\Gamma$-Sensitivity and Additive TV Slack

When true propensity $e(a \mid x)$ is unknown and estimated by $\hat e(a \mid x) \ge \hat\kappa$ on $\mathcal{D}_{\text{tr}}$:
1. **Exact Finite-Sample $\Gamma$-Sensitivity (Assumption A5b):** Suppose $\Gamma^{-1} \le \hat e(a \mid x)/e(a \mid x) \le \Gamma$ ($\Gamma \ge 1$) for all $(x, a)$.
   - *Baseline-VaR CI:* Replace $S_0(t)$ bounds with $\underline S_0^\Gamma(t) := \Gamma^{-1}\underline{\hat S}_0(t)$ and $\overline S_0^\Gamma(t) := \min(1, \Gamma \overline{\hat S}_0(t))$ in Lemma 1.
   - *Localized Contrast Inflation:* For any nonnegative score $g(L) \ge 0$ (covering $(L-\eta)_+$, $\mathbf{1}\{L > \eta\}$, and $H(Y)$), define
     $$Z_\lambda^\Gamma(g) := \Gamma(\Delta \hat w_\lambda)_+\,g(L) - \Gamma^{-1}(\Delta \hat w_\lambda)_-\,g(L) \in \left[-\frac{\|g\|_\infty}{\Gamma \hat\kappa},\; +\frac{\Gamma\|g\|_\infty}{\hat\kappa}\right].$$
     Because $\Delta w_\lambda g(L) \le Z_\lambda^\Gamma(g)$ almost surely while preserving $\widetilde S_\lambda^c$ cancellation, Theorem 1 applies at zero quantile tolerance, and to the CVaR and extensive-margin scores, under (A5b). With $\varepsilon_Q>0$, inflate $Z^{Q,\varepsilon_Q}$ as $\Gamma(\hat Z^{Q,\varepsilon_Q})_+-\Gamma^{-1}(\hat Z^{Q,\varepsilon_Q})_-$, where $\hat Z$ uses estimated propensities in both terms. The true score is the estimated score multiplied by $\hat e(A\mid X)/e(A\mid X)$, so the inequality still holds. At positive tolerance the shared-region term can be nonzero.
2. **Additive Localized Propensity-Error Slack:** Without a uniform $\Gamma$ bound, for any bounded $g \ge 0$,
   $$\Big|\mathbb{E}\big[(\Delta w_\lambda - \Delta \hat w_\lambda)\,g(L)\big]\Big| \le \frac{2\|g\|_\infty}{\hat\kappa}\,\mathbb{E}\Big[\mathbf{1}\{X \in \widetilde S_\lambda\}\,|e(1 \mid X) - \hat e(1 \mid X)|\Big],$$
   This follows by conditioning on $X$ and $A$ and using $e(a\mid X)|1/e(a\mid X)-1/\hat e(a\mid X)|=|e(a\mid X)-\hat e(a\mid X)|/\hat e(a\mid X)$ for each of the two differing actions. This slack is a bound on propensity bias in the risk-score units. It requires an independent upper bound on the displayed propensity-error expectation before it can calibrate a certificate. The baseline survival interval and positive-tolerance quantile term need their respective nonlocalized propensity-error allowances as well.

---

## 6. Exact LCT-LTT Algorithm Summary

1. **Inputs:** Calibration set $\mathcal{D}_{\text{cal}} = \{(X_i, A_i, L_i)\}_{i=1}^n$; fixed $(\pi_0, \hat\pi, s, e)$ from $\mathcal{D}_{\text{tr}}$; decreasing grid $\Lambda = \{\lambda_1 > \dots > \lambda_M\}$; risk set $\mathcal{J} \subseteq \{\text{CVaR}_\beta, Q_\beta, R_{\text{ext}}\}$; tolerances $\{\varepsilon_j\}$; level $\delta \in (0, 1)$ and split $\delta_1 \in (0, \delta)$ (default $\delta_1 = 0.2\delta$ if $\text{CVaR}_\beta$ or $Q_\beta \in \mathcal{J}$, else $0$), $\delta_2 = \delta - \delta_1$; cap $B$ (only if $\text{CVaR}_\beta \in \mathcal{J}$); optional deterministic grid $\{u_g\}_{g=0}^G$ on $[0, B]$ (for Betting CVaR).
2. **Step 1 (Baseline-VaR CI):** If $\text{CVaR}_\beta \in \mathcal{J}$ or $Q_\beta \in \mathcal{J}$, compute $[\underline\eta_0, \overline\eta_0]$ at level $1-\delta_1$ via Lemma 1 using survival-complement indicators $w_{0, i}\mathbf{1}\{L_i > v_k\}$ across distinct baseline-arm losses $\{v_k\}$.
3. **Step 2 (Fixed-Sequence Localized Testing):** Initialize $\hat\lambda \leftarrow +\infty$. For $m = 1, \dots, M$:
   - Compute localized weights $\Delta w_{\lambda_m, i}$.
   - For each $j \in \mathcal{J}$, run the exact level-$\delta_2$ localized test over $[\underline\eta_0, \overline\eta_0]$ (Section 3.1--3.3).
   - If ALL $j \in \mathcal{J}$ reject, set $\hat\lambda \leftarrow \lambda_m$; else `break`.
4. **Output.** Return the certified threshold $\hat\lambda$ and policy $\pi_{\hat\lambda}$. An all-zero sample contrast does not establish equality of candidate and baseline. Structural equality can be used when the policies are known to agree on the target support.


---

## 10. Path A Addendum: Training-Optimized GRO E-Values, Training-Anchored Fixed-Sequence LTT, AIPW-Localized Contrasts, and Propensity-Gated Support Bounds

This addendum records four exact finite-sample upgrades developed after auditing the v1 Gate G3 failure (where `aGRAPA`'s miscalibrated variance prior `0.0625`, bottom-up ladder entry at `5%`/`15%`, unreduced extensive-margin variance, and hard-coded joint learner scoring strangled Capped CVaR, weak-overlap MEPS, and joint deployment power). Every upgrade preserves exact finite-sample FWER control at level $\delta$ under the same sample-splitting assumptions (Assumptions A1–A4).

### 10.1 Theorem A1 (Training-Optimized GRO E-Value Betting & Exact Sweep Validity)

Let $\mathcal{D}_{\mathrm{tr}}$ and $\mathcal{D}_{\mathrm{cal}}$ be independent i.i.d. draws from the logged distribution $P$, and let $Z_i \in [a, b]$ ($i \in \mathcal{D}_{\mathrm{cal}}$) be any bounded calibration contrast whose definition depends only on $\mathcal{D}_{\mathrm{tr}}$ and $(X_i, A_i, L_i)$, with null hypothesis $H_0: \mathbb{E}[Z_i \mid \mathcal{D}_{\mathrm{tr}}] > \tau$ for $a < \tau < b$. Let $\hat\theta \in [0, 1/(b - \tau))$ be **any** scalar measurable with respect to $\mathcal{D}_{\mathrm{tr}}$ alone, specifically the shrunk empirical growth-rate optimal (GRO) stake
$$
\hat\theta \;=\; c_{\mathrm{shrink}} \cdot \arg\max_{\theta \in [0,\, c_{\max}/(b - \tau)]} \sum_{j \in \mathcal{D}_{\mathrm{tr}}} \log\!\bigl(1 + \theta(\tau - Z_j^{\mathrm{tr}})\bigr),
\qquad c_{\mathrm{shrink}} \in (0, 1],\; c_{\max} \in (0, 1).
$$
Define the calibration e-value $E_n(\hat\theta) = \prod_{i \in \mathcal{D}_{\mathrm{cal}}} \bigl(1 + \hat\theta(\tau - Z_i)\bigr)$. Then:

1. **Exact Finite-Sample Validity:** Under $H_0$, $1 + \hat\theta(\tau - Z_i) > 0$ almost surely and $\mathbb{E}[E_n(\hat\theta) \mid \mathcal{D}_{\mathrm{tr}}] = \prod_{i=1}^{n_{\mathrm{cal}}} (1 + \hat\theta(\tau - \mathbb{E}[Z_i \mid \mathcal{D}_{\mathrm{tr}}])) \le 1$. By Markov's inequality, $\mathbb{P}(E_n(\hat\theta) \ge 1/\delta \mid \mathcal{D}_{\mathrm{tr}}) \le \delta$, hence unconditionally $\mathbb{P}(E_n(\hat\theta) \ge 1/\delta) \le \delta$.
2. **Exact Permutation Invariance:** $E_n(\hat\theta)$ is a symmetric product over $\mathcal{D}_{\mathrm{cal}}$, eliminating the order dependence of sequential adaptive bets (`aGRAPA`) and the $\log K$ mixture penalty of multi-point grids.
3. **Airtight Intersection-Union Sweeps Over $[\underline\eta_0, \overline\eta_0]$:**
   - **Relative Quantile Test:** Because calibration breakpoints $\{k_m\}$ depend on $\mathcal{D}_{\mathrm{cal}}$, we compute **one** $\mathcal{D}_{\mathrm{tr}}$-fixed stake $\hat\theta_\lambda$ per candidate $\lambda$ (evaluated on $\mathcal{D}_{\mathrm{tr}}$ at the training IPW baseline quantile $\hat\eta_0^{\mathrm{tr}}$). Since $\hat\theta_\lambda$ does not vary with $\eta$, passing the GRO test at every calibration knot $k_m \in [\underline\eta_0, \overline\eta_0]$ deterministically implies passing at the single unknown population quantile $\eta_0^*$ whenever $\eta_0^* \in [\underline\eta_0, \overline\eta_0]$. At $\tau_Q = 0$, the exact minimum over all knots in $[\underline\eta_0, \overline\eta_0]$ reduces to one $O(n_{\mathrm{cal}} \log n_{\mathrm{cal}})$ prefix sum of $\log(1 - \hat\theta_\lambda \cdot \mathrm{inflate}(\Delta w_i, \gamma))$.
   - **Relative Capped CVaR Test:** Partition $[0, B]$ into a **deterministic** grid $0 = g_0 < g_1 < \dots < g_M = B$ independent of $\mathcal{D}_{\mathrm{cal}}$. For each cell $c$ covering $[\underline\eta_0, \overline\eta_0]$, compute $\hat\theta_{\lambda, c}$ strictly from $\mathcal{D}_{\mathrm{tr}}$ on the conservative cell score $Z_j^{(c)} = \gamma(\Delta w_j)_+(Y_j \wedge B - g_c)_+ + \gamma^{-1}(\Delta w_j)_-(Y_j \wedge B - g_{c+1})_+$. Because the true $\eta_0^* \wedge B$ belongs to a single deterministic cell $c^*$ independent of $\mathcal{D}_{\mathrm{cal}}$, passing all cells intersecting $[\underline\eta_0, \overline\eta_0]$ implies passing at $(c^*, \hat\theta_{\lambda, c^*})$ on the event $\{\eta_0^* \in [\underline\eta_0, \overline\eta_0]\}$, yielding FWER $\le \delta_1 + \delta_2 = \delta$.

*Groundings:* Single-stake fixed-time betting e-values (Waudby-Smith & Ramdas, 2024, *JRSS-B*, Section 2; Grünwald, de Heide & Koolen, 2024, *JRSS-B*, GRO e-variables) combined with the Phase 2 intersection-union localization theorem (Theorem 1 above).

### 10.2 Proposition A2 (Training-Anchored Fixed-Sequence LTT Ordering)

Let $\Lambda = (\lambda_1, \dots, \lambda_K)$ be a finite candidate family fixed on $\mathcal{D}_{\mathrm{tr}}$, and let $\sigma$ be **any** injection into $\{1, \dots, K\}$ determined solely from $\mathcal{D}_{\mathrm{tr}}$ and the known deterministic sample size $n_{\mathrm{cal}}$ (specifically, $\sigma = (\hat k_{\mathrm{anchor}}, \hat k_{\mathrm{anchor}} + 1, \dots, K)$, where $\hat k_{\mathrm{anchor}}$ is the smallest index whose bottleneck training-predicted GRO calibration pass $z$-score clears $z_{\mathrm{anchor}} = 3.0$, or $\arg\max_k z_k$ if none does). Testing $\lambda_{\sigma(1)}, \lambda_{\sigma(2)}, \dots$ at unadjusted level $\delta_2$ (with shared $\delta_1$ VaR confidence interval) and stopping at the first failure guarantees FWER $\le \delta_1 + \delta_2 = \delta$ simultaneously over all certified candidates.

*Proof:* Conditionally on $\mathcal{D}_{\mathrm{tr}}$, $\sigma$ is a fixed deterministic sequence. Let $r^*$ be the smallest position in $\sigma$ such that $\lambda_{\sigma(r^*)}$ violates any required risk constraint (overall or in a required subgroup). On $\mathcal{E}_{\mathrm{VaR}}$, certifying any unsafe policy requires rejecting the true null at the single deterministic index $\lambda_{\sigma(r^*)}$ and its violated component/group, which occurs with probability at most $\delta_2$ (Angelopoulos et al., 2022, Theorem 1; Hsu & Berger, 1999). Taking union bound with $\mathcal{E}_{\mathrm{VaR}}^c$ and expectation over $\mathcal{D}_{\mathrm{tr}}$ yields FWER $\le \delta$. $\square$

### 10.3 Proposition A3 (AIPW-Localized Contrast With Zero Range Inflation)

For any bounded outcome transform $g(L) \in [0, M]$ and any $\mathcal{D}_{\mathrm{tr}}$-measurable predictors $\hat m_0(x), \hat m_1(x) \in [0, M]$, define the localized AIPW score with propensity sensitivity $\gamma \ge 1$:
$$
\psi_\lambda^{\mathrm{AIPW}}(X, A, L; \gamma) \;=\; \mathbf{1}_{\widetilde S_\lambda}(X)\Bigl[\bigl(\hat m_{\hat\pi_\lambda}(X) - \hat m_{\pi_0}(X)\bigr) + \mathrm{inflate}_\gamma\!\bigl(w_{\hat\pi_\lambda}(g(L) - \hat m_{\hat\pi_\lambda}(X)) - w_{\pi_0}(g(L) - \hat m_{\pi_0}(X))\bigr)\Bigr],
$$
where $\mathrm{inflate}_\gamma(r) = \gamma r \mathbf{1}\{r \ge 0\} + \gamma^{-1} r \mathbf{1}\{r < 0\}$. Then:

1. **Exact $\widetilde S_\lambda^c$ Cancellation:** $\psi_\lambda^{\mathrm{AIPW}} = 0$ whenever $\hat\pi_\lambda(X) = \pi_0(X)$.
2. **Exact Dominance & Unbiasedness:** Under multiplicative propensity ratio bound $\gamma^{-1} \le e_{\mathrm{nom}}(a\mid x)/e_{\mathrm{true}}(a\mid x) \le \gamma$, $\mathbb{E}[\psi_\lambda^{\mathrm{AIPW}} \mid \mathcal{D}_{\mathrm{tr}}] \ge \mathbb{E}[g(L(\hat\pi_\lambda)) - g(L(\pi_0)) \mid \mathcal{D}_{\mathrm{tr}}]$, with exact equality at $\gamma = 1$ for arbitrarily misspecified $\hat m_0, \hat m_1 \in [0, M]$.
3. **Zero Range Inflation:** On $\widetilde S_\lambda$ with $w_{\hat\pi_\lambda} \le w_{1,\max}$ and $w_{\pi_0} \le w_{0,\max}$, let $w_{\max} = \max(w_{0,\max}, w_{1,\max})$. Then
$$
\psi_\lambda^{\mathrm{AIPW}}(X, A, L; \gamma) \;\in\; \bigl[-\max(w_{\max}/\gamma,\, 1)\, M,\; \gamma w_{\max} M\bigr] \quad \text{almost surely.}
$$
In particular, at $\gamma = 1$ (where $w_{\max} \ge 2$), $\psi_\lambda^{\mathrm{AIPW}} \in [-w_{\max} M, +w_{\max} M]$, identically matching the plain IPW support width while reducing variance whenever $\mathrm{Var}(g(L) \mid X, A) < \mathrm{Var}(g(L) \mid A)$ (notably for the extensive-margin indicator $g(L) = \mathbf{1}\{L > 0\}$).

### 10.4 Proposition A4 (Propensity-Gated Localized Support Bounds)

When the logging propensity function $e(a \mid x)$ is known, restricting switches to $\{x : e(\hat\pi(x)\mid x) \ge \kappa_1,\; e(\pi_0(x)\mid x) \ge \kappa_0\}$ for deterministic thresholds $(\kappa_0, \kappa_1) \in [\kappa, 0.5]^2$ chosen solely on $\mathcal{D}_{\mathrm{tr}}$ defines a valid deterministic policy family $\hat\pi_\lambda^{\mathrm{gate}}(x)$ whose contrast weights satisfy $\Delta w_\lambda(X, A) \in [-1/\kappa_0, +1/\kappa_1]$ almost surely on $\mathcal{D}_{\mathrm{cal}}$, eliminating global worst-case overlap inflation ($1/\kappa$) in weak-overlap regimes such as MEPS ($\kappa = 0.056$).
