# Phase 4 Protocol v2: Confirmatory Gate G3 Evaluation of Path A Upgrades

**Status:** Protocol v2 (transparent post-audit revision; v1 artifacts in `results/phase4_sim/` and `docs/phase4_preregistration.md` are preserved unmodified).

## 1. Motivation for Protocol v2

Post-mortem audit of the v1 Gate G3 `KILL` verdict identified four methodological and experimental defects:

1. **Self-defeating comparator (`B5+ext`):** In v1, `B5+ext` inherited LCT-LTT's localized signed contrast, betting engine, two-part learner, fixed-sequence LTT, and extensive-margin guard at full $\delta = 0.05$ (vs. $\delta_2 = 0.04$ for LCT-LTT), making LCT-LTT's certified prefix a deterministic subset of `B5+ext`'s prefix whenever $\hat\eta_0^{\mathrm{tr}} \in [\underline\eta_0, \overline\eta_0]$ (149/150 focal runs).
2. **Non-conflicting focal DGPs:** None of the four v1 focal cells produced a genuine marginal mean–tail conflict (uncertified `B4` had 0/50 unsafe runs on DGP 5, DGP 6, and OHIE, and 5/50 on MEPS), nor did any test survival-curve crossings near $\eta_0^*$ or catastrophic-tail severity traps.
3. **Algorithmic power bugs:** `aGRAPA`'s hard-coded variance prior `0.0625` (2,300× larger than scaled CVaR variance) strangled Capped CVaR (0.40%); bottom-up `5%`/`15%`-first ordering killed Capped CVaR and MEPS weak overlap; unreduced extensive-margin variance strangled joint tests; and `TwoPartDistributionalLearner.predict` hard-coded `min(tail, ext)` even for quantile-only runs.
4. **Premature drop of Capped CVaR:** Fixing `aGRAPA` via Training-Optimized GRO e-values and Training-Anchored ordering restores Capped CVaR power from 0.40% to ~35% at $P(S^*)=0.35, B=50$ while strictly outperforming fixed-cutoff `B5` and quantile-only prior art on severity traps.

## 2. Path A Method Suite

All methods share identical `(train, calibration, evaluation)` splits per seed (`seeds = 1000..1049`, $R = 50$ paired replications per design):

- **`LCT-LTT-GRO`** (flagship): Training-Optimized GRO e-value betting (`bound="gro"`), Training-Anchored fixed-sequence ordering (`anchored=True`), target-aligned `TwoPartDistributionalLearner.scorer(spec)`, and AIPW-localized extensive-margin control variates (`ext_control`) whenever extensive margin is active.
- **`LCT-LTT-GRO-Gate`** (on weak-overlap MEPS): Adds deterministic training-derived propensity gating (`propensity_gated_family`) with verified localized support bounds.
- **`LCT-LTT-aGRAPA`** (v1 ablation): Original v1 adaptive betting without GRO/anchoring/AIPW, quantifying the direct Path A improvement.
- **`B1-survival` / `B1-opra`**: Separate two-policy finite-sample bounds on the same target-aligned candidate family.
- **`B2-GRO`**: Mean causal forest (`grf`) certified with the exact same GRO + anchored localized certifier (isolates learner ranking advantage).
- **`B3`**: Uncertified plug-in CVaR ITR.
- **`B4`**: Uncertified mean causal forest (`grf`).
- **`B5-GRO` / `B5+ext-GRO`**: Fixed training cutoff $\hat\eta_0^{\mathrm{tr}}$ indicator LTT using the same GRO engine (completely fair matched engine comparison; exposed to survival-crossing and catastrophic-tail failures!).
- **`Oracle`**: Full-information population benchmark over the oracle nested family.

## 3. Confirmatory Design Suite (7 Designs × 50 Seeds = 350 Paired Replications)

1. **`v2-d05-conflict`** ($n_{\mathrm{cal}}=5000, \beta=0.90, \kappa=0.25$): Repaired genuine marginal mean–tail conflict ($R_1: u < 0.35$ safe; $R_2: 0.35 \le u < 0.70$ larger mean gain, extensive-margin reduction, but strongly elevated tail survival so $R_1 \cup R_2$ is marginally unsafe on $Q_{0.90}$; $R_3: u \ge 0.70$ uniformly harmful). Tests **Joint (Q + AIPW-Ext)** and **Q-only**.
2. **`v2-cvar-trap`** ($n_{\mathrm{cal}}=5000, \beta=0.90, B=50, \kappa=0.25$): Catastrophic-tail severity trap ($R_1: u < 0.35$ safe across all metrics; $R_2: u \ge 0.35$ reduces mean, extensive margin, and $Q_{0.90}$, while a 7% catastrophic loss at $L_1=45$ severely increases $\mathrm{CVaR}_{0.90}$). Tests **Capped CVaR ($B=50$)** against `B5`, `B5+ext`, `B4`, `B3`, `B1-opra`, and `LCT-LTT-aGRAPA`.
3. **`v2-cvar-power`** ($n_{\mathrm{cal}}=5000, \beta=0.90, \kappa=0.25$): Clean Capped CVaR power benchmark ($P(S^*)=0.35$) across cap frontier $B \in \{30, 50, 100\}$ ($B=50$ primary; $B=30, 100$ transparently reported to document the honest cap boundary).
4. **`v2-survival-crossing`** ($n_{\mathrm{tr}}=1000, n_{\mathrm{cal}}=5000, \beta=0.90, \kappa=0.25$): Uniformly unsafe treatment on $Q_{0.90}$ whose survival curve crosses $S_0(t)$ sharply above $\eta_0^*$ while reducing extensive margin, breaking plug-in fixed-cutoff `B5` and `B5+ext` while testing `LCT-LTT-GRO` FWER.
5. **`v2-d09-meps`** ($n_{\mathrm{cal}}=5000, \beta=0.90, \kappa=0.056$): Exact MEPS baseline & weak-overlap propensity model with repaired marginal conflict in $R_2$, evaluating both global `LCT-LTT-GRO` and propensity-gated `LCT-LTT-GRO-Gate` (Joint and Q-only).
6. **`v2-d09-ohie`** ($n_{\mathrm{cal}}=5000, \beta=0.90, \kappa=0.28$): Exact OHIE baseline & 4-stratum propensity model with repaired marginal conflict in $R_2$, evaluating Joint (Q + AIPW-Ext) and Q-only.
7. **`v2-null-strict`** ($n_{\mathrm{cal}}=5000, \beta=0.90, \kappa=0.25$): Uniformly strict near-null auditing finite-sample FWER $\le 0.05$ across Joint (Q + AIPW-Ext), Q-only, and Capped CVaR ($B=50$).

## 4. Revised Gate G3 Decision Criteria

Proceed to Phase 5 (Applied Study) if and only if ALL of the following hold across the 50-seed confirmatory runs:

1. **Safety (FWER):** `LCT-LTT-GRO` empirical prefix FWER $\le 0.065$ (nominal $\delta = 0.05$, 50-seed binomial upper guard) on EVERY design and risk target (`v2-null-strict`, `v2-survival-crossing`, `v2-cvar-trap`, `v2-d05-conflict`, `v2-d09-meps`, `v2-d09-ohie`, `v2-cvar-power`).
2. **Decisive Separation vs. Separate Bounds (`B1`):** `LCT-LTT-GRO` beats the best `B1` method by $\ge +15$ percentage points deployment (and $>25\%$ safe welfare) on `v2-d05-conflict`, `v2-cvar-power` ($B=50$), `v2-cvar-trap`, and `v2-d09-ohie`.
3. **Decisive Separation vs. Mean-Only Learners (`B2` & `B4`):** `LCT-LTT-GRO` beats `B2-GRO` in safe welfare on the repaired conflict designs (`v2-d05-conflict`, `v2-d09-ohie`, `v2-cvar-trap`) while uncertified `B4` suffers severe safety failure ($\ge 50\%$ unsafe).
4. **Decisive Separation vs. Fixed-Cutoff `B5` / `B5+ext`:** Fixed-cutoff `B5` / `B5+ext` suffers clear safety failure ($> 5\%$ FWER) on BOTH `v2-cvar-trap` (catastrophic-tail severity) and `v2-survival-crossing` (pilot-to-deployment quantile crossing), where `LCT-LTT-GRO` maintains finite-sample safety while delivering nontrivial safe welfare on `v2-cvar-trap` and `v2-cvar-power`.
5. **Weak-Overlap Viability (MEPS):** `LCT-LTT-GRO` (and/or `LCT-LTT-GRO-Gate`) delivers nontrivial safe deployment ($\ge 15\%$) on `v2-d09-meps` with FWER $\le 0.065$ (resolving v1's 0.00% collapse).
