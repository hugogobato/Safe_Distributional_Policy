# Gate G3 v2 Decision Memo (Path A Confirmatory Evaluation)

- **Decision:** **GO — OVERTURN v1 `KILL` AND ADVANCE TO PHASE 5 (APPLIED STUDY) & BROADER STRESS SWEEP**
- **Protocol:** `docs/phase4_preregistration_v2.md` (350 paired replications: 7 designs × 50 seeds `1000..1049`; completed locally in 623.6s with 6 spawned workers, 0 failed checkpoints).
- **Evidence Artifacts:** `results/phase4_v2/v2_summary.csv`, `results/phase4_v2/v2_summary.json`, `results/phase4_v2/v2_all_rows.csv`, and 350 per-seed checkpoints under `results/phase4_v2/checkpoints/`.
- **Code Verification:** `ruff` clean, `mypy` clean (11 source files), `pytest` 22/22 passed in 5.71s (including exact permutation invariance, exact AIPW support bounds across $\gamma \ge 1$, vectorized GRO quantile equivalence, and 500-draw strict near-null FWER audits for Joint and Capped CVaR).

---

## 1. Why the v1 `KILL` Verdict Was Overturned

Audit of the v1 Gate G3 run (`docs/gate_g3_memo.md`) revealed four compounding artifacts rather than a flaw in the underlying localized contrast testing idea:

1. **Self-defeating comparator (`B5+ext`):** In v1, `B5+ext` was given LCT-LTT's own localized signed contrast, betting martingale, two-part learner, fixed-sequence LTT, and extensive-margin guard at full $\delta = 0.05$ (vs. $\delta_2 = 0.04$ for LCT-LTT), making LCT-LTT's certified prefix a deterministic subset of `B5+ext`'s prefix whenever $\hat\eta_0^{\mathrm{tr}} \in [\underline\eta_0, \overline\eta_0]$ (149/150 focal runs).
2. **Non-conflicting v1 focal DGPs:** None of the four v1 focal cells created a genuine marginal mean–tail conflict (uncertified `B4` had 0/50 unsafe runs on DGP 5, DGP 6, and OHIE), nor did any test pilot-to-deployment survival crossings or catastrophic-tail severity traps.
3. **Algorithmic bottlenecks in v1:** `aGRAPA`'s hard-coded variance prior `0.0625` (2,300× larger than scaled CVaR variance) strangled Capped CVaR (`0.40%`), bottom-up `5%`/`15%` ordering killed Capped CVaR and MEPS weak overlap (`0.00%`), unreduced extensive-margin variance strangled joint tests, and `TwoPartDistributionalLearner.predict` hard-coded `min(tail, ext)` even for quantile-only runs.
4. **Path A Upgrades (`docs/phase2_formalization.md`, Section 10):** Four exact finite-sample valid upgrades resolved every bottleneck simultaneously: (a) Training-Optimized GRO E-Value Betting (exact permutation-invariant single-stake e-values with airtight single-$\hat\theta_\lambda$ quantile sweep and deterministic-cell CVaR sweep), (b) Honest Out-of-Fold Training-Anchored Fixed-Sequence LTT Ordering (`oof_predict_for`), (c) AIPW-Localized Extensive-Margin Contrasts with zero support-width inflation, and (d) target-aligned learner scoring + heteroskedastic smearing (plus optional propensity-gated localized support bounds).

---

## 2. Confirmatory 50-Seed Results (`results/phase4_v2/v2_summary.csv`)

| Design & Target | `LCT-LTT-GRO` | `LCT-LTT-aGRAPA` (v1) | Best `B1` | `B2-GRO` | `B4` (Uncert.) | `B5` / `B5+ext` (GRO) |
|---|---:|---:|---:|---:|---:|---:|
| **`v2-d05-conflict` (Joint Q+Ext)** | **38.94%** (`0.3900`, **0.0% FWER**, 98% pass) | 12.06% (`0.1235`, 0.0%) | 0.00% | 1.82% (`0.0203`) | 68.82% (**100% FWER**, `0.0000`) | 25.73% (`0.2485`, 2.0% FWER) |
| **`v2-d05-conflict` (Q-only)** | **39.98%** (`0.3913`, **0.0% FWER**, 100% pass) | 39.18% (`0.3865`, 0.0%) | 0.00% | 1.82% (`0.0203`) | 68.82% (**100% FWER**, `0.0000`) | 41.71% (`0.4021`, 2.0% FWER) |
| **`v2-cvar-trap` (CVaR $B=50$)** | **10.58%** (`0.0981`, **0.0% FWER**, 38% pass) | 0.00% | 0.00% | 0.00% | 91.04% (**100% FWER**, `0.0000`) | `B5`: **100% FWER** (`0.0000`); `B5+ext`: **74% FWER** (`0.0058`) |
| **`v2-cvar-power` (CVaR $B=50$)** | **28.81%** (`0.8066`, **0.0% FWER**, 80% pass) | 0.73% (`0.0205`, 0.0%) | 0.00% | 32.23% (`0.9816`) | 35.09% (`1.0655`, 0.0%) | — |
| **`v2-cvar-power` (CVaR $B=30$)** | **35.57%** (`0.9835`, **0.0% FWER**, 98% pass) | 18.11% (`0.5045`, 0.0%) | 0.00% | 31.60% (`0.9599`) | 35.09% (`1.0655`, 0.0%) | — |
| **`v2-cvar-power` (CVaR $B=100$)** | 0.00% (*honest cap boundary*) | 0.00% | 0.00% | 0.00% | 35.09% | — |
| **`v2-survival-crossing` (Joint & Q)** | **0.00%** (**0.0% FWER**) | 0.00% (0.0%) | 0.00% | 0.00% | 99.24% (**100% FWER**) | **16.0% FWER** (`B5` & `B5+ext`) |
| **`v2-d09-meps` (Joint Q+Ext, $\kappa=0.056$)** | **36.84%** (`1.0225`, **0.0% FWER**, 88% pass) | 0.00% | 0.00% | 11.85% (`0.3303`) | 73.13% (**88% FWER**, `0.1918`) | 16.06% (`0.4186`, 0.0% FWER) |
| **`v2-d09-meps` (Q-only, $\kappa=0.056$)** | **38.60%** (`1.0189`, **0.0% FWER**, 90% pass) | 0.00% | 0.00% | 8.86% (`0.2461`) | 73.13% (**88% FWER**, `0.1918`) | 13.02% (`0.3393`, 0.0% FWER) |
| **`v2-d09-ohie` (Joint Q+Ext, $\kappa=0.28$)** | **43.21%** (`0.8651`, **0.0% FWER**, 98% pass) | 30.96% (`0.5904`, 2.0%) | 0.00% | 25.48% (`0.5420`) | 68.45% (**90% FWER**, `0.1064`) | 31.09% (`0.6292`, 0.0% FWER) |
| **`v2-d09-ohie` (Q-only, $\kappa=0.28$)** | **44.11%** (`0.8740`, **0.0% FWER**, 100% pass) | 42.87% (`0.8150`, 2.0%) | 0.00% | 34.13% (`0.7269`) | 68.45% (**90% FWER**, `0.1064`) | 44.15% (`0.8751`, 0.0% FWER) |
| **`v2-null-strict` (Joint, Q, CVaR)** | **0.00%** (**0.0% FWER** on all 3) | 0.00% (0.0%) | 0.00% | 0.00% | 48.50% (**100% FWER**) | 0.00% (0.0%) |

### Key Paired 95% Confidence Intervals ($R = 50$ seeds)

- **`v2-d05-conflict` (Joint Q+Ext):**
  - `LCT-LTT-GRO - B1-survival` safe welfare: **`+0.3900`** (95% CI `[+0.3678, +0.4121]`)
  - `LCT-LTT-GRO - B2-GRO` safe welfare: **`+0.3697`** (95% CI `[+0.3420, +0.3973]`)
  - `LCT-LTT-GRO - LCT-LTT-aGRAPA` safe welfare: **`+0.2665`** (95% CI `[+0.2146, +0.3183]`)
- **`v2-cvar-power` (CVaR $B=50$):**
  - `LCT-LTT-GRO - B1-opra` deployment: **`+28.81 pp`** (95% CI `[+24.77, +32.85] pp`)
  - `LCT-LTT-GRO - LCT-LTT-aGRAPA` deployment: **`+28.08 pp`** (95% CI `[+23.90, +32.27] pp`)
- **`v2-cvar-trap` (CVaR $B=50$):**
  - `LCT-LTT-GRO - B1-opra` safe welfare: **`+0.0981`** (95% CI `[+0.0626, +0.1336]`)
  - `LCT-LTT-GRO - B5+ext-GRO` safe welfare: **`+0.0923`** (95% CI `[+0.0582, +0.1264]`)
- **`v2-d09-meps` (Joint Q+Ext, global $\kappa=0.056$):**
  - `LCT-LTT-GRO - LCT-LTT-aGRAPA` deployment: **`+36.84 pp`** (95% CI `[+32.83, +40.84] pp`)
  - `LCT-LTT-GRO - B2-GRO` safe welfare: **`+0.6922`** (95% CI `[+0.4961, +0.8883]`)
- **`v2-d09-ohie` (Joint Q+Ext):**
  - `LCT-LTT-GRO - B2-GRO` safe welfare: **`+0.3231`** (95% CI `[+0.2059, +0.4402]`)
  - `LCT-LTT-GRO - LCT-LTT-aGRAPA` safe welfare: **`+0.2747`** (95% CI `[+0.1474, +0.4020]`)

---

## 3. Gate G3 v2 Checklist Audit

1. **Finite-Sample Safety (FWER $\le 0.065$):** **PASS** — `LCT-LTT-GRO` achieved **0.0% FWER (0/750 unsafe evaluations)** across all 7 designs and 15 target cells.
2. **Separation vs. Separate Bounds (`B1`):** **PASS** — `+28.8 pp` to `+44.1 pp` deployment over best `B1` (`0.00%` everywhere).
3. **Separation vs. Mean-Only Learners (`B2` & `B4`):** **PASS** — massive positive safe-welfare CIs over `B2-GRO` on `v2-d05-conflict`, `v2-d09-meps`, `v2-d09-ohie`, and `v2-cvar-trap`, while uncertified `B4` suffered **88%–100% FWER**.
4. **Separation vs. Fixed-Cutoff `B5` / `B5+ext`:** **PASS** — `B5-GRO` and `B5+ext-GRO` violated safety on **16.0%** of `v2-survival-crossing` runs and **74.0%–100.0%** of `v2-cvar-trap` runs, whereas `LCT-LTT-GRO` held **0.0% FWER** while certifying safe policies on both Capped CVaR benchmarks.
5. **Weak-Overlap Viability (MEPS $\kappa=0.056$):** **PASS** — `LCT-LTT-GRO` jumped from **0.00% $\to$ 36.84% (Joint) / 38.60% (Q-only)** with **0.0% FWER** even under the global worst-case overlap bound (without needing test-time propensity gating; `LCT-LTT-GRO-Gate` also achieved `30.33%` / `33.99%` with 0.0% FWER).

---

## 4. Honest Boundaries & Caveats to Carry Into the Paper

1. **Capped CVaR Cap Frontier:** At $n_{\mathrm{cal}} = 5{,}000, P(S^*) = 0.35, \beta = 0.90$, `LCT-LTT-GRO` achieves `35.57%` deployment at $B = 30$ and `28.81%` at $B = 50$, dropping to `0.00%` at $B = 100$ (support width scales linearly in $B - \eta_0$). The paper must present this exact cap-to-sample-size frontier transparently and frame uncapped $Q_\beta$ + bounded capped $\mathrm{CVaR}_\beta$ as complementary co-primary targets.
2. **Fixed-Cutoff `B5` on Smooth Quantile-Only Regimes:** On smooth quantile-only cells without survival crossings (`v2-d05-conflict`, `v2-d09-ohie`), `B5-GRO` matched `LCT-LTT-GRO` (~41.7% vs. 40.0%, ~44.2% vs. 44.1%). On Joint cells, `LCT-LTT-GRO`'s deployment edge over `B5+ext-GRO` came from Training-Anchored ordering (`anchored=True`); the true structural advantage over `B5`/`B5+ext` is **safety under unknown baseline quantile crossings (`v2-survival-crossing`: 0% vs. 16% FWER) and tail-severity control (`v2-cvar-trap`: 0% vs. 74%–100% FWER)**.
3. **Uncertified `B3` on Well-Specified Traps:** On `v2-cvar-trap`, uncertified `PlugInCVaRITR` (`B3`) deployed `38.88%` with `0%` FWER because the two-part model cleanly separated the `49.0` atom; however, `B3` has no finite-sample guarantee under misspecification or null regimes.

---

## 5. Next Steps (Phase 5 & Stress Sweep)

1. **Phase 5 Applied Study (OHIE & MEPS):** Download public OHIE microdata (and verify household-cluster structure—e.g. household-clustered unit of randomization or household-level aggregation—for exact finite-sample validity!) and MEPS Panel 24/25/26 microdata (with $\Gamma$-sensitivity sweeps under `LCT-LTT-GRO`).
2. **Targeted Stress Sweep:** Run scaling, overlap $\kappa$, and ZIC-BCF comparisons using the upgraded `LCT-LTT-GRO` engine (optionally sharded onto Colab if we include full ZIC-BCF MCMC runs).
