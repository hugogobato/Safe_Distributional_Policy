# Phase 4 Final Completion Report (Protocol v2 — Path A)

- **Status:** **COMPLETE — 3,334 / 3,334 PAIRED REPLICATIONS FINISHED WITH ZERO FAILED CHECKPOINTS**
- **Decision:** **GO TO PHASE 5 (OHIE & MEPS APPLIED STUDIES)**
- **Global Safety Audit:** Across **8,500+ certified evaluations** of `LCT-LTT-GRO` on known-propensity and $\Gamma$-adjusted regimes (Gate G3 v2 7 designs + Core DGPs 1–4 + DGP 6 full 48-cell factorial + DGP 7 $\Gamma$-sensitivity + DGP 8 Pareto heavy tails + DGP 10 scaling up to $n_{\mathrm{cal}}=100{,}000$ + ZIC-BCF sensitivity), **empirical prefix FWER was `0.00%` in every single cell** (0 violations across all 3,334 replications, corroborated by 500-draw strict near-null unit tests).

---

## 1. Execution Inventory (3,334 Completed Replications)

| Block | Scope | Seeds | Replications | Execution Venue | Artifacts |
|---|---|---|---:|---|---|
| **Gate G3 v2 Core Suite** | 7 decision-critical designs (`v2-d05-conflict`, `v2-cvar-trap`, `v2-cvar-power`, `v2-survival-crossing`, `v2-d09-meps`, `v2-d09-ohie`, `v2-null-strict`) | 1000–1049 | 350 | Local (6 spawn workers, 10.4m) | `results/phase4_v2/v2_summary.csv`, `docs/gate_g3_v2_memo.md` |
| **Block A (`core_1_4`)** | DGPs 1–4 (`d01-main`, `d02-main`, `d03-equality`, `d04-main`) | 1000–1049 | 200 | Local (4 spawn workers, 8.4m) | `results/phase4_v2_stress/core_1_4_summary.csv` |
| **Block B (`dgp6_crossover`)** | Full 48-cell factorial $n_{\mathrm{cal}} \in \{1000, 2500, 5000, 10000\} \times s \in \{.10, .25, .50, .80\} \times \beta \in \{.80, .90, .95\}$ | 1000–1049 | 2,400 | Google Colab Chrome `u/6..u/9` (12 GitHub-backed 7.5 KB shards, ~55m) | `results/phase4_v2_stress/dgp6_merged/dgp6_summary.csv` |
| **Block C (`dgp7_misspec`)** | Omitted propensity predictors & $\Gamma$-sensitivity ($\kappa \in \{.05, .10, .25\}$) | 1000–1049 | 150 | Local (4 spawn workers, 1.3m) | `results/phase4_v2_stress/dgp7_misspec_summary.csv` |
| **Block D (`dgp8_pareto`)** | Heavy-tail Pareto ($\alpha = 1.8$) across caps $\{1, 2, 5\} \times Q_{0.99}$ | 1000–1049 | 150 | Local (4 spawn workers, 6.3m) | `results/phase4_v2_stress/dgp8_pareto_summary.csv` |
| **Block E (`dgp10_scaling`)** | Runtime & peak RSS scaling $n_{\mathrm{cal}} \in \{1\text{k}, 5\text{k}, 20\text{k}, 100\text{k}\} \times p \in \{10, 50\}$ | 1000–1002 | 24 | Local (1 isolated worker, 35.8m) | `results/phase4_v2_stress/dgp10_scaling_profiles.csv` |
| **Block F (`zic_sensitivity`)** | Production ZIC-BCF Bayesian projection sensitivity (`v2-d05-conflict`, `v2-d09-meps`, `v2-d09-ohie`) | 1000–1019 | 60 | Local (4 spawn workers, 10.5m) | `results/phase4_v2_stress/zic_sensitivity_summary.csv` |
| **Total** | | | **3,334** | | Figures in `results/phase4_v2_stress/figures/` |

---

## 2. Headline Empirical Findings

### 2.1 Block B — DGP 6 Full 48-Cell Crossover Factorial (2,400 Replications, 7,200 Certified Evaluations)

Averaged across all 48 cells of $(n_{\mathrm{cal}}, s, \beta)$:

| Target | `LCT-LTT-GRO` (Path A) | `LCT-LTT-aGRAPA` (v1) | `B2-GRO` | `B5` / `B5+ext` (GRO) | Best `B1` | `B4` (Uncert.) |
|---|---:|---:|---:|---:|---:|---:|
| **Joint Q+Ext** | **27.97%** (`0.7016`, **0.00% FWER**) | 10.72% (`0.2763`, 0.00%) | 22.18% (`0.5924`, 0.00%) | 18.17% (`0.4600`, 0.00%) | **0.00%** (0/48 cells) | 39.12% (max 8.0% FWER) |
| **Uncapped Q-only** | **30.57%** (`0.7568`, **0.00% FWER**) | 22.22% (`0.5587`, 0.00%) | 27.29% (`0.7251`, 0.00%) | 27.90% (`0.6883`, 0.04%) | **0.00%** (0/48 cells) | 39.12% (max 8.0% FWER) |
| **Capped CVaR ($B=50$)** | **21.39%** (`0.5500`, **0.00% FWER**) | 7.21% (`0.1890`, 0.00%) | 15.33% (`0.4101`, 0.00%) | — | **0.00%** (0/48 cells) | — |

Key crossover boundaries at $\beta = 0.90$ (`LCT-LTT-GRO` vs. v1 `aGRAPA`, with `B1 = 0.0%` everywhere):
- **Joint Q+Ext:** At $n_{\mathrm{cal}}=2{,}500$: `39.4%` vs `4.5%` ($s=0.50$), `74.8%` vs `1.6%` ($s=0.80$). At $n_{\mathrm{cal}}=5{,}000$: `24.9%` vs `8.6%` ($s=0.25$), `49.9%` vs `15.6%` ($s=0.50$), `79.2%` vs `26.0%` ($s=0.80$).
- **Capped CVaR ($B=50$):** At $n_{\mathrm{cal}}=5{,}000$: `48.9%` vs `1.0%` ($s=0.50$), `80.8%` vs `1.6%` ($s=0.80$). At $n_{\mathrm{cal}}=10{,}000$: `24.8%` vs `17.2%` ($s=0.25$), `50.3%` vs `36.2%` ($s=0.50$), `80.4%` vs `61.1%` ($s=0.80$). Even at extreme $\beta=0.95, n_{\mathrm{cal}}=10{,}000$: `50.3%` vs `4.0%` ($s=0.50$) and `80.4%` vs `6.4%` ($s=0.80$).
- **Honest power floor:** At tiny safe mass ($s = 0.10$) and $n_{\mathrm{cal}} \le 5{,}000$, finite-sample certification properly abstains (`0.0%–3.1%`), activating cleanly by $n_{\mathrm{cal}} = 10{,}000$ (`7.9%–10.2%`).

### 2.2 Gate G3 v2 Flagship Regimes (Conflict, Traps, Crossings, Proxies)

- **Mean–Tail Conflict (`v2-d05-conflict`, `v2-d09-meps`, `v2-d09-ohie`):** Uncertified `B4` violates safety on **88%–100%** of runs; mean-ranking `B2-GRO` loses heavily in safe welfare; `B1` is `0.00%` everywhere; `LCT-LTT-GRO` certifies **36.8%–44.1%** deployment with **0.0% FWER** (including **36.84% Joint / 38.60% Q-only** on weak-overlap MEPS $\kappa=0.056$, up from `0.00%` in v1).
- **Fixed-Cutoff `B5` / `B5+ext` Failures:** Plug-in fixed-cutoff `B5-GRO` and `B5+ext-GRO` violate safety on **16.0%** of `v2-survival-crossing` runs (unknown baseline quantile crossing) and **74.0%–100.0%** of `v2-cvar-trap` runs (catastrophic-tail severity trap), whereas `LCT-LTT-GRO` holds **0.0% FWER** across both while certifying safe policies on `v2-cvar-trap` (`10.58%`) and `v2-cvar-power` (`28.81%` at $B=50$, `35.57%` at $B=30$).

### 2.3 Core Controls, Heavy Tails, Sensitivity, Scaling, and ZIC-BCF

- **Core Controls (`core_1_4`):** Full recovery on `d01-main` (`50.04%`, 0% FWER); 8.6× Joint jump on smooth `d02-main` (`16.40%` vs v1 `1.90%`); exact `0.00%` abstention on `d03-equality`; scoring bug resolved on `d04-main` (`60.62%` Q-only vs `B2-GRO` `61.35%`).
- **Propensity Misspecification (`dgp7_misspec`):** Unadjusted misspecified propensities leak `2.0%` at $\kappa=0.05$; $\Gamma$-inflated `LCT-LTT-GRO-gamma` restores **0.0% FWER** across all $\kappa \in \{.05, .10, .25\}$ on both learned and adverse fixed ladders.
- **Heavy-Tail Pareto $\alpha=1.8$ (`dgp8_pareto`):** Uncapped `LCT-LTT-GRO` achieves **33.20% Joint / 33.43% Q-only** (0% FWER, 4.4× v1 Joint `7.56%`) completely insensitive to tail truncation, while Capped CVaR achieves **23.76%** at `1×Q99` (4.7× v1 `5.05%`) and honestly drops to `0.00%` at `5×Q99`.
- **Scaling (`dgp10_scaling`):** At $n_{\mathrm{cal}}=100{,}000, p=50$, `TwoPartDistributionalLearner` fits in **9.33s** (vs `grf` `330.14s`), `LCT-LTT-GRO` certifies in **1.16s (Q-only) / 2.83s (Joint)** with peak Python RSS **389.1 MiB**, achieving **56.62% deployment (100% pass, 0% FWER)** while `B1` remains **0.00%**.
- **ZIC-BCF Projection (`zic_sensitivity`):** Both engines hold **0.0% FWER**. Two-part boosting (`~0.6s`) dominates ZIC-BCF (`~67s`) on `v2-d05-conflict` (`0.4233` vs `0.3923`) and `v2-d09-meps` (`0.9913` vs `0.4992`), while ZIC-BCF edges `v2-d09-ohie` (`0.9214` vs `0.8870`, `+3.9%`). Two-part boosting is retained as the primary engine, with ZIC-BCF reported as a Bayesian sensitivity check.

---

## 3. Ready for Phase 5

All Phase 4 simulation evidence, crossover figures (`results/phase4_v2_stress/figures/fig1_dgp6_crossover_heatmaps.{png,pdf}`, `fig2_gate_g3_v2_summary.{png,pdf}`), and verification tests are complete. Proceed to **Phase 5 (OHIE & MEPS real-data applied study)**.
