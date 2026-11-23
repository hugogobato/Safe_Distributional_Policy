# Gate G3 decision

**KILL. Stop the research program at Gate G3.** The matched fixed-cutoff indicator baseline B5+ext achieved the same measured safety with equal or higher deployment on every one of the 200 paired runs in the four primary comparison cells. LCT-LTT's finite-sample guarantee for the comparative quantile remains distinct. The funded empirical premise was a practically useful advantage over a simpler, equally constrained alternative, and that premise failed in the preregistered regime. [Frozen protocol](phase4_preregistration.md), [paired focal contrasts](../results/phase4_sim/focal/gate_contrasts.json)

## 1. Preregistered expectations

After Gate G2, the primary risk was the intersection of uncapped upper-quantile and positive-loss probability noninferiority at zero tolerance. We tested whether localization breaks two-arm abstention, whether distributional ranking improves safety-adjusted welfare over mean ranking, and whether the nonlinear quantile certificate improves safety or welfare over an indicator test at a training-estimated cutoff. B5+ext combines that indicator test with the same localized extensive-margin constraint and offers the same policies in the same order. The primary contrast applies to DGP 5, the focal DGP 6 configuration and both application proxies. DGP 4 is the prespecified mean-favorable control established by the scenario definition. Capped CVaR was secondary after G2. [Gate G2](gate_g2_memo.md), [preregistration](phase4_preregistration.md)

## 2. Evidence

The completed core contains 50 seeds in each of six cells, including separate equality and strict near-null variants of DGP 3. The three additional focal cells also use all 50 seeds. There are 450 completed replications and 6,150 method rows. All comparisons use the same training, calibration and independent 100,000-covariate evaluation splits within a replication. [Core CSV](../results/phase4_sim/core_dgp1_5_r50.csv), [core parquet](../results/phase4_sim/core_dgp1_5_r50.parquet), [core figure](../results/phase4_sim/figures/core_method_comparison.pdf), [focal CSV](../results/phase4_sim/focal/stress_dgp6_10.csv), [focal parquet](../results/phase4_sim/focal/stress_dgp6_10.parquet)

The table gives mean deployment. Each cell has 50 pairs. The last column is the mean and paired 95% Monte Carlo interval for $W_{\mathrm{LCT}}-1.15W_{\mathrm{B5+ext}}$, in each DGP's utility units. Positive values meet the 15% welfare margin.

| Cell | LCT-LTT | Stronger B1 | Mean B2 | Matched B5+ext | Welfare above the B5+ext margin, mean [95% interval] |
|---|---:|---:|---:|---:|---:|
| DGP 5, mean/tail conflict | 6.25% | 0.00% | 0.00% | 8.15% | −0.03219 [−0.05676, −0.00762] |
| DGP 6, 5,000 calibration rows, 25% safe region, β=.90 | 8.54% | 0.00% | 6.41% | 8.54% | −0.03096 [−0.04262, −0.01931] |
| DGP 9, MEPS proxy | 0.00% | 0.00% | 0.00% | 2.41% | −0.05884 [−0.12650, +0.00882] |
| DGP 9, OHIE proxy | 8.24% | 0.00% | 4.41% | 8.45% | −0.03016 [−0.04747, −0.01284] |

No LCT-LTT unsafe certified prefix was detected in any completed cell, including with the possible-unsafe classification. B5+ext also had zero definite, plug-in and possible unsafe prefixes in all four focal cells. The strict near-null fixed ladder abstained on all 50 replications, and the equality ladder selected no positive deployment. A rate of 0/50 has a two-sided 95% exact binomial interval of approximately [0%, 7.11%]. The synthetic criterion uses the observed rate. Formal calibration coverage follows from the Phase 2 argument under its assumptions. [Core method summary](../results/phase4_sim/method_summary.csv), [focal method summary](../results/phase4_sim/focal/method_summary.csv), [validity proof](phase2_formalization.md)

## 3. Execution and deviations

All 300 core tasks and 150 focal tasks succeeded. The exact core source remained unchanged across its 300 tasks and is archived. The focal version tightened the handling of structural equality and uniform harm, added the B5 cutoff-containment diagnostic, and distinguished capped-CVaR diagnostics from quantile-plus-extensive truth. A replay of all 300 core evaluation populations verified agreement of the actual global sign and equality checks with the structural model. No selected certified core policy had zero observed held-out switches. Earlier certified prefixes were saved as aggregate flags, so the postrun audit checks the sign branches and selected policies rather than replaying earlier fitted scores. All versions and result digests are preserved. [Core source snapshot](../results/phase4_sim/core_source_snapshot.zip), [truth audit](../results/phase4_sim/core_truth_revision_audit.json), [core integrity manifest](../results/phase4_sim/core_merge_status.json), [focal integrity manifest](../results/phase4_sim/focal/focal_tasks_merge_status.json), [focal source snapshot](../results/phase4_sim/focal/source_snapshot.zip)

Seventeen tests passed. Core execution took 1,160.9 seconds, with aggregate cgroup peak 1,735,483,392 bytes. The three focal experiments took 460.3 seconds, with peak 1,587,843,072 bytes. Both scopes imposed a 4,294,967,296-byte hard limit with swap disabled. [Core memory log](../results/phase4_sim/core_cgroup_memory_bytes.txt), [focal memory log](../results/phase4_sim/focal/cgroup_memory_bytes.txt)

## 4. Strongest comparator cases

In easy recovery, LCT-LTT and B5+ext returned identical policies on all 50 seeds, each deploying 50.07% on average. B2 deployed 49.95%. In the DGP 4 location-shift control, the extensive-margin effect is exactly zero. Both joint methods therefore had little power at the boundary. With quantile only, standard mean ranking deployed 59.70%, compared with 24.47% for the two-part ranking. The control demonstrates a regime where the simpler ranking is substantially more effective. The two B1 implementations abstained across the completed designs under their matched constraints. [Core comparator table](../results/phase4_sim/method_summary.csv)

## 5. Failure regions

On the smooth hurdle DGP 2, deployment was 2.18% for LCT-LTT, 2.83% for B2 and 2.60% for matched B5+ext. In the MEPS proxy, LCT-LTT never deployed. It also fell short of the 20-point deployment-gain threshold over B1 in DGP 5 and focal DGP 6. B1's mean welfare was zero in both cells, so relative percentages have no finite denominator. The corresponding absolute safe-welfare gains were 0.0645 and 0.2064, respectively. These findings bound the attainable empirical benefit at the chosen sample size and tolerance. The broad small-sample, extreme-tail, weak-overlap and runtime frontiers remain unmeasured. [Core and focal paired contrasts](../results/phase4_sim/focal/gate_contrasts.json)

## 6. Mechanism ablations

When B5's training-fixed cutoff falls in LCT-LTT's baseline-quantile confidence interval, each threshold that LCT-LTT certifies has already passed the same localized indicator score at that cutoff, and its extensive-margin component has passed at the tighter level. At the looser matched B5+ext level, both tests still pass. This yields prefix containment for the common offered sequence. In all 149 of 150 focal stress replications where the cutoff lay within the interval, the recorded matched B5+ext prefix contained LCT-LTT's prefix. It also contained the LCT-LTT prefix in the one remaining replication. This is a statement about the realized same-score tests. A fixed-cutoff guarantee does not imply a comparative quantile guarantee, since survival contrasts may cross between the chosen cutoff and the true baseline quantile. The matched comparator displayed no such measurable safety cost in these four cells. [Focal profile table](../results/phase4_sim/focal/resource_profiles.csv), [source and theorem audit](phase2_e3_prior_art_check.md)

The 15%-first order improved mean deployment over 5%-first in DGP 5, from 2.94% to 6.25%. The paired difference was +3.32 percentage points, with 95% interval [+0.56, +6.08]. The empirical-Bernstein version deployed zero in all six core cells. Plain B5, which tests the cutoff alone, deployed 34.26% in DGP 5 and had one plug-in and possible unsafe prefix, with zero definite cases. Matching the extensive guard returned deployment to 8.15%, with zero possible unsafe prefixes. The measured safety comparison therefore uses B5+ext. [Core ablation table](../results/phase4_sim/method_summary.csv)

## 7. Practical effect sizes and uncertainty

Across the 200 primary pairs, LCT-LTT deployed less on 12, tied on 188 and never deployed more. In DGP 5, deployment difference from matched B5 was −1.90 points [−3.84, +0.05] and welfare difference was −0.01958 [−0.03940, +0.00024]. The positive 15% welfare margin is excluded by the paired intervals in DGP 5, focal DGP 6 and the OHIE proxy. The MEPS proxy's welfare interval is wider and admits a small positive margin. Its observed LCT-LTT deployment was 0/50, so it supplies no corroborating deployment advantage. These are Monte Carlo intervals for specific simulation regimes. [Focal comparisons](../results/phase4_sim/focal/gate_contrasts.json)

## 8. Terminal scope

The planned primary empirical advantage is absent against matched B5+ext on all complete primary comparison cells, with zero detectable safety advantage in those comparisons. The exact finite-sample quantile certificate remains mathematically separate from a fixed-threshold safety claim. The decision concerns the scoped field-paper program and the gain demonstrated under its preregistered benchmarks.

The early stop leaves 47 nonfocal DGP 6 cells, DGPs 7, 8 and 10, and the production ZIC-BCF slice unrun. The application proxies are synthetic, and the confirmatory OHIE and MEPS studies were not started. The planned full crossover figure and final stress table require the unrun cells. The focal table explicitly documents the three completed stress cells. No Colab run is needed for the terminal decision. [Focal task list](../results/phase4_sim/focal/focal_tasks.json), [application feasibility](phase5_data_feasibility.md)
