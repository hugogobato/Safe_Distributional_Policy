# Phase 3 prototype readiness

**Decision.** WP-3.1 and WP-3.2 pass. The prototype implements the Gate G2 pivot to joint uncapped upper-quantile and extensive-margin safety. Capped CVaR remains available as a secondary risk test and is evaluated as a diagnostic in the smoke results.

## Implementation contract

Every benchmark fits on one immutable training split, certifies on a second split, and reports outcomes on an independent evaluation split with both potential outcomes. The nuisance estimators, training-scale loss cutoff and candidate ladder use training observations only. The test procedures use the training-fixed ladder beginning at a nominal 15% deployment. Tied scores stay together. Preassigned row IDs restore calibration order before the adaptive bets and are never derived from realized outcomes, treatment or covariates. After a failed candidate test, the certificate retains the accepted prefix and returns the last accepted candidate, or the exact structural baseline fallback. [Data and split classes](../src/safe_dist_policy/data.py), [certification](../src/safe_dist_policy/certification.py), [benchmark](../src/safe_dist_policy/eval.py).

The localized test intersects the quantile and extensive-margin tests at each offered policy. The baseline VaR confidence interval consumes 20% of the level by default. The remaining level applies to each component of the intersection. When the user prespecifies groups, the VaR allocation is shared across the overall and group intervals and the same offered policy must pass all tests. Fixed-sequence testing checks every selected prefix element against the family-wise claim. The optional Gamma sensitivity inflates signed scores and expands the VaR interval according to the Phase 2 propensity ratio bound. [Formalization](phase2_formalization.md), [bounds](../src/safe_dist_policy/bounds.py).

| Method | Training and calibration |
|---|---|
| LCT-LTT | Two-part boosting for the extensive margin and positive log-severity, with a training-fixed candidate ladder and localized joint tests |
| B1-survival | Same fitted score and candidate as LCT-LTT, with non-oracle separate-policy survival/quantile and extensive-margin tests |
| B1-OPRA | Same score and candidate, with separate-policy clipped CDF envelopes and separate extensive-margin tests |
| B2 | Native grf causal forest for the mean contrast, using the same localized joint tests and ladder design |
| B3 | Training-only plug-in capped CVaR policy, obtained by optimizing the Rockafellar–Uryasev objective on a prespecified training grid |
| B4 | The same native grf fit as B2, deployed without certification |
| B5 | Same score and candidate as LCT-LTT, with a fixed training-derived exceedance threshold and bounded indicator LTT |
| Oracle | Known synthetic score on the same nominal ladder. The evaluation distribution selects the best safe offered policy by welfare |

B5 instantiates bounded indicator loss under known propensities and provides a high-probability guarantee for that loss via LTT. CPC controls expected risk of its likelihood-ratio-regulated policy under different assumptions. Oracle policies are diagnostic and use evaluation information unavailable to the fitted methods. Two-part and native grf tuning both use two training hyperparameter candidates and the same three fold assignments. [Learner implementation](../src/safe_dist_policy/learners.py), [CPC primary text, Theorem 4.6](https://arxiv.org/html/2603.02196v4).

## Smoke evidence

The smoke example follows the DGP-1 recovery description in Section 8.2 of the research plan. Its low-noise hurdle has a known, sharp region where both loss margins improve. Spatial mean–tail conflict is reserved for the Phase 4 DGP-5 comparison. Each seed uses 1,600 training, 5,000 calibration, and 50,000 independent evaluation observations.

| Method | Mean deployment | Mean safe welfare gain | Unsafe prefixes |
|---|---:|---:|---:|
| Oracle | 50.20% | 1.0112 | 0/5 |
| LCT-LTT | 50.54% | 0.9995 | 0/5 |
| B1-survival | 0.00% | 0.0000 | 0/5 |
| B1-OPRA | 0.00% | 0.0000 | 0/5 |
| B2 | 50.07% | 1.0081 | 0/5 |
| B3 | 50.09% | 1.0084 | 0/5 |
| B4 | 50.07% | 1.0081 | 0/5 |
| B5 | 50.54% | 0.9995 | 0/5 |

Mean methods and B5 match the localized two-part method in this recovery case. Deployments can exceed the oracle's near-50% welfare-maximizing deployment while attaining lower welfare. The oracle has the highest safe welfare on all five seeds. The test with conditionally deterministic losses recovers the exact safe offered set with both empirical Bernstein and betting. The strict near-null has a truly unsafe positive deployment at every candidate and certified none in 500 draws. The equality null also certified no positive candidates in 500 draws. The 13 deterministic tests cover exact fallback and recovery, invariance to input row permutation, the atom-at-zero CVaR identity, exact CDF ties, agreement with the dense-grid Bernstein supremum, positive tolerances, subgroup fallback, and 500 strict near-null trials. [Smoke CSV](../results/phase3_smoke/smoke_summary.csv), [null checks](../results/phase3_smoke/null_checks.csv), [correctness tests](../tests/test_certification_correctness.py).

## R bridge and resources

R 4.5.2, grf 2.6.1 and zicbcf 1.0.0 were available. The ZIC-BCF-Smear bridge fits the reference R estimator on training data, exports fitted posterior summaries, then projects those summaries to new covariates using a separately fitted random forest. The reference source at commit `b410fcc30cf127343c56614e74f2ad7369daa394` does not expose a native new-row prediction method for its fitted hurdle and severity model. The projection uses posterior tail probabilities from the native Gaussian severity draws and Duan-smearing posterior means. Its 10-burn-in, 12-draw smoke checked the bridge and finite predictions for 25 held-out covariate rows. It does not measure converged MCMC runtime or projection accuracy. [R bridge](../src/safe_dist_policy/forest_bridge.R), [resource profile](../results/phase3_smoke/resource_profile.json).

The five smoke seeds took 5.57–6.29 seconds each. The user scope enforced a 4 GiB aggregate cap with swap disabled and registered a 451,457,024-byte peak. The largest Python-plus-R process peak sum was 404.74 MiB. Both linter and static type checks passed. The checkpoint files and result CSVs have recorded SHA-256 digests. [Pinned environment](../uv.lock), [resource profile and checksums](../results/phase3_smoke/resource_profile.json).

Reproduce from the project root with the pinned environment. Rscript must have the two R packages installed. Give the smoke driver a new output path to preserve the saved run.

```sh
uv sync --locked
uv run --locked ruff check src tests scripts/run_phase3_smoke.py
uv run --locked mypy src
systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/pytest -q
systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/run_phase3_smoke.py --out-dir results/phase3_smoke_reproduction
```
