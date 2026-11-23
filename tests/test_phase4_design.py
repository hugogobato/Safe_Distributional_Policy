"""Deterministic checks for the Phase 4 estimands and configuration contract."""

import numpy as np

from safe_dist_policy.phase4_design import (
    CORE,
    DESIGNS,
    STRESS,
    baseline_quantile,
    population,
    propensity,
)
from safe_dist_policy.phase4_run import task_grid


def test_frozen_task_grid_has_no_duplicate_checkpoints():
    assert len(CORE) == 6
    assert len(STRESS) == 64
    assert len(task_grid("core")) == 300
    remote = task_grid("remote")
    assert len(remote) == len(set(remote)) == 2884
    assert set(task_grid("all")) == set(task_grid("core") + remote)


def test_meps_proxy_matches_tabulated_tail_ratio():
    design = DESIGNS["d09-meps-proxy"]
    q90, q99 = (baseline_quantile(design, beta) for beta in (0.90, 0.99))
    np.testing.assert_allclose([q90, q99 / q90], [10.412, 6.42], rtol=1e-10)


def test_conflict_stratum_has_larger_mean_gain_and_tail_harm():
    design = DESIGNS["d05-main"]
    law = population(design, np.array([[0.15, 0.5, 0.5], [0.45, 0.5, 0.5]]))
    means = law.loss_mean()
    saving = means[:, 0] - means[:, 1]
    tail = law.survival(baseline_quantile(design, design.beta))
    assert saving[1] > saving[0] > 0
    assert tail[0, 1] < tail[0, 0] < tail[1, 1]
    assert np.all(law.positive[:, 1] < law.positive[:, 0])


def test_true_propensity_floor_is_stronger_than_declared_bound():
    design = DESIGNS["d07-k05"]
    x = np.array([[1.0, 0.75, 0.0], [1.0, 0.25, 1.0]])
    want = design.kappa + (1 - 2 * design.kappa) * np.array([0.14, 0.86])
    np.testing.assert_allclose(propensity(design, x), want)
