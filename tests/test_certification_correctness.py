from dataclasses import replace
from math import inf

import numpy as np
import pytest

from safe_dist_policy.bounds import (
    adaptive_reject,
    cvar_eb_max,
    eb_upper,
    gro_reject,
    inflate_scores,
    quantile_interval,
)
from safe_dist_policy.certification import (
    PolicyCertificate,
    PolicyFamily,
    PolicyScores,
    RiskSpec,
    _aipw_extensive_score,
    _relative_quantile_test,
    certify_indicator,
    certify_localized,
    certify_separate,
    fixed_sequence,
    propensity_gated_family,
)
from safe_dist_policy.data import EvaluationSample, ExperimentSplit, LoggedSample
from safe_dist_policy.dgps import deterministic_recovery, null_recovery, smoke_split
from safe_dist_policy.eval import evaluate_policy, sample_cvar
from safe_dist_policy.learners import TwoPartDistributionalLearner


def test_structural_fallback_is_exact_for_covariate_dependent_baseline():
    split = smoke_split(5, n_train=80, n_cal=100, n_eval=200)

    def baseline(x):
        return (x[:, 0] > 0).astype(np.int64)

    def forbidden_prediction(_):
        pytest.fail("Structural fallback must bypass candidate scores.")

    family = PolicyFamily(forbidden_prediction, (), baseline)
    cert = fixed_sequence(family, lambda _: False, "fallback", "joint", 0.05)
    x = split.evaluation.x
    np.testing.assert_array_equal(family.actuate(x, cert.threshold), baseline(x))
    result = evaluate_policy(split.evaluation, family, cert, RiskSpec())
    assert cert.threshold == inf
    assert result.deployment == result.delta_quantile == result.delta_cvar == result.delta_extensive == 0


def test_no_observed_switch_does_not_establish_structural_equality():
    n = 50

    def score(x):
        return PolicyScores(np.ones(len(x), dtype=np.int64), x[:, 0])

    family = PolicyFamily(score, (1.0,))
    y = np.zeros(n)
    cal = LoggedSample(np.zeros((n, 1)), np.zeros(n), y, np.full(n, 0.5), np.arange(n), -y)
    spec = RiskSpec(quantile_tolerance=None, extensive_tolerance=0)
    assert certify_localized(cal, family, spec, 0.5).threshold == inf


def test_atom_at_zero_cvar_mean_reduction():
    y = np.r_[np.zeros(95), np.full(5, 8.0)]
    beta = 0.90
    assert sample_cvar(y, beta, 50) == pytest.approx(y.mean() / (1 - beta), abs=1e-12)


@pytest.mark.parametrize("method", ["eb", "betting"])
def test_exact_known_truth_policy_recovery_and_row_permutation(method):
    cal, family = deterministic_recovery()
    spec = RiskSpec(bound=method)
    cert = certify_localized(cal, family, spec, 0.50)
    assert cert.threshold == 1.0
    assert cert.passed == (1.0,)
    np.testing.assert_array_equal(family.actuate(cal.x, cert.threshold), cal.x[:, 0] < 0.5)
    permuted = cal.take(np.random.default_rng(524).permutation(cal.n))
    assert certify_localized(permuted, family, spec, 0.50) == cert


def test_cvar_bernstein_breakpoint_max_equals_independent_dense_calculation():
    rng = np.random.default_rng(155)
    y = rng.integers(0, 21, 2500) / 4
    dw = rng.choice([-2., 0., 2.], size=len(y), p=(0.57, 0.25, 0.18))
    cap, lo, hi, wmax, delta = 5.0, 0.25, 4.75, 2.0, 0.04
    grid = np.linspace(lo, hi, 4501)
    brute = max(
        eb_upper(dw * np.maximum(y - eta, 0), -(cap - eta) * wmax, (cap - eta) * wmax, delta)
        for eta in grid
    )
    assert cvar_eb_max(y, dw, lo, hi, cap, wmax, delta) == pytest.approx(brute, abs=1e-10)


def test_positive_tolerance_shared_region_and_cvar_grid_endpoints():
    cal, family = deterministic_recovery()
    for method in ("eb", "betting"):
        spec = RiskSpec(bound=method, quantile_tolerance=0.05, cvar_tolerance=0.0, cvar_cap=4.0, cvar_cells=16)
        cert = certify_localized(cal, family, spec, 0.50)
        assert cert.threshold == 1.0


def test_weighted_quantile_interval_respects_ties_and_cap():
    losses = np.repeat([0., 1., 2.], [4000, 5300, 700])
    w = np.ones_like(losses)
    for method in ("eb", "betting"):
        actual = quantile_interval(losses, w, 0.90, 0.01, 1.0, method)
        assert actual == (1.0, 1.0)


def test_adaptive_betting_matches_dense_pilot_algebra():
    rng = np.random.default_rng(45)
    for tau in (0.0, 0.5, -0.6):
        for shift in (-0.35, -0.03, 0.07):
            draw = np.where(rng.random(500) < 0.80, tau, np.clip(rng.normal(tau + shift, 0.5, 500), -2, 2))
            a, b, delta = -2., 2., 0.037
            u = (draw - a) / (b - a)
            m = (tau - a) / (b - a)
            wealth = 0.
            direct = False
            for step, observation in enumerate(u, start=1):
                prior = u[:step - 1]
                mu = (m + prior.sum()) / step
                v = (0.0625 + ((prior - mu)**2).sum()) / step
                fraction = np.clip((m - mu) / (v + (m - mu)**2), 0, 0.75 / (1 - m))
                wealth += np.log1p(fraction * (m - observation))
                direct |= bool(wealth >= np.log(1 / delta))
            assert adaptive_reject(draw, a, b, tau, delta) == direct


def test_joint_group_certificate_requires_support_for_all_declared_groups():
    cal, family = deterministic_recovery()
    result = certify_localized(
        cal, family, RiskSpec(), 0.50, groups=lambda x: (x[:, 0] > 0.5).astype(np.int64),
        required_groups=(2,),
    )
    assert result.threshold == inf and result.tested == 1


def test_certifiers_reject_false_range_claim_and_split_rejects_overlap():
    cal, family = deterministic_recovery()
    invalid = replace(cal, propensity=np.full(cal.n, 0.1))
    spec = RiskSpec()
    with pytest.raises(ValueError, match="overlap"):
        certify_localized(invalid, family, spec, 0.5)
    with pytest.raises(ValueError, match="overlap"):
        certify_separate(invalid, family, spec, 0.5)
    with pytest.raises(ValueError, match="overlap"):
        certify_indicator(invalid, family, 1.0, spec, 0.5)
    eval_set = EvaluationSample(cal.x, np.zeros((cal.n, 2)), np.zeros((cal.n, 2)), cal.row_id + cal.n)
    with pytest.raises(ValueError, match="disjoint"):
        ExperimentSplit(cal, cal, eval_set, 0.50)


def test_prefix_flag_detects_unsafe_earlier_policy_before_later_safe_policy():
    x = np.r_[np.zeros(100), np.ones(100)][:, None]
    losses = np.c_[np.where(x[:, 0] == 0, 3., 0.), np.where(x[:, 0] == 0, 0., 4.)]
    evaluation = EvaluationSample(x, losses, -losses, np.arange(len(x)))
    family = PolicyFamily(
        lambda features: PolicyScores(np.ones(len(features), dtype=np.int64), features[:, 0]),
        (1.0, 0.0),
    )
    # Move the second group's treatment down to baseline's quantile so full
    # deployment is safe for Q, while the first candidate alone increases
    # the extensive rate. Treating both groups restores the extensive rate.
    both = evaluation.potential_loss.copy()
    both[:, 1] = np.where(x[:, 0] == 0, 0., 2.)
    evaluation = EvaluationSample(x, both, -both, np.arange(len(x)))
    result = evaluate_policy(
        evaluation, family, PolicyCertificate("fixture", 0.0, (1.0, 0.0), 2, "joint", 0.05), RiskSpec(),
    )
    assert result.unsafe_prefix and not result.selected_unsafe


def test_strict_near_null_fwer_across_500_draws():
    false_passes = 0
    for seed in range(500):
        cal, family = null_recovery(seed, unsafe=True)
        cert = certify_localized(cal, family, RiskSpec(), 0.5)
        # In this DGP every positive candidate mass strictly increases both risks.
        false_passes += bool(cert.passed)
    assert false_passes / 500 <= 0.065


def test_gro_permutation_invariance_and_vectorized_quantile_match():
    cal, family = deterministic_recovery(seed=21, n=8000)
    tr_raw, _ = deterministic_recovery(seed=99, n=4000)
    tr = replace(tr_raw, row_id=tr_raw.row_id + 10_000)
    spec = RiskSpec(bound="gro", quantile_tolerance=0.0, extensive_tolerance=0.0, cvar_tolerance=0.0, cvar_cap=4.0, cvar_cells=32)
    cert = certify_localized(cal, family, spec, 0.50, train=tr, anchored=True)
    assert cert.threshold == 1.0
    # GRO e-values are strictly permutation-invariant even if row IDs are relabeled after shuffling.
    perm = np.random.default_rng(777).permutation(cal.n)
    permuted = replace(cal.take(perm), row_id=np.arange(cal.n))
    assert certify_localized(permuted, family, spec, 0.50, train=tr, anchored=True) == cert

    # Verify O(n log n) vectorized GRO quantile sweep matches brute-force knot-by-knot evaluation.
    rng = np.random.default_rng(123)
    n = 1200
    y = rng.lognormal(1.2, 0.7, n)
    w0 = np.where(rng.random(n) < 0.5, 2.0, 0.0)
    w1 = np.where((w0 == 0.0) & (rng.random(n) < 0.6), 2.0, 0.0)
    dw = w1 - w0
    interval = (2.0, 6.0)
    for gamma in (1.0, 1.3):
        for theta in (0.02, 0.12, 0.25):
            fast = _relative_quantile_test(y, w1, w0, interval, 0.0, 2.0, 2.0, 0.04, "gro", gamma, gro_stake=theta)
            knots = np.unique(np.r_[interval[0], y[(dw != 0) & (y > interval[0]) & (y < interval[1])], interval[1]])
            brute = all(
                gro_reject(inflate_scores(dw * (y > eta), gamma), -2.0 / gamma, 2.0 * gamma, 0.0, 0.04, theta)
                for eta in knots
            )
            assert fast == brute


def test_aipw_support_bounds_and_unbiasedness_for_all_gamma():
    rng = np.random.default_rng(321)
    n = 20000
    e = rng.uniform(0.2, 0.8, n)
    w_max = 1.0 / 0.2
    a_obs = (rng.random(n) < e).astype(np.int64)
    p0_true = rng.uniform(0.1, 0.9, n)
    p1_true = rng.uniform(0.1, 0.9, n)
    y0 = (rng.random(n) < p0_true).astype(np.float64)
    y1 = (rng.random(n) < p1_true).astype(np.float64)
    ext = np.where(a_obs == 1, y1, y0)
    base_act = np.zeros(n, dtype=np.int64)
    prop_act = (rng.random(n) < 0.6).astype(np.int64)
    w0 = (a_obs == base_act) / np.where(a_obs == 1, e, 1.0 - e)
    w1 = (a_obs == prop_act) / np.where(a_obs == 1, e, 1.0 - e)
    # Deliberately misspecified control variates in [0, 1]
    m0 = rng.uniform(0.0, 1.0, n)
    m1 = rng.uniform(0.0, 1.0, n)
    z1 = _aipw_extensive_score(prop_act, base_act, w1, w0, ext, m0, m1, gamma=1.0)
    assert np.all(z1 >= -w_max - 1e-12) and np.all(z1 <= w_max + 1e-12)
    assert np.all(z1[prop_act == base_act] == 0.0)
    true_contrast = float(np.mean((prop_act != base_act) * (p1_true - p0_true)))
    assert abs(float(np.mean(z1)) - true_contrast) < 0.03

    # Exhaustive corner check for gamma >= 1
    for gamma in (1.0, 1.5, 5.0, 20.0):
        lo_bound = -max(w_max / gamma, 1.0)
        hi_bound = w_max * gamma
        corners = np.array(
            [(g, c0, c1, ao, ew) for g in (0.0, 1.0) for c0 in (0.0, 0.5, 1.0) for c1 in (0.0, 0.5, 1.0)
             for ao in (0, 1) for ew in (0.2, 0.5, 0.8)]
        )
        m = len(corners)
        ao = corners[:, 3].astype(np.int64)
        ew = corners[:, 4]
        pa = np.ones(m, dtype=np.int64)
        ba = np.zeros(m, dtype=np.int64)
        pw = (ao == pa) / np.where(ao == 1, ew, 1.0 - ew)
        bw = (ao == ba) / np.where(ao == 1, ew, 1.0 - ew)
        scores = _aipw_extensive_score(pa, ba, pw, bw, corners[:, 0], corners[:, 1], corners[:, 2], gamma)
        assert float(np.min(scores)) >= lo_bound - 1e-12
        assert float(np.max(scores)) <= hi_bound + 1e-12


def test_gro_enforces_disjoint_train_and_localized_bounds():
    cal, family = deterministic_recovery(seed=19, n=1000)
    spec = RiskSpec(bound="gro")
    with pytest.raises(ValueError, match="disjoint"):
        certify_localized(cal, family, spec, 0.50, train=cal)
    with pytest.raises(ValueError, match="disjoint"):
        certify_indicator(cal, family, 1.0, spec, 0.50, train=cal)
    tr = replace(cal, row_id=cal.row_id + 10_000)
    # Invalid tight localized bound must raise
    with pytest.raises(ValueError, match="localized support bound"):
        certify_localized(cal, family, spec, 0.50, train=tr, localized_bounds=lambda _: (1.5, 1.5))

    # Propensity-gated family produces valid deterministic bounds
    def pfn(x):
        return 0.25 + 0.5 * x[:, 0]

    gated_fam, bounds_fn = propensity_gated_family(
        family.predict, tr, (0.25, 0.50), pfn, 0.25, floor_quantile=0.10,
    )
    cal_var = replace(cal, propensity=pfn(cal.x))
    tr_var = replace(tr, propensity=pfn(tr.x))
    cert = certify_localized(
        cal_var, gated_fam, spec, 0.25, train=tr_var, anchored=True, localized_bounds=bounds_fn,
    )
    assert cert.guarantee == "joint"


def test_learner_scorer_aligns_with_active_risk_spec():
    split = smoke_split(7, n_train=600, n_cal=400, n_eval=200)
    learner = TwoPartDistributionalLearner(0.90, seed=7, leaves=(5,)).fit(split.train)
    q_spec = RiskSpec(quantile_tolerance=0.0, extensive_tolerance=None)
    cvar_spec = RiskSpec(quantile_tolerance=None, extensive_tolerance=None, cvar_tolerance=0.0, cvar_cap=50.0)
    q_scores = learner.predict_for(split.evaluation.x, q_spec)
    j_scores = learner.predict(split.evaluation.x)
    c_scores = learner.predict_for(split.evaluation.x, cvar_spec)
    m0, m1 = learner.extensive_control_variate(split.evaluation.x)
    assert np.all(q_scores.score >= j_scores.score - 1e-12)
    assert np.all(np.isfinite(c_scores.score))
    assert np.all((m0 >= 0.0) & (m0 <= 1.0) & (m1 >= 0.0) & (m1 <= 1.0))


def test_gro_strict_near_null_fwer_across_500_draws():
    false_joint = 0
    false_cvar = 0
    joint_spec = RiskSpec(bound="gro", quantile_tolerance=0.0, extensive_tolerance=0.0)
    cvar_spec = RiskSpec(
        bound="gro", quantile_tolerance=None, extensive_tolerance=None,
        cvar_tolerance=0.0, cvar_cap=50.0, cvar_cells=64,
    )

    def dummy_ctrl(x):
        return np.full(len(x), 0.60), np.full(len(x), 0.60)

    for seed in range(500):
        tr_raw, _ = null_recovery(seed + 10_000, unsafe=True, n=1000)
        tr = replace(tr_raw, row_id=tr_raw.row_id + 100_000)
        cal, family = null_recovery(seed, unsafe=True, n=1500)
        c_joint = certify_localized(cal, family, joint_spec, 0.5, train=tr, ext_control=dummy_ctrl, anchored=True)
        c_cvar = certify_localized(cal, family, cvar_spec, 0.5, train=tr, anchored=True)
        false_joint += bool(c_joint.passed)
        false_cvar += bool(c_cvar.passed)
    assert false_joint / 500 <= 0.065
    assert false_cvar / 500 <= 0.065
