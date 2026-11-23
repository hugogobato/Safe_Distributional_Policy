"""Quantile-first joint certificates and separate-policy benchmarks."""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, inf, log
from typing import Literal, TypeAlias

import numpy as np

from .bounds import (
    adaptive_reject,
    betting_upper,
    cvar_eb_max,
    cvar_from_cdf,
    eb_upper,
    empirical_bernstein,
    gro_pass_zscore,
    gro_reject,
    gro_theta,
    inflate_scores,
    inverse_cdf,
    moments,
    monotone_reject,
    opra_cdf_radius,
    quantile_interval,
    tail_sums,
    weighted_cdf,
)
from .data import F64, I64, LoggedSample, validate_overlap


@dataclass(frozen=True)
class PolicyScores:
    action: I64
    score: F64

    def __post_init__(self):
        action = np.asarray(self.action)
        score = np.asarray(self.score, dtype=np.float64)
        if action.ndim != 1 or score.shape != action.shape or not np.isin(action, (0, 1)).all():
            raise ValueError("A policy needs one binary action and one score per row.")
        if not np.isfinite(score).all():
            raise ValueError("Policy scores must be finite.")
        object.__setattr__(self, "action", action.astype(np.int64, copy=False))
        object.__setattr__(self, "score", score)


Baseline: TypeAlias = int | Callable[[F64], I64]


@dataclass(frozen=True)
class PolicyFamily:
    predict: Callable[[F64], PolicyScores]
    thresholds: tuple[float, ...]
    baseline: Baseline = 0

    def __post_init__(self):
        if not callable(self.baseline) and self.baseline not in (0, 1):
            raise ValueError("The baseline action must be binary.")
        if any(np.isnan(t) or t == inf for t in self.thresholds):
            raise ValueError("The no-switch threshold is reserved for fallback.")
        if any(a <= b for a, b in zip(self.thresholds, self.thresholds[1:])):
            raise ValueError("Policy thresholds must be strictly decreasing.")

    def baseline_actions(self, x: F64) -> I64:
        if not callable(self.baseline):
            return np.full(len(x), self.baseline, dtype=np.int64)
        choices = np.asarray(self.baseline(x))
        if choices.shape != (len(x),) or not np.isin(choices, (0, 1)).all():
            raise ValueError("A baseline rule must return one binary action per row.")
        return choices.astype(np.int64, copy=False)

    def actuate(self, x: F64, threshold: float) -> I64:
        baseline = self.baseline_actions(x)
        return baseline if threshold == inf else gated_actions(self.predict(x), threshold, baseline)


def gated_actions(scores: PolicyScores, thresh: float, baseline: int | I64) -> I64:
    return np.where(scores.score >= thresh, scores.action, baseline).astype(np.int64)


def offer_from_training(
    predict: Callable[[F64], PolicyScores],
    train: LoggedSample,
    masses: tuple[float, ...],
    baseline: Baseline = 0,
) -> PolicyFamily:
    """Freeze candidate order from training ranks. Tied scores stay together."""
    if not masses or any(not 0 < m <= 1 for m in masses):
        raise ValueError("Prespecify one or more positive deployment masses.")
    if any(a >= b for a, b in zip(masses, masses[1:])):
        raise ValueError("Deployment masses must be strictly increasing.")
    pred = predict(train.x)
    base_action = PolicyFamily(predict, (), baseline).baseline_actions(train.x)
    eligible = np.sort(pred.score[pred.action != base_action])[::-1]
    offered: list[float] = []
    for mass in masses:
        rank = ceil(mass * train.n) - 1
        threshold = float(eligible[rank]) if rank < len(eligible) and mass < 1 else -inf
        if not offered or threshold != offered[-1]:
            offered.append(threshold)
    return PolicyFamily(predict, tuple(offered), baseline)


@dataclass(frozen=True)
class RiskSpec:
    beta: float = 0.90
    delta: float = 0.05
    bound: Literal["betting", "eb", "gro"] = "betting"
    quantile_tolerance: float | None = 0.0
    extensive_tolerance: float | None = 0.0
    cvar_tolerance: float | None = None
    cvar_cap: float = 50.0
    extensive_event: Literal["positive", "zero"] = "positive"
    var_budget_fraction: float = 0.20
    cvar_cells: int = 250

    def __post_init__(self):
        if not (0 < self.beta < 1 and 0 < self.delta < 1 and 0 < self.var_budget_fraction < 1):
            raise ValueError("Beta, delta and the VaR budget fraction must be interior.")
        if self.bound not in ("betting", "eb", "gro") or self.extensive_event not in ("positive", "zero"):
            raise ValueError("Unknown bound or extensive event.")
        tolerances = self.quantile_tolerance, self.extensive_tolerance, self.cvar_tolerance
        if all(t is None for t in tolerances) or any(
            t is not None and (not np.isfinite(t) or t < 0) for t in tolerances
        ):
            raise ValueError("Supply at least one finite, nonnegative risk tolerance.")
        if not np.isfinite(self.cvar_cap) or self.cvar_cap <= 0 or self.cvar_cells < 1:
            raise ValueError("Use a positive finite CVaR cap and at least one cell.")

    def events(self, loss: F64) -> F64:
        indicator = loss > 0 if self.extensive_event == "positive" else loss == 0
        return indicator.astype(np.float64)


@dataclass(frozen=True)
class PolicyCertificate:
    method: str
    threshold: float
    passed: tuple[float, ...]
    tested: int
    guarantee: Literal["joint", "indicator", "none", "oracle"]
    delta: float | None

    @property
    def deployed(self) -> bool:
        return self.threshold != inf


def fixed_sequence(
    family: PolicyFamily,
    test: Callable[[float], bool],
    method: str,
    guarantee: Literal["joint", "indicator"],
    delta: float,
    *,
    order: tuple[int, ...] | None = None,
) -> PolicyCertificate:
    """Stop at the first failed test in the predetermined sequence order."""
    indices = tuple(range(len(family.thresholds))) if order is None else order
    if len(set(indices)) != len(indices) or any(not 0 <= i < len(family.thresholds) for i in indices):
        raise ValueError("Fixed-sequence order must contain distinct valid candidate indices.")
    passed: list[float] = []
    n_tested = 0
    for idx in indices:
        threshold = family.thresholds[idx]
        n_tested += 1
        if not test(threshold):
            break
        passed.append(threshold)
    selected = min(passed) if passed else inf
    return PolicyCertificate(method, selected, tuple(passed), n_tested, guarantee, delta)


def policy_weight(data: LoggedSample, actions: I64) -> F64:
    observed_propensity = np.where(data.action == 1, data.propensity, 1 - data.propensity)
    return (data.action == actions) / observed_propensity


def _training_ipw_quantile(data: LoggedSample, baseline_action: I64, beta: float) -> float:
    w0 = policy_weight(data, baseline_action)
    active = w0 > 0
    if not active.any():
        return 0.0
    order = np.argsort(data.loss[active])
    loss = data.loss[active][order]
    weights = w0[active][order]
    cdf = np.cumsum(weights) / weights.sum()
    return float(loss[min(len(loss) - 1, int(np.searchsorted(cdf, beta)))])


def _aipw_extensive_score(
    proposed_action: I64,
    baseline_action: I64,
    proposed_w: F64,
    baseline_w: F64,
    ext: F64,
    m0: F64,
    m1: F64,
    gamma: float,
) -> F64:
    """Localized AIPW contrast with exact S_lambda^c cancellation and zero range inflation."""
    switched = proposed_action != baseline_action
    m_prop = np.where(proposed_action == 1, m1, m0)
    m_base = np.where(baseline_action == 1, m1, m0)
    plugin = m_prop - m_base
    residual = proposed_w * (ext - m_prop) - baseline_w * (ext - m_base)
    return np.where(switched, plugin + inflate_scores(residual, gamma), 0.0)


def propensity_gated_family(
    predict: Callable[[F64], PolicyScores],
    train: LoggedSample,
    masses: tuple[float, ...],
    propensity_fn: Callable[[F64], F64],
    overlap: float,
    *,
    baseline: Baseline = 0,
    floor_quantile: float = 0.05,
    reference_mass: float = 0.30,
) -> tuple[PolicyFamily, Callable[[float], tuple[float, float]]]:
    """Gate switches on a known propensity model to tighten deterministic support bounds.

    Determines propensity floors (kappa0, kappa1) purely on D_tr from the top
    reference_mass candidate units, then restricts switches to units satisfying
    e(pi_hat(x)|x) >= kappa1 and e(pi_0(x)|x) >= kappa0. Because the gate is
    deterministic given D_tr and applied identically to calibration and evaluation,
    the resulting localized support bounds (1/kappa0, 1/kappa1) are exact.
    """
    if not (0.0 <= floor_quantile < 1.0 and 0.0 < reference_mass <= 1.0):
        raise ValueError("floor_quantile must be in [0, 1) and reference_mass in (0, 1].")
    base_tr = PolicyFamily(predict, (), baseline).baseline_actions(train.x)
    pred_tr = predict(train.x)
    e_tr = np.asarray(propensity_fn(train.x), dtype=np.float64)
    eligible = pred_tr.action != base_tr
    kappa0 = overlap
    kappa1 = overlap
    if eligible.any():
        elig_scores = pred_tr.score[eligible]
        top_k = max(1, min(len(elig_scores), ceil(reference_mass * train.n)))
        cutoff = float(np.sort(elig_scores)[::-1][top_k - 1])
        top_mask = eligible & (pred_tr.score >= cutoff)
        if top_mask.any():
            e_prop = np.where(pred_tr.action[top_mask] == 1, e_tr[top_mask], 1.0 - e_tr[top_mask])
            e_base = np.where(base_tr[top_mask] == 1, e_tr[top_mask], 1.0 - e_tr[top_mask])
            kappa1 = float(np.clip(np.quantile(e_prop, floor_quantile), overlap, 0.5))
            kappa0 = float(np.clip(np.quantile(e_base, floor_quantile), overlap, 0.5))

    base_resolver = PolicyFamily(predict, (), baseline)

    def gated_predict(x: F64) -> PolicyScores:
        raw = predict(x)
        base_x = base_resolver.baseline_actions(x)
        e_x = np.asarray(propensity_fn(x), dtype=np.float64)
        e_prop = np.where(raw.action == 1, e_x, 1.0 - e_x)
        e_base = np.where(base_x == 1, e_x, 1.0 - e_x)
        allowed = (e_prop >= kappa1) & (e_base >= kappa0)
        action = np.where(allowed, raw.action, base_x).astype(np.int64)
        return PolicyScores(action, raw.score)

    family = offer_from_training(gated_predict, train, masses, baseline)
    bounds = (1.0 / kappa0, 1.0 / kappa1)
    return family, lambda _: bounds


def _reject(z: F64, a: float, b: float, tau: float, delta: float, method: str) -> bool:
    if tau >= b:
        return True
    if method == "eb":
        return eb_upper(z, a, b, delta) <= tau
    return adaptive_reject(z, a, b, tau, delta)


def _relative_quantile_test(
    y: F64, proposed: F64, baseline: F64, interval: tuple[float, float], tolerance: float,
    w0_max: float, w1_max: float, delta: float, method: str, gamma: float,
    *, gro_stake: float | None = None,
) -> bool:
    lo, hi = interval
    candidate_only = np.where(baseline == 0, proposed, 0.0)
    baseline_only = np.where(proposed == 0, baseline, 0.0)
    shared = np.where(proposed > 0, baseline, 0.0)
    if tolerance == 0:
        score_changes = y[(candidate_only > 0) | (baseline_only > 0)]
    else:
        score_changes = np.r_[y[proposed > 0] - tolerance, y[baseline > 0]]
    inside = score_changes[(score_changes > lo) & (score_changes < hi)]
    knots = np.unique(np.r_[lo, inside, hi] if np.isfinite(hi) else np.r_[lo, inside])
    if not np.isfinite(lo):
        return False

    a, b = -w0_max / gamma, w1_max * gamma
    if method == "gro":
        if gro_stake is None or gro_stake <= 0.0:
            return False
        if gro_stake >= 1.0 / b:
            raise ValueError("GRO stake must be strictly less than 1 / b.")
        if tolerance == 0:
            dw = proposed - baseline
            active = dw != 0
            if not active.any():
                return False
            inflated = inflate_scores(dw[active], gamma)
            factors = 1.0 - gro_stake * inflated
            if np.any(factors <= 0.0):
                return False
            wealth_at_knots = tail_sums(y[active], np.log(factors), knots)[0]
            return bool(np.min(wealth_at_knots) >= log(1.0 / delta))
        for eta in knots:
            z = proposed * (y - tolerance > eta) - baseline * (y > eta)
            if not gro_reject(inflate_scores(z, gamma), a, b, 0.0, delta, gro_stake):
                return False
        return True

    pos1, pos2 = tail_sums(y - tolerance, candidate_only, knots)
    neg1, neg2 = tail_sums(y, baseline_only, knots)
    if tolerance > 0:
        all1, all2 = tail_sums(y, shared, knots)
        shifted1, shifted2 = tail_sums(y - tolerance, shared, knots)
        neg1 += all1 - shifted1
        neg2 += all2 - shifted2
    means, variances = moments(gamma * pos1 - neg1 / gamma, gamma**2 * pos2 + neg2 / gamma**2, y.size)
    if method == "eb":
        return bool(np.max(empirical_bernstein(means, variances, y.size, b - a, delta)) <= 0)
    # The order here is for checking all hypotheses, never for ordering the bets' data.
    for i in np.argsort(means)[::-1]:
        eta = knots[i]
        # Evaluate the right-continuous candidate at the breakpoint even when
        # fl(eta + tolerance) would round just below the observed loss.
        z = proposed * (y - tolerance > eta) - baseline * (y > eta)
        if not adaptive_reject(inflate_scores(z, gamma), a, b, 0, delta):
            return False
    return True


def _relative_cvar_test(
    y: F64, dw: F64, interval: tuple[float, float], spec: RiskSpec, delta: float,
    w0_max: float, w1_max: float, gamma: float,
    *, train_y: F64 | None = None, train_dw: F64 | None = None,
) -> bool:
    assert spec.cvar_tolerance is not None
    cap = spec.cvar_cap
    lo, hi = (min(cap, value) for value in interval)
    tau = (1 - spec.beta) * spec.cvar_tolerance
    capped = np.minimum(y, cap)
    if spec.bound == "eb":
        return cvar_eb_max(capped, dw, lo, hi, cap, max(w0_max, w1_max), delta, gamma) <= tau

    grid = np.linspace(0, cap, spec.cvar_cells + 1)
    left = max(0, min(spec.cvar_cells - 1, int(np.searchsorted(grid, lo, side="right")) - 1))
    right = min(spec.cvar_cells - 1, int(np.searchsorted(grid, hi, side="right")) - 1)
    positive = np.maximum(dw, 0) * gamma
    negative = np.minimum(dw, 0) / gamma
    cells = np.arange(left, right + 1)
    lower = grid[cells]
    upper = grid[cells + 1]
    pos1 = tail_sums(capped, positive * capped, lower)[0] - lower * tail_sums(capped, positive, lower)[0]
    neg1 = tail_sums(capped, negative * capped, upper)[0] - upper * tail_sums(capped, negative, upper)[0]
    order = cells[np.argsort(pos1 + neg1)[::-1]]

    if spec.bound == "gro":
        assert train_y is not None and train_dw is not None
        capped_tr = np.minimum(train_y, cap)
        pos_tr = np.maximum(train_dw, 0) * gamma
        neg_tr = np.minimum(train_dw, 0) / gamma
        for cell in order:
            low, high = float(grid[cell]), float(grid[cell + 1])
            a_cell = -(cap - high) * w0_max / gamma
            b_cell = (cap - low) * w1_max * gamma
            z_tr = pos_tr * np.maximum(capped_tr - low, 0.0) + neg_tr * np.maximum(capped_tr - high, 0.0)
            theta = gro_theta(z_tr, a_cell, b_cell, tau)
            z_cal = positive * np.maximum(capped - low, 0.0) + negative * np.maximum(capped - high, 0.0)
            if not gro_reject(z_cal, a_cell, b_cell, tau, delta, theta):
                return False
        return True

    for cell in order:
        low, high = grid[cell:cell + 2]
        z = positive * np.maximum(capped - low, 0) + negative * np.maximum(capped - high, 0)
        if not adaptive_reject(z, -(cap - high) * w0_max / gamma, (cap - low) * w1_max * gamma, tau, delta):
            return False
    return True


def _localized_test(
    data: LoggedSample,
    scores: PolicyScores,
    baseline_action: I64,
    spec: RiskSpec,
    w_max: float,
    gamma: float,
    delta1: float,
    delta2: float,
    *,
    train: LoggedSample | None = None,
    train_scores: PolicyScores | None = None,
    train_baseline_action: I64 | None = None,
    ext_control: Callable[[F64], tuple[F64, F64]] | None = None,
    localized_bounds: Callable[[float], tuple[float, float]] | None = None,
) -> tuple[Callable[[float], bool], Callable[[float, int], float]]:
    if data.n < 2:
        return (lambda _: False), (lambda _t, _n: -inf)
    baseline = policy_weight(data, baseline_action)
    interval = (0.0, inf)
    if delta1:
        iv_method = "eb" if spec.bound == "eb" else "betting"
        interval = quantile_interval(data.loss, baseline, spec.beta, delta1, w_max, iv_method, gamma=gamma)
    ext = spec.events(data.loss)
    m0_cal, m1_cal = ext_control(data.x) if ext_control is not None else (None, None)

    train_w0: F64 | None = None
    train_ext: F64 | None = None
    train_cutoff = 0.0
    m0_tr: F64 | None = None
    m1_tr: F64 | None = None
    if train is not None and train.n >= 2:
        assert train_baseline_action is not None
        train_w0 = policy_weight(train, train_baseline_action)
        train_ext = spec.events(train.loss)
        if spec.quantile_tolerance is not None or spec.cvar_tolerance is not None:
            train_cutoff = _training_ipw_quantile(train, train_baseline_action, spec.beta)
        if ext_control is not None:
            m0_tr, m1_tr = ext_control(train.x)

    def resolve_bounds(threshold: float, proposed_action: I64, proposed_w: F64) -> tuple[float, float]:
        if localized_bounds is None:
            return w_max, w_max
        w0_b, w1_b = localized_bounds(threshold)
        if w0_b < 1.0 or w1_b < 1.0 or not (np.isfinite(w0_b) and np.isfinite(w1_b)):
            raise ValueError("Localized support bounds must be finite and at least 1.")
        switched = proposed_action != baseline_action
        if switched.any():
            eps = 1e-9
            if np.any(proposed_w[switched] > w1_b + eps) or np.any(baseline[switched] > w0_b + eps):
                raise ValueError("Observed calibration weights violate the declared localized support bound.")
        return float(w0_b), float(w1_b)

    def train_zscore(threshold: float, projected_n_cal: int) -> float:
        if train is None or train.n < 2 or train_scores is None or train_baseline_action is None:
            return -inf
        assert train_w0 is not None and train_ext is not None
        w0_b, w1_b = (w_max, w_max) if localized_bounds is None else localized_bounds(threshold)
        prop_act_tr = gated_actions(train_scores, threshold, train_baseline_action)
        prop_w_tr = policy_weight(train, prop_act_tr)
        dw_tr = prop_w_tr - train_w0
        z_scores: list[float] = []
        if spec.extensive_tolerance is not None:
            if m0_tr is not None and m1_tr is not None:
                z_ext_tr = _aipw_extensive_score(
                    prop_act_tr, train_baseline_action, prop_w_tr, train_w0, train_ext, m0_tr, m1_tr, gamma,
                )
                w_hi = max(w0_b, w1_b)
                a_ext, b_ext = -max(w_hi / gamma, 1.0), w_hi * gamma
            else:
                z_ext_tr = inflate_scores(dw_tr * train_ext, gamma)
                a_ext, b_ext = -w0_b / gamma, w1_b * gamma
            th_ext = gro_theta(z_ext_tr, a_ext, b_ext, spec.extensive_tolerance)
            z_scores.append(gro_pass_zscore(z_ext_tr, spec.extensive_tolerance, th_ext, projected_n_cal, delta2))
        if spec.quantile_tolerance is not None:
            tol = spec.quantile_tolerance
            z_q_tr = inflate_scores(
                prop_w_tr * (train.loss - tol > train_cutoff) - train_w0 * (train.loss > train_cutoff), gamma,
            )
            a_q, b_q = -w0_b / gamma, w1_b * gamma
            th_q = gro_theta(z_q_tr, a_q, b_q, 0.0)
            z_scores.append(gro_pass_zscore(z_q_tr, 0.0, th_q, projected_n_cal, delta2))
        if spec.cvar_tolerance is not None:
            cap = spec.cvar_cap
            tau = (1.0 - spec.beta) * spec.cvar_tolerance
            grid = np.linspace(0.0, cap, spec.cvar_cells + 1)
            cell = max(0, min(spec.cvar_cells - 1, int(np.searchsorted(grid, min(cap, train_cutoff), side="right")) - 1))
            low, high = float(grid[cell]), float(grid[cell + 1])
            capped_tr = np.minimum(train.loss, cap)
            z_c_tr = (
                np.maximum(dw_tr, 0.0) * gamma * np.maximum(capped_tr - low, 0.0)
                + np.minimum(dw_tr, 0.0) / gamma * np.maximum(capped_tr - high, 0.0)
            )
            a_c, b_c = -(cap - high) * w0_b / gamma, (cap - low) * w1_b * gamma
            th_c = gro_theta(z_c_tr, a_c, b_c, tau)
            z_scores.append(gro_pass_zscore(z_c_tr, tau, th_c, projected_n_cal, delta2))
        return min(z_scores) if z_scores else -inf

    def tests(threshold: float) -> bool:
        if interval[0] > interval[1]:
            return False
        prop_act = gated_actions(scores, threshold, baseline_action)
        proposed = policy_weight(data, prop_act)
        w0_b, w1_b = resolve_bounds(threshold, prop_act, proposed)
        dw = proposed - baseline

        prop_act_tr: I64 | None = None
        prop_w_tr: F64 | None = None
        dw_tr: F64 | None = None
        if spec.bound == "gro":
            if train is None or train.n < 2 or train_scores is None or train_baseline_action is None:
                return False
            assert train_w0 is not None
            prop_act_tr = gated_actions(train_scores, threshold, train_baseline_action)
            prop_w_tr = policy_weight(train, prop_act_tr)
            dw_tr = prop_w_tr - train_w0

        if spec.extensive_tolerance is not None:
            if m0_cal is not None and m1_cal is not None:
                z_ext = _aipw_extensive_score(
                    prop_act, baseline_action, proposed, baseline, ext, m0_cal, m1_cal, gamma,
                )
                w_hi = max(w0_b, w1_b)
                a_ext, b_ext = -max(w_hi / gamma, 1.0), w_hi * gamma
            else:
                z_ext = inflate_scores(dw * ext, gamma)
                a_ext, b_ext = -w0_b / gamma, w1_b * gamma
            if spec.bound == "gro":
                assert train is not None and train_w0 is not None and train_ext is not None
                assert prop_act_tr is not None and prop_w_tr is not None and dw_tr is not None
                assert train_baseline_action is not None
                if m0_tr is not None and m1_tr is not None:
                    z_ext_tr = _aipw_extensive_score(
                        prop_act_tr, train_baseline_action, prop_w_tr, train_w0, train_ext, m0_tr, m1_tr, gamma,
                    )
                else:
                    z_ext_tr = inflate_scores(dw_tr * train_ext, gamma)
                th_ext = gro_theta(z_ext_tr, a_ext, b_ext, spec.extensive_tolerance)
                if not gro_reject(z_ext, a_ext, b_ext, spec.extensive_tolerance, delta2, th_ext):
                    return False
            elif not _reject(z_ext, a_ext, b_ext, spec.extensive_tolerance, delta2, spec.bound):
                return False

        if spec.quantile_tolerance is not None:
            th_q: float | None = None
            if spec.bound == "gro":
                assert train is not None and train_w0 is not None and prop_w_tr is not None
                tol = spec.quantile_tolerance
                z_q_tr = inflate_scores(
                    prop_w_tr * (train.loss - tol > train_cutoff) - train_w0 * (train.loss > train_cutoff), gamma,
                )
                th_q = gro_theta(z_q_tr, -w0_b / gamma, w1_b * gamma, 0.0)
            if not _relative_quantile_test(
                data.loss, proposed, baseline, interval, spec.quantile_tolerance,
                w0_b, w1_b, delta2, spec.bound, gamma, gro_stake=th_q,
            ):
                return False

        if spec.cvar_tolerance is not None:
            if not _relative_cvar_test(
                data.loss, dw, interval, spec, delta2, w0_b, w1_b, gamma,
                train_y=None if train is None else train.loss, train_dw=dw_tr,
            ):
                return False
        return True

    return tests, train_zscore


def certify_localized(
    calibration: LoggedSample,
    family: PolicyFamily,
    spec: RiskSpec,
    overlap: float,
    *,
    gamma: float = 1.0,
    method: str = "LCT-LTT",
    groups: Callable[[F64], I64] | None = None,
    required_groups: tuple[int, ...] = (),
    train: LoggedSample | None = None,
    train_scores: PolicyScores | None = None,
    ext_control: Callable[[F64], tuple[F64, F64]] | None = None,
    anchored: bool = False,
    anchor_z_threshold: float = 3.0,
    localized_bounds: Callable[[float], tuple[float, float]] | None = None,
) -> PolicyCertificate:
    """Certify one policy jointly overall and in prespecified covariate groups.

    Pass a justified gamma >= 1 for the multiplicative propensity sensitivity
    allowance. When spec.bound == "gro", pass a strictly disjoint training split
    `train` so betting stakes (and optionally `anchored=True` entry order) are
    determined strictly on D_tr with exact finite-sample FWER on D_cal.
    """
    if not np.isfinite(gamma) or gamma < 1:
        raise ValueError("Invalid propensity sensitivity or overlap bound.")
    if bool(groups) != bool(required_groups) or len(set(required_groups)) != len(required_groups):
        raise ValueError("Supply a group map and prespecified distinct groups together.")
    if spec.bound == "gro" and train is None:
        raise ValueError("Training-optimized GRO betting requires a disjoint `train` sample.")
    if anchored and (spec.bound != "gro" or train is None):
        raise ValueError("Training-anchored fixed-sequence ordering requires bound='gro' and `train`.")
    validate_overlap(calibration, overlap)
    if train is not None:
        validate_overlap(train, overlap)
        if np.intersect1d(train.row_id, calibration.row_id).size > 0:
            raise ValueError("Training and calibration splits must have disjoint row IDs.")
    data = calibration.ordered()
    delta1 = spec.delta * spec.var_budget_fraction if (
        spec.quantile_tolerance is not None or spec.cvar_tolerance is not None
    ) else 0.0
    samples = [data]
    train_samples: list[LoggedSample | None] = [train]
    train_score_slices: list[PolicyScores | None] = [train_scores]
    if groups is not None:
        membership = groups(data.x)
        if membership.shape != (data.n,):
            raise ValueError("Group mapping must return one label per observation.")
        samples.extend(data.take(membership == g) for g in required_groups)
        if train is not None:
            tr_membership = groups(train.x)
            if tr_membership.shape != (train.n,):
                raise ValueError("Group mapping must return one label per training observation.")
            for g in required_groups:
                mask = tr_membership == g
                train_samples.append(train.take(mask))
                if train_scores is not None:
                    train_score_slices.append(PolicyScores(train_scores.action[mask], train_scores.score[mask]))
                else:
                    train_score_slices.append(None)
        else:
            train_samples.extend(None for _ in required_groups)
            train_score_slices.extend(None for _ in required_groups)

    built = [
        _localized_test(
            sample,
            family.predict(sample.x) if sample.n >= 2 else PolicyScores(sample.action, sample.loss),
            family.baseline_actions(sample.x),
            spec,
            1 / overlap,
            gamma,
            delta1 / len(samples),
            spec.delta - delta1,
            train=tr_sample,
            train_scores=(
                None if tr_sample is None or tr_sample.n < 2
                else (tr_sc if tr_sc is not None else family.predict(tr_sample.x))
            ),
            train_baseline_action=(
                None if tr_sample is None else family.baseline_actions(tr_sample.x)
            ),
            ext_control=ext_control,
            localized_bounds=localized_bounds,
        )
        for sample, tr_sample, tr_sc in zip(samples, train_samples, train_score_slices)
    ]
    tests = [pair[0] for pair in built]
    zscorers = [pair[1] for pair in built]

    order: tuple[int, ...] | None = None
    if anchored and family.thresholds:
        assert train is not None
        # Project group calibration sizes strictly from D_tr shares and fixed total calibration.n.
        projected_ns = [
            data.n if idx == 0 else max(1, round(data.n * (tr_s.n / train.n)))
            for idx, tr_s in enumerate(train_samples)
            if tr_s is not None
        ]
        candidate_z = [
            min(zfn(t, n_proj) for zfn, n_proj in zip(zscorers, projected_ns))
            for t in family.thresholds
        ]
        confident = [idx for idx, z in enumerate(candidate_z) if z >= anchor_z_threshold]
        if confident:
            anchor = confident[0]
        else:
            # Prefer interior candidates (not the unguarded -inf endpoint) to avoid
            # cliff-edge training optimism when selecting the fixed-sequence entry point.
            interior = [
                idx for idx, t in enumerate(family.thresholds)
                if np.isfinite(t) and candidate_z[idx] > -inf
            ]
            if interior:
                anchor = max(interior, key=lambda idx: candidate_z[idx])
            else:
                best = max(candidate_z)
                anchor = int(np.argmax(candidate_z)) if best > -inf else 0
        order = tuple(range(anchor, len(family.thresholds)))

    return fixed_sequence(
        family, lambda threshold: all(t(threshold) for t in tests), method, "joint", spec.delta, order=order,
    )


def certify_indicator(
    calibration: LoggedSample,
    family: PolicyFamily,
    cutoff: float,
    spec: RiskSpec,
    overlap: float,
    *,
    tolerance: float = 0.0,
    train: LoggedSample | None = None,
) -> PolicyCertificate:
    """B5. Fixed training threshold, bounded indicator loss and LTT.

    This is the off-policy indicator analogue, using known propensities and a
    high-probability LTT guarantee for a policy contrast. It is distinct from
    CPC's likelihood-ratio-regulated randomized action policy and expected risk.
    """
    if not np.isfinite(cutoff) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("The prespecified cutoff and nonnegative tolerance must be finite.")
    if spec.bound == "gro" and train is None:
        raise ValueError("GRO indicator certification requires a disjoint `train` sample.")
    validate_overlap(calibration, overlap)
    if train is not None:
        validate_overlap(train, overlap)
        if np.intersect1d(train.row_id, calibration.row_id).size > 0:
            raise ValueError("Training and calibration splits must have disjoint row IDs.")
    data = calibration.ordered()
    if data.n < 2:
        return fixed_sequence(family, lambda _: False, "B5", "indicator", spec.delta)
    pred = family.predict(data.x)
    base_action = family.baseline_actions(data.x)
    w0 = policy_weight(data, base_action)
    indicator = data.loss > cutoff
    w_max = 1 / overlap

    pred_tr = None if train is None else family.predict(train.x)
    base_tr = None if train is None else family.baseline_actions(train.x)
    w0_tr = None if train is None or base_tr is None else policy_weight(train, base_tr)
    ind_tr = None if train is None else (train.loss > cutoff)

    def test(threshold: float) -> bool:
        w = policy_weight(data, gated_actions(pred, threshold, base_action))
        z = (w - w0) * indicator
        if spec.bound == "gro":
            assert train is not None and pred_tr is not None and base_tr is not None
            assert w0_tr is not None and ind_tr is not None
            w_tr = policy_weight(train, gated_actions(pred_tr, threshold, base_tr))
            z_tr = (w_tr - w0_tr) * ind_tr
            theta = gro_theta(z_tr, -w_max, w_max, tolerance)
            return gro_reject(z, -w_max, w_max, tolerance, spec.delta, theta)
        return _reject(z, -w_max, w_max, tolerance, spec.delta, spec.bound)

    return fixed_sequence(family, test, "B5", "indicator", spec.delta)


def certify_separate(
    calibration: LoggedSample,
    family: PolicyFamily,
    spec: RiskSpec,
    overlap: float,
    *,
    engine: Literal["opra", "survival"] = "survival",
    weight_second_moment: Callable[[float], float] | None = None,
) -> PolicyCertificate:
    """B1. Separate policy bounds with the same trained scores and fixed sequence.

    Supply an analytical second-moment upper bound as a callback, where inf
    denotes the baseline; otherwise OPRA uses the design's envelope.
    """
    validate_overlap(calibration, overlap)
    data = calibration.ordered()
    if data.n < 2:
        return fixed_sequence(family, lambda _: False, f"B1-{engine}", "joint", spec.delta)
    if engine not in ("opra", "survival"):
        raise ValueError("Unknown separate policy engine.")
    scores = family.predict(data.x)
    w_max = 1 / overlap
    base_action = family.baseline_actions(data.x)
    w0 = policy_weight(data, base_action)
    alpha = spec.delta / 2
    ext = spec.events(data.loss)
    ext_lower = -betting_upper(-w0 * ext, -w_max, 0, alpha)
    q_lower = 0.0
    y_cap = np.minimum(data.loss, spec.cvar_cap)
    cvar_lower = 0.0

    def radius(threshold: float) -> float:
        moment = weight_second_moment(threshold) if weight_second_moment is not None else None
        return opra_cdf_radius(data.n, w_max, alpha, moment)

    if engine == "opra" and spec.quantile_tolerance is not None:
        knots, cdf = weighted_cdf(data.loss, w0)
        q_lower = inverse_cdf(knots, np.minimum(1, cdf + radius(inf)), spec.beta)
    if engine == "survival" and spec.quantile_tolerance is not None:
        q_lower = quantile_interval(data.loss, w0, spec.beta, spec.delta, w_max, "betting")[0]
    if spec.cvar_tolerance is not None:
        knots, cdf = weighted_cdf(y_cap, w0)
        cvar_lower = cvar_from_cdf(knots, cdf + radius(inf), spec.beta, spec.cvar_cap)

    def tests(threshold: float) -> bool:
        w = policy_weight(data, gated_actions(scores, threshold, base_action))
        if spec.extensive_tolerance is not None and not monotone_reject(
            w * ext, 0, w_max, ext_lower + spec.extensive_tolerance, alpha,
        ):
            return False
        if spec.quantile_tolerance is not None:
            target = q_lower + spec.quantile_tolerance
            if not np.isfinite(target):
                return False
            if engine == "survival" and not monotone_reject(
                w * (data.loss > target), 0, w_max, 1 - spec.beta, alpha,
            ):
                return False
            if engine == "opra":
                knots, cdf = weighted_cdf(data.loss, w)
                if inverse_cdf(knots, np.maximum(0, cdf - radius(threshold)), spec.beta) > target:
                    return False
        if spec.cvar_tolerance is not None:
            knots, cdf = weighted_cdf(y_cap, w)
            u = cvar_from_cdf(knots, cdf - radius(threshold), spec.beta, spec.cvar_cap)
            if u > cvar_lower + spec.cvar_tolerance:
                return False
        return True

    return fixed_sequence(family, tests, f"B1-{engine}", "joint", spec.delta)
