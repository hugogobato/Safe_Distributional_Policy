"""Finite-sample pointwise bounds and exact survival/stop-loss profiles.

The constructions port the checked Phase 2 pilot. Direct adaptive bets need a
preassigned observation order. Monotone fixed-stake mixtures are used for
inversion, where monotonicity of adaptive bets would be insufficient.
"""

from math import inf, log

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .data import F64

BET_FRACTIONS = np.array([0.005, 0.015, 0.03, 0.06, 0.10, 0.20, 0.40, 0.70, 0.95])


def empirical_bernstein(mean, variance, n, width, delta):
    """Maurer–Pontil one-sided UCB using the unbiased variance."""
    if n < 2:
        raise ValueError("Empirical Bernstein requires at least two observations.")
    factor = log(2 / delta)
    return mean + np.sqrt(2 * np.maximum(0, variance) * factor / n) + 7 * width * factor / (3 * (n - 1))


def moments(sum_z, sum_z2, n):
    mean = sum_z / n
    return mean, np.maximum(0, (sum_z2 - n * mean**2) / (n - 1))


def eb_upper(z: F64, a: float, b: float, delta: float) -> float:
    variance = float(z.var(ddof=1)) if len(z) > 1 else 0.0
    return float(empirical_bernstein(z.mean(), variance, len(z), b - a, delta))


def adaptive_reject(z: F64, a: float, b: float, tau: float, delta: float) -> bool:
    """One-sided aGRAPA-style rejection of the unsafe mean, up to time n."""
    if tau >= b:
        return True
    if tau <= a:
        return False
    target = (tau - a) / (b - a)
    # Work in coordinates centered at the null to avoid cancellation in the
    # predictable variance. Null-valued increments have wealth factor exactly 1.
    varying = np.flatnonzero(z != tau)
    centered = (tau - z[varying]) / (b - a)
    if centered.size == 0:
        return False
    index = varying + 1
    past = np.r_[0.0, np.cumsum(centered[:-1])]
    past_sq = np.r_[0.0, np.cumsum(centered[:-1] ** 2)]
    error = past / index
    variance = np.maximum(1e-12, (0.0625 + past_sq - (index + 1) * error**2) / index)
    stake = np.clip(error / (variance + error**2), 0, 0.75 / (1 - target))
    return bool(np.max(np.cumsum(np.log1p(stake * centered))) >= log(1 / delta))


def monotone_reject(z: F64, a: float, b: float, tau: float, delta: float) -> bool:
    """Fixed-time mixture to test an upper bound, monotone in the threshold and scores."""
    if tau >= b:
        return True
    if tau <= a:
        return False
    nonzero = z != 0
    scaled = (tau - z[nonzero]) / (b - tau)
    terms = np.log1p(BET_FRACTIONS[:, None] * scaled).sum(axis=1)
    terms += (z.size - scaled.size) * np.log1p(BET_FRACTIONS * tau / (b - tau))
    return bool(logsumexp(terms) >= log(len(BET_FRACTIONS) / delta))


def gro_theta(
    z_train: F64,
    a: float,
    b: float,
    tau: float,
    *,
    shrink: float = 0.85,
    max_fraction: float = 0.90,
) -> float:
    """Training-optimized Growth-Rate Optimal (GRO) betting stake.

    Maximizes sum_j log(1 + theta * (tau - z_j)) on D_tr alone, strictly inside
    [0, max_fraction / (b - tau)], then shrinks toward zero to guard against
    training-split optimism. Because theta is deterministic given D_tr, the
    resulting calibration e-value prod_i (1 + theta * (tau - Z_i)) is exact
    finite-sample valid and permutation-invariant.
    """
    if not (0.0 < shrink <= 1.0 and 0.0 < max_fraction < 1.0):
        raise ValueError("shrink must lie in (0, 1] and max_fraction in (0, 1).")
    if z_train.size == 0 or tau >= b or tau <= a:
        return 0.0
    diff = tau - z_train
    if float(np.mean(diff)) <= 0.0:
        return 0.0
    upper = max_fraction / (b - tau)
    # Guard against numerical edge cases where a training observation exceeds b.
    z_max = float(np.max(z_train))
    if z_max > tau:
        upper = min(upper, 0.95 / (z_max - tau))
    if upper <= 0.0:
        return 0.0

    def neg_log_wealth(theta: float) -> float:
        return -float(np.sum(np.log1p(theta * diff)))

    result = minimize_scalar(neg_log_wealth, bounds=(0.0, upper), method="bounded")
    return float(max(0.0, shrink * float(result.x)))


def gro_log_wealth(z: F64, tau: float, theta: float) -> float:
    """Permutation-invariant log e-value sum_i log(1 + theta * (tau - z_i))."""
    if theta <= 0.0 or z.size == 0:
        return 0.0
    factors = 1.0 + theta * (tau - z)
    if np.any(factors <= 0.0):
        return -inf
    return float(np.sum(np.log(factors)))


def gro_reject(z: F64, a: float, b: float, tau: float, delta: float, theta: float) -> bool:
    """Exact finite-sample test using a training-determined GRO betting stake."""
    if tau >= b:
        return True
    if tau <= a or theta <= 0.0:
        return False
    if theta >= 1.0 / (b - tau):
        raise ValueError("GRO stake must be strictly less than 1 / (b - tau).")
    return gro_log_wealth(z, tau, theta) >= log(1.0 / delta)


def gro_pass_zscore(z_train: F64, tau: float, theta: float, n_cal: int, delta: float) -> float:
    """Training-only predicted calibration pass z-score for anchor selection."""
    if theta <= 0.0 or z_train.size < 2 or n_cal < 1:
        return -inf
    factors = 1.0 + theta * (tau - z_train)
    if np.any(factors <= 0.0):
        return -inf
    terms = np.log(factors)
    mean = float(np.mean(terms))
    sd = float(np.std(terms, ddof=1))
    threshold = log(1.0 / delta)
    if sd <= 1e-12:
        return inf if n_cal * mean >= threshold else -inf
    return float((n_cal * mean - threshold) / (np.sqrt(n_cal) * sd))


def betting_upper(z: F64, a: float, b: float, delta: float) -> float:
    """Bisection rounded outward to retain coverage at the requested size."""
    lower, upper = a, b
    for _ in range(24):
        middle = (a + b) / 2
        # Inversion updates the search interval, not the declared support of z.
        if monotone_reject(z, lower, upper, middle, delta):
            b = middle
        else:
            a = middle
    return b


def tail_sums(loss: F64, w: F64, cuts: F64) -> tuple[F64, F64]:
    """First and second raw sums of w * 1{loss > cut}, including tied losses."""
    active = w != 0
    order = np.argsort(loss[active])[::-1]
    y = loss[active][order]
    w = w[active][order]
    count = np.searchsorted(-y, -cuts, side="left")
    return np.r_[0.0, np.cumsum(w)][count], np.r_[0.0, np.cumsum(w**2)][count]


def quantile_interval(
    loss: F64,
    w: F64,
    beta: float,
    delta: float,
    w_max: float,
    method: str,
    *,
    gamma: float = 1.0,
    cap: float = inf,
) -> tuple[float, float]:
    """Phase 2 Lemma 1 inversion, including the after-last-knot interval.

    For a known ratio bound, compare the lower weighted survival bound to
    gamma * (1-beta) and the upper bound to (1-beta)/gamma.
    """
    knots = np.unique(np.r_[0.0, loss[w > 0]])
    lower_test_target = gamma * (1 - beta)
    upper_test_target = (1 - beta) / gamma
    alpha = delta / 2
    if method == "eb":
        s1, s2 = tail_sums(loss, w, knots)
        estimates, variances = moments(s1, s2, len(loss))
        radius = empirical_bernstein(0.0, variances, len(loss), w_max, alpha)
        lower = np.flatnonzero(estimates - radius <= lower_test_target)
        bad_upper = np.flatnonzero(estimates + radius > upper_test_target)
        i = int(lower[0]) if len(lower) else len(knots)
        j = int(bad_upper[-1]) + 1 if len(bad_upper) else 0
    elif method == "betting":
        def endpoint(lower_side: bool) -> int:
            lo, hi = 0, len(knots)
            while lo < hi:
                mid = (lo + hi) // 2
                z = np.where(loss > knots[mid], w, 0.0)
                if lower_side:
                    crossed = not monotone_reject(-z, -w_max, 0, -lower_test_target, alpha)
                else:
                    crossed = monotone_reject(z, 0, w_max, upper_test_target, alpha)
                if crossed:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        i, j = endpoint(True), endpoint(False)
    else:
        raise ValueError(f"Unknown confidence bound method: {method}.")

    lo = min(cap, float(knots[i])) if i < len(knots) else cap
    hi = min(cap, float(knots[j])) if j < len(knots) else cap
    return lo, hi


def cvar_eb_max(
    loss: F64, dw: F64, lo: float, hi: float, cap: float, w_max: float, delta: float,
    gamma: float = 1.0,
) -> float:
    """Exact max UCB over the VaR interval by convexity on each loss segment."""
    active = dw != 0
    signed = inflate_scores(dw[active], gamma)
    y = loss[active]
    knots = np.unique(np.r_[lo, hi, y[(y > lo) & (y < hi)]])
    order = np.argsort(y)[::-1]
    y, signed = y[order], signed[order]
    count = np.searchsorted(-y, -knots, side="left")

    def prefix(values: F64) -> F64:
        return np.r_[0.0, np.cumsum(values)][count]

    squares = signed**2
    s1 = prefix(signed * y) - knots * prefix(signed)
    s2 = prefix(squares * y**2) - 2 * knots * prefix(squares * y) + knots**2 * prefix(squares)
    mean, variance = moments(s1, s2, len(loss))
    width = (gamma + 1 / gamma) * (cap - knots) * w_max
    return float(np.max(empirical_bernstein(mean, variance, len(loss), width, delta)))


def inflate_scores(z: F64, gamma: float) -> F64:
    return np.where(z >= 0, gamma * z, z / gamma)


def opra_cdf_radius(n: int, w_max: float, delta: float, w2_bound: float | None = None) -> float:
    """Huang et al. Theorem 5.1 with deterministic bounds fixed by the design.

    The default uses E[w^2] <= w_max E[w] = w_max.
    """
    factor = log(4 / delta)
    second = w_max if w2_bound is None else w2_bound
    return min(
        np.sqrt(8 * w_max**2 * factor / n),
        4 * w_max * factor / n + 2 * np.sqrt(2 * second * factor / n),
    )


def weighted_cdf(loss: F64, weights: F64) -> tuple[F64, F64]:
    """Clipped IS CDF at its knots on nonnegative support."""
    active = weights > 0
    y = np.r_[0.0, loss[active]]
    w = np.r_[0.0, weights[active]]
    knots, index = np.unique(y, return_inverse=True)
    values = np.bincount(index, weights=w) / len(loss)
    return knots, np.minimum(1, np.cumsum(values))


def cvar_from_cdf(knots: F64, cdf: F64, beta: float, cap: float) -> float:
    """Upper-tail CVaR as the integral of 1-max(F, beta) over [0, cap]."""
    widths = np.diff(np.r_[0.0, knots, cap])
    cdf_on_intervals = np.r_[0.0, np.clip(cdf, 0.0, 1.0)]
    return float(widths @ (1 - np.maximum(cdf_on_intervals, beta)) / (1 - beta))


def inverse_cdf(knots: F64, cdf: F64, beta: float, upper_end: float = inf) -> float:
    i = np.searchsorted(cdf, beta, side="left")
    return float(knots[i]) if i < len(knots) else upper_end
