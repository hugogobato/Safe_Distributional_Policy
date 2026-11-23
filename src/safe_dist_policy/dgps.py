"""Known-truth fixtures and the Phase 3 smoke data-generating process."""

import numpy as np

from .certification import PolicyFamily, PolicyScores
from .data import F64, EvaluationSample, ExperimentSplit, LoggedSample


def _logged(x: F64, loss: F64, utility: F64, propensity: F64, start: int, rng) -> LoggedSample:
    n = len(x)
    rows = np.arange(n)
    action = (rng.random(n) < propensity).astype(np.int64)
    return LoggedSample(
        x, action, loss[rows, action], propensity, rows + start, utility[rows, action],
    )


def _near_noiseless(n: int, rng):
    x = rng.uniform(-1, 1, (n, 3))
    safe = x[:, 0] < 0
    active0 = rng.random(n) >= 0.40
    active1 = rng.random(n) >= np.where(safe, 0.74, 0.30)
    loss0 = active0 * rng.lognormal(1.40, 0.22, n)
    loss1 = active1 * rng.lognormal(np.where(safe, 0.6, 2.0), np.where(safe, 0.08, 0.25), n)
    loss = np.column_stack((loss0, loss1))
    utility = -loss
    propensity = 0.5 + 0.20 * x[:, 0]
    return x, loss, utility, propensity


def smoke_split(seed: int, n_train: int = 1600, n_cal: int = 5000, n_eval: int = 50_000) -> ExperimentSplit:
    streams = np.random.SeedSequence([1, seed]).spawn(3)
    rng_train, rng_cal, rng_eval = [np.random.default_rng(s) for s in streams]
    xt, yt, ut, et = _near_noiseless(n_train, rng_train)
    xc, yc, uc, ec = _near_noiseless(n_cal, rng_cal)
    xe, ye, ue, _ = _near_noiseless(n_eval, rng_eval)
    return ExperimentSplit(
        train=_logged(xt, yt, ut, et, 0, rng_train),
        calibration=_logged(xc, yc, uc, ec, n_train, rng_cal),
        evaluation=EvaluationSample(xe, ye, ue, np.arange(n_eval) + n_train + n_cal),
        overlap=0.30,
    )


def oracle_smoke_family(masses: tuple[float, ...]) -> PolicyFamily:
    def score(x: F64) -> PolicyScores:
        return PolicyScores(np.ones(len(x), dtype=np.int64), -x[:, 0])

    return PolicyFamily(score, tuple(1 - 2 * m for m in masses))


def deterministic_recovery(seed: int = 19, n: int = 8000) -> tuple[LoggedSample, PolicyFamily]:
    """Every potential outcome is constant within its known covariate cell.

    The two distinct nonbaseline candidates switch exactly the first half or
    the full population. Only the first is jointly safe at beta=.90.
    """
    rng = np.random.default_rng(np.random.SeedSequence([11, seed]))
    x = rng.random((n, 2))
    safe = x[:, 0] < 0.5
    loss0 = np.where(x[:, 1] < 0.4, 0.0, np.where(x[:, 1] < 0.93, 1.0, 2.0))
    loss1 = np.where(safe, 0.0, 3.0)
    loss = np.c_[loss0, loss1]
    calibration = _logged(x, loss, -loss, np.full(n, 0.5), 0, rng)

    def score(features: F64) -> PolicyScores:
        return PolicyScores(np.ones(len(features), dtype=np.int64), (features[:, 0] < 0.5).astype(float))

    return calibration, PolicyFamily(score, (1.0, 0.0))


def null_recovery(seed: int, unsafe: bool, n: int = 1500) -> tuple[LoggedSample, PolicyFamily]:
    """Equality null or uniformly strict near-null under a known randomized design."""
    rng = np.random.default_rng(np.random.SeedSequence([31, seed]))
    x = rng.uniform(-1, 1, (n, 3))
    u = rng.random(n)
    amount = rng.lognormal(1.6, 0.85, n)
    loss0 = np.where(u < 0.40, 0, amount)
    loss1 = np.where(u < 0.399, 0, amount * np.exp(0.002)) if unsafe else loss0
    potential = np.column_stack((loss0, loss1))
    sample = _logged(x, potential, -potential, np.full(n, 0.5), 0, rng)
    return sample, oracle_smoke_family((0.15, 0.25, 0.35, 0.5, 0.75, 1))
