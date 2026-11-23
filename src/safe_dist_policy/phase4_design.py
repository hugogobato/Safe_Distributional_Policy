"""Preregistered Phase 4 population laws, configuration grid and sampling."""

from dataclasses import dataclass
from itertools import product
from math import ceil

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, ndtr, ndtri, roots_legendre

from .certification import PolicyFamily, PolicyScores
from .data import EvaluationSample, ExperimentSplit, LoggedSample


@dataclass(frozen=True)
class Design:
    dgp: int
    label: str = "main"
    n_cal: int = 5000
    p: int = 3
    beta: float = 0.90
    safe_mass: float = 0.35
    kappa: float = 0.25
    cap_factor: float = 0.0

    @property
    def name(self):
        return f"d{self.dgp:02d}-{self.label}"

    @property
    def seeds(self):
        return range(1000, 1003 if self.dgp == 10 else 1050)


CORE = (
    Design(1, safe_mass=0.50),
    Design(2),
    Design(3, "equality"),
    Design(3, "strict"),
    Design(4),
    Design(5, safe_mass=0.30),
)
STRESS = (
    tuple(
        Design(6, f"n{n}-s{int(s * 100):03d}-q{int(b * 100)}", n_cal=n, beta=b, safe_mass=s)
        for n, s, b in product((1000, 2500, 5000, 10000), (0.10, 0.25, 0.50, 0.80), (0.80, 0.90, 0.95))
    )
    + tuple(Design(7, f"k{int(k * 100):02d}", kappa=k) for k in (0.05, 0.10, 0.25))
    + tuple(Design(8, f"cap{c}", cap_factor=c) for c in (1, 2, 5))
    + (
        Design(9, "meps-proxy", p=7, safe_mass=0.30, kappa=0.056),
        Design(9, "ohie-proxy", p=7, safe_mass=0.30, kappa=0.28),
    )
    + tuple(Design(10, f"n{n}-p{p}", n_cal=n, p=p) for n, p in product((1000, 5000, 20000, 100000), (10, 50)))
)
DESIGNS = {design.name: design for design in (*CORE, *STRESS)}
ZIC_DESIGNS = ("d05-main", "d09-meps-proxy", "d09-ohie-proxy")


@dataclass(frozen=True)
class Arms:
    """Two action-specific hurdle and severity parameters for every X row."""

    positive: np.ndarray
    location: np.ndarray
    shape: np.ndarray
    pareto: bool = False

    def survival(self, cutoff: float):
        if cutoff <= 0:
            return self.positive
        if self.pareto:
            return self.positive * np.minimum(1.0, self.location / cutoff) ** self.shape
        return self.positive * ndtr((self.location - np.log(cutoff)) / self.shape)

    def loss_mean(self):
        if self.pareto:
            return self.positive * self.location * self.shape / (self.shape - 1)
        return self.positive * np.exp(self.location + 0.5 * self.shape**2)

    def sample(self, rng):
        active = rng.random((len(self.positive), 1)) < self.positive
        if self.pareto:
            severity = self.location * np.exp(rng.exponential(size=(len(active), 1)) / self.shape)
        else:
            severity = np.exp(self.location + self.shape * rng.normal(size=(len(active), 1)))
        return active * severity


def covariates(design: Design, n: int, rng):
    x = rng.random((n, design.p))
    if design.dgp == 9:
        x[:, 1] = x[:, 1] > 0.65
        x[:, 2] = np.floor(4 * x[:, 2])
        x[:, 3] = np.exp(2 * x[:, 3])
        x[:, 4] = np.floor(3 * x[:, 4])
        x[:, 5] = x[:, 5] > 0.30
    return x


def _regions(x, boundary, groups):
    p, mu, sigma = np.asarray(groups, dtype=float).T
    index = np.searchsorted(boundary, x, side="right")
    return p[index], mu[index], sigma[index]


def population(design: Design, x: np.ndarray) -> Arms:
    u = x[:, 0]
    p0 = 0.60
    mu0: float | np.ndarray = 1.60
    sigma0 = 0.85
    dgp = design.dgp
    if dgp == 1:
        p0, mu0, sigma0 = 0.60, 1.40, 0.22
        p1, mu1, sigma1 = _regions(u, (0.50,), ((0.26, 0.60, 0.08), (0.70, 2.00, 0.25)))
    elif dgp in (2, 10):
        mu0 = 1.60 + 0.40 * (u - 0.50)
        p1, mu1, sigma1 = 0.48 + 0.24 * u, mu0 - 0.85 + 1.40 * u, 0.53 + 0.60 * u
    elif dgp == 3:
        p1, mu1, sigma1 = (p0, mu0, sigma0) if design.label == "equality" else (0.601, 1.602, 0.85)
    elif dgp == 4:
        sigma0 = 0.65
        p1, mu1, sigma1 = p0, mu0 - 0.70 + 1.00 * u, sigma0
    elif dgp == 5:
        sigma0 = 0.65
        p1, mu1, sigma1 = _regions(
            u, (0.30, 0.60), ((0.48, 1.65, 0.20), (0.18, 2.50, 0.15), (0.67, 1.82, 0.75)),
        )
    elif dgp == 6:
        p1, mu1, sigma1 = _regions(
            u, (design.safe_mass,), ((0.45, 1.00, 0.70), (0.65, 2.15, 1.00)),
        )
    elif dgp == 7:
        mu0 = 0.80 + 1.25 * (x[:, 1] >= 0.5)
        sigma0 = 0.65
        p1, mu1, sigma1 = 0.601, mu0 + 0.01, sigma0
    elif dgp == 8:
        mu0, sigma0 = 2.2, 1.8
        p1, mu1, sigma1 = _regions(u, (0.35,), ((0.47, 1.5, 2.2), (0.63, 2.8, 1.45)))
    elif dgp == 9:
        if design.label == "meps-proxy":
            p0 = 0.451
            z90, z99 = ndtri(1 - np.array([0.10, 0.01]) / p0)
            sigma0 = np.log(6.42) / (z99 - z90)
            mu0 = np.log(10.412) - sigma0 * z90
            groups = ((0.34, 1.35, 0.45), (0.17, 2.53, 0.16), (0.54, 1.95, 1.25))
        else:
            p0 = 0.555
            groups = ((0.42, 1.12, 0.65), (0.63, 0.92, 0.75), (0.65, 2.00, 1.05))
        p1, mu1, sigma1 = _regions(u, (0.30, 0.60), groups)
    else:
        raise ValueError(f"Unknown DGP {dgp}.")

    def column_pair(a, b):
        return np.column_stack((np.broadcast_to(a, u.shape), np.broadcast_to(b, u.shape)))

    return Arms(column_pair(p0, p1), column_pair(mu0, mu1), column_pair(sigma0, sigma1), dgp == 8)


def propensity(design: Design, x: np.ndarray):
    u = x[:, 0]
    if design.dgp == 7:
        shape = 0.82 - 0.64 * (x[:, 1] >= 0.5) + 0.08 * (u - 0.5) * (2 * x[:, 2] - 1)
        return design.kappa + (1 - 2 * design.kappa) * shape
    if design.dgp == 9:
        if design.label == "meps-proxy":
            return 0.056 + 0.888 * expit(5 * (u - 0.5) + 0.33 * (x[:, 2] - 1.5) + 0.60 * (x[:, 1] - 0.35))
        return np.array([0.28, 0.42, 0.60, 0.70])[x[:, 2].astype(int)]
    return 0.25 + 0.5 * u


def baseline_quantile(design: Design, beta: float):
    if design.dgp in (2, 10):
        u, weights = roots_legendre(64)
        x = np.column_stack(((u + 1) / 2, np.full((64, 2), 0.5)))
        weights = weights / 2
    elif design.dgp == 7:
        x = np.array([[0.5, 0.25, 0.5], [0.5, 0.75, 0.5]])
        weights = np.array([0.5, 0.5])
    else:
        x = np.full((1, 3), 0.25)
        weights = np.ones(1)

    population_arms = population(design, x)
    target = 1 - beta
    if weights @ population_arms.positive[:, 0] <= target:
        return 0.0
    if design.dgp == 8:
        prob = population_arms.positive[0, 0]
        xmin = population_arms.location[0, 0]
        shape = population_arms.shape[0, 0]
        return float(xmin * (prob / target) ** (1 / shape))
    return float(brentq(lambda q: weights @ population_arms.survival(q)[:, 0] - target, 0.0, 100_000.0))


def simulated_data(design: Design, seed: int, n_eval: int = 100_000):
    n_train = ceil(2 * design.n_cal / 3)
    r_train, r_cal, r_eval = [
        np.random.default_rng(child) for child in np.random.SeedSequence([design.dgp, seed]).spawn(3)
    ]

    def logged(n, offset, rng):
        x = covariates(design, n, rng)
        e = propensity(design, x)
        row = np.arange(n)
        loss = population(design, x).sample(rng)
        observed = (rng.random(n) < e).astype(np.int64)
        outcome = loss[row, observed]
        return LoggedSample(x, observed, outcome, e, row + offset, -outcome)

    train = logged(n_train, 0, r_train)
    calibration = logged(design.n_cal, n_train, r_cal)
    x_eval = covariates(design, n_eval, r_eval)
    conditional = population(design, x_eval)
    loss = conditional.sample(r_eval)
    ids = np.arange(n_eval) + n_train + design.n_cal
    evaluation = EvaluationSample(x_eval, loss, -loss, ids)
    return ExperimentSplit(train, calibration, evaluation, design.kappa), conditional


def oracle_family(design: Design, masses: tuple[float, ...]):
    def scores(x):
        p = population(design, x)
        gain = p.loss_mean()[:, 0] - p.loss_mean()[:, 1]
        action = (gain > 0).astype(np.int64)
        return PolicyScores(action, -x[:, 0])

    boundaries = tuple(-m for m in masses)
    return PolicyFamily(scores, boundaries)


def always_treat_family(masses: tuple[float, ...]):
    def scores(x):
        return PolicyScores(np.ones(len(x), dtype=np.int64), -x[:, 0])

    return PolicyFamily(scores, tuple(-m for m in masses))
