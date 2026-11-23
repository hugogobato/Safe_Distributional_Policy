"""Phase 4 Protocol v2 designs and paired confirmatory evaluation for Path A."""

import zlib
from dataclasses import dataclass, replace
from math import inf
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, ndtr, ndtri

from .certification import (
    PolicyCertificate,
    PolicyFamily,
    RiskSpec,
    certify_indicator,
    certify_localized,
    certify_separate,
    gated_actions,
    offer_from_training,
    propensity_gated_family,
)
from .data import EvaluationSample, ExperimentSplit, LoggedSample
from .eval import DEFAULT_MASSES, higher_quantile, sample_cvar
from .learners import MeanCausalForest, PlugInCVaRITR, TwoPartDistributionalLearner


@dataclass(frozen=True)
class V2Design:
    name: str
    n_train: int = 3334
    n_cal: int = 5000
    n_eval: int = 80_000
    p: int = 3
    beta: float = 0.90
    kappa: float = 0.25
    cvar_cap: float = 50.0


V2_DESIGNS: dict[str, V2Design] = {
    d.name: d
    for d in (
        V2Design("v2-d05-conflict"),
        V2Design("v2-cvar-trap", cvar_cap=50.0),
        V2Design("v2-cvar-power", cvar_cap=50.0),
        V2Design("v2-survival-crossing", n_train=1000, n_cal=5000),
        V2Design("v2-d09-meps", p=7, kappa=0.056),
        V2Design("v2-d09-ohie", p=7, kappa=0.28),
        V2Design("v2-null-strict"),
    )
}


def _v2_covariates(design: V2Design, n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.random((n, design.p))
    if design.name in ("v2-d09-meps", "v2-d09-ohie"):
        x[:, 1] = x[:, 1] > 0.65
        x[:, 2] = np.floor(4 * x[:, 2])
        x[:, 3] = np.exp(2 * x[:, 3])
        x[:, 4] = np.floor(3 * x[:, 4])
        x[:, 5] = x[:, 5] > 0.30
    return x


def v2_propensity(design: V2Design, x: np.ndarray) -> np.ndarray:
    u = x[:, 0]
    if design.name == "v2-d09-meps":
        return 0.056 + 0.888 * expit(5 * (u - 0.5) + 0.33 * (x[:, 2] - 1.5) + 0.60 * (x[:, 1] - 0.35))
    if design.name == "v2-d09-ohie":
        return np.array([0.28, 0.42, 0.60, 0.70])[x[:, 2].astype(int)]
    return 0.25 + 0.50 * u


def _regions(u: np.ndarray, boundaries: tuple[float, ...], groups: tuple[tuple[float, float, float], ...]):
    p, mu, sigma = np.asarray(groups, dtype=float).T
    idx = np.searchsorted(boundaries, u, side="right")
    return p[idx], mu[idx], sigma[idx]


def _sample_potential_losses(design: V2Design, x: np.ndarray, rng: np.random.Generator):
    """Return potential_loss (n, 2), conditional_mean (n, 2), and conditional_p_pos (n, 2)."""
    u = x[:, 0]
    n = len(u)

    if design.name == "v2-cvar-trap":
        # Same baseline as v2-cvar-power (eta0=11.27, mean0=4.273, CVaR_0.90=19.3) so R1 is well-powered.
        # R1 (u < 0.35): eliminates >99.9% of baseline tail above eta0 (dCV ~ -2.85, gain = +0.92).
        # R2 (u >= 0.35): 93.5% zero loss, 6.5% catastrophic loss at 49.0 -> larger mean gain (+1.09 > +0.92),
        # strictly improves Q_0.90 (S1=0.065 < 0.10) and ext (0.065 < 0.60), yet explodes CVaR_0.90 (+5.6 on R1+R2)!
        p0, mu0, sig0 = 0.60, 1.60, 0.85
        act0 = rng.random(n) < p0
        l0 = act0 * rng.lognormal(mu0, sig0, n)
        mean0 = np.full(n, p0 * np.exp(mu0 + 0.5 * sig0**2))
        pos0 = np.full(n, p0)

        safe = u < 0.35
        act1_safe = rng.random(n) < 0.52
        l1_safe = act1_safe * rng.lognormal(1.85, 0.15, n)
        mean1_safe = 0.52 * np.exp(1.85 + 0.5 * 0.15**2)
        catastrophic = rng.random(n) < 0.065
        l1_trap = np.where(catastrophic, 49.0, 0.0)
        mean1_trap = 0.065 * 49.0

        l1 = np.where(safe, l1_safe, l1_trap)
        mean1 = np.where(safe, mean1_safe, mean1_trap)
        pos1 = np.where(safe, 0.52, 0.065)
        return np.column_stack((l0, l1)), np.column_stack((mean0, mean1)), np.column_stack((pos0, pos1))

    if design.name == "v2-d05-conflict":
        p0, mu0, sig0 = 0.60, 1.60, 0.65
        # eta0=9.29, mean0=3.67. R1: gain=+1.06, dS=-0.099. R2: gain=+1.23 > R1, S1=0.228 -> R1+R2 dS=+0.015 > 0!
        p1, mu1, sig1 = _regions(
            u, (0.35, 0.70), ((0.46, 1.72, 0.18), (0.23, 2.36, 0.08), (0.70, 1.90, 0.75)),
        )
    elif design.name == "v2-cvar-power":
        p0, mu0, sig0 = 0.60, 1.60, 0.85
        p1, mu1, sig1 = _regions(
            u, (0.35,), ((0.40, 0.90, 0.60), (0.65, 2.15, 1.00)),
        )
    elif design.name == "v2-survival-crossing":
        p0, mu0, sig0 = 0.60, 1.60, 0.85
        # eta0=11.27, mean0=4.27. Treatment has strong mean gain (+1.72) and ext reduction (-0.38),
        # is UNSAFE on Q_0.90 (S1(11.27) ~ 0.150 > 0.10, Q_0.90=11.65 > 11.27), yet S1(t) << S0(t) for t >= 12.2!
        p1 = np.full(n, 0.22)
        mu1 = np.full(n, 2.45)
        sig1 = np.full(n, 0.06)
    elif design.name == "v2-d09-meps":
        p0 = 0.451
        z90, z99 = ndtri(1 - np.array([0.10, 0.01]) / p0)
        sig0 = float(np.log(6.42) / (z99 - z90))
        mu0 = float(np.log(10.412) - sig0 * z90)
        # eta0=10.412 (VaR CI ~ [7.7, 17.7]). Keep R1 well below 7.7 (exp(1.35)=3.86) so R1 never crosses S0!
        # R2 (0.35 <= u < 0.75, mass 0.40): large positive mean gain, ext safe, S1(10.41)=0.22 -> R1+R2 dS > 0!
        p1, mu1, sig1 = _regions(
            u, (0.35, 0.75), ((0.30, 1.35, 0.40), (0.23, 2.52, 0.12), (0.56, 2.00, 1.25)),
        )
    elif design.name == "v2-d09-ohie":
        p0, mu0, sig0 = 0.555, 1.60, 0.85
        # eta0=10.78 (VaR CI ~ [9.3, 13.3]). Keep R1 well below 9.3 (exp(1.25)=3.49) and clearly separated from R2!
        # R2 (0.35 <= u < 0.75, mass 0.40): strong positive mean gain, ext safe, S1(10.78)=0.23 -> R1+R2 dS > 0!
        p1, mu1, sig1 = _regions(
            u, (0.35, 0.75), ((0.38, 1.25, 0.45), (0.24, 2.55, 0.12), (0.66, 2.05, 1.05)),
        )
    elif design.name == "v2-null-strict":
        p0, mu0, sig0 = 0.60, 1.60, 0.85
        p1 = np.full(n, 0.601)
        mu1 = np.full(n, 1.602)
        sig1 = np.full(n, 0.85)
    else:
        raise ValueError(f"Unknown v2 design {design.name}.")

    act0 = rng.random(n) < p0
    act1 = rng.random(n) < p1
    l0 = act0 * rng.lognormal(mu0, sig0, n)
    l1 = act1 * rng.lognormal(mu1, sig1, n)
    mean0 = np.full(n, p0 * np.exp(mu0 + 0.5 * sig0**2))
    mean1 = p1 * np.exp(mu1 + 0.5 * sig1**2)
    pos0 = np.full(n, p0)
    return np.column_stack((l0, l1)), np.column_stack((mean0, mean1)), np.column_stack((pos0, p1))


def v2_baseline_quantile(design: V2Design, beta: float | None = None) -> float:
    b = design.beta if beta is None else beta
    if design.name == "v2-d05-conflict":
        p0, mu0, sig0 = 0.60, 1.60, 0.65
    elif design.name == "v2-d09-meps":
        p0 = 0.451
        z90, z99 = ndtri(1 - np.array([0.10, 0.01]) / p0)
        sig0 = float(np.log(6.42) / (z99 - z90))
        mu0 = float(np.log(10.412) - sig0 * z90)
    elif design.name == "v2-d09-ohie":
        p0, mu0, sig0 = 0.555, 1.60, 0.85
    else:
        p0, mu0, sig0 = 0.60, 1.60, 0.85
    target = 1.0 - b
    if p0 <= target:
        return 0.0
    return float(brentq(lambda q: p0 * ndtr((mu0 - np.log(q)) / sig0) - target, 1e-9, 100_000.0))


def v2_simulated_split(design: V2Design, seed: int) -> tuple[ExperimentSplit, np.ndarray, np.ndarray]:
    key = zlib.crc32(design.name.encode("utf-8")) % 100_000
    r_tr, r_cal, r_ev = [
        np.random.default_rng(c) for c in np.random.SeedSequence([2026, key, seed]).spawn(3)
    ]

    def make_logged(n: int, offset: int, rng: np.random.Generator) -> LoggedSample:
        x = _v2_covariates(design, n, rng)
        e = v2_propensity(design, x)
        loss, _, _ = _sample_potential_losses(design, x, rng)
        act = (rng.random(n) < e).astype(np.int64)
        obs = loss[np.arange(n), act]
        return LoggedSample(x, act, obs, e, np.arange(n) + offset, -obs)

    train = make_logged(design.n_train, 0, r_tr)
    cal = make_logged(design.n_cal, design.n_train, r_cal)
    x_ev = _v2_covariates(design, design.n_eval, r_ev)
    loss_ev, mean_ev, pos_ev = _sample_potential_losses(design, x_ev, r_ev)
    ev = EvaluationSample(x_ev, loss_ev, -loss_ev, np.arange(design.n_eval) + design.n_train + design.n_cal)
    return ExperimentSplit(train, cal, ev, design.kappa), mean_ev, pos_ev


def _cached(fn):
    cache = {}

    def wrapped(x):
        k = id(x)
        if k not in cache:
            cache[k] = x, fn(x)
        return cache[k][1]

    return wrapped


def _exact_survival_contrast(design: V2Design, x: np.ndarray, eta0: float) -> np.ndarray:
    """Analytical conditional S_1(eta0 | x) - S_0(eta0 | x) for low-variance holdout truth."""
    u = x[:, 0]
    if design.name == "v2-cvar-trap":
        s0 = 0.60 * ndtr((1.60 - np.log(eta0)) / 0.85)
        s1 = np.where(u < 0.35, 0.52 * ndtr((1.85 - np.log(eta0)) / 0.15), 0.065)
        return s1 - s0
    if design.name == "v2-d05-conflict":
        s0 = 0.60 * ndtr((1.60 - np.log(eta0)) / 0.65)
        p1, mu1, sig1 = _regions(u, (0.35, 0.70), ((0.46, 1.72, 0.18), (0.23, 2.36, 0.08), (0.70, 1.90, 0.75)))
        return p1 * ndtr((mu1 - np.log(eta0)) / sig1) - s0
    if design.name == "v2-cvar-power":
        s0 = 0.60 * ndtr((1.60 - np.log(eta0)) / 0.85)
        p1, mu1, sig1 = _regions(u, (0.35,), ((0.40, 0.90, 0.60), (0.65, 2.15, 1.00)))
        return p1 * ndtr((mu1 - np.log(eta0)) / sig1) - s0
    if design.name == "v2-survival-crossing":
        s0 = 0.60 * ndtr((1.60 - np.log(eta0)) / 0.85)
        s1 = 0.22 * ndtr((2.45 - np.log(eta0)) / 0.06)
        return np.full(len(u), s1 - s0)
    if design.name == "v2-d09-meps":
        p0 = 0.451
        z90, z99 = ndtri(1 - np.array([0.10, 0.01]) / p0)
        sig0 = float(np.log(6.42) / (z99 - z90))
        mu0 = float(np.log(10.412) - sig0 * z90)
        s0 = p0 * ndtr((mu0 - np.log(eta0)) / sig0)
        p1, mu1, sig1 = _regions(u, (0.35, 0.75), ((0.30, 1.35, 0.40), (0.23, 2.52, 0.12), (0.56, 2.00, 1.25)))
        return p1 * ndtr((mu1 - np.log(eta0)) / sig1) - s0
    if design.name == "v2-d09-ohie":
        s0 = 0.555 * ndtr((1.60 - np.log(eta0)) / 0.85)
        p1, mu1, sig1 = _regions(u, (0.35, 0.75), ((0.38, 1.25, 0.45), (0.24, 2.55, 0.12), (0.66, 2.05, 1.05)))
        return p1 * ndtr((mu1 - np.log(eta0)) / sig1) - s0
    if design.name == "v2-null-strict":
        s0 = 0.60 * ndtr((1.60 - np.log(eta0)) / 0.85)
        s1 = 0.601 * ndtr((1.602 - np.log(eta0)) / 0.85)
        return np.full(len(u), s1 - s0)
    raise ValueError(design.name)


def _evaluate_cert(
    design: V2Design,
    ev: EvaluationSample,
    mean_ev: np.ndarray,
    pos_ev: np.ndarray,
    surv_diff: np.ndarray,
    family: PolicyFamily,
    cert: PolicyCertificate,
    spec: RiskSpec,
    *,
    q0: float,
    cv0: float,
) -> dict[str, object]:
    if not cert.deployed:
        return {
            "deployment": 0.0,
            "welfare_gain": 0.0,
            "safe_welfare": 0.0,
            "delta_quantile": 0.0,
            "delta_cvar": 0.0,
            "delta_extensive": 0.0,
            "tail_survival_delta": 0.0,
            "selected_unsafe": False,
            "prefix_unsafe": False,
        }
    scores = family.predict(ev.x)
    utility_gain = mean_ev[:, 0] - mean_ev[:, 1]
    ext_diff = pos_ev[:, 1] - pos_ev[:, 0]
    uniform_harm = design.name == "v2-null-strict"

    def check_threshold(thresh: float):
        sw = gated_actions(scores, thresh, 0).astype(bool)
        if not sw.any():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        loss = np.where(sw, ev.potential_loss[:, 1], ev.potential_loss[:, 0])
        dq = higher_quantile(loss, spec.beta) - q0
        dc = sample_cvar(loss, spec.beta, spec.cvar_cap) - cv0
        ds = float(np.mean(sw * surv_diff))
        de = float(np.mean(sw * ext_diff))
        gain = float(np.mean(sw * utility_gain))
        unsafe = False
        if spec.quantile_tolerance is not None and (uniform_harm or ds > 1e-12):
            unsafe = True
        if spec.extensive_tolerance is not None and (uniform_harm or de > spec.extensive_tolerance + 1e-12):
            unsafe = True
        if spec.cvar_tolerance is not None and (uniform_harm or dc > spec.cvar_tolerance + 1e-9):
            unsafe = True
        return float(np.mean(sw)), gain, dq, dc, ds, de, unsafe

    dep, gain, dq, dc, ds, de, unsafe = check_threshold(cert.threshold)
    prefix_unsafe = unsafe
    if cert.passed and not prefix_unsafe:
        for t in cert.passed:
            if t != cert.threshold and check_threshold(t)[-1]:
                prefix_unsafe = True
                break
    return {
        "deployment": dep,
        "welfare_gain": gain,
        "safe_welfare": 0.0 if unsafe else gain,
        "delta_quantile": dq,
        "delta_cvar": dc,
        "delta_extensive": de,
        "tail_survival_delta": ds,
        "selected_unsafe": unsafe,
        "prefix_unsafe": prefix_unsafe,
    }


def run_v2_replication(design_name: str, seed: int) -> list[dict[str, object]]:
    """Run one paired seed for a Phase 4 Protocol v2 design."""
    design = V2_DESIGNS[design_name]
    split, mean_ev, pos_ev = v2_simulated_split(design, seed)
    train, cal, ev = split.train, split.calibration, split.evaluation
    eta0 = v2_baseline_quantile(design)
    surv_diff = _exact_survival_contrast(design, ev.x, eta0)
    q0 = higher_quantile(ev.potential_loss[:, 0], design.beta)
    cv0_50 = sample_cvar(ev.potential_loss[:, 0], design.beta, design.cvar_cap)

    # Fit shared learners ONCE on D_tr
    dist = TwoPartDistributionalLearner(design.beta, seed).fit(train)
    mean_cf = MeanCausalForest(seed, trees=100).fit(train)
    mean_pred = _cached(mean_cf.predict)
    mean_family = offer_from_training(mean_pred, train, DEFAULT_MASSES)
    ext_ctrl = dist.extensive_control_variate

    rows: list[dict[str, object]] = []

    def record(target: str, method: str, family: PolicyFamily, spec: RiskSpec, cert: PolicyCertificate, secs: float, cv0: float):
        ev_res = _evaluate_cert(design, ev, mean_ev, pos_ev, surv_diff, family, cert, spec, q0=q0, cv0=cv0)
        rows.append({
            "design": design.name,
            "seed": seed,
            "target": target,
            "method": method,
            "guarantee": cert.guarantee,
            "deployed": cert.deployed,
            "tested": cert.tested,
            "prefix_size": len(cert.passed),
            "cert_seconds": secs,
            **ev_res,
        })

    def run_method(target: str, method: str, family: PolicyFamily, spec: RiskSpec, fn, cv0: float):
        t0 = perf_counter()
        cert = fn()
        record(target, method, family, spec, cert, perf_counter() - t0, cv0)

    def eval_quantile_suite(target_name: str, joint: bool):
        spec_gro = RiskSpec(
            beta=design.beta, bound="gro", quantile_tolerance=0.0,
            extensive_tolerance=0.0 if joint else None,
        )
        spec_v1 = replace(spec_gro, bound="betting")
        pred_fn = _cached(dist.scorer(spec_gro))
        fam = offer_from_training(pred_fn, train, DEFAULT_MASSES)
        oof_sc = dist.oof_predict_for(train, spec_gro)
        ctrl = ext_ctrl if joint else None

        run_method(
            target_name, "LCT-LTT-GRO", fam, spec_gro,
            lambda: certify_localized(
                cal, fam, spec_gro, split.overlap, train=train, train_scores=oof_sc,
                ext_control=ctrl, anchored=True, method="LCT-LTT-GRO",
            ),
            cv0_50,
        )
        run_method(
            target_name, "LCT-LTT-aGRAPA", fam, spec_v1,
            lambda: certify_localized(cal, fam, spec_v1, split.overlap, method="LCT-LTT-aGRAPA"),
            cv0_50,
        )
        run_method(
            target_name, "B1-survival", fam, spec_v1,
            lambda: certify_separate(cal, fam, spec_v1, split.overlap, engine="survival"),
            cv0_50,
        )
        run_method(
            target_name, "B1-opra", fam, spec_v1,
            lambda: certify_separate(cal, fam, spec_v1, split.overlap, engine="opra"),
            cv0_50,
        )
        run_method(
            target_name, "B2-GRO", mean_family, spec_gro,
            lambda: certify_localized(cal, mean_family, spec_gro, split.overlap, train=train, ext_control=ctrl, anchored=True, method="B2-GRO"),
            cv0_50,
        )
        run_method(
            target_name, "B4", mean_family, spec_gro,
            lambda: PolicyCertificate("B4", -inf, (), 0, "none", None),
            cv0_50,
        )

        # Matched fixed-cutoff B5 (and B5+ext when joint) using same GRO engine
        def run_b5():
            c_ind = certify_indicator(cal, fam, dist.cutoff, spec_gro, split.overlap, train=train)
            if not joint:
                return c_ind
            ext_spec = RiskSpec(beta=design.beta, bound="gro", quantile_tolerance=None, extensive_tolerance=0.0)
            c_ext = certify_localized(cal, fam, ext_spec, split.overlap, train=train, ext_control=ext_ctrl, anchored=False)
            prefix = c_ind.passed[:len(c_ext.passed)]
            return PolicyCertificate(
                "B5+ext-GRO", min(prefix) if prefix else inf, prefix, min(c_ind.tested, c_ext.tested), "indicator", spec_gro.delta,
            )

        run_method(target_name, "B5+ext-GRO" if joint else "B5-GRO", fam, spec_gro, run_b5, cv0_50)

        if design.name == "v2-d09-meps":
            gated_fam, bounds_fn = propensity_gated_family(
                pred_fn, train, DEFAULT_MASSES, lambda x: v2_propensity(design, x), split.overlap,
                floor_quantile=0.05, reference_mass=0.35,
            )
            run_method(
                target_name, "LCT-LTT-GRO-Gate", gated_fam, spec_gro,
                lambda: certify_localized(
                    cal, gated_fam, spec_gro, split.overlap, train=train, ext_control=ctrl,
                    anchored=True, localized_bounds=bounds_fn, method="LCT-LTT-GRO-Gate",
                ),
                cv0_50,
            )

    def eval_cvar_suite(target_name: str, cap: float, *, include_b5_and_b3: bool = True):
        cv0 = sample_cvar(ev.potential_loss[:, 0], design.beta, cap)
        spec_gro = RiskSpec(
            beta=design.beta, bound="gro", quantile_tolerance=None, extensive_tolerance=None,
            cvar_tolerance=0.0, cvar_cap=cap, cvar_cells=120,
        )
        spec_v1 = replace(spec_gro, bound="betting")
        pred_fn = _cached(dist.scorer(spec_gro))
        fam = offer_from_training(pred_fn, train, DEFAULT_MASSES)
        oof_sc = dist.oof_predict_for(train, spec_gro)

        run_method(
            target_name, "LCT-LTT-GRO", fam, spec_gro,
            lambda: certify_localized(
                cal, fam, spec_gro, split.overlap, train=train, train_scores=oof_sc,
                anchored=True, method="LCT-LTT-GRO",
            ),
            cv0,
        )
        run_method(
            target_name, "LCT-LTT-aGRAPA", fam, spec_v1,
            lambda: certify_localized(cal, fam, spec_v1, split.overlap, method="LCT-LTT-aGRAPA"),
            cv0,
        )
        run_method(
            target_name, "B1-opra", fam, spec_v1,
            lambda: certify_separate(cal, fam, spec_v1, split.overlap, engine="opra"),
            cv0,
        )
        run_method(
            target_name, "B2-GRO", mean_family, spec_gro,
            lambda: certify_localized(cal, mean_family, spec_gro, split.overlap, train=train, anchored=True, method="B2-GRO"),
            cv0,
        )
        run_method(
            target_name, "B4", mean_family, spec_gro,
            lambda: PolicyCertificate("B4", -inf, (), 0, "none", None),
            cv0,
        )
        if include_b5_and_b3:
            plugin = PlugInCVaRITR(dist, cap, design.beta).fit(train)
            plugin_fam = PolicyFamily(_cached(plugin.predict), (-inf,))
            run_method(
                target_name, "B3", plugin_fam, spec_gro,
                lambda: PolicyCertificate("B3", -inf, (), 0, "none", None),
                cv0,
            )
            q_spec_gro = RiskSpec(beta=design.beta, bound="gro", quantile_tolerance=0.0, extensive_tolerance=None)
            q_fam = offer_from_training(_cached(dist.scorer(q_spec_gro)), train, DEFAULT_MASSES)
            run_method(
                target_name, "B5-GRO", q_fam, spec_gro,
                lambda: certify_indicator(cal, q_fam, dist.cutoff, q_spec_gro, split.overlap, train=train),
                cv0,
            )
            # Also evaluate B5+ext on CVaR truth!
            def run_b5_ext():
                c_ind = certify_indicator(cal, q_fam, dist.cutoff, q_spec_gro, split.overlap, train=train)
                ext_spec = RiskSpec(beta=design.beta, bound="gro", quantile_tolerance=None, extensive_tolerance=0.0)
                c_ext = certify_localized(cal, q_fam, ext_spec, split.overlap, train=train, ext_control=ext_ctrl, anchored=False)
                prefix = c_ind.passed[:len(c_ext.passed)]
                return PolicyCertificate(
                    "B5+ext-GRO", min(prefix) if prefix else inf, prefix, min(c_ind.tested, c_ext.tested), "indicator", q_spec_gro.delta,
                )

            run_method(target_name, "B5+ext-GRO", q_fam, spec_gro, run_b5_ext, cv0)

    if design.name in ("v2-d05-conflict", "v2-d09-meps", "v2-d09-ohie"):
        eval_quantile_suite("Joint-Q-Ext", joint=True)
        eval_quantile_suite("Q-only", joint=False)
    elif design.name == "v2-survival-crossing":
        eval_quantile_suite("Joint-Q-Ext", joint=True)
        eval_quantile_suite("Q-only", joint=False)
    elif design.name == "v2-cvar-trap":
        eval_cvar_suite("CVaR-cap50", 50.0, include_b5_and_b3=True)
    elif design.name == "v2-cvar-power":
        eval_cvar_suite("CVaR-cap50", 50.0, include_b5_and_b3=False)
        eval_cvar_suite("CVaR-cap30", 30.0, include_b5_and_b3=False)
        eval_cvar_suite("CVaR-cap100", 100.0, include_b5_and_b3=False)
    elif design.name == "v2-null-strict":
        eval_quantile_suite("Joint-Q-Ext", joint=True)
        eval_quantile_suite("Q-only", joint=False)
        eval_cvar_suite("CVaR-cap50", 50.0, include_b5_and_b3=False)

    return rows
