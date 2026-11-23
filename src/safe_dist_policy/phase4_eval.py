"""Holdout truth evaluation and paired Phase 4 method comparisons."""

from dataclasses import replace
from math import inf
from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression

from .bounds import quantile_interval
from .certification import (
    PolicyCertificate,
    RiskSpec,
    certify_indicator,
    certify_localized,
    certify_separate,
    gated_actions,
    offer_from_training,
    policy_weight,
)
from .eval import DEFAULT_MASSES, higher_quantile, sample_cvar
from .learners import (
    MeanCausalForest,
    PlugInCVaRITR,
    TwoPartDistributionalLearner,
    ZICBCFBridge,
)
from .phase4_design import (
    always_treat_family,
    baseline_quantile,
    oracle_family,
    population,
    simulated_data,
)

EARLY_MASSES = (0.05, 0.10, *DEFAULT_MASSES)
B1_ENGINES: tuple[Literal["survival", "opra"], ...] = ("survival", "opra")
# At most 32 distinct method/ladder variants with 16 inspected candidates each.
TRUTH_LIMIT = float(ndtri(1 - 0.05 / (2 * 32 * 16 * 2)))


def _cached_predict(model):
    cache = {}

    def predict(x):
        key = id(x)
        if key not in cache:
            cache[key] = x, model.predict(x)
        return cache[key][1]

    return predict


class HoldoutTruth:
    """Paired conditional scores for Q/occurrence truth and realized welfare."""

    def __init__(self, evaluation, law, design, cap):
        self.data = evaluation
        self.exact_equal = np.array([
            design.dgp == 3 and design.label == "equality",
            (design.dgp == 3 and design.label == "equality") or design.dgp == 4,
        ])
        self.uniform_harm = design.dgp == 7 or (design.dgp == 3 and design.label == "strict")
        eta = baseline_quantile(design, design.beta)
        at_eta = law.survival(eta)
        mean = law.loss_mean()
        self.contrasts = np.column_stack((
            at_eta[:, 1] - at_eta[:, 0],
            law.positive[:, 1] - law.positive[:, 0],
        ))
        self.utility_gain = mean[:, 0] - mean[:, 1]
        self.middle = design.dgp == 5 or design.dgp == 9
        self.cap, self.beta = cap, design.beta
        self.q0 = higher_quantile(evaluation.potential_loss[:, 0], self.beta)
        self.cv0 = sample_cvar(evaluation.potential_loss[:, 0], self.beta, cap)
        self.cache = {}
        self.pareto_groups = None
        if design.dgp == 8:
            arms = population(design, np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]))
            self.pareto_groups = tuple(
                np.array([matrix[0, 0], matrix[0, 1], matrix[1, 1]])
                for matrix in (arms.positive, arms.location, arms.shape)
            )
            self.safe_pareto_region = evaluation.x[:, 0] < 0.35
            self.cvar_model0 = self._pareto_cvar(np.array([1.0, 0.0, 0.0]))

    def _pareto_cvar(self, weights):
        if self.pareto_groups is None:
            return None
        probability, minimum, exponent = self.pareto_groups
        target = 1 - self.beta

        def survival(t):
            return weights @ (probability * np.minimum(1.0, minimum / t) ** exponent)

        if survival(self.cap) >= target:
            return self.cap
        eta = (
            float(brentq(lambda t: survival(t) - target, 1e-12, self.cap))
            if weights @ probability > target else 0.0
        )
        mass_before_tail = np.maximum(0.0, np.minimum(minimum, self.cap) - eta)
        edge = np.minimum(self.cap, np.maximum(eta, minimum))
        tail = minimum**exponent * (edge**(1 - exponent) - self.cap**(1 - exponent)) / (exponent - 1)
        integral = weights @ (probability * (mass_before_tail + tail))
        return float(eta + integral / target)

    def _conditional_cvar_change(self, switched):
        if self.pareto_groups is None:
            return None
        safe_switch = float(np.mean(switched & self.safe_pareto_region))
        harmful_switch = float(np.mean(switched & ~self.safe_pareto_region))
        shares = np.array([1 - safe_switch - harmful_switch, safe_switch, harmful_switch])
        return self._pareto_cvar(shares) - self.cvar_model0

    def _components(self, switched):
        if not switched.any():
            # A nonfallback fitted rule could have a switch region missed by
            # finite evaluation. This remains possible unless equality is structural.
            return np.zeros(2), np.zeros(2, bool), ~self.exact_equal, self._conditional_cvar_change(switched)
        values = self.contrasts * switched[:, None]
        estimate = np.mean(values, axis=0)
        se = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
        definite = (estimate - TRUTH_LIMIT * se) > 1e-12
        possible = (estimate + TRUTH_LIMIT * se) > 1e-12
        if self.uniform_harm:
            # DGP 3 strict and DGP 7 increase both the positive probability
            # and severity at every X, so every nonempty switch is harmful.
            definite[:] = True
            possible[:] = True
        definite[self.exact_equal] = False
        possible[self.exact_equal] = False
        return estimate, definite, possible, self._conditional_cvar_change(switched)

    def evaluate(self, family, cert):
        if cert.threshold == inf:
            return {
                "deployment": 0.0, "welfare_gain": 0.0, "safe_welfare": 0.0,
                "safe_welfare_lower": 0.0, "delta_quantile": 0.0, "delta_cvar": 0.0,
                "delta_extensive": 0.0, "tail_survival_delta": 0.0,
                "selected_unsafe": False, "selected_possible": False,
                "q_unsafe": False, "q_definite": False, "q_possible": False,
                "ext_unsafe": False, "ext_definite": False, "ext_possible": False,
                "prefix_unsafe": False, "prefix_definite": False, "prefix_possible": False,
                "trap_switch_fraction": 0.0, "heldout_switch_observed": False,
                "cvar_conditional_delta": 0.0 if self.pareto_groups is not None else None,
                "cvar_conditional_prefix_unsafe": False if self.pareto_groups is not None else None,
            }
        key = id(family)
        if key not in self.cache:
            self.cache[key] = family, family.predict(self.data.x), {}
        _, scores, records = self.cache[key]

        def components(t):
            if t not in records:
                switches = gated_actions(scores, t, 0).astype(bool)
                records[t] = self._components(switches)
            return records[t]

        estimate, definite, possible, cvar_change = components(cert.threshold)
        visited = [components(t) for t in (cert.passed or (cert.threshold,))]
        selected = gated_actions(scores, cert.threshold, 0).astype(bool)
        loss = np.where(selected, self.data.potential_loss[:, 1], self.data.potential_loss[:, 0])
        gain = float(np.mean(selected * self.utility_gain))
        unsafe = bool(np.any(estimate > 1e-12))
        possible_unsafe = bool(np.any(possible))
        trap = (self.data.x[:, 0] >= 0.30) & (self.data.x[:, 0] < 0.60)

        return {
            "deployment": float(np.mean(selected)), "welfare_gain": gain,
            "safe_welfare": 0.0 if unsafe else gain,
            "safe_welfare_lower": 0.0 if possible_unsafe else gain,
            "delta_quantile": higher_quantile(loss, self.beta) - self.q0,
            "delta_cvar": sample_cvar(loss, self.beta, self.cap) - self.cv0,
            "delta_extensive": float(estimate[1]), "tail_survival_delta": float(estimate[0]),
            "selected_unsafe": unsafe, "selected_possible": possible_unsafe,
            "q_unsafe": bool(estimate[0] > 1e-12), "q_definite": bool(definite[0]),
            "q_possible": bool(possible[0]),
            "ext_unsafe": bool(estimate[1] > 1e-12), "ext_definite": bool(definite[1]),
            "ext_possible": bool(possible[1]),
            "prefix_unsafe": any(np.any(r[0] > 1e-12) for r in visited),
            "prefix_definite": any(np.any(r[1]) for r in visited),
            "prefix_possible": any(np.any(r[2]) for r in visited),
            "trap_switch_fraction": float(np.mean(selected & trap)) if self.middle else 0.0,
            "heldout_switch_observed": bool(selected.any()),
            "cvar_conditional_delta": cvar_change,
            "cvar_conditional_prefix_unsafe": (
                None if cvar_change is None else any(r[3] > 1e-12 for r in visited)
            ),
        }


def run_replication(design, seed: int, study: str, *, reduced_zic: bool = False):
    started = perf_counter()
    data, law = simulated_data(design, seed)
    train, cal = data.train, data.calibration
    cap = baseline_quantile(design, 0.99) * design.cap_factor if design.cap_factor else 50.0
    spec = RiskSpec(beta=design.beta, cvar_cap=cap)
    q_spec = replace(spec, extensive_tolerance=None)
    ext_spec = replace(spec, quantile_tolerance=None)
    truth = HoldoutTruth(data.evaluation, law, design, cap)
    rows = []

    def collect(label, family, operation):
        time0 = perf_counter()
        cert = operation()
        duration = perf_counter() - time0
        time0 = perf_counter()
        row = truth.evaluate(family, cert)
        threshold = cert.threshold if np.isfinite(cert.threshold) else None
        rows.append({
            "method": label, "guarantee": cert.guarantee, "threshold": threshold,
            "policy_selected": cert.deployed, "tested": cert.tested, "prefix_size": len(cert.passed),
            "cert_seconds": duration, "evaluation_seconds": perf_counter() - time0, **row,
        })
        return cert

    def local(family, option=spec, sample=cal, gamma=1.0):
        return certify_localized(sample, family, option, data.overlap, gamma=gamma)

    def fit_family(learner):
        fit_time = perf_counter()
        learner.fit(train)
        predict = _cached_predict(learner)
        family = offer_from_training(predict, train, DEFAULT_MASSES)
        early = offer_from_training(predict, train, EARLY_MASSES)
        return family, early, perf_counter() - fit_time

    dist = TwoPartDistributionalLearner(design.beta, seed)
    dist_family, dist_early, fit_dist = fit_family(dist)
    localized = collect("LCT-LTT", dist_family, lambda: local(dist_family))
    metrics = {"fit_two_part_seconds": fit_dist}
    if study == "zic":
        burn, draws, trees = (10, 12, 12) if reduced_zic else (500, 1000, 200)
        zic = ZICBCFBridge(design.beta, seed, burn=burn, draws=draws, trees=trees)
        zic_family, _, metrics["fit_zic_seconds"] = fit_family(zic)
        collect("LCT-ZIC-proj", zic_family, lambda: local(zic_family))
    else:
        mean = MeanCausalForest(seed, timeout=3600.0 if design.dgp == 10 else 300.0)
        mean_family, mean_early, metrics["fit_mean_seconds"] = fit_family(mean)
        time0 = perf_counter()
        plugin = PlugInCVaRITR(dist, cap, design.beta).fit(train)
        plugin_family = replace(dist_family, predict=_cached_predict(plugin), thresholds=(-inf,))
        metrics["fit_plugin_seconds"] = perf_counter() - time0

        oracle = oracle_family(design, DEFAULT_MASSES)
        def select_oracle():
            candidates = [PolicyCertificate("Oracle", t, (), 0, "oracle", None) for t in (inf, *oracle.thresholds)]
            values = [truth.evaluate(oracle, c) for c in candidates]
            scores = [v["safe_welfare"] if not v["selected_unsafe"] else -inf for v in values]
            return candidates[int(np.argmax(scores))]

        collect("Oracle", oracle, select_oracle)
        for engine in B1_ENGINES:
            collect(
                f"B1-{engine}", dist_family,
                lambda: certify_separate(cal, dist_family, spec, data.overlap, engine=engine),
            )
        collect("B2", mean_family, lambda: local(mean_family))
        collect("B3", plugin_family, lambda: PolicyCertificate("B3", -inf, (), 0, "none", None))
        collect("B4", mean_family, lambda: PolicyCertificate("B4", -inf, (), 0, "none", None))

        def indicator(family, matched=False):
            cert = certify_indicator(cal, family, dist.cutoff, spec, data.overlap)
            if not matched:
                return cert
            ext = local(family, ext_spec)
            prefix = cert.passed[:len(ext.passed)]
            return PolicyCertificate(
                "B5+ext", prefix[-1] if prefix else inf, prefix, min(cert.tested, ext.tested), "indicator", spec.delta,
            )

        collect("B5", dist_family, lambda: indicator(dist_family))
        matched = collect("B5+ext", dist_family, lambda: indicator(dist_family, True))
        fixed = policy_weight(cal, np.zeros(cal.n, dtype=np.int64))
        lo, hi = quantile_interval(cal.loss, fixed, spec.beta, spec.delta * spec.var_budget_fraction,
                                   1 / data.overlap, spec.bound)
        inside = lo <= dist.cutoff <= hi
        metrics["b5_training_cutoff"] = dist.cutoff
        metrics["b5_cutoff_in_lct_interval"] = inside
        metrics["b5_passes_all_localized_prefix"] = len(matched.passed) >= len(localized.passed)
        if inside and not metrics["b5_passes_all_localized_prefix"]:
            raise AssertionError("A fixed-cutoff test inside the VaR interval missed a localized certificate.")
        collect("LCT-LTT-eb", dist_family, lambda: local(dist_family, replace(spec, bound="eb")))
        collect("LCT-LTT-5pct", dist_early, lambda: local(dist_early))
        collect("B2-5pct", mean_early, lambda: local(mean_early))
        collect("B5+ext-5pct", dist_early, lambda: indicator(dist_early, True))
        if design.dgp == 4:
            collect("LCT-LTT-Qonly", dist_family, lambda: local(dist_family, q_spec))
            collect("B2-Qonly", mean_family, lambda: local(mean_family, q_spec))
            for engine in B1_ENGINES:
                collect(
                    f"B1-{engine}-Qonly", dist_family,
                    lambda: certify_separate(cal, dist_family, q_spec, data.overlap, engine=engine),
                )
        if design.dgp in (3, 7):
            always = always_treat_family(DEFAULT_MASSES)
            collect("LCT-LTT-fixed-ladder", always, lambda: local(always))
        if design.dgp == 7:
            time0 = perf_counter()
            misspec = LogisticRegression().fit(train.x[:, :1], train.action)
            prop = misspec.predict_proba(cal.x[:, :1])[:, 1]
            estimated = replace(cal, propensity=np.clip(prop, design.kappa, 1 - design.kappa))
            gamma = (1 - design.kappa) / design.kappa
            metrics["fit_propensity_seconds"] = perf_counter() - time0
            for label, family in (("", dist_family), ("-fixed-ladder", always)):
                collect(f"LCT-LTT-misspec{label}", family, lambda: local(family, sample=estimated))
                collect(f"LCT-LTT-gamma{label}", family, lambda: local(family, sample=estimated, gamma=gamma))
        if design.dgp == 8:
            cvar = replace(spec, quantile_tolerance=None, extensive_tolerance=None, cvar_tolerance=0.0)
            collect("LCT-LTT-Qonly", dist_family, lambda: local(dist_family, q_spec))
            collect("B2-Qonly", mean_family, lambda: local(mean_family, q_spec))
            collect("LCT-LTT-CVaR", dist_family, lambda: local(dist_family, cvar))
            collect(
                "B1-opra-CVaR", dist_family,
                lambda: certify_separate(cal, dist_family, cvar, data.overlap, engine="opra"),
            )
    metrics.update({
        "seconds": perf_counter() - started,
        "peak_process_rss_mib": getrusage(RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_child_rss_mib": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024.0,
        "n_train": train.n, "n_cal": cal.n, "n_eval": len(truth.data.x), "cap": cap,
    })
    return rows, metrics
