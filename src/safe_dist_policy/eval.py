"""End-to-end shared split benchmark and oracle evaluation."""

from dataclasses import dataclass
from math import inf
from time import perf_counter

import numpy as np

from .certification import (
    PolicyCertificate,
    PolicyFamily,
    PolicyScores,
    RiskSpec,
    certify_indicator,
    certify_localized,
    certify_separate,
    gated_actions,
    offer_from_training,
)
from .data import F64, EvaluationSample, ExperimentSplit
from .dgps import oracle_smoke_family
from .learners import MeanCausalForest, PlugInCVaRITR, TwoPartDistributionalLearner

DEFAULT_MASSES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00)


def higher_quantile(values: F64, beta: float) -> float:
    """Right-inverse empirical CDF, including atoms."""
    return float(np.quantile(values, beta, method="inverted_cdf"))


def sample_cvar(loss: F64, beta: float, cap: float) -> float:
    capped = np.minimum(loss, cap)
    eta = higher_quantile(capped, beta)
    return eta + float(np.maximum(capped - eta, 0).mean()) / (1 - beta)


@dataclass(frozen=True)
class PolicyEvaluation:
    deployment: float
    delta_quantile: float
    delta_extensive: float
    delta_cvar: float
    welfare_gain: float
    safe_welfare_gain: float
    selected_unsafe: bool
    unsafe_prefix: bool


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    guarantee: str
    threshold: float
    tested: int
    prefix_length: int
    evaluation: PolicyEvaluation
    certificate_seconds: float


def _risk_vector(loss: F64, spec: RiskSpec) -> F64:
    return np.array([
        higher_quantile(loss, spec.beta), spec.events(loss).mean(),
        sample_cvar(loss, spec.beta, spec.cvar_cap),
    ])


def _is_unsafe(change: F64, spec: RiskSpec) -> bool:
    limits = spec.quantile_tolerance, spec.extensive_tolerance, spec.cvar_tolerance
    return any(tol is not None and val > tol + 1e-12 for val, tol in zip(change, limits))


def evaluate_policy(
    evaluation: EvaluationSample,
    family: PolicyFamily,
    certificate: PolicyCertificate,
    spec: RiskSpec,
    predictions: PolicyScores | None = None,
) -> PolicyEvaluation:
    rows = np.arange(len(evaluation.x))
    if not certificate.deployed:
        return PolicyEvaluation(0., 0., 0., 0., 0., 0., False, False)
    baseline = family.baseline_actions(evaluation.x)
    baseline_loss = _risk_vector(evaluation.potential_loss[rows, baseline], spec)
    baseline_utility = evaluation.potential_utility[rows, baseline].mean()
    forecast = family.predict(evaluation.x) if predictions is None else predictions

    def contrasts(threshold: float):
        action = gated_actions(forecast, threshold, baseline)
        delta = _risk_vector(evaluation.potential_loss[rows, action], spec) - baseline_loss
        gain = float(evaluation.potential_utility[rows, action].mean() - baseline_utility)
        return delta, gain, float(np.mean(action != baseline))

    change, gain, deployment = contrasts(certificate.threshold)
    unsafe = _is_unsafe(change, spec)
    measured_prefix = any(
        _is_unsafe(contrasts(t)[0], spec) for t in certificate.passed if t != certificate.threshold
    )
    prefix_unsafe = measured_prefix or bool(certificate.passed and unsafe)
    if certificate.guarantee == "none":
        prefix_unsafe = unsafe
    return PolicyEvaluation(
        deployment, float(change[0]), float(change[1]), float(change[2]), gain,
        0.0 if unsafe else gain, unsafe, prefix_unsafe,
    )


def _oracular(
    evaluation: EvaluationSample, family: PolicyFamily, spec: RiskSpec, pred: PolicyScores,
) -> PolicyCertificate:
    best = PolicyCertificate("Oracle", inf, (), 0, "oracle", None)
    best_value = 0.0
    for threshold in family.thresholds:
        proposal = PolicyCertificate("Oracle", threshold, (), 0, "oracle", None)
        result = evaluate_policy(evaluation, family, proposal, spec, pred)
        if not result.selected_unsafe and result.welfare_gain > best_value:
            best, best_value = proposal, result.welfare_gain
    return best


def run_benchmark(
    split: ExperimentSplit,
    seed: int,
    *,
    spec: RiskSpec = RiskSpec(),
    masses: tuple[float, ...] = DEFAULT_MASSES,
    leaves: tuple[int, ...] = (5, 12),
    forest_trees: int = 100,
    oracle_family: PolicyFamily | None = None,
    include_opra: bool = True,
) -> list[BenchmarkRow]:
    """Train on D_tr, select on D_cal, and report all methods on shared D_eval."""
    train = split.train
    distribution = TwoPartDistributionalLearner(spec.beta, seed, leaves, spec.extensive_event).fit(train)
    mean = MeanCausalForest(seed, leaves=leaves, trees=forest_trees).fit(train)
    plugin = PlugInCVaRITR(distribution, spec.cvar_cap, spec.beta).fit(train)
    dist_family = offer_from_training(distribution.predict, train, masses)
    mean_family = offer_from_training(mean.predict, train, masses)
    plugin_family = PolicyFamily(plugin.predict, (-inf,))
    if oracle_family is None:
        oracle_family = oracle_smoke_family(masses)

    calibration = split.calibration
    benchmark = [
        (dist_family, lambda: certify_localized(calibration, dist_family, spec, split.overlap)),
        (dist_family, lambda: certify_separate(calibration, dist_family, spec, split.overlap)),
        (mean_family, lambda: certify_localized(calibration, mean_family, spec, split.overlap, method="B2")),
        (plugin_family, lambda: PolicyCertificate("B3", -inf, (), 0, "none", None)),
        (mean_family, lambda: PolicyCertificate("B4", -inf, (), 0, "none", None)),
        (dist_family, lambda: certify_indicator(calibration, dist_family, distribution.cutoff, spec, split.overlap)),
    ]
    if include_opra:
        benchmark.insert(2, (
            dist_family, lambda: certify_separate(calibration, dist_family, spec, split.overlap, engine="opra"),
        ))

    predictions: dict[int, PolicyScores] = {}

    def forecast(family: PolicyFamily) -> PolicyScores:
        key = id(family)
        if key not in predictions:
            predictions[key] = family.predict(split.evaluation.x)
        return predictions[key]

    oracle = _oracular(split.evaluation, oracle_family, spec, forecast(oracle_family))
    results = [
        BenchmarkRow(
            oracle.method, oracle.guarantee, oracle.threshold, oracle.tested, len(oracle.passed),
            evaluate_policy(split.evaluation, oracle_family, oracle, spec, forecast(oracle_family)), 0.,
        )
    ]
    for family, certify in benchmark:
        start = perf_counter()
        cert = certify()
        seconds = perf_counter() - start
        result = evaluate_policy(split.evaluation, family, cert, spec, forecast(family))
        results.append(BenchmarkRow(
            cert.method, cert.guarantee, cert.threshold, cert.tested, len(cert.passed), result, seconds,
        ))
    return results
