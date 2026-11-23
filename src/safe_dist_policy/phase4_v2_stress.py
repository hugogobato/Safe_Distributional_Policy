"""Phase 4 Protocol v2 full stress & sensitivity suite (Blocks A-F)."""

from dataclasses import replace
from math import inf
from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from .certification import (
    PolicyCertificate,
    PolicyFamily,
    RiskSpec,
    certify_indicator,
    certify_localized,
    certify_separate,
    gated_actions,
    offer_from_training,
)
from .eval import DEFAULT_MASSES, higher_quantile, sample_cvar
from .learners import MeanCausalForest, TwoPartDistributionalLearner, ZICBCFBridge
from .phase4_design import (
    CORE,
    DESIGNS,
    STRESS,
    Design,
    always_treat_family,
    baseline_quantile,
    population,
    simulated_data,
)
from .phase4_v2 import (
    V2_DESIGNS,
    _exact_survival_contrast,
    v2_baseline_quantile,
    v2_simulated_split,
)


def _cached(fn):
    cache = {}

    def wrapped(x):
        k = id(x)
        if k not in cache:
            cache[k] = x, fn(x)
        return cache[k][1]

    return wrapped


class StressTruth:
    """Analytical conditional truth for Phase 4 DGPs 1-4, 6-8, 10."""

    def __init__(self, evaluation, law, design: Design, cap: float):
        self.ev = evaluation
        self.design = design
        self.cap = cap
        self.beta = design.beta
        self.exact_equal_q = design.dgp == 3 and design.label == "equality"
        self.exact_equal_ext = self.exact_equal_q or design.dgp == 4
        self.uniform_harm = design.dgp == 7 or (design.dgp == 3 and design.label == "strict")
        eta = baseline_quantile(design, design.beta)
        at_eta = law.survival(eta)
        means = law.loss_mean()
        self.surv_diff = at_eta[:, 1] - at_eta[:, 0]
        self.ext_diff = law.positive[:, 1] - law.positive[:, 0]
        self.utility_gain = means[:, 0] - means[:, 1]
        self.q0 = higher_quantile(evaluation.potential_loss[:, 0], design.beta)
        self.cv0 = sample_cvar(evaluation.potential_loss[:, 0], design.beta, cap)
        self.pareto_groups = None
        if design.dgp == 8:
            arms = population(design, np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]))
            self.pareto_groups = tuple(
                np.array([m[0, 0], m[0, 1], m[1, 1]])
                for m in (arms.positive, arms.location, arms.shape)
            )
            self.safe_pareto = evaluation.x[:, 0] < 0.35
            self.cvar_model0 = self._pareto_cvar(np.array([1.0, 0.0, 0.0]))

    def _pareto_cvar(self, weights: np.ndarray) -> float:
        assert self.pareto_groups is not None
        prob, xmin, expn = self.pareto_groups
        target = 1.0 - self.beta

        def surv(t: float) -> float:
            return float(weights @ (prob * np.minimum(1.0, xmin / t) ** expn))

        if surv(self.cap) >= target:
            return self.cap
        eta = float(brentq(lambda t: surv(t) - target, 1e-12, self.cap)) if float(weights @ prob) > target else 0.0
        before = np.maximum(0.0, np.minimum(xmin, self.cap) - eta)
        edge = np.minimum(self.cap, np.maximum(eta, xmin))
        tail = xmin**expn * (edge ** (1.0 - expn) - self.cap ** (1.0 - expn)) / (expn - 1.0)
        integral = float(weights @ (prob * (before + tail)))
        return float(eta + integral / target)

    def _pareto_delta(self, sw: np.ndarray) -> float | None:
        if self.pareto_groups is None:
            return None
        s_safe = float(np.mean(sw & self.safe_pareto))
        s_harm = float(np.mean(sw & ~self.safe_pareto))
        return self._pareto_cvar(np.array([1.0 - s_safe - s_harm, s_safe, s_harm])) - self.cvar_model0

    def evaluate(self, family: PolicyFamily, cert: PolicyCertificate, spec: RiskSpec) -> dict[str, object]:
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
        scores = family.predict(self.ev.x)

        def eval_thresh(t: float):
            sw = gated_actions(scores, t, 0).astype(bool)
            if not sw.any():
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
            loss = np.where(sw, self.ev.potential_loss[:, 1], self.ev.potential_loss[:, 0])
            dq = higher_quantile(loss, spec.beta) - self.q0
            dc_emp = sample_cvar(loss, spec.beta, spec.cvar_cap) - self.cv0
            dc_exact = self._pareto_delta(sw)
            dc = dc_emp if dc_exact is None else dc_exact
            ds = 0.0 if self.exact_equal_q else float(np.mean(sw * self.surv_diff))
            de = 0.0 if self.exact_equal_ext else float(np.mean(sw * self.ext_diff))
            gain = float(np.mean(sw * self.utility_gain))
            unsafe = False
            if spec.quantile_tolerance is not None and not self.exact_equal_q and (self.uniform_harm or ds > 1e-12):
                unsafe = True
            if (
                spec.extensive_tolerance is not None
                and not self.exact_equal_ext
                and (self.uniform_harm or de > spec.extensive_tolerance + 1e-12)
            ):
                unsafe = True
            if spec.cvar_tolerance is not None and not self.exact_equal_q and (self.uniform_harm or dc > spec.cvar_tolerance + 1e-9):
                unsafe = True
            return float(np.mean(sw)), gain, dq, dc, ds, de, unsafe

        dep, gain, dq, dc, ds, de, unsafe = eval_thresh(cert.threshold)
        prefix_unsafe = unsafe
        if cert.passed and not prefix_unsafe:
            for t in cert.passed:
                if t != cert.threshold and eval_thresh(t)[-1]:
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


def stress_block_tasks(block: str) -> list[tuple[str, str, int]]:
    """Return list of (block, design_name, seed) tasks for a stress block."""
    if block == "core_1_4":
        designs = [d for d in CORE if d.dgp in (1, 2, 3, 4) and not (d.dgp == 3 and d.label == "strict")]
        return [(block, d.name, s) for d in designs for s in range(1000, 1050)]
    if block == "dgp6_crossover":
        designs = [d for d in STRESS if d.dgp == 6]
        return [(block, d.name, s) for d in designs for s in range(1000, 1050)]
    if block == "dgp7_misspec":
        designs = [d for d in STRESS if d.dgp == 7]
        return [(block, d.name, s) for d in designs for s in range(1000, 1050)]
    if block == "dgp8_pareto":
        designs = [d for d in STRESS if d.dgp == 8]
        return [(block, d.name, s) for d in designs for s in range(1000, 1050)]
    if block == "dgp10_scaling":
        designs = [d for d in STRESS if d.dgp == 10]
        return [(block, d.name, s) for d in designs for s in range(1000, 1003)]
    if block == "zic_sensitivity":
        return [(block, name, s) for name in ("v2-d05-conflict", "v2-d09-meps", "v2-d09-ohie") for s in range(1000, 1020)]
    raise ValueError(f"Unknown stress block {block}.")


def run_stress_task(block: str, design_name: str, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Execute one paired replication for any Phase 4 v2 stress block."""
    t_total = perf_counter()
    if block == "zic_sensitivity":
        return _run_zic_task(design_name, seed)

    design = DESIGNS[design_name]
    n_eval = 50_000 if design.dgp in (6, 10) else 80_000
    split, law = simulated_data(design, seed, n_eval=n_eval)
    train, cal = split.train, split.calibration
    cap = baseline_quantile(design, 0.99) * design.cap_factor if design.cap_factor else 50.0
    truth = StressTruth(split.evaluation, law, design, cap)

    t0 = perf_counter()
    dist = TwoPartDistributionalLearner(design.beta, seed).fit(train)
    fit_dist_s = perf_counter() - t0

    mean_family = None
    fit_mean_s = 0.0
    if design.dgp != 7:
        t0 = perf_counter()
        mean_cf = MeanCausalForest(seed, trees=100, timeout=3600.0 if design.dgp == 10 else 300.0).fit(train)
        fit_mean_s = perf_counter() - t0
        mean_family = offer_from_training(_cached(mean_cf.predict), train, DEFAULT_MASSES)

    ext_ctrl = dist.extensive_control_variate
    rows: list[dict[str, object]] = []

    def emit(target: str, method: str, fam: PolicyFamily, spec: RiskSpec, fn):
        ts = perf_counter()
        cert = fn()
        dur = perf_counter() - ts
        ev = truth.evaluate(fam, cert, spec)
        rows.append({
            "block": block,
            "design": design.name,
            "seed": seed,
            "target": target,
            "method": method,
            "guarantee": cert.guarantee,
            "deployed": cert.deployed,
            "tested": cert.tested,
            "prefix_size": len(cert.passed),
            "cert_seconds": dur,
            **ev,
        })

    def run_q_suite(target_name: str, joint: bool, *, include_b5: bool = True, include_b1: bool = True):
        spec_gro = RiskSpec(
            beta=design.beta, bound="gro", quantile_tolerance=0.0,
            extensive_tolerance=0.0 if joint else None, cvar_cap=cap,
        )
        spec_v1 = replace(spec_gro, bound="betting")
        pred_fn = _cached(dist.scorer(spec_gro))
        fam = offer_from_training(pred_fn, train, DEFAULT_MASSES)
        oof_sc = dist.oof_predict_for(train, spec_gro)
        ctrl = ext_ctrl if joint else None

        emit(
            target_name, "LCT-LTT-GRO", fam, spec_gro,
            lambda: certify_localized(
                cal, fam, spec_gro, split.overlap, train=train, train_scores=oof_sc,
                ext_control=ctrl, anchored=True, method="LCT-LTT-GRO",
            ),
        )
        emit(
            target_name, "LCT-LTT-aGRAPA", fam, spec_v1,
            lambda: certify_localized(cal, fam, spec_v1, split.overlap, method="LCT-LTT-aGRAPA"),
        )
        if include_b1:
            emit(
                target_name, "B1-survival", fam, spec_v1,
                lambda: certify_separate(cal, fam, spec_v1, split.overlap, engine="survival"),
            )
            emit(
                target_name, "B1-opra", fam, spec_v1,
                lambda: certify_separate(cal, fam, spec_v1, split.overlap, engine="opra"),
            )
        if mean_family is not None:
            emit(
                target_name, "B2-GRO", mean_family, spec_gro,
                lambda: certify_localized(
                    cal, mean_family, spec_gro, split.overlap, train=train,
                    ext_control=ctrl, anchored=True, method="B2-GRO",
                ),
            )
            emit(
                target_name, "B4", mean_family, spec_gro,
                lambda: PolicyCertificate("B4", -inf, (), 0, "none", None),
            )
        if include_b5:
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

            emit(target_name, "B5+ext-GRO" if joint else "B5-GRO", fam, spec_gro, run_b5)

    def run_cvar_suite(target_name: str):
        spec_gro = RiskSpec(
            beta=design.beta, bound="gro", quantile_tolerance=None, extensive_tolerance=None,
            cvar_tolerance=0.0, cvar_cap=cap, cvar_cells=120,
        )
        spec_v1 = replace(spec_gro, bound="betting")
        pred_fn = _cached(dist.scorer(spec_gro))
        fam = offer_from_training(pred_fn, train, DEFAULT_MASSES)
        oof_sc = dist.oof_predict_for(train, spec_gro)
        emit(
            target_name, "LCT-LTT-GRO", fam, spec_gro,
            lambda: certify_localized(
                cal, fam, spec_gro, split.overlap, train=train, train_scores=oof_sc,
                anchored=True, method="LCT-LTT-GRO",
            ),
        )
        emit(
            target_name, "LCT-LTT-aGRAPA", fam, spec_v1,
            lambda: certify_localized(cal, fam, spec_v1, split.overlap, method="LCT-LTT-aGRAPA"),
        )
        emit(
            target_name, "B1-opra", fam, spec_v1,
            lambda: certify_separate(cal, fam, spec_v1, split.overlap, engine="opra"),
        )
        if mean_family is not None:
            emit(
                target_name, "B2-GRO", mean_family, spec_gro,
                lambda: certify_localized(
                    cal, mean_family, spec_gro, split.overlap, train=train, anchored=True, method="B2-GRO",
                ),
            )

    if block == "core_1_4":
        run_q_suite("Joint-Q-Ext", joint=True)
        run_q_suite("Q-only", joint=False)
    elif block == "dgp6_crossover":
        run_q_suite("Joint-Q-Ext", joint=True)
        run_q_suite("Q-only", joint=False)
        run_cvar_suite("CVaR-cap50")
    elif block == "dgp7_misspec":
        spec_gro = RiskSpec(beta=design.beta, bound="gro", quantile_tolerance=0.0, extensive_tolerance=0.0)
        pred_fn = _cached(dist.scorer(spec_gro))
        fam = offer_from_training(pred_fn, train, DEFAULT_MASSES)
        oof_sc = dist.oof_predict_for(train, spec_gro)
        always = always_treat_family(DEFAULT_MASSES)
        misspec = LogisticRegression().fit(train.x[:, :1], train.action)
        e_hat_cal = np.clip(misspec.predict_proba(cal.x[:, :1])[:, 1], design.kappa, 1.0 - design.kappa)
        e_hat_tr = np.clip(misspec.predict_proba(train.x[:, :1])[:, 1], design.kappa, 1.0 - design.kappa)
        cal_est = replace(cal, propensity=e_hat_cal)
        tr_est = replace(train, propensity=e_hat_tr)
        gamma = (1.0 - design.kappa) / design.kappa
        for suffix, f_use, tr_sc in (("", fam, oof_sc), ("-fixed", always, None)):
            emit(
                "Joint-Q-Ext", f"LCT-LTT-GRO-true{suffix}", f_use, spec_gro,
                lambda f=f_use, s=tr_sc: certify_localized(
                    cal, f, spec_gro, split.overlap, train=train, train_scores=s,
                    ext_control=ext_ctrl, anchored=True,
                ),
            )
            emit(
                "Joint-Q-Ext", f"LCT-LTT-GRO-misspec{suffix}", f_use, spec_gro,
                lambda f=f_use, s=tr_sc: certify_localized(
                    cal_est, f, spec_gro, split.overlap, train=tr_est, train_scores=s,
                    ext_control=ext_ctrl, anchored=True,
                ),
            )
            emit(
                "Joint-Q-Ext", f"LCT-LTT-GRO-gamma{suffix}", f_use, spec_gro,
                lambda f=f_use, s=tr_sc: certify_localized(
                    cal_est, f, spec_gro, split.overlap, gamma=gamma, train=tr_est, train_scores=s,
                    ext_control=ext_ctrl, anchored=True,
                ),
            )
    elif block == "dgp8_pareto":
        run_q_suite("Joint-Q-Ext", joint=True)
        run_q_suite("Q-only", joint=False)
        run_cvar_suite(f"CVaR-cap{int(design.cap_factor)}xQ99")
    elif block == "dgp10_scaling":
        run_q_suite("Joint-Q-Ext", joint=True, include_b5=False, include_b1=True)
        run_q_suite("Q-only", joint=False, include_b5=False, include_b1=False)
    else:
        raise ValueError(f"Unhandled block {block}.")

    profile = {
        "block": block,
        "design": design.name,
        "seed": seed,
        "n_train": train.n,
        "n_cal": cal.n,
        "p": design.p,
        "cap": cap,
        "fit_dist_seconds": fit_dist_s,
        "fit_mean_seconds": fit_mean_s,
        "total_seconds": perf_counter() - t_total,
        "peak_process_rss_mib": getrusage(RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_child_rss_mib": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024.0,
    }
    return rows, profile


def _run_zic_task(design_name: str, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    t_total = perf_counter()
    design = V2_DESIGNS[design_name]
    split, mean_ev, pos_ev = v2_simulated_split(design, seed)
    train, cal, ev = split.train, split.calibration, split.evaluation
    eta0 = v2_baseline_quantile(design)
    surv_diff = _exact_survival_contrast(design, ev.x, eta0)
    q0 = higher_quantile(ev.potential_loss[:, 0], design.beta)
    cv0 = sample_cvar(ev.potential_loss[:, 0], design.beta, design.cvar_cap)

    t0 = perf_counter()
    dist = TwoPartDistributionalLearner(design.beta, seed).fit(train)
    fit_dist_s = perf_counter() - t0

    t0 = perf_counter()
    zic = ZICBCFBridge(design.beta, seed, burn=500, draws=1000, trees=200).fit(train)
    fit_zic_s = perf_counter() - t0

    rows: list[dict[str, object]] = []
    utility_gain = mean_ev[:, 0] - mean_ev[:, 1]
    ext_diff = pos_ev[:, 1] - pos_ev[:, 0]

    def eval_one(fam: PolicyFamily, cert: PolicyCertificate, spec: RiskSpec) -> dict[str, object]:
        if not cert.deployed:
            return {
                "deployment": 0.0, "welfare_gain": 0.0, "safe_welfare": 0.0,
                "delta_quantile": 0.0, "delta_cvar": 0.0, "delta_extensive": 0.0,
                "tail_survival_delta": 0.0, "selected_unsafe": False, "prefix_unsafe": False,
            }
        scores = fam.predict(ev.x)

        def check_t(t: float):
            sw = gated_actions(scores, t, 0).astype(bool)
            if not sw.any():
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
            loss = np.where(sw, ev.potential_loss[:, 1], ev.potential_loss[:, 0])
            dq = higher_quantile(loss, spec.beta) - q0
            dc = sample_cvar(loss, spec.beta, spec.cvar_cap) - cv0
            ds = float(np.mean(sw * surv_diff))
            de = float(np.mean(sw * ext_diff))
            gain = float(np.mean(sw * utility_gain))
            unsafe = (
                (spec.quantile_tolerance is not None and ds > 1e-12)
                or (spec.extensive_tolerance is not None and de > spec.extensive_tolerance + 1e-12)
            )
            return float(np.mean(sw)), gain, dq, dc, ds, de, unsafe

        dep, gain, dq, dc, ds, de, unsafe = check_t(cert.threshold)
        prefix_unsafe = unsafe or any(check_t(t)[-1] for t in cert.passed if t != cert.threshold)
        return {
            "deployment": dep, "welfare_gain": gain, "safe_welfare": 0.0 if unsafe else gain,
            "delta_quantile": dq, "delta_cvar": dc, "delta_extensive": de,
            "tail_survival_delta": ds, "selected_unsafe": unsafe, "prefix_unsafe": prefix_unsafe,
        }

    for target_name, joint in (("Joint-Q-Ext", True), ("Q-only", False)):
        spec = RiskSpec(beta=design.beta, bound="gro", quantile_tolerance=0.0, extensive_tolerance=0.0 if joint else None)
        fam_dist = offer_from_training(_cached(dist.scorer(spec)), train, DEFAULT_MASSES)
        fam_zic = offer_from_training(_cached(lambda x, s=spec: zic.predict_for(x, s)), train, DEFAULT_MASSES)
        oof_dist = dist.oof_predict_for(train, spec)

        for label, fam, tr_sc, ctrl in (
            ("LCT-LTT-GRO", fam_dist, oof_dist, dist.extensive_control_variate if joint else None),
            ("LCT-ZIC-GRO", fam_zic, None, zic.extensive_control_variate if joint else None),
        ):
            ts = perf_counter()
            cert = certify_localized(
                cal, fam, spec, split.overlap, train=train, train_scores=tr_sc,
                ext_control=ctrl, anchored=True, method=label,
            )
            dur = perf_counter() - ts
            rows.append({
                "block": "zic_sensitivity",
                "design": design.name,
                "seed": seed,
                "target": target_name,
                "method": label,
                "guarantee": cert.guarantee,
                "deployed": cert.deployed,
                "tested": cert.tested,
                "prefix_size": len(cert.passed),
                "cert_seconds": dur,
                **eval_one(fam, cert, spec),
            })

    profile = {
        "block": "zic_sensitivity",
        "design": design.name,
        "seed": seed,
        "n_train": train.n,
        "n_cal": cal.n,
        "p": design.p,
        "fit_dist_seconds": fit_dist_s,
        "fit_zic_seconds": fit_zic_s,
        "total_seconds": perf_counter() - t_total,
        "peak_process_rss_mib": getrusage(RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_child_rss_mib": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024.0,
    }
    return rows, profile
