"""Training-only learners for the candidate policies and comparator score families."""

import subprocess
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold

from .certification import PolicyScores, RiskSpec
from .data import F64, LoggedSample


def _training_folds(n: int, seed: int):
    return list(KFold(3, shuffle=True, random_state=seed).split(np.arange(n)))


def training_baseline_cutoff(train: LoggedSample, beta: float) -> float:
    """Training IPW quantile used for scoring and B5, never its certification."""
    control = train.action == 0
    if not control.any():
        raise ValueError("Training must contain baseline observations.")
    order = np.argsort(train.loss[control])
    loss = train.loss[control][order]
    weights = 1 / (1 - train.propensity[control][order])
    cdf = np.cumsum(weights) / weights.sum()
    return float(loss[min(len(loss) - 1, np.searchsorted(cdf, beta))])


class _Regression:
    def __init__(self, x: F64, y: F64, leaf_nodes: int, seed: int):
        self.constant = float(np.mean(y)) if len(y) else 0.0
        self.model = None
        if len(y) >= 20 and np.any(y != y[0]):
            self.model = HistGradientBoostingRegressor(
                max_iter=60, max_leaf_nodes=leaf_nodes, min_samples_leaf=12,
                early_stopping=False, random_state=seed,
            ).fit(x, y)

    def predict(self, x: F64) -> F64:
        return np.full(len(x), self.constant) if self.model is None else self.model.predict(x)


class _Hurdle:
    def __init__(self, x: F64, y: F64, leaf_nodes: int, seed: int):
        if len(y) == 0:
            raise ValueError("Every training fold must contain observations on both actions.")
        positive = y > 0
        self.share = float(positive.mean())
        self.classifier = None
        if len(y) >= 24 and 0 < self.share < 1:
            self.classifier = HistGradientBoostingClassifier(
                max_iter=60, max_leaf_nodes=leaf_nodes, min_samples_leaf=12,
                early_stopping=False, random_state=seed,
            ).fit(x, positive)
        log_loss = np.log(y[positive])
        self.location = _Regression(x[positive], log_loss, leaf_nodes, seed)
        residual = log_loss - self.location.predict(x[positive])
        self.log_variance = _Regression(x[positive], np.log(residual**2 + 0.0025), leaf_nodes, seed)
        # Calibrate the conditional variance proxy by the average squared standardized residual.
        local = np.exp(self.log_variance.predict(x[positive]))
        self.scale_factor = max(0.1, float(np.mean(residual**2 / local))) if len(residual) else 1.0
        local_var = np.clip(self.scale_factor * local, 0.0004, 16.0)
        self.smear = float(np.mean(np.exp(np.clip(residual - 0.5 * local_var, -20.0, 20.0)))) if len(residual) else 1.0

    def parameters(self, x: F64) -> tuple[F64, F64, F64]:
        if self.classifier is None:
            active = np.full(len(x), self.share)
        else:
            active = self.classifier.predict_proba(x)[:, 1]
        location = self.location.predict(x)
        sigma = np.sqrt(np.clip(self.scale_factor * np.exp(self.log_variance.predict(x)), 0.0004, 16.0))
        return active, location, sigma

    def summaries(self, x: F64, cutoff: float) -> tuple[F64, F64, F64]:
        active, mu, sig = self.parameters(x)
        tail = active if cutoff == 0 else active * ndtr((mu - np.log(cutoff)) / sig)
        mean = active * np.exp(np.clip(mu + 0.5 * sig**2, -20.0, 20.0)) * self.smear
        return active, tail, mean

    def capped_excess(self, x: F64, eta: F64, cap: float) -> F64:
        active, mu, sig = (v[:, None] for v in self.parameters(x))
        with np.errstate(divide="ignore"):
            z = (np.log(eta) - mu) / sig
        z_cap = (np.log(cap) - mu) / sig
        trunc_excess = (
            np.exp(mu + sig**2 / 2) * (ndtr(z_cap - sig) - ndtr(z - sig))
            - eta * (ndtr(z_cap) - ndtr(z))
            + (cap - eta) * ndtr(-z_cap)
        )
        return active * np.maximum(0, trunc_excess)


class TwoPartDistributionalLearner:
    """Arm-specific hurdle boosting, log-severity models and Duan mean smearing."""

    def __init__(
        self, beta: float, seed: int, leaves: tuple[int, ...] = (5, 12),
        extensive_event: str = "positive",
    ):
        self.beta = beta
        self.seed = seed
        self.leaves = leaves
        self.extensive_event = extensive_event
        self.fitted = False

    def _fit_arms(self, train: LoggedSample, leaf_nodes: int):
        models = []
        utilities = []
        distinct_utility = np.any(train.utility != -train.loss)
        for action in (0, 1):
            use = train.action == action
            models.append(_Hurdle(train.x[use], train.loss[use], leaf_nodes, self.seed))
            if distinct_utility:
                utilities.append(_Regression(train.x[use], train.utility[use], leaf_nodes, self.seed))
        return models, utilities

    def fit(self, train: LoggedSample) -> "TwoPartDistributionalLearner":
        self.cutoff = training_baseline_cutoff(train, self.beta)
        if len(self.leaves) == 1:
            self.leaf_nodes = self.leaves[0]
            splits = _training_folds(train.n, self.seed)
            oof_cache: list[tuple[np.ndarray, list[_Hurdle], list[_Regression], float]] = []
            for fit_rows, holdout_rows in splits:
                fitting = train.take(fit_rows)
                cutoff = training_baseline_cutoff(fitting, self.beta)
                models, utilities = self._fit_arms(fitting, self.leaf_nodes)
                oof_cache.append((holdout_rows, models, utilities, cutoff))
            self._oof_folds = oof_cache
        else:
            validation = []
            all_oof: list[list[tuple[np.ndarray, list[_Hurdle], list[_Regression], float]]] = []
            splits = _training_folds(train.n, self.seed)
            for leaf_nodes in self.leaves:
                loss = 0.0
                fold_models: list[tuple[np.ndarray, list[_Hurdle], list[_Regression], float]] = []
                for fit_rows, holdout_rows in splits:
                    fitting = train.take(fit_rows)
                    held = train.take(holdout_rows)
                    cutoff = training_baseline_cutoff(fitting, self.beta)
                    models, utilities = self._fit_arms(fitting, leaf_nodes)
                    fold_models.append((holdout_rows, models, utilities, cutoff))
                    for action, model in enumerate(models):
                        use = held.action == action
                        if not use.any():
                            continue
                        active, tail, _ = model.summaries(held.x[use], cutoff)
                        outcomes = held.loss[use]
                        loss += np.mean((active - (outcomes > 0))**2 + (tail - (outcomes > cutoff))**2)
                        if utilities:
                            y = held.utility[use]
                            pred = utilities[action].predict(held.x[use])
                            loss += np.mean((pred - y)**2) / max(1, np.var(y))
                validation.append(loss)
                all_oof.append(fold_models)
            best_idx = int(np.argmin(validation))
            self.leaf_nodes = self.leaves[best_idx]
            self._oof_folds = all_oof[best_idx]
        self._train_n = train.n
        self.arm, self.utility_models = self._fit_arms(train, self.leaf_nodes)
        self.fitted = True
        return self

    def predict(self, x: F64) -> PolicyScores:
        if not self.fitted:
            raise RuntimeError("Fit the two-part learner on training data first.")
        active0, tail0, mean0 = self.arm[0].summaries(x, self.cutoff)
        active1, tail1, mean1 = self.arm[1].summaries(x, self.cutoff)
        utility_gain = mean0 - mean1
        if self.utility_models:
            utility_gain = self.utility_models[1].predict(x) - self.utility_models[0].predict(x)
        ext_gain = active0 - active1 if self.extensive_event == "positive" else active1 - active0
        return PolicyScores((utility_gain > 0).astype(np.int64), np.minimum(tail0 - tail1, ext_gain))

    def predict_for(self, x: F64, spec: RiskSpec) -> PolicyScores:
        """Align safety-priority scores strictly with the active components of spec."""
        if not self.fitted:
            raise RuntimeError("Fit the two-part learner on training data first.")
        return self._score_with_arms(x, self.arm, self.utility_models, self.cutoff, spec)

    @staticmethod
    def _score_with_arms(
        x: F64,
        arms: list[_Hurdle],
        utilities: list[_Regression],
        cutoff: float,
        spec: RiskSpec,
    ) -> PolicyScores:
        active0, tail0, mean0 = arms[0].summaries(x, cutoff)
        active1, tail1, mean1 = arms[1].summaries(x, cutoff)
        utility_gain = mean0 - mean1
        if utilities:
            utility_gain = utilities[1].predict(x) - utilities[0].predict(x)
        components: list[F64] = []
        if spec.quantile_tolerance is not None:
            components.append(tail0 - tail1)
        if spec.extensive_tolerance is not None:
            ext_gain = active0 - active1 if spec.extensive_event == "positive" else active1 - active0
            components.append(ext_gain)
        if spec.cvar_tolerance is not None:
            eta_arr = np.array([min(cutoff, spec.cvar_cap)], dtype=np.float64)
            ex0 = arms[0].capped_excess(x, eta_arr, spec.cvar_cap)[:, 0]
            ex1 = arms[1].capped_excess(x, eta_arr, spec.cvar_cap)[:, 0]
            cvar_gain = ex0 - ex1
            if components:
                scale = max(spec.cvar_cap - float(eta_arr[0]), 1e-6)
                cvar_gain = cvar_gain / scale
            components.append(cvar_gain)
        score = components[0]
        for extra in components[1:]:
            score = np.minimum(score, extra)
        return PolicyScores((utility_gain > 0).astype(np.int64), score)

    def oof_predict_for(self, train: LoggedSample, spec: RiskSpec) -> PolicyScores:
        """Honest 3-fold out-of-fold PolicyScores on D_tr for unbiased GRO stake/anchor selection."""
        if not self.fitted or train.n != self._train_n:
            raise RuntimeError("oof_predict_for requires the exact training sample used in fit().")
        actions = np.empty(train.n, dtype=np.int64)
        scores = np.empty(train.n, dtype=np.float64)
        for holdout_rows, models, utilities, cutoff in self._oof_folds:
            ps = self._score_with_arms(train.x[holdout_rows], models, utilities, cutoff, spec)
            actions[holdout_rows] = ps.action
            scores[holdout_rows] = ps.score
        return PolicyScores(actions, scores)

    def scorer(self, spec: RiskSpec) -> Callable[[F64], PolicyScores]:
        """Return a target-aligned PolicyScores callable for offer_from_training."""
        return lambda x: self.predict_for(x, spec)

    def extensive_control_variate(self, x: F64, extensive_event: str | None = None) -> tuple[F64, F64]:
        """Training-fitted conditional extensive-margin probabilities (m0, m1) in [0, 1]."""
        if not self.fitted:
            raise RuntimeError("Fit the two-part learner on training data first.")
        event = self.extensive_event if extensive_event is None else extensive_event
        active0 = np.clip(self.arm[0].parameters(x)[0], 0.0, 1.0)
        active1 = np.clip(self.arm[1].parameters(x)[0], 0.0, 1.0)
        if event == "positive":
            return active0, active1
        return 1.0 - active0, 1.0 - active1


class PlugInCVaRITR:
    """B3. Minimize the plug-in capped Rockafellar–Uryasev objective on training."""

    def __init__(self, learner: TwoPartDistributionalLearner, cap: float, beta: float):
        self.learner = learner
        self.cap = cap
        self.beta = beta

    def fit(self, train: LoggedSample) -> "PlugInCVaRITR":
        upper = np.minimum(train.loss, self.cap)
        eta = np.unique(np.r_[np.quantile(upper, np.linspace(0, 1, 41)), self.cap])
        scores0, scores1 = (model.capped_excess(train.x, eta, self.cap) for model in self.learner.arm)
        objective = eta + np.mean(np.minimum(scores0, scores1), axis=0) / (1 - self.beta)
        self.eta = float(eta[np.argmin(objective)])
        return self

    def predict(self, x: F64) -> PolicyScores:
        eta = np.array([self.eta])
        z0, z1 = (model.capped_excess(x, eta, self.cap)[:, 0] for model in self.learner.arm)
        change = z0 - z1
        return PolicyScores((change > 0).astype(np.int64), change)


def _run_r(rscript: str, mode: str, *args: str, timeout: float = 300.0):
    bridge = Path(__file__).with_name("forest_bridge.R")
    proc = subprocess.run(
        [rscript, "--vanilla", str(bridge), mode, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode:
        raise RuntimeError(f"R {mode} failed (status {proc.returncode}): {proc.stderr[-4000:]}\n{proc.stdout[-2000:]}")


class MeanCausalForest:
    """Native grf benchmark. B2 and B4 share this exact trained candidate."""

    def __init__(
        self, seed: int, leaves: tuple[int, ...] = (5, 12), trees: int = 100,
        rscript: str = "Rscript", timeout: float = 300.0,
    ):
        self.seed = seed
        self.leaves = leaves
        self.trees = trees
        self.rscript = rscript
        self.timeout = timeout
        self._model: bytes | None = None

    def fit(self, train: LoggedSample) -> "MeanCausalForest":
        fold_id = np.empty(train.n, dtype=np.int64)
        for k, (_, holdout) in enumerate(_training_folds(train.n, self.seed)):
            fold_id[holdout] = k
        with TemporaryDirectory(prefix="safe-policy-grf-") as directory:
            tr = Path(directory) / "train.csv"
            model = Path(directory) / "model.rds"
            np.savetxt(tr, np.c_[train.action, train.loss, train.utility, train.propensity, fold_id, train.x], delimiter=",")
            _run_r(
                self.rscript, "mean-fit", str(tr), str(model), str(self.seed), str(self.trees),
                ",".join(str(x) for x in self.leaves), timeout=self.timeout,
            )
            self._model = model.read_bytes()
        return self

    def predict(self, x: F64) -> PolicyScores:
        if self._model is None:
            raise RuntimeError("Fit the mean causal forest on training data first.")
        with TemporaryDirectory(prefix="safe-policy-grf-") as directory:
            new_data = Path(directory) / "x.csv"
            model = Path(directory) / "model.rds"
            output = Path(directory) / "predictions.csv"
            model.write_bytes(self._model)
            np.savetxt(new_data, x, delimiter=",")
            _run_r(self.rscript, "mean-predict", str(model), str(new_data), str(output), timeout=self.timeout)
            effects = np.loadtxt(output, delimiter=",", ndmin=2)
        return PolicyScores((effects[:, 1] > 0).astype(np.int64), -effects[:, 0])


class ZICBCFBridge:
    """Training-only ZIC-BCF-Smear fit followed by an explicit held-out projection.

    The reference hurdle fit exports fitted posterior draws but no saved tree
    traces or newdata method. A secondary forest projects training-only
    posterior summaries to new X. Severity exceedance integrates the native
    Gaussian log-severity draws. Duan smearing is used for the means.
    """

    def __init__(
        self, beta: float, seed: int, *, burn: int = 500, draws: int = 1000,
        trees: int = 200, rscript: str = "Rscript", extensive_event: str = "positive",
    ):
        self.beta, self.seed = beta, seed
        self.burn, self.draws, self.trees = burn, draws, trees
        self.rscript = rscript
        self.extensive_event = extensive_event
        self.projection = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=8, n_jobs=1, random_state=seed,
        )
        self.utility_models: list[_Regression] = []

    def fit(self, train: LoggedSample) -> "ZICBCFBridge":
        cutoff = training_baseline_cutoff(train, self.beta)
        with TemporaryDirectory(prefix="safe-policy-zicbcf-") as directory:
            tr = Path(directory) / "train.csv"
            output = Path(directory) / "posterior_targets.csv"
            np.savetxt(tr, np.c_[train.action, train.loss, train.propensity, train.x], delimiter=",")
            _run_r(
                self.rscript, "zicbcf-fit", str(tr), str(output), str(self.seed), str(cutoff),
                str(self.burn), str(self.draws), str(self.trees), timeout=3600,
            )
            target = np.loadtxt(output, delimiter=",", ndmin=2)
        if target.shape != (train.n, 6) or not np.isfinite(target).all():
            raise ValueError("Invalid zicbcf posterior projection targets.")
        self.projection.fit(train.x, target)
        self.utility_models = []
        if np.any(train.utility != -train.loss):
            for a in (0, 1):
                keep = train.action == a
                self.utility_models.append(_Regression(train.x[keep], train.utility[keep], 12, self.seed))
        return self

    def predict(self, x: F64) -> PolicyScores:
        act0, act1, tail0, tail1, mean0, mean1 = self.projection.predict(x).T
        utility_gain = mean0 - mean1
        if self.utility_models:
            utility_gain = self.utility_models[1].predict(x) - self.utility_models[0].predict(x)
        ext_gain = act0 - act1 if self.extensive_event == "positive" else act1 - act0
        return PolicyScores((utility_gain > 0).astype(np.int64), np.minimum(tail0 - tail1, ext_gain))

    def predict_for(self, x: F64, spec: RiskSpec) -> PolicyScores:
        act0, act1, tail0, tail1, mean0, mean1 = self.projection.predict(x).T
        utility_gain = mean0 - mean1
        if self.utility_models:
            utility_gain = self.utility_models[1].predict(x) - self.utility_models[0].predict(x)
        components: list[F64] = []
        if spec.quantile_tolerance is not None:
            components.append(tail0 - tail1)
        if spec.extensive_tolerance is not None:
            ext_gain = act0 - act1 if spec.extensive_event == "positive" else act1 - act0
            components.append(ext_gain)
        if not components:
            components.append(tail0 - tail1)
        score = components[0]
        for extra in components[1:]:
            score = np.minimum(score, extra)
        return PolicyScores((utility_gain > 0).astype(np.int64), score)

    def extensive_control_variate(self, x: F64, extensive_event: str | None = None) -> tuple[F64, F64]:
        event = self.extensive_event if extensive_event is None else extensive_event
        act0, act1 = self.projection.predict(x).T[:2]
        act0 = np.clip(act0, 0.0, 1.0)
        act1 = np.clip(act1, 0.0, 1.0)
        if event == "positive":
            return act0, act1
        return 1.0 - act0, 1.0 - act1
