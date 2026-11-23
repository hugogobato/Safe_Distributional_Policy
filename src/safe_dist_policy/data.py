"""Immutable data crossing the training, calibration and oracle evaluation boundary."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

F64: TypeAlias = NDArray[np.float64]
I64: TypeAlias = NDArray[np.int64]


def _read_only(values, dtype):
    owned = np.array(values, dtype=dtype, copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True)
class LoggedSample:
    x: F64
    action: I64
    loss: F64
    propensity: F64
    row_id: I64
    utility: F64

    def __post_init__(self):
        if not np.isin(self.action, (0, 1)).all():
            raise ValueError("Actions must be 0 or 1.")
        for name in ("x", "loss", "propensity", "utility"):
            object.__setattr__(self, name, _read_only(getattr(self, name), np.float64))
        for name in ("action", "row_id"):
            object.__setattr__(self, name, _read_only(getattr(self, name), np.int64))
        n = self.loss.size
        if self.x.ndim != 2 or self.x.shape[0] != n or self.x.shape[1] < 1:
            raise ValueError("Covariates must have shape (n, p), p >= 1.")
        for name in ("loss", "action", "propensity", "row_id", "utility"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one value per row.")
        if any(not np.isfinite(a).all() for a in (self.x, self.loss, self.propensity, self.utility)):
            raise ValueError("Logged data must be finite.")
        if np.any(self.loss < 0) or np.any((self.propensity <= 0) | (self.propensity >= 1)):
            raise ValueError("Losses must be nonnegative and propensities strictly interior.")
        if np.unique(self.row_id).size != n:
            raise ValueError("Row IDs must be unique within a split.")

    def take(self, index):
        return LoggedSample(
            self.x[index], self.action[index], self.loss[index], self.propensity[index],
            self.row_id[index], self.utility[index],
        )

    def ordered(self):
        """Restore a preassigned, calibration-independent draw order."""
        if self.row_id.size < 2 or np.all(self.row_id[:-1] < self.row_id[1:]):
            return self
        return self.take(np.argsort(self.row_id))

    @property
    def n(self):
        return self.loss.size


@dataclass(frozen=True)
class EvaluationSample:
    x: F64
    potential_loss: F64
    potential_utility: F64
    row_id: I64

    def __post_init__(self):
        for name in ("x", "potential_loss", "potential_utility"):
            object.__setattr__(self, name, _read_only(getattr(self, name), np.float64))
        object.__setattr__(self, "row_id", _read_only(self.row_id, np.int64))
        n = self.row_id.size
        if self.x.ndim != 2 or self.x.shape[0] != n or self.row_id.shape != (n,):
            raise ValueError("Invalid evaluation covariates or IDs.")
        if self.potential_loss.shape != (n, 2) or self.potential_utility.shape != (n, 2):
            raise ValueError("Evaluation requires one potential loss and utility per arm.")
        if any(not np.isfinite(a).all() for a in (self.x, self.potential_loss, self.potential_utility)):
            raise ValueError("Evaluation arrays must be finite.")
        if np.any(self.potential_loss < 0) or np.unique(self.row_id).size != n:
            raise ValueError("Invalid evaluation losses or duplicate row IDs.")


@dataclass(frozen=True)
class ExperimentSplit:
    train: LoggedSample
    calibration: LoggedSample
    evaluation: EvaluationSample
    overlap: float

    def __post_init__(self):
        if not 0 < self.overlap <= 0.5:
            raise ValueError("Overlap must lie in (0, 0.5].")
        if min(self.train.n, self.calibration.n, self.evaluation.row_id.size) < 2:
            raise ValueError("Each experiment split needs at least two rows.")
        features = (self.train.x.shape[1], self.calibration.x.shape[1], self.evaluation.x.shape[1])
        if len(set(features)) != 1:
            raise ValueError("All splits must use the same covariates.")
        ids = np.concatenate((self.train.row_id, self.calibration.row_id, self.evaluation.row_id))
        if np.unique(ids).size != ids.size:
            raise ValueError("Train, calibration and evaluation row IDs must be disjoint.")
        for sample in (self.train, self.calibration):
            validate_overlap(sample, self.overlap)


def validate_overlap(sample: LoggedSample, overlap: float) -> None:
    if not 0 < overlap <= 0.5:
        raise ValueError("Overlap must lie in (0, 0.5].")
    eps = np.finfo(float).eps
    if np.any((sample.propensity < overlap - eps) | (sample.propensity > 1 - overlap + eps)):
        raise ValueError("Observed design propensities violate the declared overlap bound.")
