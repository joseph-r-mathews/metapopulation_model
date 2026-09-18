"""Loading and training-only preprocessing for local-flow datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stratified_model import PARAMETERS_PER_LOCATION, INCIDENCE_CHANNELS, FOI_CHANNELS


SPLITS = ("train", "validation", "test")


def load_split(directory, split):
    """Load one simulation-ID split from incremental NPZ shards."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    shards = sorted(Path(directory).glob(f"{split}_simulation_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No {split} shards found under {directory}")
    collected = {
        key: []
        for key in ("theta", "Y", "h", "covariates", "simulation_id", "location_id")
    }
    for shard in shards:
        with np.load(shard) as values:
            if (values["theta"].ndim != 2 or values["theta"].shape[1] != PARAMETERS_PER_LOCATION
                    or values["Y"].ndim != 3 or values["Y"].shape[1] != INCIDENCE_CHANNELS
                    or values["h"].ndim != 3 or values["h"].shape[1] != FOI_CHANNELS):
                raise ValueError(f"Dataset does not match the active model: {shard}")
            for key in collected:
                collected[key].append(values[key])
    return {key: np.concatenate(value) for key, value in collected.items()}


def _stable_standard_deviation(values, axis):
    standard_deviation = np.std(values, axis=axis)
    return np.where(standard_deviation > 1e-8, standard_deviation, 1.0)


@dataclass
class PreprocessingStats:
    theta_mean: np.ndarray
    theta_sd: np.ndarray
    incidence_mean: np.ndarray
    incidence_sd: np.ndarray
    foi_mean: np.ndarray
    foi_sd: np.ndarray
    covariate_mean: np.ndarray
    covariate_sd: np.ndarray

    @classmethod
    def from_training_split(cls, training):
        """Compute every statistic using training examples only."""
        theta = training["theta"].astype(np.float64)
        incidence = np.log1p(training["Y"].astype(np.float64))
        foi = np.log1p(training["h"].astype(np.float64))
        covariates = training["covariates"].astype(np.float64)
        return cls(
            theta_mean=theta.mean(axis=0),
            theta_sd=_stable_standard_deviation(theta, axis=0),
            incidence_mean=incidence.mean(axis=(0, 2)),
            incidence_sd=_stable_standard_deviation(incidence, axis=(0, 2)),
            foi_mean=foi.mean(axis=(0, 2)),
            foi_sd=_stable_standard_deviation(foi, axis=(0, 2)),
            covariate_mean=covariates.mean(axis=0),
            covariate_sd=_stable_standard_deviation(covariates, axis=0),
        )

    def as_dict(self):
        return {
            name: np.asarray(getattr(self, name), dtype=np.float32)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, values):
        return cls(
            **{
                name: np.asarray(values[name], dtype=np.float32)
                for name in cls.__dataclass_fields__
            }
        )

    @property
    def log_absolute_theta_jacobian(self):
        """log |d theta / d z| for theta = mean + sd*z."""
        return float(np.log(self.theta_sd).sum())

    def transform_numpy(self, values):
        return {
            "theta": (
                values["theta"].astype(np.float32) - self.theta_mean
            )
            / self.theta_sd,
            "Y": (
                np.log1p(values["Y"].astype(np.float32))
                - self.incidence_mean[None, :, None]
            )
            / self.incidence_sd[None, :, None],
            "h": (
                np.log1p(values["h"].astype(np.float32))
                - self.foi_mean[None, :, None]
            )
            / self.foi_sd[None, :, None],
            "covariates": (
                values["covariates"].astype(np.float32) - self.covariate_mean
            )
            / self.covariate_sd,
        }
