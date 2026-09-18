"""Reusable conditional spline flow; no trained model or dataset is supplied."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from torch import nn
from torch.nn import functional as F

from local_flow_data import PreprocessingStats
from stratified_model import PARAMETERS_PER_LOCATION, INCIDENCE_CHANNELS, FOI_CHANNELS


@dataclass
class FlowConfig:
    dimension: int = PARAMETERS_PER_LOCATION
    transforms: int = 6
    hidden_features: int = 128
    hidden_layers: int = 2
    spline_bins: int = 8
    tail_bound: float = 3.0
    incidence_embedding: int = 128
    foi_embedding: int = 128
    context_features: int = 128
    incidence_channels: int = INCIDENCE_CHANNELS
    foi_channels: int = FOI_CHANNELS
    covariate_features: int = 0  # supplied by future known-input encoding


class TemporalEncoder(nn.Module):
    def __init__(self, input_channels, first_kernel, first_stride, output_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                32,
                kernel_size=first_kernel,
                stride=first_stride,
                padding=first_kernel // 2,
            ),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_features),
            nn.GELU(),
        )

    def forward(self, values):
        return self.network(values)


class ConditioningEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.incidence = TemporalEncoder(config.incidence_channels, 5, 2, config.incidence_embedding)
        self.foi = TemporalEncoder(config.foi_channels, 9, 3, config.foi_embedding)
        combined = (
            config.incidence_embedding
            + config.foi_embedding
            + config.covariate_features
        )
        self.projection = nn.Sequential(
            nn.Linear(combined, 256),
            nn.GELU(),
            nn.Linear(256, config.context_features),
            nn.GELU(),
        )

    def forward(self, incidence, foi, covariates):
        combined = torch.cat(
            (self.incidence(incidence), self.foi(foi), covariates), dim=1
        )
        return self.projection(combined)


class ConditionalSplineFlow(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FlowConfig()
        self.encoder = ConditioningEncoder(self.config)
        transforms = []
        for _ in range(self.config.transforms):
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=self.config.dimension,
                    hidden_features=self.config.hidden_features,
                    context_features=self.config.context_features,
                    num_bins=self.config.spline_bins,
                    tails="linear",
                    tail_bound=self.config.tail_bound,
                    num_blocks=self.config.hidden_layers,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=F.gelu,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                )
            )
            transforms.append(ReversePermutation(features=self.config.dimension))
        self.flow = Flow(
            CompositeTransform(transforms),
            StandardNormal([self.config.dimension]),
        )

    def context(self, incidence, foi, covariates):
        return self.encoder(incidence, foi, covariates)

    def log_prob_standardized(self, theta, incidence, foi, covariates):
        context = self.context(incidence, foi, covariates)
        return self.flow.log_prob(theta, context=context)


class LocalPosteriorFlow:
    """Public original-coordinate sampling and normalized-density interface."""

    def __init__(self, model, preprocessing, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.preprocessing = preprocessing
        self._stats = {
            name: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for name, value in preprocessing.as_dict().items()
        }

    @classmethod
    def new(cls, preprocessing, config=None, device=None):
        if config is None:
            config = FlowConfig(covariate_features=len(preprocessing.covariate_mean))
        if (config.dimension != PARAMETERS_PER_LOCATION
                or config.incidence_channels != INCIDENCE_CHANNELS
                or config.foi_channels != FOI_CHANNELS):
            raise ValueError("Flow dimensions do not match the active scientific model")
        if (len(preprocessing.theta_mean) != config.dimension
                or len(preprocessing.incidence_mean) != config.incidence_channels
                or len(preprocessing.foi_mean) != config.foi_channels
                or len(preprocessing.covariate_mean) != config.covariate_features):
            raise ValueError("Preprocessing shapes do not match the flow configuration")
        return cls(ConditionalSplineFlow(config), preprocessing, device=device)

    @staticmethod
    def _batch(values, expected_rank, device):
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        single = tensor.ndim == expected_rank - 1
        if single:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != expected_rank:
            raise ValueError(f"Expected rank {expected_rank - 1} or {expected_rank}")
        return tensor, single

    def preprocess_context(self, Y, h, covariates):
        Y, single_y = self._batch(Y, 3, self.device)
        h, single_h = self._batch(h, 3, self.device)
        covariates, single_covariates = self._batch(covariates, 2, self.device)
        config = self.model.config
        if (Y.shape[1] != config.incidence_channels or h.shape[1] != config.foi_channels
                or covariates.shape[1] != config.covariate_features):
            raise ValueError("Expected flattened incidence/age-variant FoI channels and configured covariates")
        if not (len(Y) == len(h) == len(covariates)):
            raise ValueError("Y, h, and covariates must have matching batches")
        incidence = (
            torch.log1p(Y) - self._stats["incidence_mean"][None, :, None]
        ) / self._stats["incidence_sd"][None, :, None]
        foi = (
            torch.log1p(h) - self._stats["foi_mean"][None, :, None]
        ) / self._stats["foi_sd"][None, :, None]
        covariates = (
            covariates - self._stats["covariate_mean"]
        ) / self._stats["covariate_sd"]
        return incidence, foi, covariates, single_y and single_h and single_covariates

    def standardize_theta(self, theta):
        theta, single = self._batch(theta, 2, self.device)
        standardized = (
            theta - self._stats["theta_mean"]
        ) / self._stats["theta_sd"]
        return standardized, single

    @property
    def log_absolute_theta_jacobian(self):
        return torch.log(self._stats["theta_sd"]).sum()

    def log_prob(self, theta, Y, h, covariates):
        """Return normalized log q with respect to original theta coordinates."""
        standardized, single_theta = self.standardize_theta(theta)
        incidence, foi, covariates, single_context = self.preprocess_context(
            Y, h, covariates
        )
        if len(incidence) == 1 and len(standardized) > 1:
            incidence = incidence.expand(len(standardized), -1, -1)
            foi = foi.expand(len(standardized), -1, -1)
            covariates = covariates.expand(len(standardized), -1)
        if len(standardized) != len(incidence):
            raise ValueError("theta and context batch sizes do not match")
        self.model.eval()
        with torch.inference_mode():
            result = self.model.log_prob_standardized(
                standardized, incidence, foi, covariates
            ) - self.log_absolute_theta_jacobian
        values = result.detach().cpu().numpy()
        return float(values[0]) if single_theta and single_context else values

    def sample(self, Y, h, covariates, num_samples, rng=None):
        """Draw original-coordinate theta samples for one or more contexts."""
        incidence, foi, covariates, single = self.preprocess_context(
            Y, h, covariates
        )
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        seed = None
        if isinstance(rng, np.random.Generator):
            seed = int(rng.integers(0, 2**31 - 1))
        elif rng is not None:
            seed = int(rng)
        context_manager = (
            torch.random.fork_rng(devices=[self.device])
            if seed is not None and self.device.type == "cuda"
            else torch.random.fork_rng()
            if seed is not None
            else nullcontext()
        )
        self.model.eval()
        with context_manager, torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            context = self.model.context(incidence, foi, covariates)
            standardized = self.model.flow.sample(num_samples, context=context)
            values = (
                standardized * self._stats["theta_sd"]
                + self._stats["theta_mean"]
            )
        values = values.detach().cpu().numpy()
        return values[0] if single else values

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": asdict(self.model.config),
                "preprocessing": {
                    name: value.tolist()
                    for name, value in self.preprocessing.as_dict().items()
                },
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = FlowConfig(**checkpoint["config"])
        preprocessing = PreprocessingStats.from_dict(checkpoint["preprocessing"])
        wrapper = cls.new(preprocessing, config=config, device=device)
        wrapper.model.load_state_dict(checkpoint["model_state"])
        wrapper.model.eval()
        return wrapper, checkpoint.get("extra", {})
