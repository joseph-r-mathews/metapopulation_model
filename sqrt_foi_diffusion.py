"""Distributional conditional diffusion for K=16 square-root FPCA scores."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from foi_basis import BASIS_PATH, K, N_STATES, reconstruct_external_foi


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "foi_sqrt_fpca_k16_m4_best.pt"
N_WEEKS = 52


def fit_transforms(coefficients, observations):
    coefficient = np.asarray(coefficients, dtype=np.float32)
    log_y = np.log1p(np.asarray(observations, dtype=np.float32))
    return {
        "coefficient_mean": coefficient.mean(axis=0, keepdims=True),
        "coefficient_std": coefficient.std(axis=0, keepdims=True).clip(1e-6),
        "y_mean": log_y.mean(axis=(0, 2), keepdims=True),
        "y_std": log_y.std(axis=(0, 2), keepdims=True).clip(1e-6),
    }


def transform_coefficients(coefficients, stats):
    values = (np.asarray(coefficients, dtype=np.float32) - stats["coefficient_mean"])
    values = values / stats["coefficient_std"]
    return torch.from_numpy(values.astype(np.float32))


def inverse_coefficients(coefficients, stats):
    return (
        np.asarray(coefficients, dtype=np.float64) * stats["coefficient_std"]
        + stats["coefficient_mean"]
    )


def transform_y(observations, stats):
    values = np.log1p(np.asarray(observations, dtype=np.float32))
    values = (values - stats["y_mean"]) / stats["y_std"]
    return torch.from_numpy(values.astype(np.float32))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time(F.silu(time))[:, :, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(hidden)))


class CoefficientUNet(nn.Module):
    """Existing m=4 denoiser adapted to C:[batch, 51, 16]."""

    def __init__(self, base=64):
        super().__init__()
        self.base = base
        self.condition = nn.Sequential(
            nn.Linear(N_WEEKS, 64), nn.SiLU(), nn.Linear(64, K)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(base, 4 * base),
            nn.SiLU(),
            nn.Linear(4 * base, 4 * base),
        )
        self.input = nn.Conv1d(3 * N_STATES, base, 3, padding=1)
        self.enc1 = ResidualBlock(base, base, 4 * base)
        self.down1 = nn.Conv1d(base, 2 * base, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(2 * base, 2 * base, 4 * base)
        self.down2 = nn.Conv1d(2 * base, 2 * base, 4, stride=2, padding=1)
        self.middle = ResidualBlock(2 * base, 2 * base, 4 * base)
        self.dec2 = ResidualBlock(4 * base, 2 * base, 4 * base)
        self.dec1 = ResidualBlock(3 * base, base, 4 * base)
        self.output = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv1d(base, N_STATES, 3, padding=1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, coefficient, step, observations, xi):
        if coefficient.shape[1:] != (N_STATES, K) or xi.shape != coefficient.shape:
            raise ValueError("Coefficient and xi must have shape [batch, 51, 16]")
        if observations.shape != (len(coefficient), N_STATES, N_WEEKS):
            raise ValueError("Y condition must have shape [batch, 51, 52]")
        frequency = torch.exp(
            -math.log(10000)
            * torch.arange(self.base // 2, device=coefficient.device)
            / (self.base // 2)
        )
        phase = step[:, None].float() * frequency[None]
        time = self.time_mlp(torch.cat((phase.cos(), phase.sin()), dim=1))
        condition = self.condition(observations)
        first = self.enc1(
            self.input(torch.cat((coefficient, condition, xi), dim=1)), time
        )
        second = self.enc2(self.down1(first), time)
        hidden = self.middle(self.down2(second), time)
        hidden = self.dec2(
            torch.cat((F.interpolate(hidden, size=8, mode="nearest"), second), dim=1),
            time,
        )
        hidden = self.dec1(
            torch.cat((F.interpolate(hidden, size=K, mode="nearest"), first), dim=1),
            time,
        )
        return self.output(hidden)


def diffusion_schedule(steps, device):
    time = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999).float().to(device)
    alpha = 1 - beta
    alpha_bar = alpha.cumprod(0)
    previous = torch.cat((torch.ones(1, device=device), alpha_bar[:-1]))
    variance = beta * (1 - previous) / (1 - alpha_bar)
    return beta, alpha, alpha_bar, variance


def corrupt(clean, step, noise, schedule):
    alpha_bar = schedule[2][step, None, None]
    return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise


ENERGY_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def energy_loss(truth, *draws):
    """The established m=4, beta=1, lambda=1 energy-score loss."""
    if len(draws) != 4 or truth.shape[1:] != (N_STATES, K):
        raise ValueError("Energy loss requires truth and four [batch, 51, 16] draws")
    truth_term = sum(
        torch.linalg.vector_norm((draw - truth).flatten(1), dim=1) for draw in draws
    ) / 4
    pair_term = sum(
        torch.linalg.vector_norm((draws[j] - draws[k]).flatten(1), dim=1)
        for j, k in ENERGY_PAIRS
    ) / 12
    return (truth_term - pair_term).mean()


def bridge_coefficients(schedule):
    beta, alpha, alpha_bar, variance = schedule
    previous = torch.cat((torch.ones_like(alpha_bar[:1]), alpha_bar[:-1]))
    clean_weight = previous.sqrt() * beta / (1 - alpha_bar)
    noisy_weight = alpha.sqrt() * (1 - previous) / (1 - alpha_bar)
    clean_weight[0], noisy_weight[0] = 1, 0
    return clean_weight, noisy_weight, variance


@torch.no_grad()
def reverse_sample(model, observations, schedule, *, generator=None):
    def normal():
        return torch.randn(
            (len(observations), N_STATES, K),
            dtype=observations.dtype,
            device=observations.device,
            generator=generator,
        )

    clean_weight, noisy_weight, variance = bridge_coefficients(schedule)
    coefficient = normal()
    for level in reversed(range(len(schedule[0]))):
        step = torch.full(
            (len(observations),), level, device=observations.device, dtype=torch.long
        )
        clean = model(coefficient, step, observations, normal())
        coefficient = clean_weight[level] * clean + noisy_weight[level] * coefficient
        if level > 0:
            coefficient = coefficient + variance[level].sqrt() * normal()
    return coefficient


def load_foi_model(checkpoint=DEFAULT_CHECKPOINT, *, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    expected = {
        "method": "distributional_energy_sqrt_fpca",
        "target": "continuous_instantaneous_external_foi",
        "K": K,
        "target_dimension": N_STATES * K,
        "m": 4,
        "beta_energy": 1,
        "lambda_energy": 1,
        "steps": 100,
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Checkpoint {key} must be {value!r}")
    model = CoefficientUNet(saved["base"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    stats = {key: value.numpy() for key, value in saved["stats"].items()}
    return model, stats, diffusion_schedule(saved["steps"], device)


def draw_coefficients(
    observations,
    n_samples,
    *,
    checkpoint=DEFAULT_CHECKPOINT,
    device=None,
    seed=None,
    batch_size=64,
):
    Y = np.asarray(observations)
    if Y.shape != (N_STATES, N_WEEKS):
        raise ValueError("Y must have shape [51, 52]")
    if not np.isfinite(Y).all() or np.any(Y < 0):
        raise ValueError("Y must contain finite nonnegative counts")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be positive")
    model, stats, schedule = load_foi_model(checkpoint, device=device)
    target_device = next(model.parameters()).device
    condition = transform_y(Y[None], stats).to(target_device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=target_device).manual_seed(seed)
    batches = []
    for start in range(0, n_samples, batch_size):
        count = min(batch_size, n_samples - start)
        normalized = reverse_sample(
            model, condition.repeat(count, 1, 1), schedule, generator=generator
        )
        batches.append(normalized.cpu().numpy())
    return inverse_coefficients(np.concatenate(batches), stats)


@dataclass
class ExternalFoISamples:
    """Posterior coefficient draws that can be evaluated at continuous times."""

    coefficients: np.ndarray
    basis_path: Path = BASIS_PATH

    def evaluate(self, times):
        return reconstruct_external_foi(self.coefficients, times, self.basis_path)

    __call__ = evaluate


def sample_external_foi(observations, n_samples, **kwargs):
    """Sample C|Y and return callable continuous physical FoI curves."""
    coefficients = draw_coefficients(observations, n_samples, **kwargs)
    return ExternalFoISamples(coefficients)


__all__ = [
    "DEFAULT_CHECKPOINT",
    "ExternalFoISamples",
    "draw_coefficients",
    "reconstruct_external_foi",
    "sample_external_foi",
]
