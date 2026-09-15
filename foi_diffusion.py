"""Canonical m=4 conditional distributional model for external FoI."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent / "checkpoints" / "foi_distributional_m4_best.pt"
)


def fit_transforms(foi_ext, Y):
    """Pool over training simulations/weeks, retaining one scale per state.

    The log floor approximates exact zeros by 1e-10/day. exp is nonnegative
    everywhere, so reverse samples need no projection onto nonnegative curves.
    """
    log_foi = np.log(np.maximum(foi_ext, 1e-10))
    log_y = np.log1p(Y.astype(np.float32))
    stats = {"foi_mean": log_foi.mean((0, 2), keepdims=True),
            "foi_std": log_foi.std((0, 2), keepdims=True).clip(1e-6),
            "y_mean": log_y.mean((0, 2), keepdims=True),
            "y_std": log_y.std((0, 2), keepdims=True).clip(1e-6)}
    return stats


def transform_y(Y, stats):
    """Transform weekly reported incidence using training-set statistics."""
    y = (np.log1p(np.asarray(Y, dtype=np.float32)) - stats["y_mean"]) / stats["y_std"]
    return torch.from_numpy(y.astype(np.float32))


def transform(foi_ext, Y, stats):
    z = (np.log(np.maximum(foi_ext, 1e-10)) - stats["foi_mean"]) / stats["foi_std"]
    return torch.from_numpy(z.astype(np.float32)), transform_y(Y, stats)


def inverse_transform(z, stats):
    return np.exp(z.astype(np.float64) * stats["foi_std"] + stats["foi_mean"])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time))[:, :, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class TemporalUNet(nn.Module):
    """Stochastic denoiser G_theta(x_t, t, y, xi), with states as channels."""
    def __init__(self, base=64):
        super().__init__()
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, 4*base), nn.SiLU(), nn.Linear(4*base, 4*base))
        self.input = nn.Conv1d(153, base, 3, padding=1)
        self.enc1 = ResidualBlock(base, base, 4*base)
        self.down1 = nn.Conv1d(base, 2*base, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(2*base, 2*base, 4*base)
        self.down2 = nn.Conv1d(2*base, 2*base, 4, stride=2, padding=1)
        self.middle = ResidualBlock(2*base, 2*base, 4*base)
        self.dec2 = ResidualBlock(4*base, 2*base, 4*base)
        self.dec1 = ResidualBlock(3*base, base, 4*base)
        self.output = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv1d(base, 51, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, z, step, y, xi):
        assert z.shape == y.shape and z.ndim == 3 and z.shape[1:] == (51, 52)
        assert xi.shape == z.shape
        frequency = torch.exp(-math.log(10000) * torch.arange(self.base//2, device=z.device) / (self.base//2))
        phase = step[:, None].float() * frequency[None]
        time = self.time_mlp(torch.cat((phase.cos(), phase.sin()), dim=1))
        h1 = self.enc1(self.input(torch.cat((z, y, xi), dim=1)), time)
        h2 = self.enc2(self.down1(h1), time)
        h = self.middle(self.down2(h2), time)
        h = self.dec2(torch.cat((F.interpolate(h, size=26, mode="nearest"), h2), dim=1), time)
        h = self.dec1(torch.cat((F.interpolate(h, size=52, mode="nearest"), h1), dim=1), time)
        return self.output(h)



def diffusion_schedule(steps, device):
    """Standard cosine schedule; index 0 denotes the first noising step."""
    # Compute in float64 on CPU; MPS does not support float64 tensors.
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    cumulative = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=.999).float().to(device)
    alpha = 1 - beta
    alpha_bar = alpha.cumprod(0)
    previous = torch.cat((torch.ones(1, device=device), alpha_bar[:-1]))
    variance = beta * (1 - previous) / (1 - alpha_bar)
    return beta, alpha, alpha_bar, variance


def corrupt(z, step, noise, schedule):
    alpha_bar = schedule[2][step, None, None]
    return alpha_bar.sqrt() * z + (1 - alpha_bar).sqrt() * noise


ENERGY_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def energy_loss(x0, xhat1, xhat2, xhat3, xhat4):
    """m=4, lambda=beta=1: joint curve norms and six unordered pairs / 12."""
    draws = (xhat1, xhat2, xhat3, xhat4)
    assert x0.ndim == 3 and x0.shape[1:] == (51, 52)
    assert all(x.shape == x0.shape for x in draws)
    truth_term = sum(torch.linalg.vector_norm((x-x0).flatten(1), dim=1)
                     for x in draws) / 4
    pair_term = sum(torch.linalg.vector_norm((draws[j]-draws[k]).flatten(1), dim=1)
                    for j, k in ENERGY_PAIRS) / 12
    return (truth_term - pair_term).mean()


def bridge_coefficients(schedule):
    """q(x_{t-1} | x_t, x0), in the existing zero-based schedule indexing."""
    beta, alpha, alpha_bar, variance = schedule
    previous = torch.cat((torch.ones_like(alpha_bar[:1]), alpha_bar[:-1]))
    clean_weight = previous.sqrt() * beta / (1 - alpha_bar)
    noisy_weight = alpha.sqrt() * (1 - previous) / (1 - alpha_bar)
    # At the last reverse step q(x0 | x_t, x0_hat) is a point mass at x0_hat.
    clean_weight[0], noisy_weight[0] = 1, 0
    return clean_weight, noisy_weight, variance


@torch.no_grad()
def sample(model, y, schedule, *, generator=None):
    """100-level stochastic-clean-data / exact Gaussian bridge sampling.

    There is no clean-state clipping and fresh xi is sampled at every level.
    """
    def normal():
        return torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator)

    clean_weight, noisy_weight, variance = bridge_coefficients(schedule)
    x = normal()
    for t in reversed(range(len(schedule[0]))):
        step = torch.full((len(y),), t, device=y.device, dtype=torch.long)
        clean = model(x, step, y, normal())
        x = clean_weight[t] * clean + noisy_weight[t] * x
        if t > 0:
            x = x + variance[t].sqrt() * normal()
    return x


def load_foi_model(checkpoint=DEFAULT_CHECKPOINT, *, device=None):
    """Load and validate the one supported m=4 FoI checkpoint."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    expected = {
        "method": "distributional_energy",
        "m": 4,
        "beta_energy": 1,
        "lambda_energy": 1,
        "steps": 100,
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Checkpoint {key} must be {value!r}, got {saved.get(key)!r}")
    model = TemporalUNet(saved["base"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    stats = {key: value.numpy() for key, value in saved["stats"].items()}
    return model, stats, diffusion_schedule(saved["steps"], device)


def sample_external_foi(
    Y, n_samples, checkpoint=DEFAULT_CHECKPOINT, *, device=None, seed=None
):
    """Sample physical external FoI given weekly observations.

    A single ``Y`` has shape ``[51, 52]`` and returns ``[n_samples, 51, 52]``.
    Batched input ``[batch, 51, 52]`` returns
    ``[n_samples, batch, 51, 52]``.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    observations = np.asarray(Y)
    single = observations.ndim == 2
    if single:
        observations = observations[None]
    if observations.ndim != 3 or observations.shape[1:] != (51, 52):
        raise ValueError("Y must have shape [51, 52] or [batch, 51, 52]")
    if not np.isfinite(observations).all() or np.any(observations < 0):
        raise ValueError("Y must contain finite, nonnegative observations")

    model, stats, schedule = load_foi_model(checkpoint, device=device)
    condition = transform_y(observations, stats).to(next(model.parameters()).device)
    batch = len(condition)
    condition = condition.unsqueeze(0).expand(n_samples, -1, -1, -1)
    condition = condition.reshape(n_samples * batch, 51, 52)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=condition.device).manual_seed(seed)
    transformed = sample(model, condition, schedule, generator=generator)
    physical = inverse_transform(transformed.cpu().numpy(), stats)
    physical = physical.reshape(n_samples, batch, 51, 52)
    if not np.isfinite(physical).all():
        raise FloatingPointError("Diffusion sampler returned nonfinite physical FoI")
    return physical[:, 0] if single else physical
