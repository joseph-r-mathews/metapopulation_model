"""Train-only transforms, temporal U-Net, and ordinary epsilon-prediction DDPM."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
    stats["z_min"] = (log_foi.min((0, 2), keepdims=True) - stats["foi_mean"]) / stats["foi_std"]
    stats["z_max"] = (log_foi.max((0, 2), keepdims=True) - stats["foi_mean"]) / stats["foi_std"]
    return stats


def transform(foi_ext, Y, stats):
    z = (np.log(np.maximum(foi_ext, 1e-10)) - stats["foi_mean"]) / stats["foi_std"]
    y = (np.log1p(Y.astype(np.float32)) - stats["y_mean"]) / stats["y_std"]
    return torch.from_numpy(z.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


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
    """52 -> 26 -> 13 weeks; states are channels, conditioning is only Y."""
    def __init__(self, base=64):
        super().__init__()
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, 4*base), nn.SiLU(), nn.Linear(4*base, 4*base))
        self.input = nn.Conv1d(102, base, 3, padding=1)
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

    def forward(self, z, step, y):
        frequency = torch.exp(-math.log(10000) * torch.arange(self.base//2, device=z.device) / (self.base//2))
        phase = step[:, None].float() * frequency[None]
        time = self.time_mlp(torch.cat((phase.cos(), phase.sin()), dim=1))
        h1 = self.enc1(self.input(torch.cat((z, y), dim=1)), time)
        h2 = self.enc2(self.down1(h1), time)
        h = self.middle(self.down2(h2), time)
        h = self.dec2(torch.cat((F.interpolate(h, size=26, mode="nearest"), h2), dim=1), time)
        h = self.dec1(torch.cat((F.interpolate(h, size=52, mode="nearest"), h1), dim=1), time)
        # A direct full-resolution skip preserves all 51 noisy input channels.
        # The learned correction still predicts epsilon under the ordinary MSE.
        return z + self.output(h)


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


@torch.no_grad()
def sample(model, y, schedule, stats):
    """Ancestral DDPM with fixed posterior variance; no smoothing or guidance.

    Estimate z_0 from epsilon, apply standard DDPM clean-sample clipping to the
    training range, then use q(z_(s-1) | z_s, estimated z_0). This prevents
    early high-noise prediction errors from exploding under the inverse log.
    Final step has no noise. Bounds are fitted without validation/test data.
    """
    beta, alpha, alpha_bar, variance = schedule
    lower = torch.as_tensor(stats["z_min"], device=y.device)
    upper = torch.as_tensor(stats["z_max"], device=y.device)
    z = torch.randn_like(y)
    for s in reversed(range(len(beta))):
        step = torch.full((len(y),), s, device=y.device, dtype=torch.long)
        epsilon = model(z, step, y)
        clean = (z - (1-alpha_bar[s]).sqrt() * epsilon) / alpha_bar[s].sqrt()
        clean = torch.maximum(lower, torch.minimum(upper, clean))
        previous = alpha_bar[s-1] if s > 0 else torch.ones((), device=y.device)
        z = (previous.sqrt() * beta[s] * clean
             + alpha[s].sqrt() * (1-previous) * z) / (1-alpha_bar[s])
        if s > 0:
            z = z + variance[s].sqrt() * torch.randn_like(z)
    return z
