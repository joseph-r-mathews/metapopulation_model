"""Likelihood, prior, and frozen-external-FoI proposal utilities.

The coupled target always uses the full continuous-time epidemic model from
``epidemic_model``. Frozen external force of infection is used only to build
local one-dimensional proposal densities.
"""

import math

import numpy as np
from scipy.special import gammaln

from epidemic_model import (
    GAMMA,
    N_STATES,
    N_WEEKS,
    P_A,
    P_REPORT,
    PRIOR_LOG_MEAN,
    PRIOR_LOG_SD,
    evaluate_external_foi,
    solve_coupled,
)


COARSE_POINTS = 501
FINAL_POINTS = 2001
PROPOSAL_FINAL_POINTS = 4001
LOCAL_SUBSTEPS_PER_WEEK = 14


def poisson_log_likelihood(observed, incidence, axis=None):
    observed = np.asarray(observed)
    mean = P_REPORT * np.asarray(incidence, dtype=np.float64)
    result = -mean - gammaln(observed + 1)
    positive = observed > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result += np.where(positive, observed * np.log(mean), 0.0)
    total = result.sum(axis=axis)
    return float(total) if np.ndim(total) == 0 else total


def log_prior_eta(eta):
    eta = np.asarray(eta)
    return (
        -0.5 * ((eta - PRIOR_LOG_MEAN) / PRIOR_LOG_SD) ** 2
        - math.log(PRIOR_LOG_SD * math.sqrt(2 * math.pi))
    )


def coupled_log_likelihood(eta, populations, mobility, I0, observations):
    solved = solve_coupled(
        np.exp(eta),
        populations=populations,
        mobility=mobility,
        I0=I0,
        dense_output=False,
    )
    return poisson_log_likelihood(observations, solved["incidence"])


def coupled_log_posterior(eta, populations, mobility, I0, observations):
    return coupled_log_likelihood(
        eta, populations, mobility, I0, observations
    ) + float(log_prior_eta(eta).sum())


def external_foi_stages(solution, beta, populations, mobility):
    """Evaluate continuous external FoI at every local RK4 stage."""
    dt = 1.0 / LOCAL_SUBSTEPS_PER_WEEK
    starts = np.arange(N_WEEKS * LOCAL_SUBSTEPS_PER_WEEK) * dt
    times = np.stack((starts, starts + 0.5 * dt, starts + dt), axis=1)
    external = evaluate_external_foi(
        solution, times.ravel(), beta, populations, mobility
    )
    return external.reshape(N_STATES, len(starts), 3)


def local_incidence_grid(eta, state, external_stages, populations, mobility, I0):
    """Solve one local epidemic with the continuous external FoI frozen."""
    eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    beta = np.exp(eta)
    population = float(populations[state])
    local_weight = 1.0 - P_A * mobility[state].sum() / population
    susceptible = np.full(len(eta), population - I0[state], dtype=np.float64)
    infectious = np.full(len(eta), I0[state], dtype=np.float64)
    incidence = np.empty((len(eta), N_WEEKS), dtype=np.float64)
    dt = 1.0 / LOCAL_SUBSTEPS_PER_WEEK

    def rhs(current_s, current_i, external):
        rate = local_weight * beta * current_i / population + external
        infection_flow = current_s * rate
        return -infection_flow, infection_flow - GAMMA * current_i

    step = 0
    for week in range(N_WEEKS):
        start_susceptible = susceptible.copy()
        for _ in range(LOCAL_SUBSTEPS_PER_WEEK):
            h0, hmid, hend = external_stages[state, step]
            k1s, k1i = rhs(susceptible, infectious, h0)
            k2s, k2i = rhs(
                susceptible + 0.5 * dt * k1s,
                infectious + 0.5 * dt * k1i,
                hmid,
            )
            k3s, k3i = rhs(
                susceptible + 0.5 * dt * k2s,
                infectious + 0.5 * dt * k2i,
                hmid,
            )
            k4s, k4i = rhs(
                susceptible + dt * k3s,
                infectious + dt * k3i,
                hend,
            )
            susceptible += dt * (k1s + 2 * k2s + 2 * k3s + k4s) / 6
            infectious += dt * (k1i + 2 * k2i + 2 * k3i + k4i) / 6
            step += 1
        incidence[:, week] = start_susceptible - susceptible
    if not np.isfinite(incidence).all() or np.min(incidence) < -1e-6:
        raise FloatingPointError(f"Invalid local integration for state {state}")
    return np.maximum(incidence, 0.0)


def local_log_likelihood(
    eta, state, external_stages, observations, populations, mobility, I0
):
    incidence = local_incidence_grid(
        eta, state, external_stages, populations, mobility, I0
    )
    return poisson_log_likelihood(
        observations[state][None], incidence, axis=1
    )


def local_log_posterior(
    eta, state, external_stages, observations, populations, mobility, I0
):
    likelihood = local_log_likelihood(
        eta,
        state,
        external_stages,
        observations,
        populations,
        mobility,
        I0,
    )
    return log_prior_eta(eta) + likelihood, likelihood


def trapezoid_distribution(grid, log_density):
    peak = np.max(log_density)
    relative = np.exp(log_density - peak)
    segment_mass = np.diff(grid) * (relative[:-1] + relative[1:]) / 2
    area = segment_mass.sum()
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("Numerical posterior has zero or nonfinite mass")
    density = relative / area
    cdf = np.r_[0.0, np.cumsum(segment_mass) / area]
    cdf[-1] = 1.0
    return density, cdf, peak + math.log(area)


def posterior_moments(grid, density):
    mean = np.trapezoid(grid * density, grid)
    variance = np.trapezoid((grid - mean) ** 2 * density, grid)
    return float(mean), float(np.sqrt(max(variance, 0.0)))


def construct_local_posterior(
    state, external_stages, observations, populations, mobility, I0
):
    half_width = 6 * PRIOR_LOG_SD
    for _ in range(10):
        coarse = np.linspace(
            PRIOR_LOG_MEAN - half_width,
            PRIOR_LOG_MEAN + half_width,
            COARSE_POINTS,
        )
        coarse_log, _ = local_log_posterior(
            coarse,
            state,
            external_stages,
            observations,
            populations,
            mobility,
            I0,
        )
        if np.isfinite(coarse_log).any():
            boundary_ratio = np.exp(
                np.max(coarse_log[[0, -1]]) - np.max(coarse_log)
            )
            if boundary_ratio <= 1e-8:
                break
        half_width *= 1.5
    else:
        raise RuntimeError(f"State {state}: coarse grid did not contain posterior")

    coarse_density, _, _ = trapezoid_distribution(coarse, coarse_log)
    _, coarse_sd = posterior_moments(coarse, coarse_density)
    mode = coarse[np.argmax(coarse_log)]
    refined_half_width = max(8 * coarse_sd, 0.05)
    for _ in range(12):
        grid = np.linspace(
            mode - refined_half_width,
            mode + refined_half_width,
            FINAL_POINTS,
        )
        log_density, log_likelihood = local_log_posterior(
            grid,
            state,
            external_stages,
            observations,
            populations,
            mobility,
            I0,
        )
        boundary_ratio = np.exp(
            np.max(log_density[[0, -1]]) - np.max(log_density)
        )
        if boundary_ratio <= 1e-8:
            break
        refined_half_width *= 1.5
    else:
        raise RuntimeError(f"State {state}: refined grid boundaries did not decay")
    density, cdf, log_normalizer = trapezoid_distribution(grid, log_density)
    return {
        "grid": grid,
        "density": density,
        "cdf": cdf,
        "log_likelihood": log_likelihood,
        "log_normalizer": log_normalizer,
        "boundary_ratio": float(boundary_ratio),
    }


def refine_local_posterior(
    posterior, state, external_stages, observations, populations, mobility, I0
):
    """Resolve a narrow local proposal on the established 4001-point grid."""
    _, initial_sd = posterior_moments(
        posterior["grid"], posterior["density"]
    )
    mode = posterior["grid"][np.argmax(posterior["density"])]
    half_width = 12 * initial_sd
    for _ in range(8):
        grid = np.linspace(
            mode - half_width, mode + half_width, PROPOSAL_FINAL_POINTS
        )
        log_density, log_likelihood = local_log_posterior(
            grid,
            state,
            external_stages,
            observations,
            populations,
            mobility,
            I0,
        )
        boundary_ratio = np.exp(
            np.max(log_density[[0, -1]]) - np.max(log_density)
        )
        if boundary_ratio <= 1e-10:
            density, cdf, log_normalizer = trapezoid_distribution(
                grid, log_density
            )
            return {
                "grid": grid,
                "density": density,
                "cdf": cdf,
                "log_likelihood": log_likelihood,
                "log_normalizer": log_normalizer,
                "boundary_ratio": float(boundary_ratio),
            }
        half_width *= 1.5
    raise RuntimeError(f"State {state}: proposal refinement did not contain posterior")


def construct_local_proposals(
    external_stages, observations, populations, mobility, I0
):
    proposals = []
    for state in range(N_STATES):
        posterior = construct_local_posterior(
            state, external_stages, observations, populations, mobility, I0
        )
        proposals.append(
            refine_local_posterior(
                posterior,
                state,
                external_stages,
                observations,
                populations,
                mobility,
                I0,
            )
        )
    return proposals


def sample_piecewise_linear_with_log_density(posterior, uniforms):
    """Sample a numerical local proposal and return its log density."""
    grid, density, cdf = (
        posterior[key] for key in ("grid", "density", "cdf")
    )
    uniforms = np.asarray(uniforms, dtype=np.float64)
    index = np.searchsorted(cdf, uniforms, side="right") - 1
    index = np.clip(index, 0, len(grid) - 2)
    width = grid[index + 1] - grid[index]
    d0, d1 = density[index], density[index + 1]
    slope = (d1 - d0) / width
    mass = uniforms - cdf[index]
    discriminant = np.maximum(d0 * d0 + 2 * slope * mass, 0.0)
    denominator = d0 + np.sqrt(discriminant)
    offset = np.where(
        np.abs(slope) < 1e-14,
        mass / np.maximum(d0, np.finfo(float).tiny),
        2 * mass / np.maximum(denominator, np.finfo(float).tiny),
    )
    offset = np.clip(offset, 0, width)
    values = grid[index] + offset
    evaluated_density = d0 + slope * offset
    log_density = np.log(
        np.maximum(evaluated_density, np.finfo(float).tiny)
    )
    return values, log_density


def evaluate_piecewise_linear_log_density(posterior, values):
    """Evaluate a normalized numerical local proposal density."""
    values = np.asarray(values, dtype=np.float64)
    density = np.interp(
        values,
        posterior["grid"],
        posterior["density"],
        left=0.0,
        right=0.0,
    )
    with np.errstate(divide="ignore"):
        return np.log(density)
