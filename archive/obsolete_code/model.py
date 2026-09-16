"""Shared weekly-scale metapopulation model and stochastic tau-leap simulator."""

import csv
from pathlib import Path

import numpy as np
from scipy.special import gammaln


N_WEEKS = 52
TAU_DT = 1 / 224
GAMMA = 1.0
P_A = 0.5
P_REPORT = 0.4
PRIOR_LOG_MEAN = np.log(2.1)
PRIOR_LOG_SD = 0.20
INITIAL_INFECTED = {"CA": 100, "TX": 100, "FL": 100, "NY": 100}


def load_data(directory):
    """Return state labels, subpopulation codes, populations, and raw mobility."""
    directory = Path(directory)
    with (directory / "geodata_2019.csv").open() as stream:
        states = list(csv.DictReader(stream))
    labels = np.array([row["USPS"] for row in states])
    subpop = np.array([row["subpop"] for row in states])
    N = np.array([int(row["population"]) for row in states], dtype=np.int64)
    index = {code: i for i, code in enumerate(subpop)}
    M = np.zeros((len(N), len(N)), dtype=np.int64)
    with (directory / "mobility_2011-2015_statelevel.csv").open() as stream:
        for row in csv.DictReader(stream):
            M[index[row["ori"]], index[row["dest"]]] += int(row["amount"])
    np.fill_diagonal(M, 0)
    return labels, subpop, N, M


def initial_infected(labels):
    return np.array([INITIAL_INFECTED.get(label, 0) for label in labels],
                    dtype=np.int64)


def force_of_infection(I, beta, N, M, p_a=P_A):
    """Return total, local, and external FoI, all with units of inverse weeks."""
    pressure = beta * I / N
    local = (1 - p_a * M.sum(axis=1) / N) * pressure
    external = p_a * (M @ pressure) / N
    return local + external, local, external


def poisson_log_likelihood(Y, predicted_incidence, p_report=P_REPORT, axis=None):
    """Return log p(Y | p_report * predicted_incidence) under one Poisson model."""
    Y = np.asarray(Y)
    mean = p_report * np.asarray(predicted_incidence, dtype=np.float64)
    if np.any(Y < 0) or np.any(Y != np.floor(Y)):
        raise ValueError("Y must contain nonnegative integer counts")
    if np.any(mean < 0) or not np.isfinite(mean).all():
        raise ValueError("Predicted Poisson means must be finite and nonnegative")
    result = -mean - gammaln(Y + 1)
    positive = Y > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = result + np.where(positive, Y * np.log(mean), 0.)
    total = result.sum(axis=axis)
    return float(total) if np.ndim(total) == 0 else total


def simulate(beta, rng, *, N, M, I0, n_weeks=N_WEEKS, dt=TAU_DT,
             gamma=GAMMA, p_a=P_A):
    """Stochastic tau leap of the continuous-time rates, reported weekly.

    The weekly external FoI is the left-endpoint tau-leap integral over the
    week. Since a week has length one, it is also the weekly average.
    """
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), (len(N),))
    steps_per_week = round(1 / dt)
    if not np.isclose(steps_per_week * dt, 1):
        raise ValueError("dt must divide one week exactly")
    if np.any(beta < 0) or not np.isfinite(beta).all():
        raise ValueError("beta must be finite and nonnegative")
    if np.any(p_a * M.sum(axis=1) > N):
        raise ValueError("Commuter weights exceed one; local weight is negative")
    S = np.empty((len(N), n_weeks + 1), dtype=np.int64)
    I = np.empty_like(S)
    R = np.empty_like(S)
    A = np.zeros((len(N), n_weeks), dtype=np.int64)
    B = np.zeros_like(A)
    local_foi = np.zeros((len(N), n_weeks), dtype=np.float64)
    external_foi = np.zeros_like(local_foi)
    total_foi = np.zeros_like(local_foi)
    current_S, current_I, current_R = N - I0, I0.copy(), np.zeros_like(I0)
    S[:, 0], I[:, 0], R[:, 0] = current_S, current_I, current_R
    recovery_probability = -np.expm1(-gamma * dt)
    for week in range(n_weeks):
        for _ in range(steps_per_week):
            total, local, external = force_of_infection(current_I, beta, N, M, p_a)
            infections = rng.binomial(current_S, -np.expm1(-total * dt))
            recoveries = rng.binomial(current_I, recovery_probability)
            current_S = current_S - infections
            current_I = current_I + infections - recoveries
            current_R = current_R + recoveries
            A[:, week] += infections
            B[:, week] += recoveries
            total_foi[:, week] += total * dt
            local_foi[:, week] += local * dt
            external_foi[:, week] += external * dt
        S[:, week + 1], I[:, week + 1], R[:, week + 1] = (
            current_S, current_I, current_R
        )
    return dict(S=S, I=I, R=R, A=A, B=B, local_foi=local_foi,
                external_foi=external_foi, total_foi=total_foi)


def observe(predicted_incidence, rng, p_report=P_REPORT):
    """Draw the shared Poisson observation model from weekly incidence."""
    return rng.poisson(p_report * np.asarray(predicted_incidence, dtype=np.float64))


def simulate_epidemic(beta, epidemic_rng, reporting_rng, *, N, M, I0,
                      n_weeks=N_WEEKS, dt=TAU_DT, gamma=GAMMA, p_a=P_A,
                      p_report=P_REPORT):
    result = simulate(beta, epidemic_rng, N=N, M=M, I0=I0, n_weeks=n_weeks,
                      dt=dt, gamma=gamma, p_a=p_a)
    result["Y"] = observe(result["A"], reporting_rng, p_report)
    result["beta_true"] = np.asarray(beta, dtype=np.float64).copy()
    return result


def check_epidemic(result, N):
    for name in ("S", "I", "R"):
        assert result[name].shape == (51, N_WEEKS + 1)
    for name in ("A", "B", "Y", "local_foi", "external_foi", "total_foi"):
        assert result[name].shape == (51, N_WEEKS)
    assert all(np.issubdtype(result[name].dtype, np.integer)
               for name in ("S", "I", "R", "A", "B", "Y"))
    assert all(np.isfinite(result[name]).all() and np.all(result[name] >= 0)
               for name in result)
    np.testing.assert_array_equal(
        result["S"] + result["I"] + result["R"],
        np.broadcast_to(N[:, None], result["S"].shape),
    )
    np.testing.assert_array_equal(result["S"][:, :-1] - result["S"][:, 1:], result["A"])
    np.testing.assert_array_equal(result["R"][:, 1:] - result["R"][:, :-1], result["B"])
    np.testing.assert_allclose(result["total_foi"],
                               result["local_foi"] + result["external_foi"],
                               rtol=0, atol=2e-15)
