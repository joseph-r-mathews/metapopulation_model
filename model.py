"""Daily stochastic metapopulation SIR; rows are times, columns are states."""

import csv
from pathlib import Path

import numpy as np


def load_data(directory):
    """Return USPS labels, subpop codes, populations N, and raw directed M."""
    directory = Path(directory)
    with (directory / "geodata_2019.csv").open() as f:
        states = list(csv.DictReader(f))
    labels = np.array([row["USPS"] for row in states])
    subpop = np.array([row["subpop"] for row in states])  # Preserve leading zeros.
    N = np.array([int(row["population"]) for row in states], dtype=np.int64)
    index = {code: i for i, code in enumerate(subpop)}
    M = np.zeros((len(N), len(N)), dtype=np.int64)
    with (directory / "mobility_2011-2015_statelevel.csv").open() as f:
        for row in csv.DictReader(f):
            M[index[row["ori"]], index[row["dest"]]] += int(row["amount"])
    np.fill_diagonal(M, 0)  # Only j != i contributes to mobility coupling.
    return labels, subpop, N, M


def force_of_infection(I, theta, N, M, p_a):
    """Return total, local, external FoI (/day); M has zero diagonal.

    M[i, j] counts residents of i commuting to j, without row normalization.
    Destination j supplies both beta_j and I_j / N_j in the external term.
    """
    pressure = theta * I / N
    local = (1 - p_a * M.sum(axis=1) / N) * pressure
    external = p_a * (M @ pressure) / N
    foi = local + external
    return foi, local, external


def simulate(theta, n_days, rng, *, N, M, I0, gamma, p_a):
    """Return S/I/R (n_days+1, states) and five daily arrays (n_days, states).

    All initial recovered counts are zero. Day t events use X(t) and produce
    X(t+1); newly infected people cannot recover in that same time step.
    gamma and p_a are inputs, with experiment defaults set in the driver.
    """
    if np.any(p_a * M.sum(axis=1) > N):
        raise ValueError("Commuter weights exceed one; local weight is negative.")
    S = np.empty((n_days + 1, len(N)), dtype=np.int64)
    I = np.empty_like(S)
    R = np.empty_like(S)
    S[0], I[0], R[0] = N - I0, I0, 0
    incidence = np.empty((n_days, len(N)), dtype=np.int64)
    recoveries = np.empty_like(incidence)
    foi = np.empty((n_days, len(N)))
    foi_local = np.empty_like(foi)
    foi_external = np.empty_like(foi)
    recovery_probability = -np.expm1(-gamma)  # 1 - exp(-gamma * 1 day)

    for t in range(n_days):
        foi[t], foi_local[t], foi_external[t] = force_of_infection(
            I[t], theta, N, M, p_a
        )
        A = rng.binomial(S[t], -np.expm1(-foi[t]))
        B = rng.binomial(I[t], recovery_probability)
        S[t + 1] = S[t] - A
        I[t + 1] = I[t] + A - B
        R[t + 1] = R[t] + B
        incidence[t], recoveries[t] = A, B

    return dict(S=S, I=I, R=R, incidence=incidence, recoveries=recoveries,
                foi=foi, foi_local=foi_local, foi_external=foi_external)


def weekly_summary(result):
    """Sum daily incidence and average daily FoI over consecutive full weeks."""
    n_days, n_states = result["incidence"].shape
    if n_days % 7:
        raise ValueError("Weekly summaries require a whole number of weeks.")
    shape = (n_days // 7, 7, n_states)
    weekly = {"incidence": result["incidence"].reshape(shape).sum(axis=1)}
    for name in ("foi", "foi_local", "foi_external"):
        weekly[name] = result[name].reshape(shape).mean(axis=1)
    return weekly


def observe(weekly_incidence, p_report, rng):
    """Independent binomial reporting, separate from disease dynamics."""
    return rng.binomial(weekly_incidence, p_report)


def simulate_epidemic(theta, n_days, rng, *, N, M, I0, gamma, p_a,
                      p_report, reporting_rng=None):
    """Return one simulated epidemic and its MCMC-relevant truth in one dict.

    Weekly arrays use the diffusion convention ``[state, week]``. ``X`` has
    shape ``[day, state, compartment]``, with S/I/R on its final axis.
    """
    daily = simulate(theta, n_days, rng, N=N, M=M, I0=I0,
                     gamma=gamma, p_a=p_a)
    weekly = weekly_summary(daily)
    reporting_rng = rng if reporting_rng is None else reporting_rng
    Y = observe(weekly["incidence"], p_report, reporting_rng).T
    return {
        "X": np.stack((daily["S"], daily["I"], daily["R"]), axis=-1),
        "Y": Y,
        "local_foi": weekly["foi_local"].T,
        "true_external_foi": weekly["foi_external"].T,
        "true_beta": np.asarray(theta).copy(),
    }
