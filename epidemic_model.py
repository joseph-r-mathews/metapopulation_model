"""Authoritative continuous deterministic metapopulation epidemic model.

External FoI always means the instantaneous continuous function h_i(t).
"""

import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


N_STATES = 51
N_WEEKS = 52
GAMMA = 1.0
P_A = 0.5
P_REPORT = 0.4
PRIOR_LOG_MEAN = np.log(2.1)
PRIOR_LOG_SD = 0.20
INITIAL_INFECTED = {"CA": 100, "TX": 100, "FL": 100, "NY": 100}
RTOL = 1e-9
ATOL = 1e-5
WEEKLY_TIMES = np.arange(N_WEEKS + 1, dtype=np.float64)


def load_population_and_mobility(directory):
    """Return labels, subpopulation codes, populations, and raw mobility."""
    directory = Path(directory)
    with (directory / "geodata_2019.csv").open() as stream:
        states = list(csv.DictReader(stream))
    labels = np.array([row["USPS"] for row in states])
    subpop = np.array([row["subpop"] for row in states])
    populations = np.array(
        [int(row["population"]) for row in states], dtype=np.int64
    )
    index = {code: i for i, code in enumerate(subpop)}
    mobility = np.zeros((len(populations), len(populations)), dtype=np.int64)
    with (directory / "mobility_2011-2015_statelevel.csv").open() as stream:
        for row in csv.DictReader(stream):
            mobility[index[row["ori"]], index[row["dest"]]] += int(row["amount"])
    np.fill_diagonal(mobility, 0)
    if len(populations) != N_STATES:
        raise ValueError(f"Expected {N_STATES} locations, found {len(populations)}")
    return labels, subpop, populations, mobility


def initial_infected(labels):
    return np.array(
        [INITIAL_INFECTED.get(label, 0) for label in labels], dtype=np.int64
    )


def external_foi(infectious, beta, populations, mobility, p_a=P_A):
    """Instantaneous continuous external FoI h_i(t), in inverse weeks."""
    infectious = np.asarray(infectious)
    beta = np.asarray(beta)
    if infectious.ndim == 1:
        pressure = beta * infectious / populations
        return p_a * (mobility @ pressure) / populations
    pressure = beta[:, None] * infectious / populations[:, None]
    return p_a * (mobility @ pressure) / populations[:, None]


def force_of_infection(infectious, beta, populations, mobility, p_a=P_A):
    """Return total, local, and instantaneous external FoI."""
    pressure = beta * infectious / populations
    local_weight = 1.0 - p_a * mobility.sum(axis=1) / populations
    local = local_weight * pressure
    external = external_foi(infectious, beta, populations, mobility, p_a)
    return local + external, local, external


def solve_coupled(
    beta,
    *,
    populations,
    mobility,
    I0,
    gamma=GAMMA,
    p_a=P_A,
    dense_output=True,
    rtol=RTOL,
    atol=ATOL,
):
    """Solve the coupled ODE and return weekly integrated incidence."""
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), (N_STATES,))
    zeros = np.zeros(N_STATES, dtype=np.float64)
    y0 = np.concatenate((populations - I0, I0, zeros, zeros))

    def rhs(_time, state):
        susceptible = state[:N_STATES]
        infectious = state[N_STATES : 2 * N_STATES]
        total, _, _ = force_of_infection(
            infectious, beta, populations, mobility, p_a
        )
        infections = susceptible * total
        recoveries = gamma * infectious
        return np.concatenate(
            (-infections, infections - recoveries, recoveries, infections)
        )

    solution = solve_ivp(
        rhs,
        (0.0, float(N_WEEKS)),
        y0,
        method="DOP853",
        t_eval=WEEKLY_TIMES,
        dense_output=dense_output,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Coupled ODE failed: {solution.message}")
    susceptible, infectious, recovered, cumulative = np.split(solution.y, 4)
    incidence = np.diff(cumulative, axis=1)
    np.testing.assert_allclose(
        susceptible + infectious + recovered,
        np.broadcast_to(populations[:, None], susceptible.shape),
        rtol=2e-9,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        incidence, susceptible[:, :-1] - susceptible[:, 1:], rtol=2e-8, atol=2e-3
    )
    return {
        "S": susceptible,
        "I": infectious,
        "R": recovered,
        "cumulative_incidence": cumulative,
        "incidence": incidence,
        "solution": solution,
    }


def evaluate_external_foi(solution, times, beta, populations, mobility, p_a=P_A):
    """Evaluate the exact instantaneous h_i(t) from a dense ODE solution."""
    if solution.sol is None:
        raise ValueError("The ODE solution does not have dense output")
    state = solution.sol(np.asarray(times, dtype=np.float64))
    infectious = state[N_STATES : 2 * N_STATES]
    return external_foi(infectious, beta, populations, mobility, p_a)


def observe(weekly_incidence, rng, p_report=P_REPORT):
    """Generate weekly observations Y_iw ~ Poisson(p_report * mu_iw)."""
    return rng.poisson(p_report * np.asarray(weekly_incidence, dtype=np.float64))
