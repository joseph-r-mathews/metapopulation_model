"""Continuous coupled simulation and instantaneous age/variant external FoI."""

from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp

from stratified_model import (
    AGE_PROPORTIONS, VACCINATION_PROPORTIONS, INITIAL_INFECTED,
    N_LOCATIONS, N_AGES, N_STATE_COMPONENTS, N_ODE_STATES, HORIZON,
    S_INDEX, R_INDEX, I_SLICE, C_SLICE, NPI_INTERVALS,
    VACCINE_EFFICACY, SIGMA, GAMMA, VACCINATION_CHANGE_TIMES,
    INTEGRATION_METHOD, INTEGRATION_RTOL, INTEGRATION_ATOL,
    checked_array, vaccination_rate, mobility_weights, transform_parameters,
    unpack_state, epidemic_rhs, force_of_infection, weekly_incidence,
)


def stratified_populations(populations):
    age = np.asarray(populations, dtype=float)[:, None] * AGE_PROPORTIONS
    return age, age[..., None] * VACCINATION_PROPORTIONS


def initial_conditions(labels, populations, age_populations=None):
    """Seed both variants in CA/TX/FL/NY, proportional to age/vaccine sizes."""
    populations = np.asarray(populations, dtype=float)
    if age_populations is None:
        age_populations, _ = stratified_populations(populations)
    stratum = age_populations[..., None] * VACCINATION_PROPORTIONS
    seeds = np.array([INITIAL_INFECTED.get(label, (0., 0.)) for label in labels])
    state = np.zeros((N_LOCATIONS, N_AGES, 2, N_STATE_COMPONENTS))
    state[..., I_SLICE] = stratum[..., None] / populations[:, None, None, None] * seeds[:, None, None, :]
    state[..., S_INDEX] = stratum - state[..., I_SLICE].sum(axis=-1)
    return state.ravel()


def make_fixed_inputs(labels, populations, mobility, *, age_populations=None,
                      npi_intervals=NPI_INTERVALS, nu_func=None,
                      vaccination_change_times=None, vaccine_efficacy=VACCINE_EFFICACY,
                      sigma=SIGMA, gamma=GAMMA, initial_state=None):
    """A plain dictionary of known inputs; nothing here is inferred.

    A custom nu_func(t) returns (51,3); supply its discontinuity times too.
    A smooth custom callable needs no vaccination_change_times.
    """
    populations = checked_array("populations", populations, (N_LOCATIONS,)).copy()
    if age_populations is None:
        age_populations, _ = stratified_populations(populations)
    age_populations = checked_array("age_populations", age_populations, (N_LOCATIONS, N_AGES)).copy()
    if np.any(age_populations <= 0) or not np.allclose(age_populations.sum(axis=1), populations, rtol=1e-12, atol=1e-8):
        raise ValueError("Positive age populations must sum to location populations")
    intervals = checked_array("npi_intervals", npi_intervals, (N_LOCATIONS, 8, 2)).copy()
    if np.any(intervals[..., 1] < intervals[..., 0]):
        raise ValueError("NPI end precedes start")
    efficacy = checked_array("vaccine_efficacy", vaccine_efficacy, (2,)).copy()
    sigma, gamma = checked_array("sigma", sigma, (2,)).copy(), checked_array("gamma", gamma, (2,)).copy()
    if np.any((efficacy < 0) | (efficacy > 1)) or np.any(sigma <= 0) or np.any(gamma <= 0):
        raise ValueError("Invalid fixed efficacy or transition rates")
    if nu_func is None:
        nu_func = vaccination_rate
        if vaccination_change_times is None:
            vaccination_change_times = VACCINATION_CHANGE_TIMES
    if vaccination_change_times is None:
        vaccination_change_times = []
    changes = np.asarray(vaccination_change_times, dtype=float)
    if changes.ndim != 1 or not np.all(np.isfinite(changes)) or not callable(nu_func):
        raise ValueError("Provide a vaccination callable and finite change times")
    if initial_state is None:
        initial_state = initial_conditions(labels, populations, age_populations)
    initial_state = checked_array("initial_state", initial_state, (N_ODE_STATES,)).copy()
    state = unpack_state(initial_state)
    if np.any(state < 0) or not np.allclose(state[..., :6].sum(axis=(2, 3)), age_populations, rtol=1e-12, atol=1e-7):
        raise ValueError("Nonnegative initial compartments must sum to age populations")
    q = mobility_weights(populations, mobility)
    q_external = q.copy()
    np.fill_diagonal(q_external, 0.)
    return dict(populations=populations, age_populations=age_populations,
                q=q, q_external=q_external, npi_intervals=intervals,
                vaccination_rate=nu_func, vaccination_change_times=changes.copy(),
                vaccine_efficacy=efficacy, sigma=sigma, gamma=gamma,
                initial_state=initial_state)


def integrate_piecewise(rhs, initial, horizon, change_times, evaluation_times=None):
    """Restart DOP853 at known jumps; join its dense continuous interpolants.

    Endpoint RHS calls use the interior one-sided limit of the current segment.
    Values assigned at isolated discontinuities do not alter the ODE solution.
    """
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0 or not horizon.is_integer():
        raise ValueError("horizon must be a positive integer number of weeks")
    changes = np.asarray(change_times, dtype=float).ravel()
    if not np.all(np.isfinite(changes)):
        raise ValueError("Change times must be finite")
    boundaries = np.unique(np.r_[0., changes[(changes > 0) & (changes < horizon)], horizon])
    weeks = np.arange(int(horizon) + 1, dtype=float)
    extra = np.asarray([] if evaluation_times is None else evaluation_times, dtype=float)
    if extra.ndim != 1 or not np.all(np.isfinite(extra)) or np.any((extra < 0) | (extra > horizon)):
        raise ValueError("evaluation_times must be finite and within the horizon")
    times = np.unique(np.r_[weeks, extra, boundaries])
    knots, interpolants = [0.], []
    current, nfev = np.asarray(initial, dtype=float), 0
    started = perf_counter()
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        inner_left, inner_right = np.nextafter(left, right), np.nextafter(right, left)

        def segment_rhs(time, state):
            return rhs(min(max(time, inner_left), inner_right), state)

        part = solve_ivp(segment_rhs, (left, right), current,
                         method=INTEGRATION_METHOD, rtol=INTEGRATION_RTOL,
                         atol=INTEGRATION_ATOL, dense_output=True)
        if not part.success:
            raise RuntimeError(f"ODE failed on [{left}, {right}]: {part.message}")
        current = part.y[:, -1]
        knots.extend(part.sol.ts[1:])
        interpolants.extend(part.sol.interpolants)
        nfev += part.nfev
    dense = OdeSolution(np.asarray(knots), interpolants)
    return SimpleNamespace(t=times, y=dense(times), sol=dense, success=True,
                           nfev=nfev, change_times=boundaries,
                           runtime_seconds=perf_counter() - started)


def solve_epidemic(theta, labels, populations, mobility, *, horizon=HORIZON,
                   evaluation_times=None, **known_inputs):
    fixed = make_fixed_inputs(labels, populations, mobility, **known_inputs)
    beta, effects = transform_parameters(theta)
    rhs = lambda time, state: epidemic_rhs(time, state, beta, effects, fixed)
    changes = np.r_[fixed["npi_intervals"].ravel(), fixed["vaccination_change_times"]]
    solution = integrate_piecewise(rhs, fixed["initial_state"], horizon, changes, evaluation_times)
    solution.fixed_inputs = fixed
    solution.theta = np.asarray(theta, dtype=float).copy()
    weeks = np.arange(int(horizon) + 1, dtype=float)
    state = unpack_state(solution.sol(weeks))
    return dict(solution=solution, fixed_inputs=fixed,
                runtime_seconds=solution.runtime_seconds, week_boundaries=weeks,
                weekly_incidence=weekly_incidence(state[:, :, :, C_SLICE, :]))


def compute_external_foi(solution, params, times):
    """Instantaneous h(t) from dense coupled state, shape (time,51,3,2).

    params is the unconstrained 1326-vector. Known inputs are retained on the
    coupled solution, so the same NPI schedules and mobility are used here.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any((times < solution.t[0]) | (times > solution.t[-1])):
        raise ValueError("FoI times must lie within the coupled solve")
    beta, effects = transform_parameters(params)
    fixed = solution.fixed_inputs
    state = unpack_state(solution.sol(times))
    external = np.empty((len(times), N_LOCATIONS, N_AGES, 2))
    for index, time in enumerate(times):
        _, _, external[index] = force_of_infection(time, state[..., index], beta, effects, fixed)
    return external


def simulate_observations(expected_incidence, rng):
    expected = np.asarray(expected_incidence, dtype=float)
    if not np.all(np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("Poisson means must be finite and nonnegative")
    return rng.poisson(expected)


def trajectory_diagnostics(solution):
    """Conservation across vaccination groups, positivity, and monotonicity."""
    state = unpack_state(solution.y)
    total = state[:, :, :, :6, :].sum(axis=(2, 3))
    error = np.max(np.abs(total - solution.fixed_inputs["age_populations"][..., None]))
    return dict(maximum_population_conservation_error=float(error),
                minimum_compartment_value=float(state[:, :, :, :6, :].min()),
                minimum_cumulative_incidence=float(state[:, :, :, C_SLICE, :].min()),
                minimum_cumulative_increment=float(np.diff(state[:, :, :, C_SLICE, :], axis=-1).min()))
