"""The same 48-state local SEIR equations under continuous external h(a,k)."""

import numpy as np

from stratified_model import (
    N_LOCATIONS, N_AGES, N_VARIANTS, HORIZON, C_SLICE,
    INTEGRATION_ATOL, checked_array, unpack_state, transform_parameters,
    transform_local_parameters, npi_multiplier, source_pressure,
    force_of_infection, compartment_derivative, weekly_incidence,
)
from stratified_simulator import integrate_piecewise


def make_exact_external_foi_function(full_solution, params, location):
    """Callable h(t)->(3,2), evaluated from the continuous coupled trajectory."""
    if not 0 <= location < N_LOCATIONS:
        raise IndexError("location is out of range")
    beta, effects = transform_parameters(params)
    fixed = full_solution.fixed_inputs

    def h_func(time):
        if not full_solution.t[0] <= time <= full_solution.t[-1]:
            raise ValueError("External forcing requested outside the coupled solve")
        state = unpack_state(full_solution.sol(float(time)))
        return force_of_infection(time, state, beta, effects, fixed)[2][location]

    # Local integration must also stop at source-location NPI jumps.
    h_func.change_times = full_solution.change_times.copy()
    return h_func


def solve_local_frozen_foi(theta_local, h_func, fixed_inputs, location, *,
                           horizon=HORIZON, evaluation_times=None,
                           forcing_change_times=None):
    """Use original known inputs; custom h discontinuities must be supplied.

    fixed_inputs is returned by make_fixed_inputs or a coupled simulation.
    The exact-forcing helper carries its own change_times automatically.
    """
    if not 0 <= location < N_LOCATIONS:
        raise IndexError("location is out of range")
    beta, effects = transform_local_parameters(theta_local)
    fixed = fixed_inputs
    age = fixed["age_populations"][location:location + 1]
    intervals = fixed["npi_intervals"][location:location + 1]
    protection = np.stack((np.ones(N_VARIANTS), 1. - fixed["vaccine_efficacy"]))
    initial = unpack_state(fixed["initial_state"])[location].ravel()

    def rhs(time, flat_state):
        state = unpack_state(flat_state, n_locations=1)
        multiplier = npi_multiplier(time, effects[None, :], intervals)
        local = fixed["q"][location, location] * source_pressure(state, beta[None, ...], multiplier, age)[0]
        external = checked_array("h_func(t)", h_func(time), (N_AGES, N_VARIANTS))
        if np.any(external < -INTEGRATION_ATOL):
            raise ValueError("External force of infection must be nonnegative")
        infection_rate = (local + external)[:, None, :] * protection[None, :, :]
        nu = checked_array("vaccination_rate(t)", fixed["vaccination_rate"](time), (N_LOCATIONS, N_AGES))
        if np.any(nu < 0):
            raise ValueError("Vaccination rates must be nonnegative")
        return compartment_derivative(state, infection_rate[None, ...], nu[location:location + 1],
                                      fixed["sigma"], fixed["gamma"]).ravel()

    jumps = getattr(h_func, "change_times", []) if forcing_change_times is None else forcing_change_times
    changes = np.r_[intervals.ravel(), fixed["vaccination_change_times"], jumps]
    solution = integrate_piecewise(rhs, initial, horizon, changes, evaluation_times)
    weeks = np.arange(int(horizon) + 1, dtype=float)
    state = unpack_state(solution.sol(weeks), n_locations=1)
    return dict(solution=solution, week_boundaries=weeks,
                weekly_incidence=weekly_incidence(state[0, :, :, C_SLICE, :]),
                runtime_seconds=solution.runtime_seconds)
