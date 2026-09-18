"""Deterministic equation, conservation, discontinuity, and local identity tests."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

import stratified_model as model
from epidemic_model import load_population_and_mobility
from stratified_simulator import (
    make_fixed_inputs, solve_epidemic, compute_external_foi,
    simulate_observations, trajectory_diagnostics,
)
from validate_local_frozen_simulator import check_frozen_identity

ROOT = Path(__file__).resolve().parent


def deterministic_theta():
    offset = 0.04 * np.sin(np.arange(51 * 18).reshape(51, 2, 3, 3))
    beta = model.BETA_PRIOR_LOG_MEAN + offset
    logits = np.broadcast_to(model.NPI_PRIOR_LOGIT_MEAN, (51, 8)).copy()
    logits += 0.03 * np.cos(np.arange(51 * 8).reshape(51, 8))
    return model.pack_parameters(beta, logits)


def validation_intervals():
    # Three different schedules exercise source-location changes and noninteger
    # jumps; all 8 intervals still overlap exactly as specified by the product.
    intervals = model.NPI_INTERVALS.copy()
    intervals += (np.arange(51) % 3)[:, None, None] * 0.125
    return intervals


def main():
    labels, _, populations, mobility = load_population_and_mobility(ROOT)
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(f"Validation failed: {name}")

    check("dimensions", (model.PARAMETERS_PER_LOCATION, model.PARAMETER_DIMENSION,
                          model.N_ODE_STATES, model.LOCAL_ODE_STATES) == (26, 1326, 2448, 48))
    theta = deterministic_theta()
    log_beta, logits = model.unpack_parameters(theta)
    check("pack_unpack_exact", np.array_equal(theta, model.pack_parameters(log_beta, logits)))
    check("parameter_order", all(
        theta[model.parameter_index(i, "log_beta", variant=k, source_age=b, recipient_age=a)] == log_beta[i, k, b, a]
        for i in range(51) for k in range(2) for b in range(3) for a in range(3)) and all(
        theta[model.parameter_index(i, "npi_logit", npi=n)] == logits[i, n]
        for i in range(51) for n in range(8)))
    beta, effects = model.transform_parameters(theta)
    check("parameter_constraints", np.all(beta > 0) and np.all((effects > 0) & (effects < 1)))
    check("prior_finite_and_reproducible", np.isfinite(model.log_prior(theta)) and np.array_equal(
        model.sample_prior(np.random.default_rng(17)), model.sample_prior(np.random.default_rng(17))))
    local_theta = model.extract_local_parameters(theta, 7)
    local_beta, local_effects = model.transform_local_parameters(local_theta)
    check("local_parameter_transform", np.array_equal(local_beta, beta[7]) and np.array_equal(local_effects, effects[7]))
    check("local_prior_dimension", model.sample_prior(np.random.default_rng(18), local=True).shape == (26,))

    intervals = validation_intervals()
    fixed = make_fixed_inputs(labels, populations, mobility, npi_intervals=intervals)
    q = fixed["q"]
    raw_q = model.MOBILITY_SCALE * mobility / populations[:, None]
    np.fill_diagonal(raw_q, 0.)
    check("mobility_exact_construction", np.array_equal(raw_q, fixed["q_external"]) and np.array_equal(np.diag(q), 1 - raw_q.sum(axis=1)))
    check("mobility_valid", np.all(np.diag(q) >= 0) and np.allclose(q.sum(axis=1), 1., rtol=0, atol=1e-15))
    try:
        model.mobility_weights(populations, mobility * 1000000)
    except ValueError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    check("negative_q_ii_rejected", invalid_rejected)

    check("zero_npi", np.array_equal(model.npi_multiplier(7., np.zeros((51, 8)), intervals), np.ones(51)))
    before = model.npi_multiplier(7., effects, intervals)
    stronger = effects.copy()
    stronger[:, 0] = 0.9
    check("overlapping_npi_product", np.allclose(before, (1-effects[:, 0]) * (1-effects[:, 1]), rtol=0, atol=1e-15))
    check("stronger_npi_reduces_transmission", np.all(model.npi_multiplier(7., stronger, intervals) < before))
    check("closed_npi_endpoints", model.npi_multiplier(4., effects[:1], intervals[:1])[0] == 1 - effects[0, 0]
          and np.isclose(model.npi_multiplier(10., effects[:1], intervals[:1])[0], (1-effects[0, 0])*(1-effects[0, 1])))

    # A nonuniform artificial infectious state makes source/recipient and
    # variant transpositions detectable independently of the solver.
    state = model.unpack_state(fixed["initial_state"]).copy()
    state[..., model.I_SLICE] = 1. + np.arange(51*3*2*2).reshape(51, 3, 2, 2) % 13
    lam, local, external = model.force_of_infection(7., state, beta, effects, fixed)
    manual_local = np.zeros_like(local)
    manual_external = np.zeros_like(external)
    for i in range(51):
        for a in range(3):
            for k in range(2):
                for j in range(51):
                    source = sum(beta[j, k, b, a] * before[j] * state[j, b, :, 3+k].sum()
                                 / fixed["age_populations"][j, b] for b in range(3))
                    if i == j:
                        manual_local[i, a, k] += q[i, j] * source
                    else:
                        manual_external[i, a, k] += q[i, j] * source
    check("source_recipient_variant_and_mobility_orientation", np.allclose(local, manual_local, rtol=1e-13, atol=1e-18)
          and np.allclose(external, manual_external, rtol=1e-13, atol=1e-18))
    check("vaccine_reduces_equal_susceptible_incidence", np.all(lam[:, :, 1] < lam[:, :, 0])
          and np.allclose(lam[:, :, 1], lam[:, :, 0] * (1-model.VACCINE_EFFICACY)))
    zero_fixed = make_fixed_inputs(labels, populations, np.zeros_like(mobility))
    check("zero_mobility_external", np.array_equal(model.force_of_infection(7., state, beta, effects, zero_fixed)[2], np.zeros((51, 3, 2))))

    nu = fixed["vaccination_rate"](12.)
    derivative = model.compartment_derivative(state, lam, nu)
    transfer = nu * (state[:, :, 0, model.S_INDEX] + state[:, :, 0, model.R_INDEX])
    budget = derivative[..., :6].sum(axis=-1)
    check("vaccination_stratum_rhs_budget", np.allclose(budget[:, :, 0], -transfer, rtol=1e-12, atol=1e-8)
          and np.allclose(budget[:, :, 1], transfer, rtol=1e-12, atol=1e-8))
    no_vaccination = model.compartment_derivative(state, lam, np.zeros_like(nu))
    check("no_E_or_I_vaccination", np.array_equal(derivative[..., 1:5], no_vaccination[..., 1:5]))
    check("cumulative_rhs_is_infections", np.array_equal(derivative[..., model.C_SLICE], state[..., 0, None] * lam))

    print("Equation checks passed; running deterministic 52-week integration and frozen identity.", flush=True)
    full = solve_epidemic(theta, labels, populations, mobility, npi_intervals=intervals,
                          evaluation_times=np.linspace(0, 52, 417))
    diagnostics = trajectory_diagnostics(full["solution"])
    check("location_age_conservation", diagnostics["maximum_population_conservation_error"] < 1e-5)
    check("nonnegative_compartments", diagnostics["minimum_compartment_value"] >= -1e-5)
    check("nonnegative_cumulative", diagnostics["minimum_cumulative_incidence"] >= -1e-5)
    check("nondecreasing_cumulative", diagnostics["minimum_cumulative_increment"] >= -1e-5)
    check("observation_shape", full["weekly_incidence"].shape == (51, 3, 2, 2, 52))
    times = np.array([0., 4., 4.125, 9.12345, 14.25, 52.])
    h = compute_external_foi(full["solution"], theta, times)
    check("external_shape_and_continuous_evaluation", h.shape == (6, 51, 3, 2) and all(
        np.array_equal(h[n], model.force_of_infection(t, model.unpack_state(full["solution"].sol(t)), beta, effects, full["fixed_inputs"])[2])
        for n, t in enumerate(times)))

    # Independent quadrature of vaccination transfers checks every U/V stratum
    # budget, rather than incorrectly requiring each vaccination group fixed.
    def transfer_at(time):
        current = model.unpack_state(full["solution"].sol(time))
        return fixed["vaccination_rate"](time) * (current[:, :, 0, 0] + current[:, :, 0, 5])

    net_transfer, _ = quad_vec(transfer_at, 0., 52., points=full["solution"].change_times,
                              epsabs=1e-6, epsrel=1e-11)
    first = model.unpack_state(full["solution"].sol(0.))[..., :6].sum(axis=-1)
    last = model.unpack_state(full["solution"].sol(52.))[..., :6].sum(axis=-1)
    transfer_error = float(np.max(np.abs(last-first-np.stack((-net_transfer, net_transfer), axis=-1))))
    check("each_vaccination_stratum_integrated_budget", transfer_error < 1e-4)

    initial = fixed["initial_state"].copy()
    initial_view = model.unpack_state(initial)
    initial_view[..., 0] += initial_view[..., 4]
    initial_view[..., 4] = 0.
    absent = solve_epidemic(theta, labels, populations, mobility, horizon=12, initial_state=initial)
    check("unseeded_variant_stays_absent", np.count_nonzero(absent["weekly_incidence"][..., 1, :]) == 0)

    # Exact vaccination-only solution independently tests callable inputs.
    initial_view[...] = 0.
    initial_view[..., 0] = fixed["age_populations"][..., None] * np.array([0.56, 0.24])
    initial_view[..., 5] = fixed["age_populations"][..., None] * np.array([0.14, 0.06])
    constant_nu = np.broadcast_to(np.array([0.01, 0.02, 0.03]), (51, 3))
    vacc_only = solve_epidemic(theta, labels, populations, mobility, horizon=4,
                               initial_state=initial, nu_func=lambda t: constant_nu)
    final = model.unpack_state(vacc_only["solution"].sol(4.))
    analytic = initial_view.copy()
    for component in (0, 5):
        analytic[:, :, 0, component] *= np.exp(-constant_nu * 4.)
        analytic[:, :, 1, component] += initial_view[:, :, 0, component] - analytic[:, :, 0, component]
    check("vaccination_only_analytic_solution", np.allclose(final, analytic, rtol=1e-10, atol=1e-6))

    observed = simulate_observations(full["weekly_incidence"], np.random.default_rng(100))
    check("poisson_reproducible", np.array_equal(observed, simulate_observations(full["weekly_incidence"], np.random.default_rng(100))))
    check("poisson_likelihood", np.isfinite(model.poisson_log_likelihood(observed, full["weekly_incidence"]))
          and model.poisson_log_likelihood(np.array([0]), np.array([0.])) == 0.
          and np.isneginf(model.poisson_log_likelihood(np.array([1]), np.array([0.]))))
    identity = check_frozen_identity(full, theta, labels)
    check("frozen_foi_identity", identity["passed"])
    summary = dict(all_passed=True, checks=checks, diagnostics=diagnostics,
                   vaccination_stratum_budget_error=transfer_error,
                   minimum_q_ii=float(np.diag(q).min()),
                   validation_integration_seconds=full["runtime_seconds"], frozen_identity=identity)
    output = ROOT / "results" / "validation_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
