"""Run exactly one 52-week prior-predictive epidemic and save its known inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

import stratified_model as model
from epidemic_model import load_population_and_mobility
from stratified_simulator import solve_epidemic, compute_external_foi, simulate_observations, trajectory_diagnostics

ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20260917


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stratified_smoke")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    labels, subpop, populations, mobility = load_population_and_mobility(ROOT)
    rng = np.random.default_rng(args.seed)
    theta = model.sample_prior(rng)
    times = np.linspace(0., model.HORIZON, 365)
    full = solve_epidemic(theta, labels, populations, mobility, evaluation_times=times)
    expected = full["weekly_incidence"]
    observed = simulate_observations(expected, rng)
    external = compute_external_foi(full["solution"], theta, times)
    fixed = full["fixed_inputs"]
    diagnostics = trajectory_diagnostics(full["solution"])
    assert diagnostics["maximum_population_conservation_error"] < 1e-5
    assert diagnostics["minimum_compartment_value"] >= -model.INTEGRATION_ATOL
    assert diagnostics["minimum_cumulative_increment"] >= -model.INTEGRATION_ATOL
    arrays = dict(theta_true=theta, mu_weekly=expected, Y_weekly=observed,
                  external_foi_dense=external, external_foi_times=times,
                  npi_intervals=fixed["npi_intervals"], vaccine_efficacy=fixed["vaccine_efficacy"],
                  vaccination_change_times=fixed["vaccination_change_times"],
                  vaccination_schedule_rates=model.VACCINATION_SCHEDULE_RATES,
                  populations=populations, age_populations=fixed["age_populations"],
                  mobility=mobility, mobility_weights=fixed["q"],
                  initial_state=fixed["initial_state"], sigma=fixed["sigma"], gamma=fixed["gamma"],
                  beta_prior_log_mean=model.BETA_PRIOR_LOG_MEAN,
                  npi_prior_logit_mean=model.NPI_PRIOR_LOGIT_MEAN)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(args.output / f"{name}.npy", values)
    summary = dict(integration_runtime_seconds=full["runtime_seconds"],
                   ode_dimension=model.N_ODE_STATES, parameter_dimension=model.PARAMETER_DIMENSION,
                   observation_shape=list(observed.shape), external_foi_shape=list(external.shape),
                   total_expected_incidence_by_variant=expected.sum(axis=(0, 1, 2, 4)).tolist(),
                   total_observed_incidence_by_variant=observed.sum(axis=(0, 1, 2, 4)).tolist(),
                   **diagnostics)
    metadata = dict(model="age_vaccination_variant_seir", seed=args.seed, horizon_weeks=model.HORIZON,
                    labels=labels.tolist(), subpopulation_codes=subpop.tolist(),
                    parameter_order="location-major: log beta[variant,source_age,recipient_age] C order, then 8 NPI logits",
                    state_shape=[51, 3, 2, 8], state_order=list(model.STATE_NAMES),
                    observation_axes=["location", "age", "vaccination", "variant", "week"],
                    external_foi_axes=["time", "location", "age", "variant"],
                    vaccination_labels=["U", "V"],
                    vaccination_schedule="rate[index] with index=searchsorted(change_times,t,side='right')",
                    npi_intervals="closed intervals; transmission jumps handled with one-sided segment integration",
                    mobility_scale=model.MOBILITY_SCALE,
                    beta_prior_log_sd=model.BETA_PRIOR_LOG_SD,
                    npi_prior_logit_sd=model.NPI_PRIOR_LOGIT_SD,
                    integrator=dict(method=model.INTEGRATION_METHOD, rtol=model.INTEGRATION_RTOL,
                                    atol=model.INTEGRATION_ATOL, dense_output=True),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [ROOT / name for name in ("stratified_model.py", "stratified_simulator.py",
                                    "epidemic_model.py", "simulate_stratified_epidemic.py",
                                    "geodata_2019.csv", "mobility_2011-2015_statelevel.csv")]},
                    output_shapes={name: list(value.shape) for name, value in arrays.items()}, **summary)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
