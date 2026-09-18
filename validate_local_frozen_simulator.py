"""Deterministic local/full identity check under continuous age/variant FoI."""

import json
from pathlib import Path

import numpy as np

from epidemic_model import load_population_and_mobility
from local_frozen_simulator import make_exact_external_foi_function, solve_local_frozen_foi
from stratified_model import extract_local_parameters, unpack_state
from stratified_simulator import solve_epidemic


def check_frozen_identity(full, theta, labels, tolerance=1e-5):
    details = []
    for label in ("CA", "TX", "NY", "WY"):
        location = int(np.flatnonzero(labels == label)[0])
        h_func = make_exact_external_foi_function(full["solution"], theta, location)
        assert h_func(9.125).shape == (3, 2)
        local = solve_local_frozen_foi(extract_local_parameters(theta, location),
                                       h_func, full["fixed_inputs"], location,
                                       horizon=int(full["solution"].t[-1]))
        reference = full["weekly_incidence"][location]
        error = float(np.linalg.norm(local["weekly_incidence"] - reference) / max(np.linalg.norm(reference), 1e-300))
        local_state = unpack_state(local["solution"].y, n_locations=1)
        assert local_state.min() >= -1e-5
        assert error < tolerance, f"Frozen identity failed for {label}: {error}"
        details.append(dict(location=label, relative_L2_incidence_error=error))
    return dict(passed=True, tolerance=tolerance, comparisons=details,
                maximum_relative_L2_incidence_error=max(d["relative_L2_incidence_error"] for d in details))


def main():
    from validate_stratified_model import deterministic_theta, validation_intervals
    root = Path(__file__).resolve().parent
    labels, _, populations, mobility = load_population_and_mobility(root)
    theta = deterministic_theta()
    full = solve_epidemic(theta, labels, populations, mobility,
                          npi_intervals=validation_intervals())
    print(json.dumps(check_frozen_identity(full, theta, labels), indent=2))


if __name__ == "__main__":
    main()
