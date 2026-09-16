"""Diagnose deterministic tau-leap convergence to the coupled ODE model."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from deterministic_model import ATOL, RTOL, solve_coupled
from model import (GAMMA, N_WEEKS, P_A, PRIOR_LOG_MEAN, PRIOR_LOG_SD,
                   force_of_infection, initial_infected, load_data)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "deterministic_target"
DTS = (1 / 7, 1 / 14, 1 / 28, 1 / 56, 1 / 112, 1 / 224)
SEED = 20260922


def deterministic_tau_leap(beta, dt, *, N, M, I0, gamma=GAMMA, p_a=P_A):
    """Mean recursion of the binomial tau leap, without random draws."""
    steps_per_week = round(1 / dt)
    if not np.isclose(steps_per_week * dt, 1.0):
        raise ValueError("dt must divide one week exactly")
    S = N.astype(np.float64) - I0
    I = I0.astype(np.float64).copy()
    R = np.zeros_like(S)
    incidence = np.zeros((len(N), N_WEEKS), dtype=np.float64)
    external_integral = np.zeros_like(incidence)
    p_rec = -np.expm1(-gamma * dt)
    for week in range(N_WEEKS):
        for _ in range(steps_per_week):
            total, _, external = force_of_infection(I, beta, N, M, p_a)
            mean_infections = S * -np.expm1(-total * dt)
            mean_recoveries = I * p_rec
            incidence[:, week] += mean_infections
            external_integral[:, week] += external * dt
            S -= mean_infections
            I += mean_infections - mean_recoveries
            R += mean_recoveries
    np.testing.assert_allclose(S + I + R, N, rtol=2e-14, atol=2e-6)
    return incidence, external_integral


def metrics(approximation, reference):
    error = approximation - reference
    return (
        float(np.linalg.norm(error.ravel()) / np.linalg.norm(reference.ravel())),
        float(np.sqrt(np.mean(error ** 2))),
        float(np.corrcoef(approximation.ravel(), reference.ravel())[0, 1]),
    )


def verify_shared_definition(beta, N, M, I0):
    I = I0.astype(np.float64)
    total, local, external = force_of_infection(I, beta, N, M, P_A)
    pressure = beta * I / N
    expected_local = (1 - P_A * M.sum(axis=1) / N) * pressure
    expected_external = P_A * (M @ pressure) / N
    np.testing.assert_array_equal(beta.shape, (51,))
    np.testing.assert_allclose(local, expected_local, rtol=0, atol=0)
    np.testing.assert_allclose(external, expected_external, rtol=0, atol=0)
    np.testing.assert_allclose(total, expected_local + expected_external,
                               rtol=0, atol=0)
    return {
        "beta": "one identical 51-vector passed to both implementations",
        "gamma_per_week": GAMMA,
        "populations": "one identical N array passed to both implementations",
        "mobility": "one identical raw M array passed to both implementations",
        "p_a": P_A,
        "initial_conditions": "S0=N-I0, I0 shared, R0=0",
        "local_foi": "[1-(p_a/N_i) sum_{j!=i} M_ij] beta_i I_i/N_i",
        "external_foi": "sum_{j!=i} (p_a M_ij/N_i) beta_j I_j/N_j",
        "tau_weekly_external_foi":
            "sum over within-week left endpoints: external_foi(t_k) * dt",
        "ode_weekly_external_foi":
            "integral from week w to w+1 of external_foi(t) dt, via augmented ODE state",
        "comparison": "both are weekly integrals (equal numerically to weekly averages because interval length=1)",
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels, _, N, M = load_data(ROOT)
    I0 = initial_infected(labels)
    beta = np.exp(np.random.default_rng(SEED).normal(
        PRIOR_LOG_MEAN, PRIOR_LOG_SD, 51
    ))
    shared = verify_shared_definition(beta, N, M, I0)
    reference = solve_coupled(beta, N=N, M=M, I0=I0)
    rows = []
    for dt in DTS:
        started = perf_counter()
        incidence, external = deterministic_tau_leap(
            beta, dt, N=N, M=M, I0=I0
        )
        runtime_seconds = perf_counter() - started
        incidence_metrics = metrics(incidence, reference["incidence"])
        foi_metrics = metrics(external, reference["external_foi"])
        rows.append({
            "dt": dt,
            "incidence_relative_L2": incidence_metrics[0],
            "incidence_RMSE": incidence_metrics[1],
            "incidence_correlation": incidence_metrics[2],
            "foi_relative_L2": foi_metrics[0],
            "foi_RMSE": foi_metrics[1],
            "foi_correlation": foi_metrics[2],
            "runtime_seconds": runtime_seconds,
        })
    path = OUTPUT / "deterministic_target_tau_leap_convergence.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    incidence_errors = np.array([row["incidence_relative_L2"] for row in rows])
    foi_errors = np.array([row["foi_relative_L2"] for row in rows])
    decreasing = bool(np.all(np.diff(incidence_errors) < 0)
                      and np.all(np.diff(foi_errors) < 0))
    report = {
        "ode_reference": {"method": "DOP853", "rtol": RTOL, "atol": ATOL},
        "shared_model_check": shared,
        "rows": rows,
        "relative_L2_errors_strictly_decrease": decreasing,
    }
    (OUTPUT / "deterministic_target_tau_leap_convergence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not decreasing:
        raise RuntimeError("Tau-leap error did not systematically decrease; model mismatch remains")


if __name__ == "__main__":
    main()
