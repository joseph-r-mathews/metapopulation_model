"""Pre-training stochastic/deterministic consistency check."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from deterministic_model import ATOL, RTOL, solve_coupled
from model import (GAMMA, P_A, PRIOR_LOG_MEAN, PRIOR_LOG_SD, TAU_DT,
                   initial_infected, load_data, simulate)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "deterministic_target"


def comparison_metrics(sample_mean, deterministic):
    difference = np.asarray(sample_mean) - np.asarray(deterministic)
    norm = np.linalg.norm(np.asarray(deterministic).ravel())
    return {
        "RMSE": float(np.sqrt(np.mean(difference ** 2))),
        "relative_L2_error": float(np.linalg.norm(difference.ravel()) / norm),
        "correlation": float(np.corrcoef(sample_mean.ravel(), deterministic.ravel())[0, 1]),
    }


def consistency_check(replicates=100, seed=20260922, output=RESULTS):
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels, _, N, M = load_data(ROOT)
    I0 = initial_infected(labels)
    rng = np.random.default_rng(seed)
    beta = np.exp(rng.normal(PRIOR_LOG_MEAN, PRIOR_LOG_SD, 51))
    deterministic = solve_coupled(beta, N=N, M=M, I0=I0)
    incidence = np.empty((replicates, 51, 52))
    external = np.empty_like(incidence)
    for replicate in range(replicates):
        stochastic = simulate(
            beta,
            np.random.default_rng(np.random.SeedSequence([seed, replicate])),
            N=N, M=M, I0=I0,
        )
        incidence[replicate] = stochastic["A"]
        external[replicate] = stochastic["external_foi"]
    rows = []
    for quantity, mean, target in (
        ("weekly_incidence", incidence.mean(0), deterministic["incidence"]),
        ("weekly_external_foi", external.mean(0), deterministic["external_foi"]),
    ):
        rows.append({"quantity": quantity, **comparison_metrics(mean, target)})
    with (output / "deterministic_target_stochastic_consistency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "replicates": replicates, "tau_dt_weeks": TAU_DT,
        "gamma_per_week": GAMMA, "p_a": P_A,
        "deterministic_solver": "scipy.solve_ivp/DOP853",
        "deterministic_rtol": RTOL, "deterministic_atol": ATOL,
        "runtime_seconds": perf_counter() - started,
        "metrics": rows,
    }
    (output / "deterministic_target_stochastic_consistency.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)
    failures = [row for row in rows
                if row["relative_L2_error"] > 0.15 or row["correlation"] < 0.98]
    if failures:
        raise RuntimeError(f"Stochastic/deterministic consistency failed: {failures}")
    return summary


if __name__ == "__main__":
    consistency_check()
