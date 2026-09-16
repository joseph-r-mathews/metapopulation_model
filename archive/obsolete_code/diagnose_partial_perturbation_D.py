"""Partial-coordinate D diagnostic for the coherent weekly oracle proposal."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import P_REPORT, initial_infected, load_data, poisson_log_likelihood
from temporary_coherent_weekly_oracle import (ROOT, RESULTS,
                                              simulate_local_grid,
                                              simulate_weekly_target)


KS = (1, 2, 4, 8, 16, 32, 51)
DRAWS_PER_K = 100
SEED = 20260928


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coupled_log_likelihood(eta, Y, N, M, I0, batch_size=250):
    values = np.empty(len(eta))
    for start in range(0, len(eta), batch_size):
        block = eta[start:start+batch_size]
        simulation = simulate_weekly_target(np.exp(block), N=N, M=M, I0=I0)
        values[start:start+len(block)] = poisson_log_likelihood(
            Y[None], simulation["mu"], P_REPORT, axis=(1, 2)
        )
    return values


def summed_local_log_likelihood(eta, H, Y, N, M, I0):
    values = np.zeros(len(eta))
    for state in range(51):
        simulation = simulate_local_grid(
            eta[:, state], state, H[state], N=N, M=M, I0=I0
        )
        values += poisson_log_likelihood(
            Y[state][None], simulation["mu"], P_REPORT, axis=1
        )
    return values


def ess(values):
    weights = np.exp(values-np.max(values))
    weights /= weights.sum()
    effective = 1/np.dot(weights, weights)
    return float(effective), float(effective/len(values))


def main():
    epidemic_path = RESULTS/"coherent_weekly_test_epidemics.npz"
    sample_path = RESULTS/"samples"/"epidemic_00.npz"
    if not epidemic_path.exists() or not sample_path.exists():
        raise FileNotFoundError("Existing epidemic-0 data or oracle proposal draws are missing")
    with np.load(epidemic_path) as saved:
        eta_true = np.asarray(saved["eta_true"][0])
        Y = np.asarray(saved["Y"][0])
        H = np.asarray(saved["H_oracle"][0])
    with np.load(sample_path) as saved:
        proposal_draws = np.asarray(saved["eta"])
    labels, _, N, M = load_data(ROOT)
    I0 = initial_infected(labels)
    rng = np.random.default_rng(SEED)

    eta = np.empty((len(KS)*DRAWS_PER_K, 51))
    metadata = []
    row_index = 0
    for k in KS:
        for draw_id in range(DRAWS_PER_K):
            selected = np.sort(rng.choice(51, size=k, replace=False))
            value = eta_true.copy()
            source_rows = rng.integers(0, len(proposal_draws), size=k)
            value[selected] = proposal_draws[source_rows, selected]
            eta[row_index] = value
            metadata.append((k, draw_id, selected))
            row_index += 1

    true_matrix = eta_true[None]
    D_true = float(
        coupled_log_likelihood(true_matrix, Y, N, M, I0)[0]
        - summed_local_log_likelihood(true_matrix, H, Y, N, M, I0)[0]
    )
    coupled = coupled_log_likelihood(eta, Y, N, M, I0)
    local = summed_local_log_likelihood(eta, H, Y, N, M, I0)
    D = coupled-local
    delta = D-D_true
    rows = []
    for index, (k, draw_id, selected) in enumerate(metadata):
        rows.append({
            "k": k, "draw_id": draw_id,
            "perturbed_coordinate_indices": ";".join(map(str, selected)),
            "perturbed_states": ";".join(labels[selected]),
            "log_L_coupled": float(coupled[index]),
            "sum_local_log_likelihoods": float(local[index]),
            "D": float(D[index]), "D_true": D_true,
            "Delta_D": float(delta[index]),
        })
    write_csv(RESULTS/"partial_perturbation_D_results.csv", rows)

    summaries = []
    for k in KS:
        values = delta[[item[0] == k for item in metadata]]
        effective, fraction = ess(values)
        summaries.append({
            "k": k, "draws": len(values), "D_true": D_true,
            "mean_Delta_D": float(values.mean()),
            "sd_Delta_D": float(values.std()),
            "median_Delta_D": float(np.median(values)),
            "Delta_D_q10": float(np.quantile(values, .10)),
            "Delta_D_q25": float(np.quantile(values, .25)),
            "Delta_D_q75": float(np.quantile(values, .75)),
            "Delta_D_q90": float(np.quantile(values, .90)),
            "min_Delta_D": float(values.min()),
            "max_Delta_D": float(values.max()),
            "ESS": effective, "ESS_fraction": fraction,
        })
    write_csv(RESULTS/"partial_perturbation_D_summary.csv", summaries)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(KS, [row["sd_Delta_D"] for row in summaries], marker="o")
    ax.set(xlabel="Number of perturbed coordinates (k)", ylabel="SD(Delta_D)")
    fig.tight_layout()
    fig.savefig(RESULTS/"partial_perturbation_sd_vs_k.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(KS, [row["ESS_fraction"] for row in summaries], marker="o")
    ax.set(xlabel="Number of perturbed coordinates (k)", ylabel="ESS fraction")
    fig.tight_layout()
    fig.savefig(RESULTS/"partial_perturbation_ess_vs_k.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
