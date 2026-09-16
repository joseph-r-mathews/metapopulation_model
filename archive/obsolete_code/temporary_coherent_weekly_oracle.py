"""Temporary oracle-only importance test for a coherent weekly-H target."""

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import (GAMMA, P_A, P_REPORT, PRIOR_LOG_MEAN, PRIOR_LOG_SD,
                   initial_infected, load_data, poisson_log_likelihood)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "coherent_weekly_oracle"
SAMPLES = RESULTS / "samples"
SEED = 20260927
WEEKS = 52
RK4_SUBSTEPS = 14
COARSE_POINTS = 501
FINAL_POINTS = 2001
TARGET_M = 20000
MINIMUM_M = 5000
MAX_PROJECTED_HOURS = 4


def log_prior_eta(eta):
    eta = np.asarray(eta)
    return (-0.5 * ((eta - PRIOR_LOG_MEAN) / PRIOR_LOG_SD) ** 2
            - math.log(PRIOR_LOG_SD * math.sqrt(2 * math.pi)))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rk4_step(S, I, R, C, beta, local_weight, H, N, dt):
    def rhs(s, infected):
        rate = local_weight * beta * infected / N + H
        infections = s * rate
        recoveries = GAMMA * infected
        return -infections, infections - recoveries, recoveries, infections

    k1 = rhs(S, I)
    k2 = rhs(S + .5*dt*k1[0], I + .5*dt*k1[1])
    k3 = rhs(S + .5*dt*k2[0], I + .5*dt*k2[1])
    k4 = rhs(S + dt*k3[0], I + dt*k3[1])
    return tuple(value + dt*(a + 2*b + 2*c + d)/6
                 for value, a, b, c, d in zip((S, I, R, C), k1, k2, k3, k4))


def simulate_weekly_target(beta, *, N, M, I0, return_states=False):
    """Full weekly model; H is recomputed at each candidate's week start."""
    beta = np.asarray(beta, dtype=np.float64)
    single = beta.ndim == 1
    if single:
        beta = beta[None]
    count = len(beta)
    S = np.broadcast_to((N-I0)[None], (count, 51)).astype(np.float64).copy()
    I = np.broadcast_to(I0[None], (count, 51)).astype(np.float64).copy()
    R = np.zeros_like(S)
    C = np.zeros_like(S)
    local_weight = 1 - P_A * M.sum(axis=1) / N
    mu = np.empty((count, 51, WEEKS), dtype=np.float64)
    H_sequence = np.empty_like(mu)
    if return_states:
        state_S = np.empty((count, 51, WEEKS+1))
        state_I = np.empty_like(state_S)
        state_R = np.empty_like(state_S)
        state_S[:, :, 0], state_I[:, :, 0], state_R[:, :, 0] = S, I, R
    dt = 1 / RK4_SUBSTEPS
    for week in range(WEEKS):
        pressure = beta * I / N
        H = P_A * (pressure @ M.T) / N
        H_sequence[:, :, week] = H
        start_C = C.copy()
        for _ in range(RK4_SUBSTEPS):
            S, I, R, C = rk4_step(
                S, I, R, C, beta, local_weight, H, N, dt
            )
        mu[:, :, week] = C - start_C
        if return_states:
            state_S[:, :, week+1] = S
            state_I[:, :, week+1] = I
            state_R[:, :, week+1] = R
    if not np.isfinite(mu).all() or np.min(mu) < -1e-6:
        raise FloatingPointError("Weekly target produced invalid incidence")
    result = {"mu": np.maximum(mu, 0.), "H": H_sequence}
    if return_states:
        result.update(S=state_S, I=state_I, R=state_R)
    if single:
        result = {key: value[0] for key, value in result.items()}
    return result


def simulate_local_grid(eta, state, H, *, N, M, I0, return_states=False):
    """Local weekly model with the supplied 52 values held exactly constant."""
    eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    beta = np.exp(eta)
    count = len(beta)
    population = float(N[state])
    local_weight = 1 - P_A * M[state].sum() / population
    S = np.full(count, population-I0[state], dtype=np.float64)
    I = np.full(count, I0[state], dtype=np.float64)
    R = np.zeros(count)
    C = np.zeros(count)
    mu = np.empty((count, WEEKS))
    if return_states:
        state_S = np.empty((count, WEEKS+1))
        state_I = np.empty_like(state_S)
        state_R = np.empty_like(state_S)
        state_S[:, 0], state_I[:, 0], state_R[:, 0] = S, I, R
    dt = 1 / RK4_SUBSTEPS
    for week in range(WEEKS):
        start_C = C.copy()
        for _ in range(RK4_SUBSTEPS):
            S, I, R, C = rk4_step(
                S, I, R, C, beta, local_weight, H[week], population, dt
            )
        mu[:, week] = C - start_C
        if return_states:
            state_S[:, week+1] = S
            state_I[:, week+1] = I
            state_R[:, week+1] = R
    result = {"mu": np.maximum(mu, 0.)}
    if return_states:
        result.update(S=state_S, I=state_I, R=state_R)
    return result


def generate_epidemics(N, M, I0):
    eta_all, beta_all, Y_all, mu_all, H_all = [], [], [], [], []
    for epidemic in range(10):
        prior_seed, observation_seed = np.random.SeedSequence(
            [SEED, epidemic]
        ).spawn(2)
        eta = np.random.default_rng(prior_seed).normal(
            PRIOR_LOG_MEAN, PRIOR_LOG_SD, 51
        )
        beta = np.exp(eta)
        result = simulate_weekly_target(beta, N=N, M=M, I0=I0)
        Y = np.random.default_rng(observation_seed).poisson(P_REPORT * result["mu"])
        eta_all.append(eta); beta_all.append(beta); Y_all.append(Y)
        mu_all.append(result["mu"]); H_all.append(result["H"])
    data = {"eta_true": np.asarray(eta_all), "beta_true": np.asarray(beta_all),
            "Y": np.asarray(Y_all), "mu_true": np.asarray(mu_all),
            "H_oracle": np.asarray(H_all)}
    np.savez_compressed(RESULTS / "coherent_weekly_test_epidemics.npz", **data)
    return data


def local_log_values(eta, state, H, Y, N, M, I0):
    mu = simulate_local_grid(eta, state, H, N=N, M=M, I0=I0)["mu"]
    likelihood = poisson_log_likelihood(Y[state][None], mu, axis=1)
    return log_prior_eta(eta) + likelihood


def normalized_grid(grid, log_density):
    peak = np.max(log_density)
    relative = np.exp(log_density - peak)
    mass = np.diff(grid) * (relative[:-1] + relative[1:]) / 2
    area = mass.sum()
    density = relative / area
    cdf = np.r_[0., np.cumsum(mass) / area]
    cdf[-1] = 1.
    return density, cdf, peak + math.log(area)


def moments(grid, density):
    mean = np.trapezoid(grid*density, grid)
    variance = np.trapezoid((grid-mean)**2*density, grid)
    return float(mean), float(np.sqrt(max(variance, 0.)))


def construct_local_posterior(state, H, Y, N, M, I0):
    half_width = 6 * PRIOR_LOG_SD
    for _ in range(10):
        coarse = np.linspace(PRIOR_LOG_MEAN-half_width,
                             PRIOR_LOG_MEAN+half_width, COARSE_POINTS)
        logp = local_log_values(coarse, state, H, Y, N, M, I0)
        boundary = np.exp(np.max(logp[[0, -1]]) - np.max(logp))
        if boundary <= 1e-8:
            break
        half_width *= 1.5
    else:
        raise RuntimeError(f"State {state}: coarse grid failed")
    density, _, _ = normalized_grid(coarse, logp)
    _, sd = moments(coarse, density)
    mode = coarse[np.argmax(logp)]
    refined_half_width = max(8*sd, .05)
    for _ in range(12):
        grid = np.linspace(mode-refined_half_width,
                           mode+refined_half_width, FINAL_POINTS)
        logp = local_log_values(grid, state, H, Y, N, M, I0)
        boundary = np.exp(np.max(logp[[0, -1]]) - np.max(logp))
        if boundary <= 1e-8:
            break
        refined_half_width *= 1.5
    else:
        raise RuntimeError(f"State {state}: refined grid failed")
    density, cdf, log_normalizer = normalized_grid(grid, logp)
    return {"grid": grid, "density": density, "cdf": cdf,
            "log_normalizer": log_normalizer, "boundary_ratio": float(boundary)}


def construct_proposal(H, Y, N, M, I0):
    return [construct_local_posterior(state, H[state], Y, N, M, I0)
            for state in range(51)]


def sample_piecewise_linear(posterior, uniforms):
    grid, density, cdf = (posterior[key] for key in ("grid", "density", "cdf"))
    index = np.clip(np.searchsorted(cdf, uniforms, side="right")-1,
                    0, len(grid)-2)
    width = grid[index+1]-grid[index]
    d0, d1 = density[index], density[index+1]
    slope = (d1-d0)/width
    remaining = uniforms-cdf[index]
    root = np.sqrt(np.maximum(d0*d0+2*slope*remaining, 0.))
    offset = np.where(np.abs(slope) < 1e-14,
                      remaining/np.maximum(d0, np.finfo(float).tiny),
                      2*remaining/np.maximum(d0+root, np.finfo(float).tiny))
    offset = np.clip(offset, 0, width)
    return grid[index]+offset, np.log(np.maximum(d0+slope*offset,
                                                  np.finfo(float).tiny))


def sample_product(proposal, count, epidemic):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, epidemic, count]))
    eta = np.empty((count, 51))
    log_q = np.zeros(count)
    for state, posterior in enumerate(proposal):
        eta[:, state], component = sample_piecewise_linear(
            posterior, rng.random(count)
        )
        log_q += component
    return eta, log_q


def evaluate_target(eta, Y, N, M, I0, batch_size=250):
    log_target = np.empty(len(eta))
    started = perf_counter()
    for start in range(0, len(eta), batch_size):
        values = eta[start:start+batch_size]
        result = simulate_weekly_target(np.exp(values), N=N, M=M, I0=I0)
        likelihood = poisson_log_likelihood(
            Y[None], result["mu"], axis=(1, 2)
        )
        log_target[start:start+len(values)] = (
            log_prior_eta(values).sum(axis=1) + likelihood
        )
    return log_target, perf_counter()-started


def diagnostics(log_weight):
    shifted = log_weight-np.max(log_weight)
    weights = np.exp(shifted); weights /= weights.sum()
    sum_squares = np.dot(weights, weights)
    ess = 1/sum_squares
    cv2 = len(weights)*sum_squares-1
    np.testing.assert_allclose(ess/len(weights), 1/(1+cv2), rtol=1e-13)
    return {"ESS": float(ess), "ESS_fraction": float(ess/len(weights)),
            "CV_squared": float(cv2),
            "max_normalized_weight": float(weights.max()),
            "sd_log_weight": float(log_weight.std()),
            "min_log_weight": float(log_weight.min()),
            "max_log_weight": float(log_weight.max())}


def evaluate_proposal(epidemic, proposal, count, Y, N, M, I0, save):
    eta, log_q = sample_product(proposal, count, epidemic)
    log_target, runtime = evaluate_target(eta, Y, N, M, I0)
    row = {"epidemic_id": epidemic, "proposal_type": "oracle_weekly_coherent",
           "M": count, **diagnostics(log_target-log_q),
           "coupled_runtime_seconds": runtime}
    if save:
        SAMPLES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SAMPLES/f"epidemic_{epidemic:02d}.npz",
                            eta=eta, beta=np.exp(eta), log_q=log_q,
                            log_target=log_target, log_weight=log_target-log_q)
    return row


def sanity_discrepancies(beta, H, full, labels, N, M, I0, epidemic):
    rows = []
    for state in range(51):
        local = simulate_local_grid([np.log(beta[state])], state, H[state],
                                    N=N, M=M, I0=I0, return_states=True)
        rows.append({
            "epidemic_id": epidemic, "state": labels[state],
            "max_abs_S": float(np.max(np.abs(local["S"][0]-full["S"][state]))),
            "max_abs_I": float(np.max(np.abs(local["I"][0]-full["I"][state]))),
            "max_abs_R": float(np.max(np.abs(local["R"][0]-full["R"][state]))),
            "max_abs_incidence": float(np.max(np.abs(local["mu"][0]-full["mu"][state]))),
        })
    return rows


def local_diagnostics(proposal, eta_true, labels):
    rows = []
    for state, posterior in enumerate(proposal):
        grid, density, cdf = (posterior[key] for key in ("grid", "density", "cdf"))
        mean, sd = moments(grid, density)
        q05, q25, median, q75, q95 = np.interp((.05,.25,.5,.75,.95), cdf, grid)
        rows.append({"state": labels[state], "posterior_mean": mean,
                     "posterior_median": float(median), "posterior_SD": sd,
                     "lower_50": float(q25), "upper_50": float(q75),
                     "lower_90": float(q05), "upper_90": float(q95),
                     "eta_true": float(eta_true[state])})
    write_csv(RESULTS/"coherent_weekly_oracle_local_posteriors.csv", rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, state_name in zip(axes.flat, ("CA","NY","DC","WY")):
        state = list(labels).index(state_name); posterior = proposal[state]
        ax.plot(posterior["grid"], posterior["density"])
        ax.axvline(eta_true[state], color="black", ls="--")
        ax.set(title=state_name, xlabel="eta = log(beta)", ylabel="Density")
    fig.tight_layout()
    fig.savefig(RESULTS/"coherent_weekly_oracle_local_posteriors.png", dpi=160)
    plt.close(fig)


def choose_m(runtime_100):
    projected = runtime_100/100*(10*TARGET_M)
    if projected <= MAX_PROJECTED_HOURS*3600:
        return TARGET_M, projected
    affordable = int(TARGET_M*MAX_PROJECTED_HOURS*3600/projected)
    return max(MINIMUM_M, 1000*(affordable//1000)), projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    labels, _, N, M = load_data(ROOT); I0 = initial_infected(labels)
    data = generate_epidemics(N, M, I0)
    sanity_rows = []
    proposals = {}
    for epidemic in range(10):
        full = simulate_weekly_target(data["beta_true"][epidemic], N=N, M=M,
                                      I0=I0, return_states=True)
        sanity_rows.extend(sanity_discrepancies(
            data["beta_true"][epidemic], data["H_oracle"][epidemic], full,
            labels, N, M, I0, epidemic
        ))
    write_csv(RESULTS/"coherent_weekly_oracle_sanity.csv", sanity_rows)

    proposals[0] = construct_proposal(data["H_oracle"][0], data["Y"][0], N, M, I0)
    local_diagnostics(proposals[0], data["eta_true"][0], labels)
    smoke = evaluate_proposal(0, proposals[0], 100, data["Y"][0], N, M, I0, False)
    print(json.dumps({"smoke": smoke}), flush=True)
    chosen_M, projected = choose_m(smoke["coupled_runtime_seconds"])
    print(json.dumps({"runtime_100_seconds": smoke["coupled_runtime_seconds"],
                      "projected_10x20000_seconds": projected,
                      "chosen_common_M": chosen_M}), flush=True)
    rows = []
    for epidemic in range(10):
        proposal = proposals.get(epidemic) or construct_proposal(
            data["H_oracle"][epidemic], data["Y"][epidemic], N, M, I0
        )
        row = evaluate_proposal(epidemic, proposal, chosen_M,
                                data["Y"][epidemic], N, M, I0, True)
        rows.append(row)
        write_csv(RESULTS/"coherent_weekly_oracle_importance_results.csv", rows)
        print(json.dumps(row), flush=True)
    ess = np.array([row["ESS_fraction"] for row in rows])
    summary = [{"proposal_type": "oracle_weekly_coherent",
                "number_of_epidemics": 10, "M": chosen_M,
                "median_ESS_fraction": float(np.median(ess)),
                "min_ESS_fraction": float(ess.min()),
                "max_ESS_fraction": float(ess.max()),
                "ESS_fraction_q10": float(np.quantile(ess,.1)),
                "ESS_fraction_q25": float(np.quantile(ess,.25)),
                "ESS_fraction_q75": float(np.quantile(ess,.75)),
                "ESS_fraction_q90": float(np.quantile(ess,.9)),
                "median_CV_squared": float(np.median([r["CV_squared"] for r in rows])),
                "median_max_normalized_weight": float(np.median(
                    [r["max_normalized_weight"] for r in rows]))}]
    write_csv(RESULTS/"coherent_weekly_oracle_importance_summary.csv", summary)
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(range(10), ess)
    ax.set(xlabel="Epidemic", ylabel="ESS fraction")
    fig.tight_layout(); fig.savefig(RESULTS/"coherent_weekly_oracle_ess.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"summary": summary[0], "RK4_substeps_per_week": RK4_SUBSTEPS,
                      "diffusion_used": False}), flush=True)


if __name__ == "__main__":
    main()
