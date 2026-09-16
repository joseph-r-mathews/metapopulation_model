"""FoI-decoupled importance proposals for the deterministic ODE posterior."""

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deterministic_model import (ATOL, RTOL, continuous_state_external,
                                 solve_coupled, solve_local,
                                 weekly_state_external)
from foi_diffusion import DEFAULT_CHECKPOINT, sample_external_foi
from generate_diffusion_data import DATA_ROOT
from model import (GAMMA, P_REPORT, PRIOR_LOG_MEAN, PRIOR_LOG_SD,
                   initial_infected, load_data, poisson_log_likelihood)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "deterministic_importance"
SAMPLES = RESULTS / "samples"
SEED = 20260926
COARSE_POINTS = 501
FINAL_POINTS = 2001
LOCAL_SUBSTEPS_PER_WEEK = 14
PRIMARY_TARGET_M = 20000
MINIMUM_M = 5000
MAX_PROJECTED_HOURS = 4
ORACLE_RELATIVE_TOL = 1e-3
_TARGET_CONTEXT = None


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


def load_test_data():
    required = ("Y", "H", "eta_true", "beta_true")
    paths = {name: DATA_ROOT / "test" / f"{name}.npy" for name in required}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required deterministic test arrays missing: {missing}")
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def local_incidence_grid(eta, state, external, N, I0):
    """RK4 integration of independent local ODEs for a vector of eta values."""
    eta = np.asarray(eta, dtype=np.float64)
    beta = np.exp(eta)
    population = float(N[state])
    local_weight = external.local_weight
    S = np.full(len(eta), population - I0[state], dtype=np.float64)
    I = np.full(len(eta), I0[state], dtype=np.float64)
    incidence = np.empty((len(eta), 52), dtype=np.float64)
    dt = 1 / LOCAL_SUBSTEPS_PER_WEEK

    def rhs(susceptible, infectious, time):
        rate = (local_weight * beta * infectious / population
                + float(external(time)))
        infection_flow = susceptible * rate
        return -infection_flow, infection_flow - GAMMA * infectious

    for week in range(52):
        start_S = S.copy()
        for substep in range(LOCAL_SUBSTEPS_PER_WEEK):
            time = week + substep * dt
            k1s, k1i = rhs(S, I, time)
            k2s, k2i = rhs(S + .5*dt*k1s, I + .5*dt*k1i,
                           time + .5*dt)
            k3s, k3i = rhs(S + .5*dt*k2s, I + .5*dt*k2i,
                           time + .5*dt)
            # Stay on the current side of a weekly piecewise-constant jump.
            right = np.nextafter(time + dt, time)
            k4s, k4i = rhs(S + dt*k3s, I + dt*k3i, right)
            S += dt * (k1s + 2*k2s + 2*k3s + k4s) / 6
            I += dt * (k1i + 2*k2i + 2*k3i + k4i) / 6
        incidence[:, week] = start_S - S
    if not np.isfinite(incidence).all() or np.min(incidence) < -1e-6:
        raise FloatingPointError(f"Invalid local integration for state {state}")
    return np.maximum(incidence, 0.)


def local_log_values(eta, state, external, Y, N, I0):
    incidence = local_incidence_grid(eta, state, external, N, I0)
    likelihood = poisson_log_likelihood(Y[state][None], incidence, axis=1)
    return log_prior_eta(eta) + likelihood, likelihood


def trapezoid_distribution(grid, log_density):
    peak = np.max(log_density)
    relative = np.exp(log_density - peak)
    segment_mass = np.diff(grid) * (relative[:-1] + relative[1:]) / 2
    area = segment_mass.sum()
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("Numerical posterior has zero or nonfinite mass")
    density = relative / area
    cdf = np.r_[0., np.cumsum(segment_mass) / area]
    cdf[-1] = 1.
    return density, cdf, peak + math.log(area)


def posterior_moments(grid, density):
    mean = np.trapezoid(grid * density, grid)
    variance = np.trapezoid((grid - mean) ** 2 * density, grid)
    return float(mean), float(np.sqrt(max(variance, 0.)))


def construct_local_posterior(state, external, Y, N, I0):
    half_width = 6 * PRIOR_LOG_SD
    for _ in range(10):
        coarse = np.linspace(PRIOR_LOG_MEAN - half_width,
                             PRIOR_LOG_MEAN + half_width, COARSE_POINTS)
        coarse_log, _ = local_log_values(coarse, state, external, Y, N, I0)
        if np.isfinite(coarse_log).any():
            boundary_ratio = np.exp(np.max(coarse_log[[0, -1]])
                                    - np.max(coarse_log))
            if boundary_ratio <= 1e-8:
                break
        half_width *= 1.5
    else:
        raise RuntimeError(f"State {state}: coarse grid did not contain posterior")
    coarse_density, _, _ = trapezoid_distribution(coarse, coarse_log)
    _, coarse_sd = posterior_moments(coarse, coarse_density)
    mode = coarse[np.argmax(coarse_log)]
    refined_half_width = max(8 * coarse_sd, 0.05)
    for _ in range(12):
        grid = np.linspace(mode - refined_half_width,
                           mode + refined_half_width, FINAL_POINTS)
        log_density, log_likelihood = local_log_values(
            grid, state, external, Y, N, I0
        )
        boundary_ratio = np.exp(np.max(log_density[[0, -1]])
                                - np.max(log_density))
        if boundary_ratio <= 1e-8:
            break
        refined_half_width *= 1.5
    else:
        raise RuntimeError(f"State {state}: refined grid boundaries did not decay")
    density, cdf, log_normalizer = trapezoid_distribution(grid, log_density)
    return {
        "grid": grid, "density": density, "cdf": cdf,
        "log_likelihood": log_likelihood,
        "log_normalizer": log_normalizer,
        "boundary_ratio": float(boundary_ratio),
    }


def construct_product_proposal(kind, H, true_solution, beta_true, Y, N, M, I0):
    posteriors = []
    for state in range(51):
        if kind == "oracle_continuous":
            external = continuous_state_external(
                true_solution, beta_true, state, N=N, M=M
            )
        else:
            external = weekly_state_external(H[state], state, N=N, M=M)
        posteriors.append(
            construct_local_posterior(state, external, Y, N, I0)
        )
    return posteriors


def sample_piecewise_linear(posterior, uniforms):
    grid, density, cdf = (posterior[key] for key in ("grid", "density", "cdf"))
    index = np.searchsorted(cdf, uniforms, side="right") - 1
    index = np.clip(index, 0, len(grid) - 2)
    width = grid[index + 1] - grid[index]
    d0, d1 = density[index], density[index + 1]
    slope = (d1 - d0) / width
    mass = uniforms - cdf[index]
    discriminant = np.maximum(d0*d0 + 2*slope*mass, 0.)
    denominator = d0 + np.sqrt(discriminant)
    offset = np.where(np.abs(slope) < 1e-14,
                      mass / np.maximum(d0, np.finfo(float).tiny),
                      2*mass / np.maximum(denominator, np.finfo(float).tiny))
    offset = np.clip(offset, 0, width)
    values = grid[index] + offset
    evaluated_density = d0 + slope * offset
    return values, np.log(np.maximum(evaluated_density, np.finfo(float).tiny)), index, offset/width


def sample_product(posteriors, count, rng):
    eta = np.empty((count, 51), dtype=np.float64)
    log_q_components = np.empty_like(eta)
    local_likelihood = np.empty_like(eta)
    for state, posterior in enumerate(posteriors):
        values, log_density, index, fraction = sample_piecewise_linear(
            posterior, rng.random(count)
        )
        eta[:, state] = values
        log_q_components[:, state] = log_density
        # The piecewise-linear normalized density is the numerical q_i used
        # for both sampling and evaluation. Its implied local likelihood keeps
        # the prior-cancellation identity exact for that same numerical q_i.
        local_likelihood[:, state] = (
            log_density + posterior["log_normalizer"] - log_prior_eta(values)
        )
    return eta, log_q_components.sum(axis=1), local_likelihood.sum(axis=1)


def initialize_target(N, M, I0, Y):
    global _TARGET_CONTEXT
    _TARGET_CONTEXT = (N, M, I0, Y)


def target_worker(eta):
    N, M, I0, Y = _TARGET_CONTEXT
    solved = solve_coupled(np.exp(eta), N=N, M=M, I0=I0,
                           rtol=RTOL, atol=ATOL, dense_output=False)
    coupled_likelihood = poisson_log_likelihood(Y, solved["incidence"], P_REPORT)
    target = log_prior_eta(eta).sum() + coupled_likelihood
    return float(target), float(coupled_likelihood)


def evaluate_target(eta, N, M, I0, Y, workers):
    started = perf_counter()
    if workers == 1:
        initialize_target(N, M, I0, Y)
        values = [target_worker(value) for value in eta]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_target,
            initargs=(N, M, I0, Y),
        ) as pool:
            values = list(pool.map(target_worker, eta, chunksize=20))
    elapsed = perf_counter() - started
    values = np.asarray(values)
    return values[:, 0], values[:, 1], elapsed


def weight_diagnostics(log_weight):
    if not np.isfinite(log_weight).all():
        raise FloatingPointError("Importance log weights are not all finite")
    shifted = log_weight - np.max(log_weight)
    normalized = np.exp(shifted)
    normalized /= normalized.sum()
    sum_squares = np.dot(normalized, normalized)
    ess = 1 / sum_squares
    cv_squared = len(log_weight) * sum_squares - 1
    np.testing.assert_allclose(ess / len(log_weight),
                               1 / (1 + cv_squared), rtol=1e-13)
    return {
        "ESS": float(ess), "ESS_fraction": float(ess / len(log_weight)),
        "CV_squared": float(cv_squared),
        "max_normalized_weight": float(normalized.max()),
        "mean_log_weight": float(log_weight.mean()),
        "sd_log_weight": float(log_weight.std()),
        "min_log_weight": float(log_weight.min()),
        "max_log_weight": float(log_weight.max()),
    }


def evaluate_proposal(epidemic_id, kind, posteriors, count, N, M, I0, Y,
                      workers, save_samples):
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, epidemic_id, sum(map(ord, kind)), count])
    )
    eta, log_q, summed_local_likelihood = sample_product(posteriors, count, rng)
    log_target, coupled_likelihood, runtime = evaluate_target(
        eta, N, M, I0, Y, workers
    )
    log_weight = log_target - log_q
    likelihood_ratio = coupled_likelihood - summed_local_likelihood
    cancellation_difference = log_weight - likelihood_ratio
    cancellation_range = float(np.ptp(cancellation_difference))
    row = {
        "epidemic_id": epidemic_id, "proposal_type": kind,
        "diffusion_draw_id": "NA", "M": count,
        **weight_diagnostics(log_weight),
        "coupled_runtime_seconds": runtime,
        "prior_cancellation_range": cancellation_range,
    }
    if save_samples:
        SAMPLES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            SAMPLES / f"epidemic_{epidemic_id:02d}_{kind}.npz",
            eta=eta, beta=np.exp(eta), log_q=log_q,
            log_target=log_target, log_weight=log_weight,
        )
    return row


def oracle_identity(true_solution, beta_true, labels, N, M, I0, epidemic_id):
    rows = []
    for state in range(51):
        external = continuous_state_external(
            true_solution, beta_true, state, N=N, M=M
        )
        local = solve_local(beta_true[state], state, external, N=N, I0=I0)
        references = {
            "S": true_solution["S"][state],
            "I": true_solution["I"][state],
            "R": true_solution["R"][state],
            "weekly_incidence": true_solution["incidence"][state],
        }
        differences = {
            "S": local["S"] - references["S"],
            "I": local["I"] - references["I"],
            "R": local["R"] - references["R"],
            "weekly_incidence": local["incidence"] - references["weekly_incidence"],
        }
        correlations = {}
        for name in references:
            reference_centered = references[name] - np.mean(references[name])
            local_values = references[name] + differences[name]
            local_centered = local_values - np.mean(local_values)
            denominator = (np.linalg.norm(reference_centered)
                           * np.linalg.norm(local_centered))
            correlations[name] = (float(np.dot(reference_centered, local_centered)
                                        / denominator)
                                  if denominator > 0 else 1.)
        rows.append({
            "epidemic_id": epidemic_id, "state": labels[state],
            **{f"max_abs_{name}": float(np.max(np.abs(differences[name])))
               for name in references},
            **{f"relative_L2_{name}": float(
                np.linalg.norm(differences[name])
                / max(np.linalg.norm(references[name]), 1.)
            ) for name in references},
            **{f"shape_correlation_{name}": correlations[name]
               for name in references},
        })
    maximum_relative = max(
        row[key] for row in rows for key in
        ("relative_L2_S", "relative_L2_I", "relative_L2_R",
         "relative_L2_weekly_incidence")
    )
    if maximum_relative > ORACLE_RELATIVE_TOL:
        raise RuntimeError(
            f"Epidemic {epidemic_id}: oracle relative identity failed "
            f"({maximum_relative} > {ORACLE_RELATIVE_TOL})"
        )
    shape_failures = [
        (row["state"], name, row[f"shape_correlation_{name}"])
        for row in rows
        for name in ("S", "I", "R", "weekly_incidence")
        if row[f"shape_correlation_{name}"] < .99
        and row[f"relative_L2_{name}"] > 1e-5
    ]
    if shape_failures:
        raise RuntimeError(
            f"Epidemic {epidemic_id}: obvious oracle trajectory-shape "
            f"discrepancy: {shape_failures}"
        )
    return rows


def local_diagnostic_rows(posteriors_by_kind, eta_true, labels):
    rows = []
    for kind, posteriors in posteriors_by_kind.items():
        for state, posterior in enumerate(posteriors):
            grid, density, cdf = (posterior[key]
                                  for key in ("grid", "density", "cdf"))
            mean, sd = posterior_moments(grid, density)
            q05, q25, median, q75, q95 = np.interp(
                (.05, .25, .5, .75, .95), cdf, grid
            )
            rows.append({
                "epidemic_id": 0, "state": labels[state],
                "proposal_type": kind, "posterior_mean": mean,
                "posterior_median": float(median), "posterior_SD": sd,
                "lower_50": float(q25), "upper_50": float(q75),
                "lower_90": float(q05), "upper_90": float(q95),
                "true_eta": float(eta_true[state]),
                "grid_points": len(grid),
                "grid_lower": float(grid[0]), "grid_upper": float(grid[-1]),
                "boundary_to_peak_density": posterior["boundary_ratio"],
            })
    return rows


def plot_local(posteriors_by_kind, eta_true, labels):
    colors = {"oracle_continuous": "#0072B2", "oracle_weekly": "#E69F00",
              "diffusion_median": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, state_name in zip(axes.flat, ("CA", "NY", "DC", "WY")):
        state = list(labels).index(state_name)
        for kind, posteriors in posteriors_by_kind.items():
            posterior = posteriors[state]
            ax.plot(posterior["grid"], posterior["density"],
                    color=colors[kind], label=kind)
        ax.axvline(eta_true[state], color="black", ls="--", label="true eta")
        ax.set(title=state_name, xlabel="eta = log(beta)", ylabel="Density")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "deterministic_target_local_posteriors.png", dpi=160)
    plt.close(fig)


def summarize(rows):
    summaries = []
    for kind in ("oracle_continuous", "oracle_weekly", "diffusion_median"):
        selected = [row for row in rows if row["proposal_type"] == kind]
        ess = np.array([row["ESS_fraction"] for row in selected])
        summaries.append({
            "proposal_type": kind, "number_of_epidemics": len(selected),
            "median_ESS_fraction": float(np.median(ess)),
            "min_ESS_fraction": float(ess.min()),
            "max_ESS_fraction": float(ess.max()),
            "ESS_fraction_q10": float(np.quantile(ess, .10)),
            "ESS_fraction_q25": float(np.quantile(ess, .25)),
            "ESS_fraction_q50": float(np.quantile(ess, .50)),
            "ESS_fraction_q75": float(np.quantile(ess, .75)),
            "ESS_fraction_q90": float(np.quantile(ess, .90)),
            "median_CV_squared": float(np.median(
                [row["CV_squared"] for row in selected]
            )),
            "median_max_normalized_weight": float(np.median(
                [row["max_normalized_weight"] for row in selected]
            )),
        })
    return summaries


def make_summary_plots(rows):
    kinds = ("oracle_continuous", "oracle_weekly", "diffusion_median")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([[row["ESS_fraction"] for row in rows
                 if row["proposal_type"] == kind] for kind in kinds],
               tick_labels=kinds)
    ax.set(ylabel="ESS fraction")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "deterministic_target_ess_by_proposal.png", dpi=160)
    plt.close(fig)

    continuous = {row["epidemic_id"]: row for row in rows
                  if row["proposal_type"] == "oracle_continuous"}
    diffusion = {row["epidemic_id"]: row for row in rows
                 if row["proposal_type"] == "diffusion_median"}
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([continuous[i]["ESS_fraction"] for i in sorted(continuous)],
               [diffusion[i]["ESS_fraction"] for i in sorted(diffusion)])
    limits = [1 / max(row["M"] for row in rows), 1]
    ax.plot(limits, limits, "k--", lw=1)
    ax.set(xscale="log", yscale="log",
           xlabel="Oracle-continuous ESS fraction",
           ylabel="Diffusion-median ESS fraction")
    fig.tight_layout()
    fig.savefig(RESULTS / "deterministic_target_oracle_vs_diffusion.png", dpi=160)
    plt.close(fig)


def prepare_epidemic(epidemic_id, test, labels, N, M, I0, device,
                     check_identity=True):
    Y = np.asarray(test["Y"][epidemic_id])
    eta_true = np.asarray(test["eta_true"][epidemic_id], dtype=np.float64)
    beta_true = np.asarray(test["beta_true"][epidemic_id], dtype=np.float64)
    np.testing.assert_allclose(np.exp(eta_true), beta_true, rtol=2e-15)
    true_solution = solve_coupled(beta_true, N=N, M=M, I0=I0,
                                  rtol=RTOL, atol=ATOL, dense_output=True)
    np.testing.assert_allclose(true_solution["external_foi"],
                               np.asarray(test["H"][epidemic_id]),
                               rtol=2e-8, atol=2e-12)
    identity = (oracle_identity(true_solution, beta_true, labels, N, M, I0,
                                epidemic_id) if check_identity else [])
    draws = sample_external_foi(Y, 500, checkpoint=DEFAULT_CHECKPOINT,
                                device=device, seed=SEED + epidemic_id)
    median = np.median(draws, axis=0)
    proposals = {
        "oracle_continuous": construct_product_proposal(
            "oracle_continuous", None, true_solution, beta_true, Y, N, M, I0
        ),
        "oracle_weekly": construct_product_proposal(
            "oracle_weekly", true_solution["external_foi"], true_solution,
            beta_true, Y, N, M, I0
        ),
        "diffusion_median": construct_product_proposal(
            "diffusion_median", median, true_solution, beta_true, Y, N, M, I0
        ),
    }
    return Y, eta_true, proposals, identity


def choose_primary_m(smoke_runtime):
    projected = smoke_runtime / 100 * (10 * 3 * PRIMARY_TARGET_M)
    if projected <= MAX_PROJECTED_HOURS * 3600:
        return PRIMARY_TARGET_M, projected
    affordable = int(PRIMARY_TARGET_M * MAX_PROJECTED_HOURS * 3600 / projected)
    chosen = max(MINIMUM_M, min(PRIMARY_TARGET_M,
                                1000 * (affordable // 1000)))
    return chosen, projected


def read_completed_results(path):
    if not path.exists():
        return []
    integer_fields = ("epidemic_id", "M")
    numeric_fields = (
        "ESS", "ESS_fraction", "CV_squared", "max_normalized_weight",
        "mean_log_weight", "sd_log_weight", "min_log_weight",
        "max_log_weight", "coupled_runtime_seconds", "prior_cancellation_range",
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    test = load_test_data()
    labels, _, N, M = load_data(ROOT)
    I0 = initial_infected(labels)
    if len(test["Y"]) < 10:
        raise RuntimeError("Deterministic test dataset has fewer than 10 examples")

    print("Target: pi(eta|Y) from the full coupled deterministic ODE.", flush=True)
    print("Local proposals: deterministic local ODEs with one fixed H per proposal.",
          flush=True)
    print(f"Grid: {COARSE_POINTS} coarse, {FINAL_POINTS} final; adaptive boundaries; "
          f"piecewise-linear density/CDF; local RK4 {LOCAL_SUBSTEPS_PER_WEEK} steps/week.",
          flush=True)

    # Check all ten decompositions before constructing any new proposal.
    identity_rows = []
    for epidemic_id in range(10):
        beta_true = np.asarray(test["beta_true"][epidemic_id], dtype=np.float64)
        true_solution = solve_coupled(
            beta_true, N=N, M=M, I0=I0, rtol=RTOL, atol=ATOL,
            dense_output=True,
        )
        identity_rows.extend(oracle_identity(
            true_solution, beta_true, labels, N, M, I0, epidemic_id
        ))
    write_csv(RESULTS / "deterministic_target_oracle_identity.csv", identity_rows)

    # Implementation smoke test on epidemic zero.
    Y0, eta0, proposals0, identity0 = prepare_epidemic(
        0, test, labels, N, M, I0, args.device, check_identity=False
    )
    diagnostic_rows = local_diagnostic_rows(proposals0, eta0, labels)
    write_csv(RESULTS / "deterministic_target_local_posteriors.csv", diagnostic_rows)
    plot_local(proposals0, eta0, labels)

    smoke_rows = []
    for kind, posteriors in proposals0.items():
        row = evaluate_proposal(0, kind, posteriors, 100, N, M, I0, Y0,
                                args.workers, save_samples=False)
        smoke_rows.append(row)
        print(json.dumps({"smoke": row}), flush=True)
    write_csv(RESULTS / "deterministic_target_importance_smoke.csv", smoke_rows)
    chosen_M, projected_20k = choose_primary_m(
        smoke_rows[0]["coupled_runtime_seconds"]
    )
    print(json.dumps({
        "benchmark_100_wall_seconds": smoke_rows[0]["coupled_runtime_seconds"],
        "projected_600000_wall_seconds": projected_20k,
        "projected_600000_hours": projected_20k / 3600,
        "chosen_common_M": chosen_M,
        "runtime_cap_hours_for_reduction": MAX_PROJECTED_HOURS,
    }), flush=True)

    results_path = RESULTS / "deterministic_target_importance_results.csv"
    rows = read_completed_results(results_path)
    completed = {(row["epidemic_id"], row["proposal_type"], row["M"])
                 for row in rows}
    for epidemic_id in range(10):
        if epidemic_id == 0:
            Y, proposals = Y0, proposals0
        else:
            Y, _, proposals, identity = prepare_epidemic(
                epidemic_id, test, labels, N, M, I0, args.device,
                check_identity=False,
            )
        for kind, posteriors in proposals.items():
            if (epidemic_id, kind, chosen_M) in completed:
                print(json.dumps({"resuming_skip_completed": {
                    "epidemic_id": epidemic_id, "proposal_type": kind,
                    "M": chosen_M,
                }}), flush=True)
                continue
            row = evaluate_proposal(
                epidemic_id, kind, posteriors, chosen_M, N, M, I0, Y,
                args.workers, save_samples=True,
            )
            rows.append(row)
            write_csv(results_path, rows)
            print(json.dumps({"primary": row}), flush=True)
    summaries = summarize(rows)
    write_csv(RESULTS / "deterministic_target_importance_summary.csv", summaries)
    make_summary_plots(rows)
    maximum_identity = {
        key: max(row[key] for row in identity_rows)
        for key in ("max_abs_S", "max_abs_I", "max_abs_R",
                    "max_abs_weekly_incidence")
    }
    maximum_identity_relative_L2 = {
        key: max(row[key] for row in identity_rows)
        for key in ("relative_L2_S", "relative_L2_I", "relative_L2_R",
                    "relative_L2_weekly_incidence")
    }
    report = {
        "target": "pi(eta|Y), full coupled deterministic ODE Poisson posterior",
        "local_proposals": "deterministic local ODE with fixed H",
        "oracle_identity_maximum_errors": maximum_identity,
        "oracle_identity_maximum_relative_L2": maximum_identity_relative_L2,
        "oracle_identity_relative_tolerance": ORACLE_RELATIVE_TOL,
        "grid": {"coarse_points": COARSE_POINTS, "final_points": FINAL_POINTS,
                 "boundary_ratio_max": 1e-8,
                 "local_RK4_substeps_per_week": LOCAL_SUBSTEPS_PER_WEEK},
        "benchmark_100_wall_seconds": smoke_rows[0]["coupled_runtime_seconds"],
        "projected_600000_wall_seconds": projected_20k,
        "actual_M": chosen_M, "primary_summary": summaries,
        "diffusion_draw_experiment_run": False,
        "numerical_issues": [],
    }
    (RESULTS / "deterministic_target_importance_final_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
