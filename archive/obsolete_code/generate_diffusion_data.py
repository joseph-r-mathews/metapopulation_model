"""Generate deterministic-ODE prior-predictive diffusion datasets."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from deterministic_model import ATOL, RTOL, solve_coupled
from model import (GAMMA, N_WEEKS, P_A, P_REPORT, PRIOR_LOG_MEAN,
                   PRIOR_LOG_SD, initial_infected, load_data, observe)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "deterministic_data"
SPECS = {
    "eta_true": (np.float64, (51,)),
    "beta_true": (np.float64, (51,)),
    "mu": (np.float64, (51, 52)),
    "Y": (np.int32, (51, 52)),
    "H": (np.float64, (51, 52)),
}


def simulate_one(seed, split_id, index, N, M, I0):
    prior_seed, observation_seed = np.random.SeedSequence(
        [seed, split_id, index]
    ).spawn(2)
    eta = np.random.default_rng(prior_seed).normal(
        PRIOR_LOG_MEAN, PRIOR_LOG_SD, 51
    )
    beta = np.exp(eta)
    started = perf_counter()
    solved = solve_coupled(beta, N=N, M=M, I0=I0, dense_output=False)
    solve_seconds = perf_counter() - started
    mu = solved["incidence"]
    H = solved["external_foi"]
    Y = observe(mu, np.random.default_rng(observation_seed), P_REPORT)
    return eta, beta, mu, Y, H, solve_seconds


def deterministic_sanity(seed=20260924):
    labels, _, N, M = load_data(ROOT)
    I0 = initial_infected(labels)
    prior_seed, observation_seed = np.random.SeedSequence([seed, 99, 0]).spawn(2)
    eta = np.random.default_rng(prior_seed).normal(
        PRIOR_LOG_MEAN, PRIOR_LOG_SD, 51
    )
    beta = np.exp(eta)
    started = perf_counter()
    solved = solve_coupled(beta, N=N, M=M, I0=I0, dense_output=False)
    solve_seconds = perf_counter() - started
    mu, H = solved["incidence"], solved["external_foi"]
    Y = observe(mu, np.random.default_rng(observation_seed), P_REPORT)
    assert solved["S"].shape == solved["I"].shape == solved["R"].shape == (51, 53)
    assert mu.shape == H.shape == Y.shape == (51, N_WEEKS)
    assert all(np.isfinite(solved[name]).all() for name in ("S", "I", "R"))
    assert np.isfinite(mu).all() and np.all(mu >= 0)
    assert np.isfinite(H).all() and np.all(H >= 0)
    assert np.issubdtype(Y.dtype, np.integer) and np.all(Y >= 0)
    np.testing.assert_array_equal(
        solved["incidence"], np.diff(solved["cumulative_incidence"], axis=1)
    )
    np.testing.assert_array_equal(
        solved["external_foi"],
        np.diff(solved["cumulative_external_foi"], axis=1),
    )
    summary = {
        "passed": True,
        "S_shape": list(solved["S"].shape),
        "I_shape": list(solved["I"].shape),
        "R_shape": list(solved["R"].shape),
        "mu_shape": list(mu.shape), "H_shape": list(H.shape),
        "Y_shape": list(Y.shape), "solve_seconds": solve_seconds,
        "incidence_definition": "C(w+1)-C(w)",
        "external_foi_definition": "Q(w+1)-Q(w), the weekly average/integral over a unit week",
        "observation_model": "Poisson(p_report * mu)",
    }
    print(json.dumps({"deterministic_sanity": summary}), flush=True)
    return summary


def generate_chunk(task):
    seed, split_id, start, count = task
    labels, _, N, M = load_data(ROOT)
    I0 = initial_infected(labels)
    arrays = {name: np.empty((count, *shape), dtype=dtype)
              for name, (dtype, shape) in SPECS.items()}
    solve_seconds = 0.
    for offset in range(count):
        eta, beta, mu, Y, H, elapsed = simulate_one(
            seed, split_id, start + offset, N, M, I0
        )
        values = {"eta_true": eta, "beta_true": beta, "mu": mu, "Y": Y, "H": H}
        for name in arrays:
            arrays[name][offset] = values[name]
        solve_seconds += elapsed
    return start, arrays, solve_seconds


def generate_split(name, count, split_id, seed, workers, output=DATA_ROOT):
    path = Path(output) / name
    complete = path / "COMPLETE"
    summary_path = path / "generation_summary.json"
    if complete.exists():
        with np.load(path / "metadata.npz", allow_pickle=False) as metadata:
            valid = (int(metadata["count"]) == count
                     and int(metadata["seed"]) == seed
                     and str(metadata["forward_model"]) == "deterministic_coupled_ode")
        if not valid:
            raise ValueError(f"Existing {path} is incompatible")
        print(f"Using complete {path} ({count} simulations)", flush=True)
        return json.loads(summary_path.read_text())
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Incomplete dataset directory exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    maps = {field: np.lib.format.open_memmap(
                path / f"{field}.npy", mode="w+", dtype=dtype,
                shape=(count, *shape)
            ) for field, (dtype, shape) in SPECS.items()}
    labels, subpop, N, M = load_data(ROOT)
    np.savez_compressed(
        path / "metadata.npz", labels=labels, subpop=subpop, N=N, M=M,
        I0=initial_infected(labels), gamma=GAMMA, p_a=P_A,
        p_report=P_REPORT, n_weeks=N_WEEKS,
        prior_log_mean=PRIOR_LOG_MEAN, prior_log_sd=PRIOR_LOG_SD,
        ode_method="DOP853", ode_rtol=RTOL, ode_atol=ATOL,
        forward_model="deterministic_coupled_ode",
        observation_model="poisson_reported_deterministic_incidence",
        external_foi_definition="weekly_average_Q_difference",
        seed=seed, split_id=split_id, count=count,
    )
    tasks = [(seed, split_id, start, min(20, count - start))
             for start in range(0, count, 20)]
    wall_started = perf_counter()
    total_solve_seconds = 0.
    if workers == 1:
        generated = map(generate_chunk, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        generated = pool.map(generate_chunk, tasks)
    try:
        for start, arrays, solve_seconds in generated:
            stop = start + len(arrays["Y"])
            for field, values in arrays.items():
                maps[field][start:stop] = values
            for array in maps.values():
                array.flush()
            total_solve_seconds += solve_seconds
            print(f"{name}: {stop}/{count}; {perf_counter()-wall_started:.1f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    del maps
    wall_seconds = perf_counter() - wall_started
    summary = {
        "split": name, "simulations": count, "workers": workers,
        "wall_runtime_seconds": wall_seconds,
        "summed_solve_seconds": total_solve_seconds,
        "average_deterministic_solve_seconds": total_solve_seconds / count,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    complete.write_text("complete\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--validation", type=int, default=2000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    deterministic_sanity(args.seed)
    if args.sanity_only:
        return
    summaries = []
    for split_id, (name, count) in enumerate((
        ("train", args.train), ("validation", args.validation), ("test", args.test)
    )):
        summaries.append(generate_split(name, count, split_id, args.seed,
                                        args.workers, DATA_ROOT))
    report = {
        "total_simulations": sum(row["simulations"] for row in summaries),
        "total_wall_runtime_seconds": sum(row["wall_runtime_seconds"] for row in summaries),
        "average_deterministic_solve_seconds": (
            sum(row["summed_solve_seconds"] for row in summaries)
            / sum(row["simulations"] for row in summaries)
        ),
        "splits": summaries,
    }
    (DATA_ROOT / "generation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
