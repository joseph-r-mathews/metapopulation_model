"""Cache independent simulation triples; never called implicitly by training."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from model import load_data, observe, simulate, weekly_summary

ROOT = Path(__file__).resolve().parent
N_DAYS, GAMMA, P_A, P_REPORT = 364, 1 / 7, 0.5, 0.4
INITIAL_INFECTED = {"CA": 100, "TX": 100, "FL": 100, "NY": 100}


def generate_chunk(task):
    seed, split, start, count, check = task
    labels, subpop, N, M = load_data(ROOT)
    I0 = np.array([INITIAL_INFECTED.get(s, 0) for s in labels], dtype=np.int64)
    theta = np.empty((count, 51), dtype=np.float32)
    foi_ext = np.empty((count, 51, 52), dtype=np.float32)
    Y = np.empty((count, 51, 52), dtype=np.int32)
    for k in range(count):
        # Split ID and simulation index make streams disjoint, even if counts change.
        streams = np.random.SeedSequence([seed, split, start + k]).spawn(3)
        prior, epidemic, reporting = [np.random.default_rng(s) for s in streams]
        beta = np.exp(prior.normal(np.log(.30), .20, 51))
        daily = simulate(beta, N_DAYS, epidemic, N=N, M=M, I0=I0,
                         gamma=GAMMA, p_a=P_A)
        weekly = weekly_summary(daily)
        observed = observe(weekly["incidence"], P_REPORT, reporting)
        theta[k], foi_ext[k], Y[k] = beta, weekly["foi_external"].T, observed.T
        if check:
            assert np.all(daily["S"] + daily["I"] + daily["R"] == N)
            assert all(np.all(a >= 0) for a in daily.values())
            np.testing.assert_allclose(daily["foi"], daily["foi_local"] + daily["foi_external"])
            assert np.array_equal(weekly["incidence"].sum(0), daily["incidence"].sum(0))
            assert np.all(observed <= weekly["incidence"])
    assert foi_ext.shape == Y.shape == (count, 51, 52)
    assert np.isfinite(foi_ext).all() and np.all(foi_ext >= 0) and np.all(Y >= 0)
    return theta, foi_ext, Y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--validation", type=int, default=2000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "diffusion_data")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    labels, subpop, N, M = load_data(ROOT)
    I0 = np.array([INITIAL_INFECTED.get(s, 0) for s in labels])
    counts = [("pilot", 100)]
    if not args.pilot_only:
        counts += [("train", args.train), ("validation", args.validation), ("test", args.test)]
    for split, (name, count) in enumerate(counts):
        path = args.output / f"{name}.npz"
        if path.exists():
            with np.load(path) as saved:
                if len(saved["theta"]) != count or int(saved["seed"]) != args.seed:
                    raise ValueError(f"{path} has different counts/seed; choose another --output.")
            print(f"Using cached {path} ({count} simulations)", flush=True)
            continue
        start = perf_counter()
        tasks = [(args.seed, split, i, min(250, count-i), name == "pilot")
                 for i in range(0, count, 250)]
        chunks = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for chunk in pool.map(generate_chunk, tasks):
                chunks.append(chunk)
                done = sum(len(c[0]) for c in chunks)
                print(f"{name}: {done}/{count}, {perf_counter()-start:.1f}s", flush=True)
        theta, foi_ext, Y = [np.concatenate([c[k] for c in chunks]) for k in range(3)]
        np.savez_compressed(path, theta=theta, foi_ext=foi_ext, Y=Y,
                            labels=labels, subpop=subpop, N=N, M=M, I0=I0,
                            gamma=GAMMA, p_a=P_A, p_report=P_REPORT, n_days=N_DAYS,
                            seed=args.seed, split_id=split,
                            simulation_index=np.arange(count))
        print(f"Saved {path}: theta={theta.shape}, foi_ext/Y={foi_ext.shape}; "
              f"{perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
