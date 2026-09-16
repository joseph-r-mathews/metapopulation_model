"""Extract continuous FoI curves and assess spline/functional PCA bases."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline

from deterministic_model import ATOL, RTOL, solve_coupled
from generate_diffusion_data import DATA_ROOT
from model import P_A, load_data


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "functional_foi"
DENSE_POINTS_PER_WEEK = 8
DENSE_TIMES = np.linspace(0., 52., 52*DENSE_POINTS_PER_WEEK+1)
SPLINE_DEGREE = 3
SPLINE_DIMENSION = 48
K_VALUES = (2, 4, 8, 12, 16)
SEED = 20260929


def spline_specification():
    interior_count = SPLINE_DIMENSION-SPLINE_DEGREE-1
    interior = np.linspace(0., 52., interior_count+2)[1:-1]
    knots = np.r_[np.repeat(0., SPLINE_DEGREE+1), interior,
                  np.repeat(52., SPLINE_DEGREE+1)]
    design = BSpline.design_matrix(DENSE_TIMES, knots, SPLINE_DEGREE).toarray()
    inverse = np.linalg.pinv(design)
    return knots, design, inverse


def extract_chunk(task):
    split, start, count = task
    beta = np.load(DATA_ROOT/split/"beta_true.npy", mmap_mode="r")
    _, _, N, M = load_data(ROOT)
    with np.load(DATA_ROOT/split/"metadata.npz", allow_pickle=False) as metadata:
        I0 = np.asarray(metadata["I0"])
    curves = np.empty((count, 51, len(DENSE_TIMES)), dtype=np.float32)
    solve_seconds = 0.
    for offset in range(count):
        started = perf_counter()
        solved = solve_coupled(np.asarray(beta[start+offset]), N=N, M=M, I0=I0,
                               rtol=RTOL, atol=ATOL, dense_output=True)
        state = solved["solution"].sol(DENSE_TIMES)
        infectious = state[51:102]
        pressure = np.asarray(beta[start+offset])[:, None]*infectious/N[:, None]
        curves[offset] = (P_A*(M@pressure)/N[:, None]).astype(np.float32)
        solve_seconds += perf_counter()-started
    return start, curves, solve_seconds


def extract_split(split, workers):
    beta = np.load(DATA_ROOT/split/"beta_true.npy", mmap_mode="r")
    count = len(beta)
    path = DATA_ROOT/split/"h_instantaneous_dense.npy"
    complete = DATA_ROOT/split/"H_INSTANTANEOUS_COMPLETE"
    if complete.exists():
        return json.loads((DATA_ROOT/split/"h_instantaneous_summary.json").read_text())
    curves = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(count, 51, len(DENSE_TIMES)))
    tasks = [(split, start, min(20, count-start))
             for start in range(0, count, 20)]
    started = perf_counter(); solve_seconds = 0.
    if workers == 1:
        generated = map(extract_chunk, tasks); pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        generated = pool.map(extract_chunk, tasks)
    try:
        for start, values, elapsed in generated:
            curves[start:start+len(values)] = values
            curves.flush(); solve_seconds += elapsed
            print(f"{split} instantaneous h: {start+len(values)}/{count}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    del curves
    summary = {"split": split, "epidemics": count,
               "wall_seconds": perf_counter()-started,
               "average_solve_seconds": solve_seconds/count,
               "dense_times": len(DENSE_TIMES)}
    (DATA_ROOT/split/"h_instantaneous_summary.json").write_text(
        json.dumps(summary, indent=2)+"\n")
    complete.write_text("complete\n")
    return summary


def fit_spline_coefficients(split, inverse):
    curves = np.load(DATA_ROOT/split/"h_instantaneous_dense.npy", mmap_mode="r")
    count = len(curves)
    path = DATA_ROOT/split/"h_spline_coefficients.npy"
    coefficients = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32,
        shape=(count, 51, SPLINE_DIMENSION)
    )
    for start in range(0, count, 100):
        values = np.asarray(curves[start:start+100], dtype=np.float64)
        coefficients[start:start+len(values)] = (
            values.reshape(-1, len(DENSE_TIMES)) @ inverse.T
        ).reshape(len(values), 51, SPLINE_DIMENSION).astype(np.float32)
    coefficients.flush(); del coefficients


def fit_state_specific_pca():
    training = np.load(DATA_ROOT/"train"/"h_spline_coefficients.npy", mmap_mode="r")
    mean = np.asarray(training, dtype=np.float64).mean(axis=0)
    components = np.empty((51, max(K_VALUES), SPLINE_DIMENSION))
    eigenvalues = np.empty((51, max(K_VALUES)))
    for state in range(51):
        centered = np.asarray(training[:, state], dtype=np.float64)-mean[state]
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        components[state] = right[:max(K_VALUES)]
        eigenvalues[state] = singular[:max(K_VALUES)]**2/(len(centered)-1)
    return mean, components, eigenvalues


def curve_errors(reconstructed, truth):
    numerator = np.linalg.norm(reconstructed-truth, axis=1)
    denominator = np.linalg.norm(truth, axis=1)
    return numerator/np.maximum(denominator, np.finfo(float).tiny)


def evaluate_reconstruction(mean, components, design):
    validation_coeff = np.load(
        DATA_ROOT/"validation"/"h_spline_coefficients.npy", mmap_mode="r"
    )
    validation_curves = np.load(
        DATA_ROOT/"validation"/"h_instantaneous_dense.npy", mmap_mode="r"
    )
    all_errors = {K: [] for K in K_VALUES}
    minima = {K: float("inf") for K in K_VALUES}
    negative = {K: 0 for K in K_VALUES}
    total = 0
    spline_errors = []
    for start in range(0, len(validation_coeff), 50):
        coefficient = np.asarray(validation_coeff[start:start+50], dtype=np.float64)
        truth = np.asarray(validation_curves[start:start+50], dtype=np.float64)
        spline = (coefficient.reshape(-1, SPLINE_DIMENSION) @ design.T)
        spline_errors.extend(curve_errors(spline, truth.reshape(-1, len(DENSE_TIMES))))
        for K in K_VALUES:
            reconstructed_coeff = np.empty_like(coefficient)
            for state in range(51):
                centered = coefficient[:, state]-mean[state]
                scores = centered@components[state, :K].T
                reconstructed_coeff[:, state] = mean[state]+scores@components[state, :K]
            reconstructed = (reconstructed_coeff.reshape(-1, SPLINE_DIMENSION)
                             @ design.T)
            all_errors[K].extend(curve_errors(
                reconstructed, truth.reshape(-1, len(DENSE_TIMES))
            ))
            minima[K] = min(minima[K], float(reconstructed.min()))
            negative[K] += int(np.count_nonzero(reconstructed < 0))
        total += truth.size
    rows = []
    for K in K_VALUES:
        errors = np.asarray(all_errors[K])
        rows.append({"K": K, "median_relative_L2": float(np.median(errors)),
                     "q90_relative_L2": float(np.quantile(errors, .90)),
                     "q99_relative_L2": float(np.quantile(errors, .99)),
                     "max_relative_L2": float(errors.max()),
                     "total_coefficient_dimension": 51*K,
                     "minimum_reconstructed_h": minima[K],
                     "fraction_reconstructed_below_zero": negative[K]/total})
    spline_errors = np.asarray(spline_errors)
    spline_summary = {
        "R": SPLINE_DIMENSION,
        "median_relative_L2": float(np.median(spline_errors)),
        "q99_relative_L2": float(np.quantile(spline_errors, .99)),
        "maximum_relative_L2": float(spline_errors.max()),
    }
    return rows, spline_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        with np.load(DATA_ROOT/split/"metadata.npz", allow_pickle=False) as metadata:
            if str(metadata["forward_model"]) != "deterministic_coupled_ode":
                raise RuntimeError(f"{split} was not generated by the continuous ODE")
    summaries = [extract_split(split, args.workers)
                 for split in ("train", "validation", "test")]
    knots, design, inverse = spline_specification()
    for split in ("train", "validation", "test"):
        fit_spline_coefficients(split, inverse)
    mean, components, eigenvalues = fit_state_specific_pca()
    rows, spline_summary = evaluate_reconstruction(mean, components, design)
    with (RESULTS/"functional_basis_reconstruction_by_K.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(
        RESULTS/"functional_basis_specification.npz",
        dense_times=DENSE_TIMES, spline_knots=knots,
        spline_degree=SPLINE_DEGREE, spline_dimension=SPLINE_DIMENSION,
        spline_design_dense=design, mean_spline_coefficients=mean,
        pca_components=components, pca_eigenvalues=eigenvalues,
        K_values=np.asarray(K_VALUES), centering="state_specific",
    )
    report = {
        "old_target": "weekly integrated external FoI Q(w+1)-Q(w); obsolete",
        "new_target": "coefficients representing instantaneous continuous h_i(t)",
        "reused_existing_arrays": ["eta_true", "beta_true", "Y", "mu"],
        "regenerated_Y_or_mu": False,
        "dense_points_per_week": DENSE_POINTS_PER_WEEK,
        "spline_representation": spline_summary,
        "functional_PCA_centering": "state-specific",
        "reconstruction_by_K": rows,
        "extraction_runtime": summaries,
        "chosen_K": None,
        "diffusion_training_started": False,
    }
    (RESULTS/"functional_basis_pretraining_report.json").write_text(
        json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
