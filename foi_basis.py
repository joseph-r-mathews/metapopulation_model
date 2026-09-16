"""Fixed K=16 square-root spline/FPCA representation of continuous FoI."""

import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "deterministic_data"
RESULTS = ROOT / "results"
BASIS_PATH = ROOT / "checkpoints" / "foi_sqrt_fpca_k16_basis.npz"
RECONSTRUCTION_PATH = RESULTS / "functional_basis_sqrt_reconstruction_by_K.csv"
DENSE_POINTS_PER_WEEK = 8
DENSE_TIMES = np.linspace(0.0, 52.0, 52 * DENSE_POINTS_PER_WEEK + 1)
SPLINE_DEGREE = 3
SPLINE_DIMENSION = 48
K = 16
N_STATES = 51
CHUNK_SIZE = 50


def spline_knots():
    interior_count = SPLINE_DIMENSION - SPLINE_DEGREE - 1
    interior = np.linspace(0.0, 52.0, interior_count + 2)[1:-1]
    return np.r_[
        np.repeat(0.0, SPLINE_DEGREE + 1),
        interior,
        np.repeat(52.0, SPLINE_DEGREE + 1),
    ]


def spline_design(times, knots=None):
    values = np.atleast_1d(np.asarray(times, dtype=np.float64))
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 52.0):
        raise ValueError("Continuous evaluation times must be finite and in [0, 52]")
    knots = spline_knots() if knots is None else np.asarray(knots)
    return BSpline.design_matrix(
        values, knots, SPLINE_DEGREE, extrapolate=False
    ).toarray()


def _project_sqrt_curves(curves, inverse_design):
    values = np.asarray(curves, dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError("True instantaneous external FoI contains negative values")
    return (
        np.sqrt(values).reshape(-1, len(DENSE_TIMES)) @ inverse_design.T
    ).reshape(len(values), N_STATES, SPLINE_DIMENSION)


def _fit_training_basis(inverse_design):
    curves = np.load(
        DATA_ROOT / "train" / "h_instantaneous_dense.npy", mmap_mode="r"
    )
    coefficients = np.empty(
        (len(curves), N_STATES, SPLINE_DIMENSION), dtype=np.float32
    )
    minimum = float("inf")
    negative = 0
    for start in range(0, len(curves), CHUNK_SIZE):
        values = np.asarray(curves[start : start + CHUNK_SIZE])
        minimum = min(minimum, float(values.min()))
        negative += int(np.count_nonzero(values < 0.0))
        coefficients[start : start + len(values)] = _project_sqrt_curves(
            values, inverse_design
        ).astype(np.float32)
    print(
        f"true_h_scan,train,minimum={minimum:.17g},negative_count={negative}",
        flush=True,
    )
    if negative:
        raise RuntimeError("Training h(t) contains negative values; preparation stopped")

    mean = np.mean(coefficients, axis=0, dtype=np.float64)
    components = np.empty((N_STATES, K, SPLINE_DIMENSION), dtype=np.float64)
    eigenvalues = np.empty((N_STATES, K), dtype=np.float64)
    for state in range(N_STATES):
        centered = np.asarray(coefficients[:, state], dtype=np.float64) - mean[state]
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        components[state] = right[:K]
        eigenvalues[state] = singular[:K] ** 2 / (len(centered) - 1)
    return mean, components, eigenvalues, coefficients


def _scores(coefficients, mean, components):
    return np.einsum(
        "nsr,skr->nsk", coefficients - mean[None], components, optimize=True
    )


def _write_targets(split, inverse_design, mean, components, training_coefficients=None):
    curves = np.load(
        DATA_ROOT / split / "h_instantaneous_dense.npy", mmap_mode="r"
    )
    target_path = DATA_ROOT / split / "sqrt_fpca_k16_coefficients.npy"
    targets = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=(len(curves), N_STATES, K)
    )
    for start in range(0, len(curves), CHUNK_SIZE):
        if training_coefficients is None:
            coefficient = _project_sqrt_curves(
                curves[start : start + CHUNK_SIZE], inverse_design
            )
        else:
            coefficient = np.asarray(
                training_coefficients[start : start + CHUNK_SIZE], dtype=np.float64
            )
        targets[start : start + len(coefficient)] = _scores(
            coefficient, mean, components
        ).astype(np.float32)
    targets.flush()
    del targets
    (DATA_ROOT / split / "SQRT_FPCA_K16_COMPLETE").write_text("complete\n")


def _relative_l2(reconstructed, truth):
    numerator = np.linalg.norm(reconstructed - truth, axis=-1)
    denominator = np.linalg.norm(truth, axis=-1)
    return numerator / np.maximum(denominator, np.finfo(float).tiny)


def _evaluate_split(split, design, inverse_design, mean, components):
    curves = np.load(
        DATA_ROOT / split / "h_instantaneous_dense.npy", mmap_mode="r"
    )
    errors = []
    minimum = float("inf")
    negative = 0
    total = 0
    for start in range(0, len(curves), CHUNK_SIZE):
        truth = np.asarray(curves[start : start + CHUNK_SIZE], dtype=np.float64)
        coefficient = _project_sqrt_curves(truth, inverse_design)
        score = _scores(coefficient, mean, components)
        reconstructed_coefficient = mean[None] + np.einsum(
            "nsk,skr->nsr", score, components, optimize=True
        )
        reconstructed = np.square(
            reconstructed_coefficient.reshape(-1, SPLINE_DIMENSION) @ design.T
        ).reshape(truth.shape)
        errors.extend(_relative_l2(reconstructed, truth).ravel())
        minimum = min(minimum, float(reconstructed.min()))
        negative += int(np.count_nonzero(reconstructed < 0.0))
        total += reconstructed.size
    values = np.asarray(errors)
    return {
        "split": split,
        "K": K,
        "median_relative_L2": float(np.median(values)),
        "q90_relative_L2": float(np.quantile(values, 0.90)),
        "q99_relative_L2": float(np.quantile(values, 0.99)),
        "max_relative_L2": float(values.max()),
        "total_coefficient_dimension": N_STATES * K,
        "minimum_reconstructed_h": minimum,
        "fraction_reconstructed_below_zero": negative / total,
    }


def _verify_previous_k16(rows):
    previous = ROOT / "results" / "functional_foi" / "functional_basis_sqrt_reconstruction_by_K.csv"
    if not previous.exists():
        return
    with previous.open(newline="") as stream:
        expected = {
            row["split"]: row
            for row in csv.DictReader(stream)
            if int(row["K"]) == K
        }
    for row in rows:
        if row["split"] not in expected:
            continue
        for metric in (
            "median_relative_L2",
            "q90_relative_L2",
            "q99_relative_L2",
            "max_relative_L2",
        ):
            if not np.isclose(
                row[metric], float(expected[row["split"]][metric]), rtol=2e-5, atol=1e-12
            ):
                raise RuntimeError(
                    f"K=16 refactor sanity check failed for {row['split']} {metric}"
                )


def prepare_basis_and_targets(force=False):
    """Fit the fixed training basis, write C targets, and verify held-out K=16."""
    target_paths = [
        DATA_ROOT / split / "sqrt_fpca_k16_coefficients.npy"
        for split in ("train", "validation", "test")
    ]
    if BASIS_PATH.exists() and all(path.exists() for path in target_paths) and not force:
        print("Using complete K=16 square-root FPCA basis and targets", flush=True)
        return

    knots = spline_knots()
    design = spline_design(DENSE_TIMES, knots)
    inverse_design = np.linalg.pinv(design)
    mean, components, eigenvalues, training_coefficients = _fit_training_basis(
        inverse_design
    )
    BASIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BASIS_PATH,
        method="sqrt_cubic_bspline_state_specific_fpca",
        target="continuous_instantaneous_external_foi",
        K=K,
        coefficient_dimension=N_STATES * K,
        dense_times=DENSE_TIMES,
        spline_knots=knots,
        spline_degree=SPLINE_DEGREE,
        spline_dimension=SPLINE_DIMENSION,
        mean_spline_coefficients=mean,
        pca_components=components,
        pca_eigenvalues=eigenvalues,
    )
    _write_targets(
        "train", inverse_design, mean, components, training_coefficients
    )
    del training_coefficients
    for split in ("validation", "test"):
        _write_targets(split, inverse_design, mean, components)
    rows = [
        _evaluate_split(split, design, inverse_design, mean, components)
        for split in ("validation", "test")
    ]
    _verify_previous_k16(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with RECONSTRUCTION_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)


def load_basis(path=BASIS_PATH):
    with np.load(Path(path), allow_pickle=False) as saved:
        if str(saved["method"]) != "sqrt_cubic_bspline_state_specific_fpca":
            raise ValueError("Not a square-root FPCA basis")
        if int(saved["K"]) != K:
            raise ValueError(f"Expected K={K}")
        return {key: np.asarray(saved[key]) for key in saved.files}


def reconstruct_external_foi(coefficients, times, basis_path=BASIS_PATH):
    """Reconstruct h(t)=z(t)^2; returns [samples, 51, len(times)]."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[1:] != (N_STATES, K):
        raise ValueError("C must have shape [51, 16] or [n_samples, 51, 16]")
    if not np.isfinite(values).all():
        raise ValueError("C must be finite")
    basis = load_basis(basis_path)
    design = spline_design(times, basis["spline_knots"])
    spline_coefficient = basis["mean_spline_coefficients"][None] + np.einsum(
        "nsk,skr->nsr", values, basis["pca_components"], optimize=True
    )
    z = np.einsum("nsr,tr->nst", spline_coefficient, design, optimize=True)
    return np.square(z)
