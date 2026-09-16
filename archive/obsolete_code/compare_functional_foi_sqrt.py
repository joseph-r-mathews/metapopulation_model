"""Compare raw-h and square-root functional FoI representations."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_diffusion_data import DATA_ROOT
from prepare_functional_foi import (
    DENSE_TIMES,
    K_VALUES,
    RESULTS,
    SPLINE_DIMENSION,
    curve_errors,
    spline_specification,
)


HELD_OUT_SPLITS = ("validation", "test")
CHUNK_SIZE = 50


def scan_true_curves():
    summaries = []
    for split in ("train",) + HELD_OUT_SPLITS:
        curves = np.load(
            DATA_ROOT / split / "h_instantaneous_dense.npy", mmap_mode="r"
        )
        minimum = float("inf")
        negative_count = 0
        for start in range(0, len(curves), CHUNK_SIZE):
            values = np.asarray(curves[start : start + CHUNK_SIZE])
            minimum = min(minimum, float(values.min()))
            negative_count += int(np.count_nonzero(values < 0.0))
        summaries.append((split, minimum, negative_count, curves.size))

    for split, minimum, negative_count, size in summaries:
        print(
            f"true_h_scan,{split},minimum={minimum:.17g},"
            f"negative_count={negative_count},fraction={negative_count / size:.17g}",
            flush=True,
        )
    if any(negative_count for _, _, negative_count, _ in summaries):
        worst = min(minimum for _, minimum, _, _ in summaries)
        raise RuntimeError(
            "True dense h contains negative values (minimum "
            f"{worst:.17g}). No clipping is permitted, so sqrt fitting stopped."
        )


def fit_sqrt_training_coefficients(inverse):
    curves = np.load(
        DATA_ROOT / "train" / "h_instantaneous_dense.npy", mmap_mode="r"
    )
    coefficients = np.empty(
        (len(curves), 51, SPLINE_DIMENSION), dtype=np.float32
    )
    for start in range(0, len(curves), CHUNK_SIZE):
        truth = np.asarray(curves[start : start + CHUNK_SIZE], dtype=np.float64)
        z = np.sqrt(truth)
        coefficients[start : start + len(truth)] = (
            z.reshape(-1, len(DENSE_TIMES)) @ inverse.T
        ).reshape(len(truth), 51, SPLINE_DIMENSION).astype(np.float32)
    return coefficients


def fit_state_specific_pca(training_coefficients):
    mean = np.mean(training_coefficients, axis=0, dtype=np.float64)
    components = np.empty((51, max(K_VALUES), SPLINE_DIMENSION))
    for state in range(51):
        centered = (
            np.asarray(training_coefficients[:, state], dtype=np.float64)
            - mean[state]
        )
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        components[state] = right[: max(K_VALUES)]
    return mean, components


def reconstruct_coefficients(coefficients, mean, components, K):
    reconstructed = np.empty_like(coefficients, dtype=np.float64)
    for state in range(51):
        centered = coefficients[:, state] - mean[state]
        scores = centered @ components[state, :K].T
        reconstructed[:, state] = mean[state] + scores @ components[state, :K]
    return reconstructed


def summarize(errors, minimum, negative_count, total, split, K):
    values = np.asarray(errors)
    return {
        "split": split,
        "K": K,
        "median_relative_L2": float(np.median(values)),
        "q90_relative_L2": float(np.quantile(values, 0.90)),
        "q99_relative_L2": float(np.quantile(values, 0.99)),
        "max_relative_L2": float(values.max()),
        "total_coefficient_dimension": 51 * K,
        "minimum_reconstructed_h": minimum,
        "fraction_reconstructed_below_zero": negative_count / total,
    }


def evaluate_split(split, inverse, design, sqrt_mean, sqrt_components,
                   raw_mean, raw_components):
    truth_curves = np.load(
        DATA_ROOT / split / "h_instantaneous_dense.npy", mmap_mode="r"
    )
    raw_coefficients = np.load(
        DATA_ROOT / split / "h_spline_coefficients.npy", mmap_mode="r"
    )
    sqrt_errors = {K: [] for K in K_VALUES}
    raw_errors = {K: [] for K in K_VALUES}
    sqrt_minimum = {K: float("inf") for K in K_VALUES}
    raw_minimum = {K: float("inf") for K in K_VALUES}
    sqrt_negative = {K: 0 for K in K_VALUES}
    raw_negative = {K: 0 for K in K_VALUES}
    total = 0

    for start in range(0, len(truth_curves), CHUNK_SIZE):
        truth = np.asarray(
            truth_curves[start : start + CHUNK_SIZE], dtype=np.float64
        )
        raw_coefficient = np.asarray(
            raw_coefficients[start : start + CHUNK_SIZE], dtype=np.float64
        )
        z = np.sqrt(truth)
        sqrt_coefficient = (
            z.reshape(-1, len(DENSE_TIMES)) @ inverse.T
        ).reshape(len(truth), 51, SPLINE_DIMENSION)
        flat_truth = truth.reshape(-1, len(DENSE_TIMES))

        for K in K_VALUES:
            reconstructed_z_coeff = reconstruct_coefficients(
                sqrt_coefficient, sqrt_mean, sqrt_components, K
            )
            reconstructed_z = (
                reconstructed_z_coeff.reshape(-1, SPLINE_DIMENSION) @ design.T
            )
            reconstructed_sqrt_h = np.square(reconstructed_z)

            reconstructed_raw_coeff = reconstruct_coefficients(
                raw_coefficient, raw_mean, raw_components, K
            )
            reconstructed_raw_h = (
                reconstructed_raw_coeff.reshape(-1, SPLINE_DIMENSION) @ design.T
            )

            sqrt_errors[K].extend(curve_errors(reconstructed_sqrt_h, flat_truth))
            raw_errors[K].extend(curve_errors(reconstructed_raw_h, flat_truth))
            sqrt_minimum[K] = min(
                sqrt_minimum[K], float(reconstructed_sqrt_h.min())
            )
            raw_minimum[K] = min(raw_minimum[K], float(reconstructed_raw_h.min()))
            sqrt_negative[K] += int(np.count_nonzero(reconstructed_sqrt_h < 0.0))
            raw_negative[K] += int(np.count_nonzero(reconstructed_raw_h < 0.0))
        total += truth.size

    sqrt_rows = [
        summarize(
            sqrt_errors[K], sqrt_minimum[K], sqrt_negative[K], total, split, K
        )
        for K in K_VALUES
    ]
    raw_rows = [
        summarize(raw_errors[K], raw_minimum[K], raw_negative[K], total, split, K)
        for K in K_VALUES
    ]
    return sqrt_rows, raw_rows, {
        K: np.asarray(sqrt_errors[K]) for K in K_VALUES
    }


def verify_existing_raw_validation(raw_rows):
    path = RESULTS / "functional_basis_reconstruction_by_K.csv"
    with path.open(newline="") as stream:
        existing = {int(row["K"]): row for row in csv.DictReader(stream)}
    evaluated = {row["K"]: row for row in raw_rows if row["split"] == "validation"}
    metrics = (
        "median_relative_L2",
        "q90_relative_L2",
        "q99_relative_L2",
        "max_relative_L2",
        "fraction_reconstructed_below_zero",
    )
    for K in K_VALUES:
        for metric in metrics:
            if not np.isclose(
                float(existing[K][metric]), evaluated[K][metric], rtol=1e-6, atol=1e-12
            ):
                raise RuntimeError(
                    f"Recomputed raw-h validation {metric} at K={K} does not "
                    "match the existing reconstruction result."
                )


def comparison_rows(sqrt_rows, raw_rows):
    sqrt_lookup = {(row["split"], row["K"]): row for row in sqrt_rows}
    rows = []
    for raw in raw_rows:
        sqrt = sqrt_lookup[(raw["split"], raw["K"])]
        rows.append({
            "split": raw["split"],
            "K": raw["K"],
            "raw_h_median_relative_L2": raw["median_relative_L2"],
            "sqrt_h_median_relative_L2": sqrt["median_relative_L2"],
            "raw_h_q90_relative_L2": raw["q90_relative_L2"],
            "sqrt_h_q90_relative_L2": sqrt["q90_relative_L2"],
            "raw_h_q99_relative_L2": raw["q99_relative_L2"],
            "sqrt_h_q99_relative_L2": sqrt["q99_relative_L2"],
            "raw_h_max_relative_L2": raw["max_relative_L2"],
            "sqrt_h_max_relative_L2": sqrt["max_relative_L2"],
            "raw_h_negative_fraction": raw["fraction_reconstructed_below_zero"],
            "sqrt_h_negative_fraction": sqrt["fraction_reconstructed_below_zero"],
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_one(coefficient, mean, components, design, K, square):
    centered = coefficient - mean
    scores = centered @ components[:K].T
    reconstructed_coefficient = mean + scores @ components[:K]
    curve = reconstructed_coefficient @ design.T
    return np.square(curve) if square else curve


def plot_examples(test_errors, inverse, design, sqrt_mean, sqrt_components,
                  raw_mean, raw_components):
    K = max(K_VALUES)
    errors = test_errors[K]
    targets = (
        ("typical", float(np.median(errors))),
        ("high", float(np.quantile(errors, 0.99))),
        ("worst-case", float(errors.max())),
    )
    indices = [int(np.argmin(np.abs(errors - target))) for _, target in targets]
    truth_curves = np.load(
        DATA_ROOT / "test" / "h_instantaneous_dense.npy", mmap_mode="r"
    )
    raw_coefficients = np.load(
        DATA_ROOT / "test" / "h_spline_coefficients.npy", mmap_mode="r"
    )
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for axis, (label, _), flat_index in zip(axes, targets, indices):
        epidemic, state = divmod(flat_index, 51)
        truth = np.asarray(truth_curves[epidemic, state], dtype=np.float64)
        sqrt_coefficient = np.sqrt(truth) @ inverse.T
        raw_h = reconstruct_one(
            np.asarray(raw_coefficients[epidemic], dtype=np.float64)[state],
            raw_mean[state], raw_components[state], design, K, False,
        )
        sqrt_h = reconstruct_one(
            sqrt_coefficient, sqrt_mean[state], sqrt_components[state],
            design, K, True,
        )
        axis.plot(DENSE_TIMES, truth, color="black", linewidth=2, label="true h(t)")
        axis.plot(DENSE_TIMES, raw_h, linewidth=1.4, label="raw-h PCA")
        axis.plot(DENSE_TIMES, sqrt_h, linewidth=1.4, label="sqrt-h PCA")
        axis.set_title(
            f"{label}: test epidemic {epidemic}, state {state}, "
            f"sqrt relative L2={errors[flat_index]:.4g}"
        )
        axis.set_ylabel("external FoI")
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("time (weeks)")
    figure.tight_layout()
    figure.savefig(
        RESULTS / "functional_basis_raw_vs_sqrt_examples.png", dpi=180
    )
    plt.close(figure)


def print_table(title, rows):
    print(title)
    print(",".join(rows[0].keys()))
    for row in rows:
        print(",".join(str(value) for value in row.values()))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    scan_true_curves()
    _, design, inverse = spline_specification()
    sqrt_training = fit_sqrt_training_coefficients(inverse)
    sqrt_mean, sqrt_components = fit_state_specific_pca(sqrt_training)
    del sqrt_training

    with np.load(RESULTS / "functional_basis_specification.npz") as raw_spec:
        raw_mean = np.asarray(raw_spec["mean_spline_coefficients"])
        raw_components = np.asarray(raw_spec["pca_components"])

    sqrt_rows = []
    raw_rows = []
    errors_by_split = {}
    for split in HELD_OUT_SPLITS:
        sqrt_split, raw_split, errors = evaluate_split(
            split, inverse, design, sqrt_mean, sqrt_components,
            raw_mean, raw_components,
        )
        sqrt_rows.extend(sqrt_split)
        raw_rows.extend(raw_split)
        errors_by_split[split] = errors

    verify_existing_raw_validation(raw_rows)
    comparisons = comparison_rows(sqrt_rows, raw_rows)
    write_csv(
        RESULTS / "functional_basis_sqrt_reconstruction_by_K.csv", sqrt_rows
    )
    write_csv(
        RESULTS / "functional_basis_raw_vs_sqrt_comparison.csv", comparisons
    )
    plot_examples(
        errors_by_split["test"], inverse, design, sqrt_mean, sqrt_components,
        raw_mean, raw_components,
    )

    high_k = [row for row in comparisons if row["K"] == max(K_VALUES)]
    absolute_worsening = max(
        sqrt_metric - raw_metric
        for row in high_k
        for raw_metric, sqrt_metric in (
            (row["raw_h_median_relative_L2"], row["sqrt_h_median_relative_L2"]),
            (row["raw_h_q90_relative_L2"], row["sqrt_h_q90_relative_L2"]),
            (row["raw_h_q99_relative_L2"], row["sqrt_h_q99_relative_L2"]),
        )
    )
    materially_degraded = absolute_worsening > 0.01
    min_sqrt_h = min(row["minimum_reconstructed_h"] for row in sqrt_rows)
    negative_fraction = max(
        row["fraction_reconstructed_below_zero"] for row in sqrt_rows
    )

    print_table("sqrt_h_reconstruction_table", sqrt_rows)
    print_table("raw_h_vs_sqrt_h_comparison_table", comparisons)
    print(
        "sqrt_h_materially_degrades_accuracy="
        f"{str(materially_degraded).lower()} "
        "(K=16 median/q90/q99 maximum absolute worsening "
        f"{absolute_worsening:.6g}; material threshold 0.01)"
    )
    print(
        "sqrt_h_nonnegativity_guaranteed_in_practice="
        f"{str(min_sqrt_h >= 0.0 and negative_fraction == 0.0).lower()} "
        f"(minimum={min_sqrt_h:.17g}, fraction_below_zero={negative_fraction:.17g})"
    )


if __name__ == "__main__":
    main()
