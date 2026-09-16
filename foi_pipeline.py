"""Prepare, train, evaluate, and finalize the sqrt-FPCA FoI pipeline."""

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from epidemic_model import (
    ATOL,
    GAMMA,
    N_STATES,
    N_WEEKS,
    P_A,
    P_REPORT,
    PRIOR_LOG_MEAN,
    PRIOR_LOG_SD,
    RTOL,
    evaluate_external_foi,
    initial_infected,
    load_population_and_mobility,
    observe,
    solve_coupled,
)
from foi_basis import (
    BASIS_PATH,
    DATA_ROOT,
    DENSE_TIMES,
    K,
    RECONSTRUCTION_PATH,
    prepare_basis_and_targets,
    reconstruct_external_foi,
)
from sqrt_foi_diffusion import (
    DEFAULT_CHECKPOINT,
    CoefficientUNet,
    corrupt,
    diffusion_schedule,
    energy_loss,
    fit_transforms,
    inverse_coefficients,
    load_foi_model,
    reverse_sample,
    transform_coefficients,
    transform_y,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRAINING_HISTORY = RESULTS / "foi_sqrt_fpca_k16_training_loss.csv"
METRICS_PATH = RESULTS / "foi_sqrt_fpca_k16_metrics.csv"
FIGURE_PATH = RESULTS / "foi_sqrt_fpca_k16_examples.png"
SUCCESS_MARKER = RESULTS / "FOI_SQRT_FPCA_K16_PIPELINE_COMPLETE"
SEED = 20261001
SPLITS = (("train", 20000), ("validation", 2000), ("test", 2000))
EVALUATION_CASES = 300
POSTERIOR_SAMPLES = 500
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 50


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _simulate_chunk(task):
    split_id, start, count = task
    labels, _, populations, mobility = load_population_and_mobility(ROOT)
    I0 = initial_infected(labels)
    arrays = {
        "eta_true": np.empty((count, N_STATES), dtype=np.float64),
        "beta_true": np.empty((count, N_STATES), dtype=np.float64),
        "mu": np.empty((count, N_STATES, N_WEEKS), dtype=np.float64),
        "Y": np.empty((count, N_STATES, N_WEEKS), dtype=np.int32),
        "h_instantaneous_dense": np.empty(
            (count, N_STATES, len(DENSE_TIMES)), dtype=np.float32
        ),
    }
    for offset in range(count):
        prior_seed, observation_seed = np.random.SeedSequence(
            [20260924, split_id, start + offset]
        ).spawn(2)
        eta = np.random.default_rng(prior_seed).normal(
            PRIOR_LOG_MEAN, PRIOR_LOG_SD, N_STATES
        )
        beta = np.exp(eta)
        solved = solve_coupled(
            beta, populations=populations, mobility=mobility, I0=I0, dense_output=True
        )
        mu = solved["incidence"]
        h = evaluate_external_foi(
            solved["solution"], DENSE_TIMES, beta, populations, mobility
        )
        arrays["eta_true"][offset] = eta
        arrays["beta_true"][offset] = beta
        arrays["mu"][offset] = mu
        arrays["Y"][offset] = observe(
            mu, np.random.default_rng(observation_seed), P_REPORT
        )
        arrays["h_instantaneous_dense"][offset] = h.astype(np.float32)
    return start, arrays


def _split_is_reusable(path, count):
    shapes = {
        "eta_true.npy": (count, N_STATES),
        "beta_true.npy": (count, N_STATES),
        "mu.npy": (count, N_STATES, N_WEEKS),
        "Y.npy": (count, N_STATES, N_WEEKS),
        "h_instantaneous_dense.npy": (count, N_STATES, len(DENSE_TIMES)),
    }
    for name, shape in shapes.items():
        file = path / name
        if not file.exists() or np.load(file, mmap_mode="r").shape != shape:
            return False
    with np.load(path / "metadata.npz", allow_pickle=False) as metadata:
        return str(metadata["forward_model"]) == "deterministic_coupled_ode"


def generate_or_reuse_data(workers=4):
    """Reuse the current continuous-ODE data, or generate the same data afresh."""
    labels, subpop, populations, mobility = load_population_and_mobility(ROOT)
    for split_id, (split, count) in enumerate(SPLITS):
        path = DATA_ROOT / split
        if path.exists() and _split_is_reusable(path, count):
            print(f"Reusing {split}: eta, beta, mu, Y, and dense h(t)", flush=True)
            continue
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Incomplete or incompatible data directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        specs = {
            "eta_true": (np.float64, (count, N_STATES)),
            "beta_true": (np.float64, (count, N_STATES)),
            "mu": (np.float64, (count, N_STATES, N_WEEKS)),
            "Y": (np.int32, (count, N_STATES, N_WEEKS)),
            "h_instantaneous_dense": (
                np.float32,
                (count, N_STATES, len(DENSE_TIMES)),
            ),
        }
        maps = {
            name: np.lib.format.open_memmap(path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            for name, (dtype, shape) in specs.items()
        }
        tasks = [
            (split_id, start, min(10, count - start))
            for start in range(0, count, 10)
        ]
        generated = map(_simulate_chunk, tasks) if workers == 1 else None
        pool = None if workers == 1 else ProcessPoolExecutor(max_workers=workers)
        if pool is not None:
            generated = pool.map(_simulate_chunk, tasks)
        try:
            for start, arrays in generated:
                stop = start + len(arrays["Y"])
                for name, values in arrays.items():
                    maps[name][start:stop] = values
                for values in maps.values():
                    values.flush()
                print(f"{split}: {stop}/{count}", flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
        del maps
        np.savez_compressed(
            path / "metadata.npz",
            labels=labels,
            subpop=subpop,
            N=populations,
            M=mobility,
            I0=initial_infected(labels),
            gamma=GAMMA,
            p_a=P_A,
            p_report=P_REPORT,
            n_weeks=N_WEEKS,
            prior_log_mean=PRIOR_LOG_MEAN,
            prior_log_sd=PRIOR_LOG_SD,
            ode_method="DOP853",
            ode_rtol=RTOL,
            ode_atol=ATOL,
            forward_model="deterministic_coupled_ode",
            observation_model="poisson_reported_weekly_integrated_incidence",
            external_foi_definition="instantaneous_continuous_h_on_dense_grid",
            dense_times=DENSE_TIMES,
            seed=20260924,
            split_id=split_id,
            count=count,
        )


def _load(split, name):
    return np.load(DATA_ROOT / split / f"{name}.npy", mmap_mode="r")


def train_diffusion(device):
    existing = None
    start_epoch = 1
    previous_runtime = 0.0
    best = float("inf")
    best_epoch = 0
    if DEFAULT_CHECKPOINT.exists():
        existing = torch.load(
            DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True
        )
        if (existing.get("completed_epochs", 0) >= MAX_EPOCHS
                or existing.get("stopped_early", False)):
            print(f"Using complete checkpoint {DEFAULT_CHECKPOINT}", flush=True)
            return
        start_epoch = int(existing.get("completed_epochs", 0)) + 1
        previous_runtime = float(existing.get("training_runtime_seconds", 0.0))
        best = float(existing["validation_energy_loss"])
        best_epoch = int(existing["epoch"])
        print(
            f"Resuming best-validation checkpoint from epoch {best_epoch}; "
            f"continuing at epoch {start_epoch} with patience "
            f"{EARLY_STOPPING_PATIENCE}",
            flush=True,
        )

    training_C, training_Y = _load("train", "sqrt_fpca_k16_coefficients"), _load("train", "Y")
    validation_C, validation_Y = _load("validation", "sqrt_fpca_k16_coefficients"), _load("validation", "Y")
    if existing is None:
        stats = fit_transforms(training_C, training_Y)
    else:
        stats = {key: value.numpy() for key, value in existing["stats"].items()}
    clean = transform_coefficients(training_C, stats)
    condition = transform_y(training_Y, stats)
    validation_clean = transform_coefficients(validation_C, stats)
    validation_condition = transform_y(validation_Y, stats)
    schedule = diffusion_schedule(100, device)

    torch.manual_seed(SEED)
    model = CoefficientUNet(64).to(device)
    if existing is not None:
        model.load_state_dict(existing["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    fixed = torch.Generator().manual_seed(SEED + 10)
    validation_step = torch.randint(100, (len(validation_clean),), generator=fixed)
    validation_noise = torch.randn(validation_clean.shape, generator=fixed)
    validation_xi = [
        torch.randn(validation_clean.shape, generator=fixed) for _ in range(4)
    ]
    history = []
    epochs_without_improvement = max(0, start_epoch - 1 - best_epoch)
    started = perf_counter()
    last_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        last_epoch = epoch
        model.train()
        training_total = 0.0
        for indices in torch.randperm(len(clean)).split(64):
            x0 = clean[indices].to(device)
            y = condition[indices].to(device)
            step = torch.randint(100, (len(x0),), device=device)
            noisy = corrupt(x0, step, torch.randn_like(x0), schedule)
            predictions = [
                model(noisy, step, y, torch.randn_like(noisy)) for _ in range(4)
            ]
            loss = energy_loss(x0, *predictions)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_total += loss.item() * len(x0)

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for start in range(0, len(validation_clean), 64):
                sl = slice(start, start + 64)
                x0 = validation_clean[sl].to(device)
                y = validation_condition[sl].to(device)
                step = validation_step[sl].to(device)
                noisy = corrupt(x0, step, validation_noise[sl].to(device), schedule)
                predictions = [
                    model(noisy, step, y, xi[sl].to(device)) for xi in validation_xi
                ]
                loss = energy_loss(x0, *predictions)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite validation loss at epoch {epoch}"
                    )
                validation_total += loss.item() * len(x0)
        row = {
            "epoch": epoch,
            "train_energy_loss": training_total / len(clean),
            "validation_energy_loss": validation_total / len(validation_clean),
            "runtime_seconds": perf_counter() - started,
        }
        history.append(row)
        _write_csv(TRAINING_HISTORY, history)
        print(json.dumps(row), flush=True)
        if row["validation_energy_loss"] < best:
            best = row["validation_energy_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "method": "distributional_energy_sqrt_fpca",
                    "target": "continuous_instantaneous_external_foi",
                    "model": model.state_dict(),
                    "stats": {
                        key: torch.from_numpy(value) for key, value in stats.items()
                    },
                    "base": 64,
                    "steps": 100,
                    "K": K,
                    "target_dimension": N_STATES * K,
                    "epoch": epoch,
                    "validation_energy_loss": best,
                    "seed": SEED,
                    "lambda_energy": 1,
                    "beta_energy": 1,
                    "m": 4,
                    "batch_size": 64,
                    "completed_epochs": epoch,
                    "max_epochs": MAX_EPOCHS,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "stopped_early": False,
                    "basis_file": BASIS_PATH.name,
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
                DEFAULT_CHECKPOINT,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}; best validation epoch "
                f"was {best_epoch}",
                flush=True,
            )
            break
    saved = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    saved["completed_epochs"] = last_epoch
    saved["max_epochs"] = MAX_EPOCHS
    saved["early_stopping_patience"] = EARLY_STOPPING_PATIENCE
    saved["stopped_early"] = stopped_early
    saved["training_runtime_seconds"] = (
        previous_runtime + perf_counter() - started
    )
    torch.save(saved, DEFAULT_CHECKPOINT)


@torch.no_grad()
def _draw_loaded(model, stats, schedule, Y, count, device, seed):
    condition = transform_y(np.asarray(Y)[None], stats).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    draws = []
    for start in range(0, count, 64):
        batch = condition.repeat(min(64, count - start), 1, 1)
        normalized = reverse_sample(model, batch, schedule, generator=generator)
        draws.append(normalized.cpu().numpy())
    return inverse_coefficients(np.concatenate(draws), stats)


def _physical_energy_score(draws, truth):
    first = np.linalg.norm(draws - truth[None], axis=(1, 2)).mean()
    half = len(draws) // 2
    second = np.linalg.norm(
        draws[:half] - draws[half : 2 * half], axis=(1, 2)
    ).mean()
    return float(first - 0.5 * second)


def _plot_example(truth, draws, labels):
    mean = draws.mean(axis=0)
    low, high = np.quantile(draws, (0.05, 0.95), axis=0)
    states = ("CA", "NY", "DC", "WY")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, name in zip(axes.flat, states):
        state = list(labels).index(name)
        axis.fill_between(DENSE_TIMES, low[state], high[state], alpha=0.25, label="90% band")
        for draw in draws[:8]:
            axis.plot(DENSE_TIMES, draw[state], color="tab:blue", alpha=0.12, linewidth=0.7)
        axis.plot(DENSE_TIMES, mean[state], color="tab:blue", linewidth=1.7, label="posterior mean")
        axis.plot(DENSE_TIMES, truth[state], color="black", linestyle="--", linewidth=1.5, label="true h(t)")
        axis.set(title=name, xlabel="Time (weeks)", ylabel="Instantaneous external FoI", ylim=(0, None))
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def evaluate_diffusion(device):
    model, stats, schedule = load_foi_model(device=device)
    test_Y = _load("test", "Y")
    test_h = _load("test", "h_instantaneous_dense")
    with np.load(DATA_ROOT / "test" / "metadata.npz", allow_pickle=False) as metadata:
        labels = np.asarray(metadata["labels"])
    indices = np.random.default_rng(SEED + 400).choice(
        len(test_Y), EVALUATION_CASES, replace=False
    )
    points = 0
    sum_prediction = sum_truth = sum_prediction_squared = sum_truth_squared = 0.0
    sum_product = squared_error = 0.0
    covered = np.zeros(len(COVERAGE_LEVELS), dtype=np.int64)
    widths = np.zeros(len(COVERAGE_LEVELS), dtype=np.float64)
    energy_scores = []
    example = None
    for number, index in enumerate(indices, 1):
        coefficient_draws = _draw_loaded(
            model, stats, schedule, test_Y[index], POSTERIOR_SAMPLES, device, SEED + int(index)
        )
        draws = reconstruct_external_foi(coefficient_draws, DENSE_TIMES)
        truth = np.asarray(test_h[index], dtype=np.float64)
        mean = draws.mean(axis=0)
        points += truth.size
        sum_prediction += float(mean.sum())
        sum_truth += float(truth.sum())
        sum_prediction_squared += float(np.square(mean).sum())
        sum_truth_squared += float(np.square(truth).sum())
        sum_product += float((mean * truth).sum())
        squared_error += float(np.square(mean - truth).sum())
        levels = np.asarray(COVERAGE_LEVELS)
        lows = np.quantile(draws, (1 - levels) / 2, axis=0)
        highs = np.quantile(draws, (1 + levels) / 2, axis=0)
        covered += ((truth[None] >= lows) & (truth[None] <= highs)).sum(axis=(1, 2))
        widths += (highs - lows).sum(axis=(1, 2))
        energy_scores.append(_physical_energy_score(draws, truth))
        if example is None:
            example = (truth, draws)
        print(f"evaluation {number}/{len(indices)}", flush=True)
    covariance = sum_product - sum_prediction * sum_truth / points
    prediction_variance = sum_prediction_squared - sum_prediction**2 / points
    truth_variance = sum_truth_squared - sum_truth**2 / points
    row = {
        "test_cases": len(indices),
        "posterior_samples_per_case": POSTERIOR_SAMPLES,
        "posterior_mean_RMSE": float(np.sqrt(squared_error / points)),
        "posterior_mean_correlation": float(
            covariance / np.sqrt(prediction_variance * truth_variance)
        ),
        "energy_score": float(np.mean(energy_scores)),
    }
    for position, level in enumerate(COVERAGE_LEVELS):
        label = int(level * 100)
        row[f"pointwise_coverage_{label}"] = float(covered[position] / points)
        row[f"mean_interval_width_{label}"] = float(widths[position] / points)
    _write_csv(METRICS_PATH, [row])
    _plot_example(example[0], example[1], labels)
    print(json.dumps(row), flush=True)
    return row


def _record_tree():
    excluded = {".git", ".venv", "__pycache__"}
    return sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.rglob("*")
        if not any(part in excluded for part in path.relative_to(ROOT).parts)
    )


def cleanup_after_success():
    """Archive obsolete source and checkpoints; remove reproducible old outputs."""
    if not SUCCESS_MARKER.exists() or not DEFAULT_CHECKPOINT.exists() or not METRICS_PATH.exists():
        raise RuntimeError("Cleanup is gated on a completed checkpoint and evaluation")
    moved = []
    deleted = []
    archive_code = ROOT / "archive" / "obsolete_code"
    archive_checkpoints = ROOT / "archive" / "obsolete_checkpoints"
    archive_code.mkdir(parents=True, exist_ok=True)
    archive_checkpoints.mkdir(parents=True, exist_ok=True)
    obsolete_source = (
        "compare_functional_foi_sqrt.py",
        "deterministic_model.py",
        "diagnose_partial_perturbation_D.py",
        "diagnose_tau_leap.py",
        "foi_diffusion.py",
        "generate_diffusion_data.py",
        "model.py",
        "prepare_functional_foi.py",
        "run_deterministic_importance.py",
        "run_experiments.py",
        "temporary_coherent_weekly_oracle.py",
        "train_deterministic_diffusion.py",
    )
    for name in obsolete_source:
        source = ROOT / name
        if source.exists():
            destination = archive_code / name
            shutil.move(str(source), str(destination))
            moved.append(f"{name} -> archive/obsolete_code/{name}")
    for name in (
        "foi_distributional_m4_deterministic_best.pt",
        "foi_distributional_m4_deterministic_best.pt.OBSOLETE.txt",
        "foi_distributional_m4_best.pt",
    ):
        source = ROOT / "checkpoints" / name
        if source.exists():
            destination = archive_checkpoints / name
            shutil.move(str(source), str(destination))
            moved.append(f"checkpoints/{name} -> archive/obsolete_checkpoints/{name}")

    for directory in (
        "coherent_weekly_oracle",
        "deterministic_diffusion",
        "deterministic_importance",
        "deterministic_target",
        "functional_foi",
    ):
        path = RESULTS / directory
        if path.exists():
            deleted.extend(
                str(file.relative_to(ROOT)).replace("\\", "/")
                for file in path.rglob("*")
                if file.is_file()
            )
            shutil.rmtree(path)
    keep_result_names = {
        RECONSTRUCTION_PATH.name,
        METRICS_PATH.name,
        FIGURE_PATH.name,
        SUCCESS_MARKER.name,
        "foi_sqrt_fpca_k16_pipeline.log",
        "foi_sqrt_fpca_k16_pipeline_errors.log",
    }
    for path in RESULTS.iterdir():
        if path.is_file() and path.name not in keep_result_names:
            deleted.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            path.unlink()

    for split, _ in SPLITS:
        path = DATA_ROOT / split
        for name in ("H.npy", "h_spline_coefficients.npy"):
            obsolete = path / name
            if obsolete.exists():
                deleted.append(str(obsolete.relative_to(ROOT)).replace("\\", "/"))
                obsolete.unlink()
        old_marker = path / "H_INSTANTANEOUS_COMPLETE"
        new_marker = path / "INSTANTANEOUS_H_COMPLETE"
        if old_marker.exists():
            old_marker.replace(new_marker)
            moved.append(
                f"deterministic_data/{split}/H_INSTANTANEOUS_COMPLETE -> "
                f"deterministic_data/{split}/INSTANTANEOUS_H_COMPLETE"
            )
        metadata_path = path / "metadata.npz"
        with np.load(metadata_path, allow_pickle=False) as loaded:
            metadata = {key: loaded[key] for key in loaded.files}
        metadata["external_foi_definition"] = "instantaneous_continuous_h_on_dense_grid"
        metadata["dense_times"] = DENSE_TIMES
        temporary = path / "metadata_active.npz"
        np.savez_compressed(temporary, **metadata)
        os.replace(temporary, metadata_path)
    obsolete_note = DATA_ROOT / "OBSOLETE_H_TARGET.txt"
    if obsolete_note.exists():
        deleted.append(str(obsolete_note.relative_to(ROOT)).replace("\\", "/"))
        obsolete_note.unlink()
    cache = ROOT / "__pycache__"
    if cache.exists():
        deleted.extend(
            str(file.relative_to(ROOT)).replace("\\", "/")
            for file in cache.rglob("*")
            if file.is_file()
        )
        shutil.rmtree(cache)

    tree = _record_tree()
    kept = [entry for entry in tree if not entry.startswith("archive/")]
    report = [
        "Square-root FPCA cleanup report",
        "",
        "Files kept:",
        *[f"  {entry}" for entry in kept],
        "",
        "Files moved to archive:",
        *[f"  {entry}" for entry in moved],
        "",
        "Files deleted:",
        *[f"  {entry}" for entry in deleted],
        "",
        "Final active directory tree:",
        *[f"  {entry}" for entry in tree],
        "",
        f"Active checkpoint: {DEFAULT_CHECKPOINT.relative_to(ROOT)}",
        "Active training command: .\\.venv\\Scripts\\python.exe -u -B foi_pipeline.py train",
        "Active evaluation command: .\\.venv\\Scripts\\python.exe -u -B foi_pipeline.py evaluate",
    ]
    (ROOT / "cleanup_manifest.txt").write_text("\n".join(report) + "\n")
    print("Cleanup complete; report written to cleanup_manifest.txt", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("data", "prepare", "train", "evaluate", "all", "cleanup"), nargs="?", default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-basis", action="store_true")
    args = parser.parse_args()
    if args.action in ("data", "all"):
        generate_or_reuse_data(args.workers)
    if args.action in ("prepare", "all"):
        prepare_basis_and_targets(force=args.force_basis)
    if args.action in ("train", "all"):
        if not torch.cuda.is_available():
            raise RuntimeError("The diffusion training run requires CUDA")
        torch.set_num_threads(2)
        train_diffusion(torch.device("cuda"))
    if args.action in ("evaluate", "all"):
        if not torch.cuda.is_available():
            raise RuntimeError("Diffusion evaluation requires CUDA")
        evaluate_diffusion(torch.device("cuda"))
    if args.action == "all":
        SUCCESS_MARKER.write_text("complete\n")
        cleanup_after_success()
    elif args.action == "cleanup":
        cleanup_after_success()


if __name__ == "__main__":
    main()
