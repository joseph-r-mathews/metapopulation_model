"""Generate deterministic data, train m=4 diffusion, and evaluate it."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from foi_diffusion import (DEFAULT_CHECKPOINT, TemporalUNet, corrupt,
                           diffusion_schedule, energy_loss, fit_transforms,
                           inverse_transform, load_foi_model, sample,
                           transform, transform_y)
from generate_diffusion_data import (DATA_ROOT, deterministic_sanity,
                                     generate_split)
from model import N_WEEKS


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "deterministic_diffusion"
SEED = 20260925
N_EVALUATION = 300
N_POSTERIOR_SAMPLES = 500
LEVELS = (.50, .70, .80, .90, .95)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_arrays(split, names):
    return {name: np.load(DATA_ROOT / split / f"{name}.npy", mmap_mode="r")
            for name in names}


def load_metadata(split):
    with np.load(DATA_ROOT / split / "metadata.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def train_diffusion(device):
    RESULTS.mkdir(parents=True, exist_ok=True)
    history_path = RESULTS / "deterministic_diffusion_training_loss.csv"
    if DEFAULT_CHECKPOINT.exists():
        saved = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
        if (saved.get("completed_epochs") == 200
                and saved.get("training_simulator") == "deterministic_coupled_ode"):
            print(f"Using complete checkpoint {DEFAULT_CHECKPOINT}", flush=True)
            return
        raise FileExistsError(f"Incomplete or incompatible checkpoint: {DEFAULT_CHECKPOINT}")
    training = load_arrays("train", ("H", "Y"))
    validation = load_arrays("validation", ("H", "Y"))
    stats = fit_transforms(training["H"], training["Y"])
    x0, y = transform(training["H"], training["Y"], stats)
    vx0, vy = transform(validation["H"], validation["Y"], stats)
    schedule = diffusion_schedule(100, device)
    torch.manual_seed(SEED)
    model = TemporalUNet(64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    fixed = torch.Generator().manual_seed(SEED + 10)
    vt = torch.randint(100, (len(vx0),), generator=fixed)
    ve = torch.randn(vx0.shape, generator=fixed)
    vxi = [torch.randn(vx0.shape, generator=fixed) for _ in range(4)]
    best, history, started = float("inf"), [], perf_counter()
    for epoch in range(1, 201):
        model.train()
        total = 0.
        for indices in torch.randperm(len(x0)).split(64):
            clean, condition = x0[indices].to(device), y[indices].to(device)
            step = torch.randint(100, (len(clean),), device=device)
            noisy = corrupt(clean, step, torch.randn_like(clean), schedule)
            predictions = [model(noisy, step, condition, torch.randn_like(noisy))
                           for _ in range(4)]
            loss = energy_loss(clean, *predictions)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(clean)
        model.eval()
        validation_total = 0.
        with torch.no_grad():
            for start in range(0, len(vx0), 64):
                sl = slice(start, start + 64)
                clean, condition = vx0[sl].to(device), vy[sl].to(device)
                step = vt[sl].to(device)
                noisy = corrupt(clean, step, ve[sl].to(device), schedule)
                predictions = [model(noisy, step, condition, xi[sl].to(device))
                               for xi in vxi]
                loss = energy_loss(clean, *predictions)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}")
                validation_total += loss.item() * len(clean)
        row = dict(epoch=epoch, train_energy_loss=total / len(x0),
                   validation_energy_loss=validation_total / len(vx0),
                   learning_rate=3e-4, runtime_seconds=perf_counter() - started)
        history.append(row)
        write_csv(history_path, history)
        print(json.dumps(row), flush=True)
        if row["validation_energy_loss"] < best:
            best = row["validation_energy_loss"]
            DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(
                method="distributional_energy", time_scale="weekly",
                observation_model="poisson",
                training_simulator="deterministic_coupled_ode",
                model=model.state_dict(),
                stats={key: torch.from_numpy(value) for key, value in stats.items()},
                base=64, steps=100, epoch=epoch, validation_energy_loss=best,
                seed=SEED, lambda_energy=1, beta_energy=1, m=4,
                batch_size=64, completed_epochs=epoch,
                trainable_parameters=sum(p.numel() for p in model.parameters()),
            ), DEFAULT_CHECKPOINT)
    saved = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    saved["completed_epochs"] = 200
    saved["training_runtime_seconds"] = perf_counter() - started
    torch.save(saved, DEFAULT_CHECKPOINT)


def draw_loaded(model, stats, schedule, Y, count, device, seed):
    condition = transform_y(np.asarray(Y)[None], stats).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    draws = []
    for start in range(0, count, 64):
        batch = condition.repeat(min(64, count - start), 1, 1)
        transformed = sample(model, batch, schedule, generator=generator)
        draws.append(inverse_transform(transformed.cpu().numpy(), stats))
    result = np.concatenate(draws)
    if result.shape != (count, 51, 52) or not np.isfinite(result).all():
        raise FloatingPointError("Invalid physical external-FoI samples")
    return result


def physical_energy_score(draws, truth):
    x = torch.as_tensor(draws, dtype=torch.float64).flatten(1)
    target = torch.as_tensor(truth, dtype=torch.float64).flatten()
    return (torch.linalg.vector_norm(x - target, dim=1).mean()
            - 0.5 * torch.pdist(x).mean()).item()


def posterior_plot(truth, draws, labels, case):
    median, low, high = np.quantile(draws, (.5, .05, .95), axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, state_name in zip(axes.flat, ("CA", "NY", "DC", "WY")):
        state = list(labels).index(state_name)
        weeks = np.arange(1, N_WEEKS + 1)
        ax.fill_between(weeks, low[state], high[state], alpha=.25,
                        label="90% interval")
        ax.plot(weeks, median[state], label="posterior median")
        ax.plot(weeks, truth[state], "k--", label="true deterministic H")
        ax.set(title=state_name, xlabel="Week", ylabel="External FoI / week",
               xlim=(1, N_WEEKS), ylim=(0, None))
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"deterministic_posterior_{case:03d}.png", dpi=160)
    plt.close(fig)


def evaluate_diffusion(device):
    model, stats, schedule = load_foi_model(device=device)
    test = load_arrays("test", ("H", "Y"))
    metadata = load_metadata("test")
    indices = np.random.default_rng(SEED + 400).choice(
        len(test["Y"]), N_EVALUATION, replace=False
    )
    truths, means, energy_scores = [], [], []
    covered = np.zeros(len(LEVELS), dtype=np.int64)
    widths = np.zeros(len(LEVELS))
    for number, index in enumerate(indices):
        draws = draw_loaded(model, stats, schedule, test["Y"][index],
                            N_POSTERIOR_SAMPLES, device, SEED + 300 + int(index))
        truth = np.asarray(test["H"][index])
        lows = np.quantile(draws, (1 - np.array(LEVELS)) / 2, axis=0)
        highs = np.quantile(draws, (1 + np.array(LEVELS)) / 2, axis=0)
        covered += ((truth[None] >= lows) & (truth[None] <= highs)).sum(axis=(1, 2))
        widths += (highs - lows).sum(axis=(1, 2))
        truths.append(truth)
        means.append(draws.mean(axis=0))
        energy_scores.append(physical_energy_score(draws, truth))
        print(f"Diffusion evaluation {number + 1}/{len(indices)}", flush=True)
    truth, mean = np.asarray(truths), np.asarray(means)
    row = dict(
        test_cases=len(indices), samples_per_case=N_POSTERIOR_SAMPLES,
        RMSE=float(np.sqrt(np.mean((mean - truth) ** 2))),
        correlation=float(np.corrcoef(mean.ravel(), truth.ravel())[0, 1]),
        multivariate_energy_score=float(np.mean(energy_scores)),
    )
    for j, level in enumerate(LEVELS):
        suffix = round(level * 100)
        row[f"coverage_{suffix}"] = float(covered[j] / truth.size)
        row[f"average_width_{suffix}"] = float(widths[j] / truth.size)
    write_csv(RESULTS / "deterministic_diffusion_test_metrics.csv", [row])
    for case in (0, 1, 2):
        draws = draw_loaded(model, stats, schedule, test["Y"][case],
                            N_POSTERIOR_SAMPLES, device, SEED + 800 + case)
        posterior_plot(np.asarray(test["H"][case]), draws,
                       metadata["labels"], case)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    sanity = deterministic_sanity()
    if args.sanity_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The 200-epoch diffusion run requires CUDA")
    generation = []
    for split_id, (name, count) in enumerate((
        ("train", 20000), ("validation", 2000), ("test", 2000)
    )):
        generation.append(generate_split(name, count, split_id, 20260924,
                                         args.workers))
    device = torch.device("cuda")
    torch.set_num_threads(2)
    train_diffusion(device)
    metrics = evaluate_diffusion(device)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    report = {
        "forward_model": "deterministic coupled ODE only",
        "stochastic_epidemic_used": False,
        "sanity": sanity,
        "generation": generation,
        "total_generation_wall_seconds": sum(x["wall_runtime_seconds"] for x in generation),
        "average_deterministic_solve_seconds": (
            sum(x["summed_solve_seconds"] for x in generation) / 24000
        ),
        "training_runtime_seconds": float(checkpoint["training_runtime_seconds"]),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_energy_loss": float(checkpoint["validation_energy_loss"]),
        "held_out_metrics": metrics,
    }
    (RESULTS / "deterministic_diffusion_final_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
