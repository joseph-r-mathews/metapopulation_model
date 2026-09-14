"""Train/evaluate cached external-FoI conditional DDPM; see --help."""

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
from torch.nn import functional as F

from foi_diffusion import (TemporalUNet, corrupt, diffusion_schedule,
                           fit_transforms, inverse_transform, sample, transform)

ROOT = Path(__file__).resolve().parent
STATES = ["CA", "NY", "DC", "WY"]


def load_split(path):
    with np.load(path) as data:
        return {k: data[k] for k in ("foi_ext", "Y", "labels")}


def plot_posterior(truth, samples, labels, path, title):
    median, low, high = np.quantile(samples, [.5, .05, .95], axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, state in zip(axes.flat, STATES):
        i = list(labels).index(state)
        weeks = np.arange(1, 53)
        ax.fill_between(weeks, low[i], high[i], alpha=.25, color="#0072B2", label="Pointwise 90% interval")
        ax.plot(weeks, median[i], color="#0072B2", label="Diffusion median")
        ax.plot(weeks, truth[i], color="black", ls="--", label="True external FoI")
        # Show an actual draw, since a smooth median can hide noisy samples.
        ax.plot(weeks, samples[0, i], color="#D55E00", alpha=.7, lw=.8, label="One diffusion draw")
        ax.set(title=state, xlabel="Week", ylabel="External FoI (per day)", xlim=(1, 52), ylim=(0, None))
        ax.grid(alpha=.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(.5, .96))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model, stats, test, schedule, device, args, output):
    """Use the first n test epidemics, fixed in advance, with no selection."""
    _, conditions = transform(test["foi_ext"], test["Y"], stats)
    n = min(args.test_cases, len(conditions))
    truth = test["foi_ext"][:n]
    means, medians, lows, highs, example_samples = [], [], [], [], []
    torch.manual_seed(args.seed + 300)
    start = perf_counter()
    for k in range(n):
        draws = []
        for j in range(0, args.samples, args.batch_size):
            y = conditions[k:k+1].repeat(min(args.batch_size, args.samples-j), 1, 1).to(device)
            draws.append(inverse_transform(sample(model, y, schedule, stats).cpu().numpy(), stats))
        draws = np.concatenate(draws)
        assert draws.shape == (args.samples, 51, 52)
        assert np.isfinite(draws).all() and np.all(draws >= 0)
        low, median, high = np.quantile(draws, [.05, .5, .95], axis=0)
        means.append(draws.mean(0)); medians.append(median)
        lows.append(low); highs.append(high)
        if k < 3:
            example_samples.append(draws)
            plot_posterior(truth[k], draws, test["labels"], output / f"posterior_{k:03d}.png",
                           f"Held-out epidemic {k}: external FoI given weekly incidence")
        print(f"Evaluation {k+1}/{n}, {args.samples} samples: {perf_counter()-start:.1f}s", flush=True)
    mean, low, high = np.array(means), np.array(lows), np.array(highs)
    metrics = []
    for state in ["All"] + STATES:
        idx = slice(None) if state == "All" else [list(test["labels"]).index(state)]
        target, estimate = truth[:, idx], mean[:, idx]
        metrics.append(dict(state=state, rmse=float(np.sqrt(np.mean((estimate-target)**2))),
                            coverage_90=float(np.mean((target >= low[:, idx]) & (target <= high[:, idx]))),
                            width_90=float(np.mean(high[:, idx]-low[:, idx])),
                            correlation=float(np.corrcoef(target.ravel(), estimate.ravel())[0, 1])))
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    np.savez_compressed(output / "posterior_summary.npz", truth=truth, mean=mean,
                        median=np.array(medians), low=low, high=high, labels=test["labels"],
                        test_indices=np.arange(n), example_samples=np.array(example_samples))
    support = inverse_transform(np.concatenate((stats["z_min"], stats["z_max"])), stats)
    details = dict(test_cases=n, samples_per_case=args.samples, runtime_seconds=perf_counter()-start,
                   target_fraction_below_log_floor=float(np.mean(truth < 1e-10)),
                   target_fraction_above_training_max=float(np.mean(truth > support[1])), metrics=metrics)
    with (output / "evaluation.json").open("w") as f:
        json.dump(details, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics, perf_counter()-start


def train(model, train_data, val_data, stats, schedule, device, args, output, tiny=False):
    z, y = transform(train_data["foi_ext"], train_data["Y"], stats)
    vz, vy = transform(val_data["foi_ext"], val_data["Y"], stats)
    # Fixed validation corruption gives comparable epoch losses; no test data used.
    generator = torch.Generator().manual_seed(args.seed + 10)
    vs = torch.randint(args.steps, (len(vz),), generator=generator)
    ve = torch.randn(vz.shape, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, history, start = float("inf"), [], perf_counter()
    rounds = (args.tiny_steps + 99)//100 if tiny else args.epochs
    for epoch in range(rounds):
        model.train()
        order = torch.randperm(len(z))
        batches = [torch.randint(len(z), (min(args.batch_size, len(z)),)) for _ in range(100)] if tiny else order.split(args.batch_size)
        total, seen = 0., 0
        for indices in batches:
            clean, condition = z[indices].to(device), y[indices].to(device)
            step = torch.randint(args.steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            prediction = model(corrupt(clean, step, noise, schedule), step, condition)
            loss = F.mse_loss(prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(clean); seen += len(clean)
        model.eval()
        validation = 0.
        with torch.no_grad():
            for j in range(0, len(vz), args.batch_size):
                clean, condition = vz[j:j+args.batch_size].to(device), vy[j:j+args.batch_size].to(device)
                step, noise = vs[j:j+args.batch_size].to(device), ve[j:j+args.batch_size].to(device)
                loss = F.mse_loss(model(corrupt(clean, step, noise, schedule), step, condition), noise)
                validation += loss.item() * len(clean)
        row = dict(epoch=epoch+1, train_loss=total/seen, validation_loss=validation/len(vz),
                   seconds=perf_counter()-start)
        history.append(row)
        print(("Tiny " if tiny else "") + json.dumps(row), flush=True)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save(dict(model=model.state_dict(), stats={k: torch.from_numpy(v) for k, v in stats.items()}, base=args.base, steps=args.steps,
                            epoch=epoch+1, validation_loss=best, seed=args.seed), output / "best.pt")
        with (output / "loss.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader(); writer.writerows(history)
    runtime = perf_counter()-start
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Training")
    ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], label="Fixed-corruption validation")
    ax.set(xlabel="100 updates" if tiny else "Epoch", ylabel="Noise prediction MSE", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "loss.png", dpi=150); plt.close(fig)
    return history, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["tiny", "train", "evaluate"], default="tiny")
    parser.add_argument("--data", type=Path, default=ROOT / "diffusion_data")
    parser.add_argument("--output", type=Path, default=ROOT / "diffusion_outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tiny-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--test-cases", type=int, default=32)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    output = args.output / "tiny" if args.phase == "tiny" else args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "tiny":
        pilot = load_split(args.data / "pilot.npz")
        train_data = {k: v[:50] if k != "labels" else v for k, v in pilot.items()}
        val_data = train_data  # Intentional overfit diagnostic, not held-out validation.
    elif args.phase == "train":
        train_data = load_split(args.data / "train.npz")
        val_data = load_split(args.data / "validation.npz")
    if args.phase != "evaluate":
        stats = fit_transforms(train_data["foi_ext"], train_data["Y"])
        model = TemporalUNet(args.base).to(device)
        schedule = diffusion_schedule(args.steps, device)
        z, y = transform(train_data["foi_ext"][:4], train_data["Y"][:4], stats)
        np.testing.assert_allclose(inverse_transform(z.numpy(), stats),
                                   np.maximum(train_data["foi_ext"][:4], 1e-10), rtol=1e-5)
        step = torch.tensor([0, 25, 50, 99], device=device).clamp(max=args.steps-1)
        noise = torch.randn_like(z).to(device)
        noisy = corrupt(z.to(device), step, noise, schedule)
        recovered = (noisy - (1-schedule[2][step, None, None]).sqrt()*noise) / schedule[2][step, None, None].sqrt()
        torch.testing.assert_close(recovered, z.to(device), atol=1e-3, rtol=1e-3)
        assert model(noisy, step, y.to(device)).shape == (4, 51, 52)
        assert sample(model, y.to(device), schedule, stats).shape == (4, 51, 52)
        print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
              f"train/validation: {len(train_data['Y'])}/{len(val_data['Y'])}; sanity checks passed", flush=True)
        history, runtime = train(model, train_data, val_data, stats, schedule, device, args, output, args.phase == "tiny")
        with (output / "training.json").open("w") as f:
            json.dump(dict(runtime_seconds=runtime, device=str(device), arguments=vars(args),
                           final=history[-1], best_validation_loss=min(r["validation_loss"] for r in history)), f, indent=2, default=str)
    checkpoint = torch.load(output / "best.pt", map_location="cpu")
    model = TemporalUNet(checkpoint["base"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = {k: v.numpy() for k, v in checkpoint["stats"].items()}
    schedule = diffusion_schedule(checkpoint["steps"], device)
    if args.phase == "tiny":
        _, y = transform(train_data["foi_ext"][:1], train_data["Y"][:1], stats)
        draws = inverse_transform(sample(model, y.repeat(32, 1, 1).to(device), schedule, stats).cpu().numpy(), stats)
        assert np.isfinite(draws).all() and np.all(draws >= 0)
        plot_posterior(train_data["foi_ext"][0], draws, train_data["labels"], output / "overfit.png",
                       "Tiny-data overfit: training epidemic 0 (32 draws)")
        np.savez_compressed(output / "overfit_samples.npz", samples=draws, truth=train_data["foi_ext"][0])
        print(f"Inspect {output / 'overfit.png'} before starting full training.", flush=True)
    else:
        evaluate(model, stats, load_split(args.data / "test.npz"), schedule, device, args, output)


if __name__ == "__main__":
    main()
