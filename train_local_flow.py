"""Reusable training utility for a future explicitly supplied local dataset.

No dataset generation or training is part of the simulator validation workflow.
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from local_flow import FlowConfig, LocalPosteriorFlow
from local_flow_data import PreprocessingStats, load_split


ROOT = Path(__file__).resolve().parent
SEED = 20261013


def tensor_dataset(values, preprocessing):
    transformed = preprocessing.transform_numpy(values)
    return TensorDataset(
        *(
            torch.as_tensor(transformed[key], dtype=torch.float32)
            for key in ("theta", "Y", "h", "covariates")
        )
    )


def mean_nll(model, loader, device, log_jacobian):
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for theta, incidence, foi, covariates in loader:
            theta = theta.to(device)
            incidence = incidence.to(device)
            foi = foi.to(device)
            covariates = covariates.to(device)
            log_probability = model.log_prob_standardized(
                theta, incidence, foi, covariates
            ) - log_jacobian
            total += float((-log_probability).sum().cpu())
            count += len(theta)
    return total / count


def write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    training = load_split(args.data, "train")
    validation = load_split(args.data, "validation")
    preprocessing = PreprocessingStats.from_training_split(training)
    training_dataset = tensor_dataset(training, preprocessing)
    validation_dataset = tensor_dataset(validation, preprocessing)
    generator = torch.Generator().manual_seed(SEED)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    config = FlowConfig(covariate_features=training["covariates"].shape[1])
    wrapper = LocalPosteriorFlow.new(preprocessing, config=config, device=device)
    model = wrapper.model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    log_jacobian = wrapper.log_absolute_theta_jacobian

    best_validation = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_total = 0.0
        training_count = 0
        for theta, incidence, foi, covariates in training_loader:
            theta = theta.to(device)
            incidence = incidence.to(device)
            foi = foi.to(device)
            covariates = covariates.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_probability = model.log_prob_standardized(
                theta, incidence, foi, covariates
            ) - log_jacobian
            loss = -log_probability.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite flow training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            training_total += float(loss.detach().cpu()) * len(theta)
            training_count += len(theta)

        training_nll = training_total / training_count
        validation_nll = mean_nll(
            model, validation_loader, device, log_jacobian
        )
        scheduler.step(validation_nll)
        learning_rate = optimizer.param_groups[0]["lr"]
        improved = validation_nll < best_validation - 1e-4
        if improved:
            best_validation = validation_nll
            best_epoch = epoch
            epochs_without_improvement = 0
            elapsed = perf_counter() - started
            wrapper.save(
                args.checkpoint,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_nll": best_validation,
                    "training_wall_seconds_at_best": elapsed,
                    "training_examples": len(training_dataset),
                    "validation_examples": len(validation_dataset),
                    "data_directory": str(args.data.resolve()),
                    "seed": SEED,
                },
            )
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "training_nll": training_nll,
            "validation_nll": validation_nll,
            "learning_rate": learning_rate,
            "best_validation_nll": best_validation,
        }
        history.append(row)
        write_history(args.history, history)
        print(json.dumps(row), flush=True)
        if epochs_without_improvement >= args.patience:
            break

    wall_seconds = perf_counter() - started
    summary = {
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation,
        "training_wall_seconds": wall_seconds,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
    }
    print("training_summary " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
