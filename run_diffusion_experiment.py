"""Train the baseline conditional DDPM or plot one held-out epidemic."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from foi_diffusion import TemporalUNet, corrupt, diffusion_schedule, fit_transforms, inverse_transform, sample, transform
ROOT = Path(__file__).resolve().parent
STATES = ['CA', 'NY', 'DC', 'WY']

def load_split(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ('foi_ext', 'Y', 'labels')}

def plot_posterior(truth, samples, labels, path, title):
    median, low, high = np.quantile(samples, [0.5, 0.05, 0.95], axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, state in zip(axes.flat, STATES):
        i = list(labels).index(state)
        weeks = np.arange(1, 53)
        ax.fill_between(weeks, low[i], high[i], alpha=0.25, color='#0072B2', label='Pointwise 90% interval')
        ax.plot(weeks, median[i], color='#0072B2', label='Posterior median')
        ax.plot(weeks, truth[i], color='black', ls='--', label='True external FoI')
        ax.plot(weeks, samples[0, i], color='#D55E00', alpha=0.7, lw=0.8, label='One posterior draw')
        ax.set(title=state, xlabel='Week', ylabel='External FoI (per day)', xlim=(1, 52), ylim=(0, None))
        ax.grid(alpha=0.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)

def train(model, train_data, val_data, stats, schedule, device, args, output):
    z, y = transform(train_data['foi_ext'], train_data['Y'], stats)
    vz, vy = transform(val_data['foi_ext'], val_data['Y'], stats)
    generator = torch.Generator().manual_seed(args.seed + 10)
    vs = torch.randint(len(schedule[0]), (len(vz),), generator=generator)
    ve = torch.randn(vz.shape, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float('inf')
    history = []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for indices in torch.randperm(len(z)).split(args.batch_size):
            clean, condition = (z[indices].to(device), y[indices].to(device))
            step = torch.randint(len(schedule[0]), (len(clean),), device=device)
            noise = torch.randn_like(clean)
            loss = F.mse_loss(model(corrupt(clean, step, noise, schedule), step, condition), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(clean)
            seen += len(clean)
        model.eval()
        validation = 0.0
        with torch.no_grad():
            for j in range(0, len(vz), args.batch_size):
                clean, condition = (vz[j:j + args.batch_size].to(device), vy[j:j + args.batch_size].to(device))
                step, noise = (vs[j:j + args.batch_size].to(device), ve[j:j + args.batch_size].to(device))
                loss = F.mse_loss(model(corrupt(clean, step, noise, schedule), step, condition), noise)
                validation += loss.item() * len(clean)
        row = dict(epoch=epoch, train_loss=total / seen, validation_loss=validation / len(vz), seconds=perf_counter() - start)
        history.append(row)
        print(json.dumps(row), flush=True)
        if row['validation_loss'] < best:
            best = row['validation_loss']
            torch.save(dict(model=model.state_dict(), stats={k: torch.from_numpy(v) for k, v in stats.items()}, base=args.base, steps=len(schedule[0]), epoch=epoch, validation_loss=best, seed=args.seed), output / 'best.pt')
        (output / 'training.json').write_text(json.dumps(dict(history=history, best_validation_loss=best, runtime_seconds=perf_counter() - start, arguments=vars(args)), indent=2, default=str))
    return history

def evaluate(model, stats, test, schedule, device, args, output):
    if not 0 <= args.case < len(test['Y']):
        raise ValueError('Held-out case index is outside the test split')
    truth = test['foi_ext'][args.case:args.case + 1]
    _, condition = transform(truth, test['Y'][args.case:args.case + 1], stats)
    torch.manual_seed(args.seed + 300)
    batches = []
    for j in range(0, args.samples, args.batch_size):
        y = condition.repeat(min(args.batch_size, args.samples - j), 1, 1).to(device)
        batches.append(inverse_transform(sample(model, y, schedule, stats).cpu().numpy(), stats))
    draws = np.concatenate(batches)
    assert draws.shape == (args.samples, 51, 52) and np.isfinite(draws).all() and (draws >= 0).all()
    path = output / f'posterior_{args.case:03d}.png'
    plot_posterior(truth[0], draws, test['labels'], path, f'Held-out epidemic {args.case}: external FoI given weekly incidence')
    print(f'Saved {path}', flush=True)
    return draws

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['train', 'evaluate'], default='evaluate')
    parser.add_argument('--data', type=Path, default=ROOT / 'diffusion_data')
    parser.add_argument('--output', type=Path, default=ROOT / 'diffusion_outputs')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--base', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--case', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260916)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.samples, args.threads) <= 0:
        parser.error('Epochs, batch size, samples and threads must be positive')
    if args.base <= 0 or args.base % 8:
        parser.error('Base width must be a positive multiple of 8')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.phase == 'train':
        training = load_split(args.data / 'train.npz')
        validation = load_split(args.data / 'validation.npz')
        stats = fit_transforms(training['foi_ext'], training['Y'])
        model = TemporalUNet(args.base).to(device)
        args.output.mkdir(parents=True, exist_ok=True)
        print(f'Training from scratch on {device}', flush=True)
        train(model, training, validation, stats, diffusion_schedule(100, device), device, args, args.output)
    else:
        path = args.checkpoint or args.output / 'best.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = TemporalUNet(checkpoint['base']).to(device)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        stats = {k: v.numpy() for k, v in checkpoint['stats'].items()}
        test = load_split(args.data / 'test.npz')
        args.output.mkdir(parents=True, exist_ok=True)
        evaluate(model, stats, test, diffusion_schedule(checkpoint['steps'], device), device, args, args.output)
if __name__ == '__main__':
    main()
