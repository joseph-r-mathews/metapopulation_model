# 51-state SIR and conditional FoI diffusion baseline

The pipeline uses five small Python files:

- `model.py`: raw CSV loading, daily stochastic coupled SIR, local/external force of infection, weekly aggregation, and binomial reporting.
- `generate_diffusion_data.py`: prior-predictive `(Y, external FoI)` simulation pairs and deterministic disk caching.
- `foi_diffusion.py`: the temporal Conv1d conditional U-Net, baseline transforms, forward diffusion, and reverse sampling.
- `run_diffusion_experiment.py`: training with validation-based checkpoint selection and one basic held-out posterior plot.
- `run_experiments.py`: the original standalone SIR numerical checks and plots.

Raw inputs remain at the project root: `geodata_2019.csv` and `mobility_2011-2015_statelevel.csv`. State ordering follows the population CSV, including string subpopulation codes with leading zeros. Mobility entries are raw directed commuter counts, with a zero diagonal; they are not row-normalized.

With `h = theta * I / N`, daily force of infection is:

```python
local = (1 - p_a * M.sum(axis=1) / N) * h
external = p_a * (M @ h) / N
foi = local + external
```

Daily infections and recoveries use independent binomial draws from the state at the start of the day. Commuting changes exposure while populations remain in their state of residence. The simulator is unchanged.

## Environment and commands

Install `requirements.txt` in a virtual environment. This Windows project's existing `.venv` already has CUDA-enabled PyTorch for the GTX 1060. Use its interpreter directly; activation is optional.

```powershell
# Generate caches only when needed; existing files are reused.
.\.venv\Scripts\python.exe generate_diffusion_data.py --train 20000 --validation 2000 --test 2000

# Explicitly start fresh training. Training does not generate data or resume.
.\.venv\Scripts\python.exe run_diffusion_experiment.py --phase train --epochs 10

# Plot one held-out epidemic using the canonical checkpoint.
.\.venv\Scripts\python.exe run_diffusion_experiment.py --phase evaluate --case 0 --samples 200
```

On macOS/Linux use `.venv/bin/python` instead. CUDA, then MPS, then CPU is selected automatically. `--help` lists the simple command-line options. Training saves `best.pt` by lowest validation noise MSE and records its history/runtime in `training.json`. Evaluation loads checkpoint weights and transforms and saves only `posterior_000.png` (or the selected case's index). Plots show truth, median, a pointwise 90% interval, and one posterior draw for CA/NY/DC/WY, without opening GUI windows.

## Cached data and baseline method

`diffusion_data/` retains the canonical `train.npz` (20,000 simulations), `validation.npz` (2,000), and `test.npz` (2,000), plus the generator's checked 100-simulation `pilot.npz`. Each split contains `theta` `(n,51)`, `foi_ext` `(n,51,52)`, and `Y` `(n,51,52)`, with state labels, physical parameters, and seeds. The target is the weekly mean external daily FoI, in /day. Weekly incidence is independently reported with probability 0.4. Defaults use 364 days, gamma=1/7, commuting participation=0.5, and 100 initial infections in CA/TX/FL/NY. Prior beta values are independent lognormal draws centered at 0.30 with log-scale standard deviation 0.20.

The U-Net is unchanged: 51 state channels, temporal resolutions 52/26/13, and conditioning only on transformed weekly `Y`. Stored `theta` is not used by the network. The baseline FoI transform is `log(max(foi_ext, 1e-10))`, standardized per state from training data only. The Y transform remains standardized `log1p(Y)`. Inversion is exponential; exact zeros and values below the floor are approximated by the floor.

The baseline uses the existing 100-step cosine diffusion schedule, epsilon MSE with AdamW, and ancestral Gaussian DDPM sampling. Each estimated clean transformed sample is clipped to that state's training range at every reverse step, with no added noise on the final step. There are no alternative transform or clipping modes.

`diffusion_outputs/best.pt` is the retained original baseline checkpoint, selected during the initial 50-epoch run. Its weights, transforms, width, and schedule can be loaded by the clean driver. Recent calibration, longer-training comparison, alternative-transform, and clipping-study scripts/results have been removed. `.gitignore` excludes environments and generated caches/outputs.
