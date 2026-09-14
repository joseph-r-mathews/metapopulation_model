# 51-state stochastic SIR experiment

Two Python files:

- `model.py`: CSV loading, explicit force of infection, daily stochastic SIR,
  weekly aggregation, and independent binomial reporting.
- `run_experiments.py`: editable experiment defaults, numerical checks, and plots.

Run from this directory with the existing environment:

```bash
.venv/bin/python run_experiments.py
```

For a fresh environment, install `requirements.txt` with pip. Only NumPy and
Matplotlib are external dependencies; CSV loading uses Python's standard library.
Plots are saved without opening GUI windows.

## Data and equations

The two supplied CSVs are unchanged. State ordering follows `geodata_2019.csv`.
`ori` and `dest` match string `subpop` codes, preserving leading zeros.
`M[i,j]` is the number of residents of i commuting to j. The diagonal is zero;
counts are never row-normalized. With `h = theta * I / N`, the explicit equation is

```python
local = (1 - p_a * M.sum(axis=1) / N) * h
external = p_a * (M @ h) / N
foi = local + external
```

Thus residents of i experience destination j's transmission rate and infectious
prevalence. Daily infections are `Binom(S, 1-exp(-foi))`; recoveries are
`Binom(I, 1-exp(-gamma))`. Both use the state at the start of the day. Counts stay
in their state of residence; commuting changes exposure, not population totals.
The model takes `gamma` and `p_a` explicitly and assumes initially zero recovered.

## Experiment and outputs

Defaults are 364 days (52 weeks), gamma = 1/7 per day, p_a = 0.5, and 100 initial
infections in each of CA, TX, FL, NY. The baseline has all 51 beta values equal to
0.30. Five more vectors have independent `log(beta_i) ~ Normal(log(0.30), 0.20²)`.
All six runs use identical initial conditions and independent epidemic RNGs, so
differences reflect both transmission parameters and stochastic epidemic events.
Prior draws, epidemics, and reporting use separately seeded NumPy Generators.
These choices are experimental defaults, not fitted estimates.

`outputs/` contains:

- `weekly_foi.png`: total weekly-average FoI for CA, NY, DC, WY across all runs.
- `foi_components.png`: local and external FoI for the same states and runs,
  on separate vertical scales so the small mobility component remains visible.
- `weekly_incidence.png`: national and selected-state weekly new infections.
- `simulations.npz`: parameters, labels, inputs, RNG seeds, and simulation arrays.

For `simulate(theta, n_days, rng, N=N, M=M, I0=I0, gamma=gamma, p_a=p_a)`, returned
`S`, `I`, `R` have shape `(365, 51)` at the default horizon, including day zero.
`incidence`, `recoveries`, `foi`, `foi_local`, `foi_external` have shape `(364, 51)`;
entry t describes the interval from t to t+1. Labels are returned separately by
`load_data`. `weekly_summary` returns `(52, 51)` arrays: incidence sums and FoI
arithmetic means, still in units of per day. Week one uses daily entries 0–6.
It requires complete weeks; the daily simulator accepts other horizons.

The archive stacks each daily/weekly array on a leading run axis (6 runs; baseline
first). Weekly names have a `weekly_` prefix. `theta` is `(6, 51)`. `Y` is
`(6, 52, 51)`, sampled separately as `Binom(weekly_incidence, 0.4)`.
For example, a state's future training triple is `theta[k]`,
`weekly_foi[k, :, i]`, `Y[k, :, i]`; the FoI curve has 52 entries.

The script checks population conservation, nonnegative integer counts, transition
identities, event bounds, daily/weekly FoI decomposition, weekly incidence totals,
and observation bounds. It prints the mobility distribution and epidemic totals.

## Conditional diffusion experiment

The epidemic simulator is reused unchanged. Three additional files implement
the experiment without an ML framework:

- `generate_diffusion_data.py`: independent simulation triples and disk caching.
- `foi_diffusion.py`: training-only transforms, a temporal convolutional U-Net,
  forward noising, and ancestral DDPM sampling.
- `run_diffusion_experiment.py`: tiny-data diagnostic, training, validation-based
  checkpoint selection, held-out evaluation, and plots.

PyTorch is the only additional dependency. On this Intel Mac the existing
`.venv` has Python 3.13 without PyTorch; the working PyTorch environment is
`/Users/jm/anaconda3/bin/python3.10`. Substitute that interpreter for `python`
in the training commands below. CUDA or MPS is used if available; this machine
runs on CPU. Data generation can use `.venv/bin/python`.

```bash
python generate_diffusion_data.py --pilot-only
python generate_diffusion_data.py --train 20000 --validation 2000 --test 2000
python run_diffusion_experiment.py --phase tiny
# Inspect diffusion_outputs/tiny/overfit.png before full training.
python run_diffusion_experiment.py --phase train
python run_diffusion_experiment.py --phase evaluate
```

Counts, output directories, training epochs, width, batch size, and sample counts
are simple command-line arguments; use `--help`. Training never generates data.
Generation checks 100 pilot simulations first, then caches the three splits in
`diffusion_data/{train,validation,test}.npz`. Existing files are reused; choose
a new `--output` directory to change counts or the dataset seed.

Each split stores `theta` `(n,51)`, `foi_ext` `(n,51,52)`, and `Y` `(n,51,52)`.
The target is the weekly arithmetic mean of **external** daily FoI, in /day.
Daily trajectories are discarded after each simulation. State labels, physical
parameters, and seed metadata are retained. NumPy `SeedSequence` uses the dataset
seed, split ID, and simulation index, then creates separate prior, disease, and
reporting streams. Changing worker count does not change the dataset. Every run
draws an independent beta vector and uses the same four-state initialization.

The network receives only noisy transformed external FoI, the diffusion step,
and transformed weekly `Y` from all 51 states. Stored `theta` is never loaded by
the training/evaluation driver. The U-Net has temporal resolutions 52, 26, 13,
residual Conv1d blocks, SiLU, GroupNorm, sinusoidal step embeddings, and skip
connections. A direct noisy-input skip preserves fine-resolution noise across
all 51 channels; the output is still trained with the ordinary epsilon MSE.

FoI is transformed as `log(max(foi_ext, 1e-10))`; `Y` uses `log1p(Y)`. Means and
standard deviations are fitted per state, pooling training epidemics and weeks
only. The inverse FoI transform is exponential and always nonnegative. Exact
zero and values below 1e-10/day are approximated by that floor; raw-scale metrics
still use the original unmodified test targets. Consequently exact-zero targets
cannot be covered by strictly positive samples, a limitation of this transform.

There are 100 cosine-schedule steps. The forward process is
`z_s = sqrt(alpha_bar_s)*z_0 + sqrt(1-alpha_bar_s)*epsilon`. Training minimizes
mean squared epsilon prediction error with AdamW. Reverse sampling uses the
standard Gaussian posterior mean and fixed posterior variance, with no noise
at the final step. The estimated clean `z_0` is clipped to each state's training
range before computing the posterior mean. This ordinary DDPM stabilization
prevents high-noise errors from exploding through the exponential; it also
restricts extrapolation beyond the training range. There is no curve smoothing,
attention, conditioning on theta, or state-specific inference.

Equations follow [Ho et al., DDPM](https://arxiv.org/abs/2006.11239); the cosine
schedule and clean-sample clipping follow the
[Improved Diffusion reference implementation](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py).

`diffusion_outputs/best.pt` saves weights, transforms, schedule length, width,
and the best validation epoch. Validation uses fixed randomly drawn timesteps
and noise for comparable epoch losses. The tiny diagnostic uses 50 pilot
simulations for both training and evaluation deliberately; its loss is an
overfit check, not a generalization estimate. Full training initializes a new
network and refits transforms on the actual training split.

Held-out evaluation uses the first 32 test epidemics, fixed before inspecting
results, with 200 independent reverse samples per epidemic. `metrics.csv`
reports raw-scale RMSE of the posterior mean, pointwise 90% coverage, mean
interval width, and flattened Pearson correlation, overall and separately for
CA, NY, DC, WY. These are pointwise intervals, not simultaneous curve bands.
The first three cases are plotted, including one actual draw so that a smooth
median cannot hide noisy samples. `posterior_summary.npz` retains the true
curves, posterior summaries, indices, labels, and draws for those three cases.
`loss.csv`, `loss.png`, and `training.json` retain loss history and runtime.
The test split is used only after selecting the checkpoint on validation loss.
