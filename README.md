# 51-state SIR and canonical external-FoI diffusion model

This repository contains the stochastic 51-state metapopulation SIR simulator
and one trained conditional distributional-diffusion model for weekly external
force of infection (FoI).

The retained diffusion model is the m=4 model with energy exponent 1, lambda 1,
and a 100-step cosine schedule. It was trained for 200 epochs; validation energy
loss selected epoch 199 as the global best checkpoint.

## Main files

- `model.py`: population/mobility loading, SIR dynamics, exact local/external
  FoI, weekly aggregation, reporting, and `simulate_epidemic`.
- `generate_diffusion_data.py`: deterministic generation of the existing
  20,000/2,000/2,000 train/validation/test simulation dataset.
- `foi_diffusion.py`: transforms, Conv1d stochastic denoiser, fixed m=4 energy
  loss, forward diffusion, reverse bridge sampler, checkpoint loader, and
  `sample_external_foi`.
- `run_experiments.py`: retained SIR numerical-check and plotting utility.

Raw data remain at the project root because the unchanged epidemic data loader
expects them there. The canonical cached dataset remains under `diffusion_data/`.

## Sampling external FoI

```python
from foi_diffusion import sample_external_foi

# Y has shape [51, 52]. The result is in physical FoI/day units.
draws = sample_external_foi(Y, n_samples=2)
assert draws.shape == (2, 51, 52)
```

For batched observations shaped `[batch, 51, 52]`, the returned shape is
`[n_samples, batch, 51, 52]`. CUDA is selected when available; pass
`device="cpu"` to override it. No beta conditioning or MCMC is implemented.

The canonical checkpoint is `checkpoints/foi_distributional_m4_best.pt`.
Final training metadata and reference posterior plots are in
`results/final_distributional/`.

## Epidemic output for future MCMC work

`model.simulate_epidemic(...)` returns a plain dictionary containing `X`, `Y`,
`local_foi`, `true_external_foi`, and `true_beta`. Weekly arrays use `[51, 52]`;
`X` uses `[day, state, S/I/R]`.

Install the dependencies in `requirements.txt`. The existing dataset and
checkpoint are already present; do not regenerate data or retrain merely to use
the model.
