# Continuous external-FoI diffusion

The active scientific model is the continuous deterministic coupled
51-location epidemic ODE. Its instantaneous force of infection is

```text
lambda_i(t) = a_i beta_i I_i(t)/N_i + h_i(t),
a_i = 1 - (p_a/N_i) sum_{j != i} M_ij,
h_i(t) = sum_{j != i} (p_a M_ij/N_i) beta_j I_j(t)/N_j.
```

Weekly incidence is integrated by an ODE state,
`mu[i,w] = C_i(w+1)-C_i(w)`, and observations follow
`Y[i,w] ~ Poisson(p_report * mu[i,w])`. External FoI `h_i(t)` is always a
continuous instantaneous function; there is no weekly-H target.

The representation uses `z_i(t)=sqrt(h_i(t))`, a cubic B-spline fit, and a
state-specific FPCA basis fixed at `K=16`. The diffusion target is the
`C.shape=[51,16]` score matrix (816 dimensions). Physical curves are recovered
as

```text
h_i(t) = [z_bar_i(t) + sum_{k=1}^16 C[i,k] phi_i,k(t)]^2.
```

`sqrt_foi_diffusion.sample_external_foi(Y, n_samples)` returns posterior draws
that are directly callable at arbitrary continuous times in `[0,52]`.
`foi_basis.reconstruct_external_foi(C, t)` is the lower-level reconstruction
function.

Commands:

```powershell
# Generate or reuse deterministic data
.\.venv\Scripts\python.exe -u -B foi_pipeline.py data --workers 4

# Fit the fixed basis and create K=16 targets
.\.venv\Scripts\python.exe -u -B foi_pipeline.py prepare

# Train the m=4, 100-step distributional diffusion (maximum 400 epochs,
# best-validation checkpointing, early-stopping patience 50)
.\.venv\Scripts\python.exe -u -B foi_pipeline.py train

# Evaluate in physical continuous-FoI space
.\.venv\Scripts\python.exe -u -B foi_pipeline.py evaluate

# Run the complete pipeline; cleanup occurs only after evaluation succeeds
.\.venv\Scripts\python.exe -u -B foi_pipeline.py all --workers 4
```

The active checkpoint is `checkpoints/foi_sqrt_fpca_k16_m4_best.pt`. The fixed
basis is `checkpoints/foi_sqrt_fpca_k16_basis.npz`. Final metrics and the single
summary figure are written under `results/`.
