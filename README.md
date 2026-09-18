# Age x vaccination x variant metapopulation SEIR

The active model has **51 locations, 3 age groups, 2 vaccination states
(U,V), 2 variants, and 8 known NPI intervals per location**. Each location has
two independent 3x3 baseline transmission matrices and eight NPI effects:
26 inferred parameters per location, 1326 in total. Everything else is fixed.
Defaults below are synthetic methodological choices, not scientific estimates.

## Equations

Matrix rows are infectious/source ages b; columns are susceptible/recipient
ages a. There is no separate age-contact matrix. Time is measured in weeks.

```text
m_i(t)       = product_l [1 - r_il * 1{start_il <= t <= end_il}]
beta_ik(t)  = beta0_ik * m_i(t)
q_ij        = 0.5 * M_ij / N_i                    (j != i)
q_ii        = 1 - sum_{j != i} q_ij
J_ibk       = I_ibUk + I_ibVk
g_iak       = sum_b beta0_ikba * m_i(t) * J_ibk / N_ib
ell_iak     = q_ii * g_iak
h_iak       = sum_{j != i} q_ij * g_jak
lambda_iavk = c_vk * (ell_iak + h_iak)
c_Uk        = 1
c_Vk        = 1 - epsilon_k

dS_iav/dt   = -S_iav * sum_k lambda_iavk + vaccination_S_iav
dE_iavk/dt  = S_iav * lambda_iavk - sigma_k * E_iavk
dI_iavk/dt  = sigma_k * E_iavk - gamma_k * I_iavk
dR_iav/dt   = sum_k gamma_k * I_iavk + vaccination_R_iav
dC_iavk/dt  = S_iav * lambda_iavk
```

Vaccination transfers S_U to S_V at nu_ia(t)*S_U and R_U to R_V at
nu_ia(t)*R_U. E and I are not vaccinated. S and R are shared across variants;
recovery gives complete immunity to both variants. There is no waning,
reinfection, cross-immunity parameter, reporting parameter, or inferred
susceptibility/infectiousness modifier.

Weekly incidence is C(w+1)-C(w), resolved by location, age, vaccination, and
variant. Observations are independent Poisson draws with these means.

Population is conserved within each location/age after summing vaccination
groups. Each vaccination-stratum budget equals its signed vaccination
transfer; its population is not constant during vaccination. M couples
infection pressure, not population transport. The raw off-diagonal mobility
matrix is not renormalized; negative q_ii raises an error.

## Fixed ordering and array shapes

The unconstrained vector is location-major. For each location its 26 entries
are, in exactly this order:

```text
 0..8:  log beta0[variant=0, b,a]:
        (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
 9..17: log beta0[variant=1, b,a], in the same order
18..25: logit r_NPI[0], ..., logit r_NPI[7]
```

`pack_parameters` and `unpack_parameters` act on these unconstrained arrays;
`transform_parameters` returns positive beta and NPI effects strictly between
zero and one. Unrepresentable extreme coordinates raise an error rather than
silently clipping. `parameter_index` provides explicit flat indices.

| Quantity | Shape / ordering |
|---|---|
| theta; local theta | `(1326,)`; `(26,)` |
| log beta / beta0 | `(51,2,3,3)` = location, variant, source age, recipient age |
| NPI logits / effects | `(51,8)` |
| NPI intervals | `(51,8,2)` = location, NPI, start/end |
| S, R | `(51,3,2)` |
| E, I, C, lambda | `(51,3,2,2)` = location, age, vaccination, variant |
| ell, h, g | `(51,3,2)` = location, age, variant |
| populations; age populations | `(51,)`; `(51,3)` |
| raw M, q | `(51,51)` |
| vaccine efficacy, sigma, gamma | `(2,)` each |
| vaccination callable output | `(51,3)` |
| vaccination change times | `(2,)`, defaults `[8,36]` |
| vaccination schedule rates | `(3,51,3)` = schedule segment, location, age |
| baseline prior log means; NPI prior means | `(2,3,3)`; `(8,)` |
| packed coupled state | `(51,3,2,8)` flattened to `(2448,)` |
| packed local state | `(3,2,8)` flattened to `(48,)` |
| component order | `S,E_variant1,E_variant2,I_variant1,I_variant2,R,C_variant1,C_variant2` |
| dense state evaluation at T times | `(2448,T)`, reshapes to `(51,3,2,8,T)` |
| weekly means and observations | `(51,3,2,2,52)` |
| local weekly means | `(3,2,2,52)` |
| external FoI evaluated at T times | `(T,51,3,2)`; smoke `(365,51,3,2)` |
| dense output times | `(365,)`, spanning weeks 0 to 52 |
| local forcing callable | `h_func(t) -> (3,2)` |
| future flattened local flow inputs | theta `(26,)`, Y `(12,52)`, h `(6,365)` |

There are 1836 epidemic state variables and 612 cumulative-incidence states,
for 2448 ODE variables. Local solves have 36 epidemic and 12 cumulative states.

## Centralized defaults

All scientific constants and priors are in `stratified_model.py`.

Age fractions are `[0.22,0.50,0.28]`; initial U/V fractions are `[0.70,0.30]`.
Vaccine efficacies are `[0.70,0.45]`. Both latent rates are 7/3 per week
(three-day mean); both infectious rates are 1 per week (one-week mean).
CA, TX, FL, and NY each start with 60 variant-1 and 40 variant-2 infections,
distributed proportionally across age/vaccination strata and subtracted from S.
E, R, and C initially vanish. These initial infections are not counted as
new infections in C.

The two positive matrices defining the log-transmission prior means are:

```text
variant 1:                 variant 2:
[[1.50, 0.35, 0.15],       [[1.65, 0.40, 0.20],
 [0.30, 1.40, 0.30],        [0.35, 1.55, 0.35],
 [0.15, 0.40, 1.45]]        [0.20, 0.45, 1.60]]
```

Each log beta has independent Normal(log(matrix entry), 0.20^2) prior.
Each NPI logit has independent Normal(logit(0.15), 0.35^2) prior.
`log_prior` is a density in these unconstrained coordinates. No prior was tuned.

The centralized NPI intervals are `[4,10], [6,14], [12,20], [18,26],
[24,32], [30,38], [36,46], [42,50]` weeks for every location by default.
Custom intervals may differ by location. Overlapping effects multiply.
The default vaccination callable is zero before week 8, uses rates
`[0.01,0.015,0.02]` per week by age on `[8,36)`, and is zero from week 36 onward.
Custom `nu_func(t)` returns `(51,3)` and accepts accompanying
`vaccination_change_times`; a smooth callable needs no breakpoints.

## Continuous integration and local forcing

Both solvers use SciPy `solve_ivp`, DOP853, rtol=1e-9, atol=1e-5,
dense_output=True, and a default 52-week horizon. These are the pre-existing
tolerances. Integration restarts at every known NPI/schedule discontinuity.
Endpoint RHS evaluations use one-sided segment limits; pointwise external
FoI evaluation follows the specified closed-interval NPI convention. Dense
interpolants are joined into a continuous trajectory. Weekly means come from
that trajectory's cumulative states, subject to numerical integration error.

`solve_epidemic(theta, labels, populations, mobility, **known_inputs)` returns
a solution, weekly means, and a plain dictionary of fixed inputs.
`compute_external_foi(solution, theta, times)` evaluates instantaneous h from
the coupled dense trajectory, including the source-location NPI multiplier.
It does not use weekly averages or an interpolated grid of FoI samples.

For a frozen local solve:

```python
from stratified_model import extract_local_parameters
from local_frozen_simulator import make_exact_external_foi_function, solve_local_frozen_foi

h_func = make_exact_external_foi_function(full["solution"], theta, location)
local = solve_local_frozen_foi(
    extract_local_parameters(theta, location), h_func,
    full["fixed_inputs"], location,
)
```

The exact-forcing helper carries source intervention breakpoints into the
local solver. A custom discontinuous h callable should supply
`forcing_change_times`. The local solver uses the same compartment derivative,
age denominators, q_ii, vaccine effects, known schedules, and initial state.

## Validation and one smoke simulation

```powershell
.\.venv\Scripts\python.exe validate_stratified_model.py
.\.venv\Scripts\python.exe simulate_stratified_epidemic.py
```

The validation script includes the frozen identity checks; there is no need
to run `validate_local_frozen_simulator.py` separately unless isolating those
checks. Validation uses deterministic parameters and location-specific NPI
intervals, including noninteger boundaries. It checks equations against
independent scalar sums, integrated vaccination budgets by quadrature, a
vaccination-only analytic solution, absence of an unseeded variant, and
four frozen local identities over the full 52 weeks.

Recorded validation: **32 checks passed**. Maximum frozen local incidence
relative-L2 error was **3.26251e-8** (requirement <1e-5); maximum population
conservation error was **2.60770e-8 people**. Every vaccination-stratum
integrated budget agreed within **3.45521e-7 people**. Minimum compartment was
zero, cumulative incidence was nonnegative/nondecreasing, and minimum q_ii
was 0.880946. Details: `results/validation_results.json`.

Exactly one smoke simulation was run after validation, using seed 20260917.
Integration took **1.22059 seconds** on this machine. Expected variant totals
were **1,295,512.799** and **151,579,155.390**; Poisson totals were
**1,295,706** and **151,574,305**. Maximum population error was
**1.67638e-8 people** and minimum compartment value was **0**.

`results/stratified_smoke/` contains theta, weekly means, observations, dense
FoI and its times, NPI intervals, vaccine efficacy, vaccination breakpoints
and rate arrays, populations, age populations, raw mobility, q, initial state,
sigma, gamma, and prior means as named `.npy` files. `metadata.json` records
all array shapes, parameter/state ordering, seed, fixed constants, prior
standard deviations, integration settings, software versions, input/source
SHA-256 hashes, runtime, and diagnostic results for reproduction.

## Retained infrastructure and directory tree

No flow dataset was generated and no normalizing flow was trained. The
reusable spline architecture, preprocessing, and training utility remain.
They obtain dimensions from the model (26 parameters, 12 incidence channels,
6 FoI channels); future datasets/checkpoints with incompatible shapes are
rejected. Covariates have no built-in obsolete encoding: their dimension is
explicitly supplied/inferred from future preprocessing. Future training must
define an encoding of the known local inputs, including schedules. The
training CLI requires explicit data, checkpoint, and history paths.

The active directory tree, excluding Git, the virtual environment, and Python
caches, is:

```text
metapopulation_model/
  .gitignore
  README.md
  requirements.txt
  geodata_2019.csv
  mobility_2011-2015_statelevel.csv
  epidemic_model.py                 # geographic input loader only
  stratified_model.py               # equations, dimensions, priors, defaults
  stratified_simulator.py           # coupled solve and continuous FoI
  local_frozen_simulator.py         # same equations with frozen h
  validate_stratified_model.py
  validate_local_frozen_simulator.py
  simulate_stratified_epidemic.py
  local_flow.py                     # reusable, untrained
  local_flow_data.py
  train_local_flow.py               # requires explicit future dataset
  archive/
    retired_model_summary.md
  checkpoints/                     # empty
  results/
    validation_results.json
    stratified_smoke/
      theta_true.npy
      mu_weekly.npy
      Y_weekly.npy
      external_foi_dense.npy
      external_foi_times.npy
      npi_intervals.npy
      vaccine_efficacy.npy
      vaccination_change_times.npy
      vaccination_schedule_rates.npy
      populations.npy
      age_populations.npy
      mobility.npy
      mobility_weights.npy
      initial_state.npy
      sigma.npy
      gamma.npy
      beta_prior_log_mean.npy
      npi_prior_logit_mean.npy
      metadata.json
```

Removed old source: model-specific dataset generation, auditing, evaluation,
and scaling scripts (`generate_local_flow_data.py`, `audit_local_flow_data.py`,
`evaluate_local_flow.py`, `run_local_flow_scaling.py`), the SIR proposal module
`local_proposals.py`, and the obsolete partition module `mobility_blocks.py`.
The former SIR equations in `epidemic_model.py` were removed, preserving only
the original geographic input reader. The stratified simulator and tests
were replaced in place.

Removed old generated artifacts: `results/local_flow_pilot/`,
`results/local_flow_scaling/` (including learning curves and figures), old
`results/stratified_smoke/`, `results/local_flow_training.csv`,
`results/local_flow_evaluation.json`, both `local_flow_scaling_pipeline*.log`
files, `results/mobility_k2_partition.csv`, `checkpoints/local_posterior_flow.pt`,
and the four checkpoints in `checkpoints/local_flow_scaling/`.
Only `archive/retired_model_summary.md` preserves the retired identifiability
experiment. Untracked generated artifacts were deleted, not moved to a
recoverable archive; previously committed source remains in Git history.
