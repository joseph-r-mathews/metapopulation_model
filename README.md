# Mobility-block epidemic inference

The active workflow is:

```text
continuous coupled epidemic model
    -> fixed mobility-only K=2 blocks
    -> block MCMC
```

The authoritative model is the continuous deterministic coupled 51-location
SIR ODE in `epidemic_model.py`. Its instantaneous force of infection is

```text
lambda_i(t) = a_i beta_i I_i(t)/N_i + h_i(t),
a_i = 1 - (p_a/N_i) sum_{j != i} M_ij,
h_i(t) = sum_{j != i} (p_a M_ij/N_i) beta_j I_j(t)/N_j.
```

Weekly incidence is integrated as an ODE state,
`mu[i,w] = C_i(w+1)-C_i(w)`, and observations follow
`Y[i,w] ~ Poisson(p_report * mu[i,w])`.

`local_proposals.py` contains the likelihood, prior, coupled log posterior,
and one-dimensional frozen-external-FoI proposal routines used by block MCMC.
Frozen FoI is only a proposal construction device: every MCMC target
evaluation must use the full coupled continuous-time model.

## Fixed mobility partition

The partition was obtained by minimizing within-block symmetric mobility

```text
C_ij = M_ij/N_i + M_ji/N_j
```

as a weighted Max-2-Cut. The solution was certified optimal and is stored in
`mobility_blocks.py`; normal inference loads it directly and never reruns the
optimization.

- Block 1 (24): WY, AK, ND, RI, NH, NM, UT, IA, OK, LA, AL, SC, WI, MD, MO,
  IN, TN, WA, VA, OH, PA, NY, FL, CA
- Block 2 (27): VT, DC, SD, DE, MT, ME, HI, ID, WV, NE, KS, NV, MS, AR, CT,
  OR, KY, MN, CO, MA, AZ, NJ, MI, NC, GA, IL, TX

The retained numerical record is
`results/mobility_k2_partition.csv`.
