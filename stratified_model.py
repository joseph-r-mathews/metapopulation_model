"""Age x vaccination x variant SEIR equations and centralized toy defaults.

Time is in weeks. Transmission matrices use (source age, recipient age).
Defaults are methodological examples, not calibrated scientific estimates.
"""

import numpy as np
from scipy.special import expit, gammaln

N_LOCATIONS = 51
N_AGES = 3
N_VACCINATION_GROUPS = 2
N_VARIANTS = 2
N_NPIS = 8
N_BETA = N_VARIANTS * N_AGES * N_AGES
PARAMETERS_PER_LOCATION = N_BETA + N_NPIS
PARAMETER_DIMENSION = N_LOCATIONS * PARAMETERS_PER_LOCATION
N_STRATA = N_LOCATIONS * N_AGES * N_VACCINATION_GROUPS
STATE_NAMES = ("S", "E_variant1", "E_variant2", "I_variant1",
               "I_variant2", "R", "C_variant1", "C_variant2")
N_STATE_COMPONENTS = len(STATE_NAMES)
N_EPIDEMIC_COMPONENTS = 6
N_ODE_STATES = N_STRATA * N_STATE_COMPONENTS
LOCAL_ODE_STATES = N_AGES * N_VACCINATION_GROUPS * N_STATE_COMPONENTS
INCIDENCE_CHANNELS = N_AGES * N_VACCINATION_GROUPS * N_VARIANTS
FOI_CHANNELS = N_AGES * N_VARIANTS
S_INDEX, R_INDEX = 0, 5
E_SLICE, I_SLICE, C_SLICE = slice(1, 3), slice(3, 5), slice(6, 8)

HORIZON = 52
INTEGRATION_METHOD = "DOP853"
INTEGRATION_RTOL = 1e-9
INTEGRATION_ATOL = 1e-5
MOBILITY_SCALE = 0.5
AGE_PROPORTIONS = np.array([0.22, 0.50, 0.28])
VACCINATION_PROPORTIONS = np.array([0.70, 0.30])
VACCINE_EFFICACY = np.array([0.70, 0.45])
SIGMA = np.array([7.0 / 3.0, 7.0 / 3.0])  # three-day latent periods
GAMMA = np.array([1.0, 1.0])  # one-week infectious periods
INITIAL_INFECTED = {"CA": (60., 40.), "TX": (60., 40.),
                    "FL": (60., 40.), "NY": (60., 40.)}

# (variant, source age, recipient age); positive matrices define log means.
# No additional age-contact matrix multiplies beta.
BETA_PRIOR_BASELINE = np.array([
    [[1.50, 0.35, 0.15], [0.30, 1.40, 0.30], [0.15, 0.40, 1.45]],
    [[1.65, 0.40, 0.20], [0.35, 1.55, 0.35], [0.20, 0.45, 1.60]],
])
BETA_PRIOR_LOG_MEAN = np.log(BETA_PRIOR_BASELINE)
BETA_PRIOR_LOG_SD = 0.20
NPI_PRIOR_LOGIT_MEAN = np.full(N_NPIS, np.log(0.15 / 0.85))
NPI_PRIOR_LOGIT_SD = 0.35

# Closed known intervals, identical across locations by default. Custom
# (51, 8, 2) arrays may differ by location. Overlap is intentional.
NPI_INTERVALS = np.broadcast_to(np.array([
    [4., 10.], [6., 14.], [12., 20.], [18., 26.],
    [24., 32.], [30., 38.], [36., 46.], [42., 50.],
]), (N_LOCATIONS, N_NPIS, 2)).copy()

# Known rates in week^-1: [0,8) none; [8,36) rollout; [36,52] none.
VACCINATION_CHANGE_TIMES = np.array([8., 36.])
VACCINATION_SCHEDULE_RATES = np.broadcast_to(np.array([
    [0., 0., 0.], [0.01, 0.015, 0.02], [0., 0., 0.],
])[:, None, :], (3, N_LOCATIONS, N_AGES)).copy()


def vaccination_rate(time):
    return VACCINATION_SCHEDULE_RATES[
        np.searchsorted(VACCINATION_CHANGE_TIMES, time, side="right")
    ]


def checked_array(name, values, shape):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return values


def pack_parameters(log_beta, npi_logits):
    """Location-major [log beta(k,b,a) row-major, NPI logits(ell)]."""
    log_beta = checked_array("log_beta", log_beta, (N_LOCATIONS, 2, 3, 3))
    npi_logits = checked_array("npi_logits", npi_logits, (N_LOCATIONS, N_NPIS))
    return np.concatenate((log_beta.reshape(N_LOCATIONS, N_BETA), npi_logits), axis=1).ravel()


def unpack_parameters(theta):
    packed = checked_array("theta", theta, (PARAMETER_DIMENSION,)).reshape(N_LOCATIONS, PARAMETERS_PER_LOCATION)
    return packed[:, :N_BETA].reshape(N_LOCATIONS, 2, 3, 3).copy(), packed[:, N_BETA:].copy()


def _transform(log_beta, logits):
    with np.errstate(over="ignore", under="ignore"):
        beta = np.exp(log_beta)
    effects = expit(logits)
    if not np.all(np.isfinite(beta) & (beta > 0)) or not np.all((effects > 0) & (effects < 1)):
        raise ValueError("Parameters exceed representable positive beta/open-unit NPI range")
    return beta, effects


def transform_parameters(theta):
    return _transform(*unpack_parameters(theta))


def extract_local_parameters(theta, location):
    if not 0 <= location < N_LOCATIONS:
        raise IndexError("location is out of range")
    return checked_array("theta", theta, (PARAMETER_DIMENSION,)).reshape(N_LOCATIONS, PARAMETERS_PER_LOCATION)[location].copy()


def transform_local_parameters(theta):
    theta = checked_array("theta_local", theta, (PARAMETERS_PER_LOCATION,))
    return _transform(theta[:N_BETA].reshape(2, 3, 3), theta[N_BETA:])


def parameter_index(location, kind, *, variant=None, source_age=None, recipient_age=None, npi=None):
    if not 0 <= location < N_LOCATIONS:
        raise IndexError("location is out of range")
    if kind == "log_beta":
        if variant not in range(N_VARIANTS) or source_age not in range(N_AGES) or recipient_age not in range(N_AGES):
            raise IndexError("invalid variant/source/recipient index")
        offset = variant * 9 + source_age * 3 + recipient_age
    elif kind == "npi_logit" and npi in range(N_NPIS):
        offset = N_BETA + npi
    else:
        raise ValueError("kind must be log_beta or npi_logit with valid indices")
    return location * PARAMETERS_PER_LOCATION + offset


def sample_prior(rng, *, local=False):
    size = () if local else (N_LOCATIONS,)
    beta = rng.normal(BETA_PRIOR_LOG_MEAN, BETA_PRIOR_LOG_SD, size=size + (2, 3, 3))
    logits = rng.normal(NPI_PRIOR_LOGIT_MEAN, NPI_PRIOR_LOGIT_SD, size=size + (N_NPIS,))
    return np.concatenate((beta.ravel(), logits)) if local else pack_parameters(beta, logits)


def log_prior(theta):
    beta, logits = unpack_parameters(theta)
    return float(sum(np.sum(-0.5 * ((x - mean) / sd)**2 - np.log(sd * np.sqrt(2 * np.pi)))
                     for x, mean, sd in ((beta, BETA_PRIOR_LOG_MEAN, BETA_PRIOR_LOG_SD),
                                         (logits, NPI_PRIOR_LOGIT_MEAN, NPI_PRIOR_LOGIT_SD))))


def mobility_weights(populations, mobility):
    populations = checked_array("populations", populations, (N_LOCATIONS,))
    mobility = checked_array("mobility", mobility, (N_LOCATIONS, N_LOCATIONS))
    if np.any(populations <= 0) or np.any(mobility < 0):
        raise ValueError("Population must be positive and mobility nonnegative")
    q = MOBILITY_SCALE * mobility / populations[:, None]
    np.fill_diagonal(q, 0.)
    diagonal = 1. - q.sum(axis=1)
    if np.any(diagonal < 0.):
        raise ValueError("Mobility implies negative q_ii; do not renormalize M")
    np.fill_diagonal(q, diagonal)
    return q


def npi_multiplier(time, effects, intervals):
    """Closed intervals, including both endpoints; zero effects are allowed."""
    active = (time >= intervals[..., 0]) & (time <= intervals[..., 1])
    return np.prod(1. - effects * active, axis=-1)


def unpack_state(state, n_locations=N_LOCATIONS):
    """State axes (location, age, vaccination, component[, time])."""
    state = np.asarray(state, dtype=np.float64)
    return state.reshape((n_locations, N_AGES, N_VACCINATION_GROUPS, N_STATE_COMPONENTS) + state.shape[1:])


def source_pressure(state, beta, multiplier, age_populations):
    infectious = state[..., I_SLICE].sum(axis=2)
    prevalence = infectious / age_populations[..., None]
    return np.einsum("ikba,ibk->iak", beta, prevalence) * multiplier[:, None, None]


def force_of_infection(time, state, beta, effects, fixed):
    """Return lambda(i,a,v,k), local(i,a,k), external(i,a,k)."""
    source = source_pressure(state, beta, npi_multiplier(time, effects, fixed["npi_intervals"]), fixed["age_populations"])
    local = np.diag(fixed["q"])[:, None, None] * source
    external = np.einsum("ij,jak->iak", fixed["q_external"], source)
    protection = np.stack((np.ones(N_VARIANTS), 1. - fixed["vaccine_efficacy"]))
    return (local + external)[:, :, None, :] * protection[None, None, :, :], local, external


def compartment_derivative(state, infection_rate, nu, sigma=SIGMA, gamma=GAMMA):
    """Shared coupled/local equations; no vaccination of E or I."""
    infections = state[..., S_INDEX, None] * infection_rate
    exposed, infectious = state[..., E_SLICE], state[..., I_SLICE]
    derivative = np.zeros_like(state)
    derivative[..., S_INDEX] = -infections.sum(axis=-1)
    derivative[..., E_SLICE] = infections - sigma * exposed
    derivative[..., I_SLICE] = sigma * exposed - gamma * infectious
    derivative[..., R_INDEX] = (gamma * infectious).sum(axis=-1)
    derivative[..., C_SLICE] = infections
    for component in (S_INDEX, R_INDEX):
        transfer = nu * state[:, :, 0, component]
        derivative[:, :, 0, component] -= transfer
        derivative[:, :, 1, component] += transfer
    return derivative


def epidemic_rhs(time, flat_state, beta, effects, fixed):
    state = unpack_state(flat_state)
    infection_rate, _, _ = force_of_infection(time, state, beta, effects, fixed)
    nu = checked_array("vaccination_rate(t)", fixed["vaccination_rate"](time), (N_LOCATIONS, N_AGES))
    if np.any(nu < 0):
        raise ValueError("Vaccination rates must be nonnegative")
    return compartment_derivative(state, infection_rate, nu, fixed["sigma"], fixed["gamma"]).ravel()


def weekly_incidence(cumulative, tolerance=INTEGRATION_ATOL):
    incidence = np.diff(cumulative, axis=-1)
    if not np.all(np.isfinite(incidence)) or np.min(incidence) < -tolerance:
        raise FloatingPointError("Invalid or decreasing cumulative incidence")
    return np.maximum(incidence, 0.)


def poisson_log_likelihood(observed, expected, omit_constant=False):
    observed, expected = np.asarray(observed), np.asarray(expected, dtype=float)
    if observed.shape != expected.shape or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(expected)):
        raise ValueError("Counts and means must be finite and have matching shapes")
    if np.any(observed < 0) or np.any(observed != np.floor(observed)) or np.any(expected < 0):
        raise ValueError("Counts must be nonnegative integers and means nonnegative")
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(observed > 0, observed * np.log(expected), 0.) - expected
    if not omit_constant:
        terms -= gammaln(observed + 1)
    return float(terms.sum())
