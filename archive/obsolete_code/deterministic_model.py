"""Deterministic coupled and fixed-external-FoI SIR integrations."""

import numpy as np
from scipy.integrate import solve_ivp

from model import GAMMA, N_WEEKS, P_A, force_of_infection


RTOL = 1e-9
ATOL = 1e-5
WEEKLY_TIMES = np.arange(N_WEEKS + 1, dtype=np.float64)


def solve_coupled(beta, *, N, M, I0, gamma=GAMMA, p_a=P_A,
                  rtol=RTOL, atol=ATOL, dense_output=True):
    """Solve the 51-state ODE with cumulative incidence and external FoI."""
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), (51,))
    zeros = np.zeros(51)
    y0 = np.concatenate((N - I0, I0, zeros, zeros, zeros))

    def rhs(time, state):
        S, I = state[:51], state[51:102]
        total, _, external = force_of_infection(I, beta, N, M, p_a)
        infections = S * total
        recoveries = gamma * I
        return np.concatenate((-infections, infections - recoveries,
                               recoveries, infections, external))

    solution = solve_ivp(rhs, (0., float(N_WEEKS)), y0, method="DOP853",
                         t_eval=WEEKLY_TIMES, dense_output=dense_output,
                         rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"Coupled ODE failed: {solution.message}")
    S, I, R, cumulative, external_integral = np.split(solution.y, 5)
    result = dict(S=S, I=I, R=R,
                  cumulative_incidence=cumulative,
                  cumulative_external_foi=external_integral,
                  incidence=np.diff(cumulative, axis=1),
                  external_foi=np.diff(external_integral, axis=1),
                  solution=solution)
    if np.min(S) < -1e-4 or np.min(I) < -1e-4 or np.min(R) < -1e-4:
        raise FloatingPointError("Coupled ODE produced materially negative states")
    np.testing.assert_allclose(S + I + R, np.broadcast_to(N[:, None], S.shape),
                               rtol=2e-9, atol=2e-3)
    np.testing.assert_allclose(result["incidence"], S[:, :-1] - S[:, 1:],
                               rtol=2e-8, atol=2e-3)
    return result


def continuous_external(solution, beta, N, M, p_a=P_A):
    """Return a callable evaluating exact coupled external FoI from dense output."""
    def evaluate(time):
        state = solution.sol(time)
        I = state[51:102] if state.ndim == 1 else state[51:102, :]
        if I.ndim == 1:
            return force_of_infection(I, beta, N, M, p_a)[2]
        return np.column_stack([
            force_of_infection(I[:, column], beta, N, M, p_a)[2]
            for column in range(I.shape[1])
        ])
    return evaluate


def weekly_hold(curve):
    """Return the shared left-constant interpolation for a 52-value FoI curve."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (N_WEEKS,):
        raise ValueError("A fixed weekly FoI curve must have shape [52]")

    def evaluate(time):
        index = np.minimum(np.floor(np.asarray(time)).astype(int), N_WEEKS - 1)
        index = np.maximum(index, 0)
        return curve[index]
    return evaluate


def solve_local(beta, state_index, external, *, N, I0, gamma=GAMMA,
                rtol=RTOL, atol=ATOL):
    """Solve one deterministic state with a fixed external-FoI callable."""
    local_weight = 1.0
    if hasattr(external, "local_weight"):
        local_weight = external.local_weight
    y0 = np.array([N[state_index] - I0[state_index], I0[state_index], 0., 0.])

    def rhs(time, state):
        S, I = state[:2]
        rate = local_weight * beta * I / N[state_index] + float(external(time))
        infections = S * rate
        recoveries = gamma * I
        return (-infections, infections - recoveries, recoveries, infections)

    solution = solve_ivp(rhs, (0., float(N_WEEKS)), y0, method="DOP853",
                         t_eval=WEEKLY_TIMES, rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"Local ODE failed for state {state_index}: {solution.message}")
    S, I, R, cumulative = solution.y
    return dict(S=S, I=I, R=R, incidence=np.diff(cumulative))


class FixedExternal:
    """Callable external FoI carrying the state's local mobility weight."""

    def __init__(self, evaluate, local_weight):
        self.evaluate = evaluate
        self.local_weight = float(local_weight)

    def __call__(self, time):
        return self.evaluate(time)


def continuous_state_external(coupled, beta, state_index, *, N, M, p_a=P_A):
    vector = continuous_external(coupled["solution"], beta, N, M, p_a)
    weight = 1 - p_a * M[state_index].sum() / N[state_index]
    return FixedExternal(lambda time: vector(time)[state_index], weight)


def weekly_state_external(curve, state_index, *, N, M, p_a=P_A):
    weight = 1 - p_a * M[state_index].sum() / N[state_index]
    return FixedExternal(weekly_hold(curve), weight)
