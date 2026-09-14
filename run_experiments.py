"""Run the 51-state experiment: python run_experiments.py (plots in outputs/)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import load_data, observe, simulate, weekly_summary


# Experiment defaults: edit here. Rates and FoI have units of 1/day.
N_DAYS = 364
N_DRAWS = 5
GAMMA = 1 / 7
P_A = 0.5
P_REPORT = 0.4
SEED = 20260914
INITIAL_INFECTED = {"CA": 100, "TX": 100, "FL": 100, "NY": 100}
REPRESENTATIVE = ["CA", "NY", "DC", "WY"]


def check_result(result, weekly, Y, N):
    S, I, R = (result[name] for name in ("S", "I", "R"))
    A, B = result["incidence"], result["recoveries"]
    assert np.all(S + I + R == N)
    assert all(np.all(values >= 0) for values in result.values())
    assert all(np.issubdtype(result[k].dtype, np.integer)
               for k in ("S", "I", "R", "incidence", "recoveries"))
    assert np.array_equal(S[:-1] - S[1:], A)
    assert np.array_equal(R[1:] - R[:-1], B)
    assert np.array_equal(I[1:], I[:-1] + A - B)
    assert np.all(A <= S[:-1]) and np.all(B <= I[:-1])
    np.testing.assert_array_equal(
        result["foi"], result["foi_local"] + result["foi_external"]
    )
    np.testing.assert_allclose(
        weekly["foi"], weekly["foi_local"] + weekly["foi_external"], atol=1e-15
    )
    assert np.array_equal(weekly["incidence"].sum(axis=0), A.sum(axis=0))
    assert np.all((Y >= 0) & (Y <= weekly["incidence"]))


def main():
    root = Path(__file__).resolve().parent
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    labels, subpop, N, M = load_data(root)
    assert len(N) == 51 and len(set(subpop)) == 51
    assert np.all(N > 0) and np.all(M >= 0) and np.all(M.diagonal() == 0)
    outgoing = M.sum(axis=1) / N
    print(f"States: {len(N)}; directed nonzero edges: {np.count_nonzero(M):,}")
    print(f"Population: {N.sum():,}; daily interstate commuters: {M.sum():,}")
    quantiles = np.quantile(outgoing, [0, .25, .5, .75, 1])
    print("Outgoing commuters / population (not normalized):")
    print("  min / Q1 / median / Q3 / max: "
          + " / ".join(f"{v:.4%}" for v in quantiles))
    print(f"  mean: {outgoing.mean():.4%}; minimum local weight: "
          f"{(1 - P_A * outgoing).min():.6f}")
    index = {label: i for i, label in enumerate(labels)}
    I0 = np.zeros(len(N), dtype=np.int64)
    for state, count in INITIAL_INFECTED.items():
        I0[index[state]] = count
    print(f"Initial infections: {INITIAL_INFECTED}; gamma={GAMMA:.6f}; p_a={P_A}")
    for state in REPRESENTATIVE:
        i = index[state]
        print(f"  {state}: N={N[i]:,}, outgoing/N={outgoing[i]:.4%}")

    # Separate streams keep reporting and prior sampling out of epidemic RNGs.
    prior_rng = np.random.default_rng(SEED)
    theta = np.vstack((np.full(len(N), 0.30),
                       np.exp(prior_rng.normal(np.log(.30), .20, (N_DRAWS, len(N))))))
    run_labels = ["Baseline"] + [f"Draw {k}" for k in range(1, N_DRAWS + 1)]
    simulation_seeds = SEED + 100 + np.arange(len(theta))
    observation_seeds = SEED + 200 + np.arange(len(theta))
    daily_runs, weekly_runs, observations = [], [], []
    for k, beta in enumerate(theta):
        result = simulate(beta, N_DAYS, np.random.default_rng(simulation_seeds[k]),
                          N=N, M=M, I0=I0, gamma=GAMMA, p_a=P_A)
        weekly = weekly_summary(result)
        Y = observe(weekly["incidence"], P_REPORT,
                    np.random.default_rng(observation_seeds[k]))
        check_result(result, weekly, Y, N)
        daily_runs.append(result)
        weekly_runs.append(weekly)
        observations.append(Y)
        national = weekly["incidence"].sum(axis=1)
        print(f"{run_labels[k]:8s}: new infections={national.sum():,}; "
              f"peak week={national.argmax() + 1}; peak/week={national.max():,}; "
              f"final infectious={result['I'][-1].sum():,}")

    # Preserve labels and all arrays; run 0 is the constant-beta baseline.
    arrays = {name: np.stack([r[name] for r in daily_runs]) for name in daily_runs[0]}
    arrays.update({f"weekly_{name}": np.stack([r[name] for r in weekly_runs])
                   for name in weekly_runs[0]})
    np.savez_compressed(output / "simulations.npz", **arrays, theta=theta,
                        Y=np.stack(observations), labels=labels, subpop=subpop,
                        N=N, M=M, I0=I0, gamma=GAMMA, p_a=P_A, p_report=P_REPORT,
                        prior_seed=SEED, simulation_seeds=simulation_seeds,
                        observation_seeds=observation_seeds, run_labels=run_labels)
    print(f"Checks passed for all {len(theta)} runs: conservation, nonnegative integer counts,")
    print("  daily transitions, event bounds, FoI decomposition, weekly totals, reporting.")
    print(f"Saved S/I/R: {arrays['S'].shape}; daily FoI: {arrays['foi'].shape}; "
          f"weekly FoI/Y: {arrays['weekly_foi'].shape} (run, time, state).")

    weeks = np.arange(1, N_DAYS // 7 + 1)
    colors = ["#222222", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    def curves(ax, name, state=None, scale=1):
        for k, weekly in enumerate(weekly_runs):
            values = weekly[name].sum(axis=1) if state is None else weekly[name][:, index[state]]
            ax.plot(weeks, values / scale, color=colors[k % len(colors)], label=run_labels[k],
                    lw=2 if k == 0 else 1.3, ls="--" if k == 0 else "-")
        ax.set(xlim=(1, len(weeks)), ylim=(0, None), xlabel="Week")
        ax.grid(alpha=.18)

    def save(fig, filename, title):
        handles, names = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .955),
                   ncol=6, frameon=False)
        fig.suptitle(title, y=.995, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .915))
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
        print(f"Saved {output / filename}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, state in zip(axes.flat, REPRESENTATIVE):
        curves(ax, "foi", state)
        ax.set(title=state, ylabel="Weekly mean FoI (per day)")
    save(fig, "weekly_foi.png", "Weekly force of infection across transmission vectors")

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for row, state in enumerate(REPRESENTATIVE):
        for col, component in enumerate(("local", "external")):
            curves(axes[row, col], f"foi_{component}", state)
            axes[row, col].set(title=f"{state} | {component}", ylabel="Weekly mean FoI (per day)")
    save(fig, "foi_components.png", "Local and external force of infection (separate vertical scales)")

    fig = plt.figure(figsize=(11, 10))
    grid = fig.add_gridspec(3, 2)
    national_ax = fig.add_subplot(grid[0, :])
    curves(national_ax, "incidence", scale=1e6)
    national_ax.set(title="United States", ylabel="New infections / week (millions)")
    for k, state in enumerate(REPRESENTATIVE):
        ax = fig.add_subplot(grid[1 + k // 2, k % 2])
        curves(ax, "incidence", state, scale=1e3)
        ax.set(title=state, ylabel="New infections / week (thousands)")
    save(fig, "weekly_incidence.png", "National and state incidence across transmission vectors")


if __name__ == "__main__":
    main()
