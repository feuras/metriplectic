"""
20_protocol_cost_vs_entropy.py

Compare two control protocols between the same pair of densities using the
metriplectic H^{-1}-type cost induced by the KKT operator.

We consider a 1D periodic domain and densities of the form
    rho(x; a) = rho0 * [ 1 + a cos(k x) ],
with |a| < 1 so that rho > 0.

We fix:
  - a_start, a_end (two amplitudes defining rho_0 and rho_1),
and define two protocols a(t) on t ∈ [0, T]:

  1. 'linear':
       a_lin(t) = a_start + (a_end - a_start) * (t / T),

  2. 'wiggle':
       a_wiggle(t) = a_start
                     + (a_end - a_start) * (t / T)
                     + delta_wiggle * sin(2π t / T),

with a small wiggle amplitude that keeps rho positive.

For each protocol we:

  * Discretise time into Nt steps t_n.
  * At each time step n:
      - Build rho_n(x) = rho(x; a_n).
      - Compute Shannon entropy S_n = -∫ rho_n log rho_n dx.
  * For each time interval [t_n, t_{n+1}]:
      - Define mid-point density rho_mid = 0.5 (rho_n + rho_{n+1}).
      - Define v = (rho_{n+1} - rho_n) / dt.
      - Solve L_{rho_mid,G} phi = v with G = 1 using kkt_solve.
      - Compute metric cost
            C_n = ∫ rho_mid |phi_x|^2 dx,
        using phi_x = dx(phi).
  * Approximate:
      - Path "action"      A = ∑ C_n dt,
      - Path "length"      L = ∑ sqrt(max(C_n, 0)) dt,
      - Total entropy change ΔS = S_final - S_initial.

We then compare the two protocols: we expect that the 'wiggle' protocol has
greater action and length for the same ΔS, illustrating a geometric cost
structure consistent with a thermodynamic speed limit picture.

Outputs:
  out/protocol_cost_vs_entropy_meta.csv
  out/protocol_cost_vs_entropy_data.npz
  out/protocol_cost_vs_entropy_plot.png
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import Grid, proj_field, dx, kkt_solve


# =============================================================================
# Configuration
# =============================================================================

# Spatial domain
L = 40.0
N = 4096
rho0_value = 1.0 / L

# Density modulation mode
mode_rho = 3  # k = 2π * mode_rho / L

# Amplitudes for start and end densities
a_start = 0.05
a_end = 0.20

# Time discretisation
T = 1.0
Nt = 200

# Wiggle amplitude (must keep |a(t)| < 1)
delta_wiggle = 0.06

# KKT solver settings
kkt_tol = 1e-10
kkt_maxit = 6000

# Parallelism not required here; time loop is modest and easier sequentially.

# Output directory
outdir = "out"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def build_rho(a, x):
    """
    Build density rho(x; a) = rho0 * [ 1 + a cos(k x) ] with mean rho0.
    Apply mild spectral projection to keep consistency with other scripts.
    """
    rho = rho0_value * (1.0 + a * np.cos(2.0 * np.pi * mode_rho * x / L))
    # Project to dealias and restore mean
    rho = proj_field(rho, mask)
    rho = rho - rho.mean() + rho0_value
    return rho


def shannon_entropy(rho, x):
    """
    Shannon entropy S = -∫ rho log rho dx.
    Use a small floor to avoid log(0).
    """
    eps = 1e-16
    rho_safe = np.maximum(rho, eps)
    integrand = -rho_safe * np.log(rho_safe)
    return np.trapezoid(integrand, x)


def protocol_linear(t):
    """
    Linear protocol for a(t) between a_start and a_end.
    """
    return a_start + (a_end - a_start) * (t / T)


def protocol_wiggle(t):
    """
    Wiggly protocol: linear ramp plus sinusoidal excursion.
    """
    return (a_start
            + (a_end - a_start) * (t / T)
            + delta_wiggle * np.sin(2.0 * np.pi * t / T))


def compute_protocol_metrics(name, a_of_t, x, k, mask):
    """
    For a given protocol a(t), compute:

      - S_n: Shannon entropy at each time step t_n,
      - C_n: metric cost on each time interval [t_n, t_{n+1}],
      - path action A = ∑ C_n dt,
      - path length L = ∑ sqrt(max(C_n,0)) dt,
      - ΔS = S_final - S_initial.

    Returns a dictionary with these and the raw arrays.
    """
    print("------------------------------------------------------------")
    print(f"Evaluating protocol '{name}'")
    t_grid = np.linspace(0.0, T, Nt + 1)
    dt = t_grid[1] - t_grid[0]

    # Build rho at each time step
    a_vals = np.array([a_of_t(t) for t in t_grid])
    rho_list = []
    for n, a in enumerate(a_vals):
        if np.abs(a) >= 0.95:
            print(f"  [warning] |a(t)| close to 1 at step {n}, a = {a:.3f}")
        rho = build_rho(a, x)
        rho_list.append(rho)
    rho_arr = np.stack(rho_list, axis=0)  # shape (Nt+1, N)

    # Entropy at each time
    S_arr = np.array([shannon_entropy(rho_arr[n], x) for n in range(Nt + 1)])
    print(f"  Entropy S(t): min = {S_arr.min():.6f}, max = {S_arr.max():.6f}")

    # Metric cost per interval
    C_arr = np.zeros(Nt, dtype=float)
    kkt_res_arr = np.zeros(Nt, dtype=float)
    kkt_it_arr = np.zeros(Nt, dtype=int)

    print("  Computing metric cost along the path...")
    for n in range(Nt):
        rho_n = rho_arr[n]
        rho_np1 = rho_arr[n + 1]
        rho_mid = 0.5 * (rho_n + rho_np1)

        v = (rho_np1 - rho_n) / dt

        # Solve L_{rho_mid,G} phi = v, with G = 1
        G = np.ones_like(rho_mid)
        phi, it, res = kkt_solve(
            rho_mid, G, v, k, mask, tol=kkt_tol, maxit=kkt_maxit
        )
        phi_x = dx(phi, k, mask)

        # Metric cost C_n = ∫ rho_mid |phi_x|^2 dx
        C_n = np.trapezoid(rho_mid * phi_x * phi_x, x)

        C_arr[n] = C_n
        kkt_res_arr[n] = res
        kkt_it_arr[n] = it

    # Aggregate quantities
    # Enforce positivity for length; small negative noise can appear in C_n.
    C_pos = np.maximum(C_arr, 0.0)
    action = np.sum(C_arr * dt)
    length = np.sum(np.sqrt(C_pos) * dt)
    delta_S = S_arr[-1] - S_arr[0]

    print(f"  KKT residuals: max = {kkt_res_arr.max():.3e}, "
          f"median = {np.median(kkt_res_arr):.3e}")
    print(f"  KKT iterations: max = {kkt_it_arr.max()}, "
          f"median = {int(np.median(kkt_it_arr))}")
    print(f"  Path action A = ∑ C_n dt = {action:.6e}")
    print(f"  Path length L = ∑ sqrt(C_n) dt = {length:.6e}")
    print(f"  ΔS (Shannon) = S(T) - S(0) = {delta_S:.6e}")

    return {
        "name": name,
        "t_grid": t_grid,
        "a_vals": a_vals,
        "rho_arr": rho_arr,
        "S_arr": S_arr,
        "C_arr": C_arr,
        "kkt_res_arr": kkt_res_arr,
        "kkt_it_arr": kkt_it_arr,
        "action": action,
        "length": length,
        "delta_S": delta_S,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("============================================================")
    print("20_protocol_cost_vs_entropy.py")
    print("Comparing metriplectic path cost and entropy change")
    print("------------------------------------------------------------")
    print(f"Spatial grid: L = {L:.3f}, N = {N}")
    print(f"rho0_value = {rho0_value:.6f}, mode_rho = {mode_rho:d}")
    print(f"a_start = {a_start:.3f}, a_end = {a_end:.3f}")
    print(f"T = {T:.3f}, Nt = {Nt}")
    print(f"delta_wiggle = {delta_wiggle:.3f}")
    print(f"KKT tol/maxit = {kkt_tol:.1e}, {kkt_maxit}")
    print("============================================================")

    # Build spatial grid and frequencies
    grid = Grid(L=L, N=N)
    global x, k, mask
    x, k, mask = grid.build()

    # Compute metrics for each protocol
    res_linear = compute_protocol_metrics("linear", protocol_linear, x, k, mask)
    res_wiggle = compute_protocol_metrics("wiggle", protocol_wiggle, x, k, mask)

    # Summarise
    print("============================================================")
    print("Summary comparison")
    for res in (res_linear, res_wiggle):
        print(f"Protocol '{res['name']}':")
        print(f"  action A   = {res['action']:.6e}")
        print(f"  length L   = {res['length']:.6e}")
        print(f"  ΔS         = {res['delta_S']:.6e}")
    # Explicit comparison
    print("------------------------------------------------------------")
    print("Relative comparison (wiggle vs linear):")
    A_lin = res_linear["action"]
    A_wig = res_wiggle["action"]
    L_lin = res_linear["length"]
    L_wig = res_wiggle["length"]
    dS_lin = res_linear["delta_S"]
    dS_wig = res_wiggle["delta_S"]

    print(f"  A_wig / A_lin = {A_wig / A_lin:.6f}")
    print(f"  L_wig / L_lin = {L_wig / L_lin:.6f}")
    print(f"  ΔS_linear     = {dS_lin:.6e}")
    print(f"  ΔS_wiggle     = {dS_wig:.6e}")
    print("------------------------------------------------------------")

    # Save data
    np.savez(
        os.path.join(outdir, "protocol_cost_vs_entropy_data.npz"),
        L=L,
        N=N,
        rho0_value=rho0_value,
        mode_rho=mode_rho,
        a_start=a_start,
        a_end=a_end,
        T=T,
        Nt=Nt,
        delta_wiggle=delta_wiggle,
        t_grid_linear=res_linear["t_grid"],
        a_vals_linear=res_linear["a_vals"],
        S_arr_linear=res_linear["S_arr"],
        C_arr_linear=res_linear["C_arr"],
        kkt_res_linear=res_linear["kkt_res_arr"],
        kkt_it_linear=res_linear["kkt_it_arr"],
        t_grid_wiggle=res_wiggle["t_grid"],
        a_vals_wiggle=res_wiggle["a_vals"],
        S_arr_wiggle=res_wiggle["S_arr"],
        C_arr_wiggle=res_wiggle["C_arr"],
        kkt_res_wiggle=res_wiggle["kkt_res_arr"],
        kkt_it_wiggle=res_wiggle["kkt_it_arr"],
    )

    meta = pd.DataFrame([
        {
            "protocol": "linear",
            "action": res_linear["action"],
            "length": res_linear["length"],
            "delta_S": res_linear["delta_S"],
        },
        {
            "protocol": "wiggle",
            "action": res_wiggle["action"],
            "length": res_wiggle["length"],
            "delta_S": res_wiggle["delta_S"],
        },
    ])
    meta_path = os.path.join(outdir, "protocol_cost_vs_entropy_meta.csv")
    meta.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path}")
    print(f"Wrote {os.path.join(outdir, 'protocol_cost_vs_entropy_data.npz')}")

    # Quick plot: C(t) and S(t) for both protocols
    plt.figure(figsize=(7, 4))
    plt.subplot(2, 1, 1)
    plt.plot(res_linear["t_grid"][:-1], res_linear["C_arr"], label="linear")
    plt.plot(res_wiggle["t_grid"][:-1], res_wiggle["C_arr"], label="wiggle")
    plt.ylabel("C(t) [metric cost]")
    plt.legend(loc="best")
    plt.title("Protocol metric cost and entropy")

    plt.subplot(2, 1, 2)
    plt.plot(res_linear["t_grid"], res_linear["S_arr"], label="linear")
    plt.plot(res_wiggle["t_grid"], res_wiggle["S_arr"], label="wiggle")
    plt.xlabel("t")
    plt.ylabel("S(t) [Shannon]")
    plt.legend(loc="best")

    plt.tight_layout()
    fig_path = os.path.join(outdir, "protocol_cost_vs_entropy_plot.png")
    plt.savefig(fig_path, dpi=180)
    print(f"Wrote {fig_path}")

    t1 = time.time()
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
