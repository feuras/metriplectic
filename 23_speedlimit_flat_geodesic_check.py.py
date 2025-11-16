"""
23_speedlimit_flat_demo.py

Two-mode H^{-1}_{rho}(G) speed–limit check in a nearly flat patch.

We restrict to a 1D periodic domain and a two-mode density family
    rho(x; a1, a2) = rho0 * [ 1 + a1 cos(k1 x) + a2 cos(2 k1 x) ],
with modest amplitudes so that rho stays positive after projection.

We fix start and end points in (a1, a2) space and compare four protocols:

  1. linear:
       straight line in (a1, a2) at constant speed.

  2. wait_then_jump:
       stay at the initial density until t = T/2, then ramp to the final
       density over the remaining half-interval.

  3. jump_then_wait:
       ramp quickly to the final density over [0, T/2], then sit there.

  4. overshoot_then_return:
       ramp past the target in (a1, a2), then return to it by time T.

For each protocol we compute the H^{-1}_{rho}(G) cost along the path via
KKT solves and the Shannon entropy change ΔS.

In a nearly flat patch the true geodesic in the induced H^{-1}_{rho}(G)
metric is expected to coincide with the straight-line protocol, and any
“compressed” or “wiggly” protocol should have strictly larger action A.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import Grid, proj_field, dx, kkt_solve


# =============================================================================
# Configuration (quick but rigorous run)
# =============================================================================

L = 40.0
N = 1024
rho0_value = 1.0 / L

# Base mode wavenumber (k1 = 2π*mode1/L, k2 = 2*k1)
mode1 = 3

# Start and end in (a1, a2): different corners but modest amplitudes
a1_start, a2_start = 0.20, 0.00
a1_end,   a2_end   = 0.00, 0.28

# Time discretisation
T = 1.0
Nt = 160

# KKT solver settings (tight tolerance, reasonable iteration cap)
kkt_tol = 1e-10
kkt_maxit = 4000

# Amplitude clipping to keep rho safely positive after projection
a_clip = 0.85

# Output directory
outdir = "out"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Grid and globals
# =============================================================================

grid = Grid(L=L, N=N)
x, k, mask = grid.build()

k1 = 2.0 * np.pi * mode1 / L
k2 = 2.0 * k1


# =============================================================================
# Helpers
# =============================================================================

def build_rho(a1, a2, x):
    """
    rho(x; a1, a2) = rho0 * [1 + a1 cos(k1 x) + a2 cos(2 k1 x)],
    then project and renormalise the mean.
    """
    rho = rho0_value * (
        1.0
        + a1 * np.cos(k1 * x)
        + a2 * np.cos(k2 * x)
    )
    rho = proj_field(rho, mask)
    rho = rho - rho.mean() + rho0_value
    return rho


def shannon_entropy(rho, x):
    """Shannon entropy S = -∫ rho log rho dx with a small floor."""
    eps = 1e-16
    rho_safe = np.maximum(rho, eps)
    integrand = -rho_safe * np.log(rho_safe)
    return np.trapezoid(integrand, x)


# =============================================================================
# Protocols
# =============================================================================

def protocol_linear(t):
    """Baseline linear protocol in (a1, a2) at constant speed."""
    s = t / T
    a1 = a1_start + (a1_end - a1_start) * s
    a2 = a2_start + (a2_end - a2_start) * s
    return a1, a2


def protocol_wait_then_jump(t):
    """
    Sit at the start density for t < T/2, then ramp to the end over [T/2, T].
    """
    if t < 0.5 * T:
        return a1_start, a2_start
    s = (t - 0.5 * T) / (0.5 * T)
    a1 = a1_start + (a1_end - a1_start) * s
    a2 = a2_start + (a2_end - a2_start) * s
    a1 = float(np.clip(a1, -a_clip, a_clip))
    a2 = float(np.clip(a2, -a_clip, a_clip))
    return a1, a2


def protocol_jump_then_wait(t):
    """
    Ramp quickly to the end density over [0, T/2], then sit there.
    """
    if t < 0.5 * T:
        s = t / (0.5 * T)
        a1 = a1_start + (a1_end - a1_start) * s
        a2 = a2_start + (a2_end - a2_start) * s
    else:
        a1, a2 = a1_end, a2_end
    a1 = float(np.clip(a1, -a_clip, a_clip))
    a2 = float(np.clip(a2, -a_clip, a_clip))
    return a1, a2


def protocol_overshoot_then_return(t):
    """
    Overshoot the target in (a1, a2), then return to it by t = T.

    We choose an overshoot factor > 1 and interpolate via a midpoint
    (a1_mid, a2_mid) before coming back to (a1_end, a2_end).
    """
    overshoot_factor = 1.4
    a1_mid = overshoot_factor * a1_end
    a2_mid = overshoot_factor * a2_end

    t_half = 0.5 * T
    if t < t_half:
        s = t / t_half
        a1 = a1_start + (a1_mid - a1_start) * s
        a2 = a2_start + (a2_mid - a2_start) * s
    else:
        s = (t - t_half) / t_half
        a1 = a1_mid + (a1_end - a1_mid) * s
        a2 = a2_mid + (a2_end - a2_mid) * s

    a1 = float(np.clip(a1, -a_clip, a_clip))
    a2 = float(np.clip(a2, -a_clip, a_clip))
    return a1, a2


# =============================================================================
# Core metric computation
# =============================================================================

def compute_protocol_metrics(name, a_of_t, x, k, mask):
    """
    For a given protocol (a1(t), a2(t)), compute:

      - S(t_n): Shannon entropy at Nt+1 time points,
      - C_n: H^{-1}_{rho}(G) metric cost on each interval [t_n, t_{n+1}],
      - action A = ∑ C_n dt,
      - length L = ∑ sqrt(max(C_n, 0)) dt,
      - ΔS = S(T) - S(0),

    along with KKT residual and iteration diagnostics.

    Returns (meta, data_dict).
    """
    print(f"Evaluating protocol '{name}'...")
    t_grid = np.linspace(0.0, T, Nt + 1)
    dt = t_grid[1] - t_grid[0]

    # Build rho at each time step
    a1_vals = np.empty(Nt + 1, dtype=float)
    a2_vals = np.empty(Nt + 1, dtype=float)
    rho_list = []

    for i, t in enumerate(t_grid):
        a1, a2 = a_of_t(t)
        a1_vals[i] = a1
        a2_vals[i] = a2
        rho = build_rho(a1, a2, x)
        rho_list.append(rho)

    rho_arr = np.stack(rho_list, axis=0)  # shape (Nt+1, N)

    # Shannon entropy at each time
    S_arr = np.array([shannon_entropy(rho_arr[n], x) for n in range(Nt + 1)])

    # Fixed non-uniform mobility G(x) (same for all protocols)
    G = 1.0 + 0.3 * np.cos(2.0 * np.pi * x / L)
    G = proj_field(G, mask)

    # Metric cost per interval
    C_arr = np.zeros(Nt, dtype=float)
    kkt_res_arr = np.zeros(Nt, dtype=float)
    kkt_it_arr = np.zeros(Nt, dtype=int)

    for n in range(Nt):
        rho_n = rho_arr[n]
        rho_np1 = rho_arr[n + 1]
        rho_mid = 0.5 * (rho_n + rho_np1)
        v = (rho_np1 - rho_n) / dt

        phi, it, res = kkt_solve(
            rho_mid, G, v, k, mask, tol=kkt_tol, maxit=kkt_maxit
        )
        phi_x = dx(phi, k, mask)

        # H^{-1}_{rho}(G) cost: ∫ rho_mid G |∂x φ|^2 dx
        C_n = np.trapezoid(rho_mid * G * phi_x * phi_x, x)

        C_arr[n] = C_n
        kkt_res_arr[n] = res
        kkt_it_arr[n] = it

    C_pos = np.maximum(C_arr, 0.0)
    action = float(np.sum(C_arr * dt))
    length = float(np.sum(np.sqrt(C_pos) * dt))
    delta_S = float(S_arr[-1] - S_arr[0])

    meta = {
        "name": name,
        "action": action,
        "length": length,
        "delta_S": delta_S,
        "max_residual": float(kkt_res_arr.max()),
        "median_residual": float(np.median(kkt_res_arr)),
        "max_iters": int(kkt_it_arr.max()),
        "median_iters": int(np.median(kkt_it_arr)),
    }
    data = {
        "t_grid": t_grid,
        "a1_vals": a1_vals,
        "a2_vals": a2_vals,
        "S_arr": S_arr,
        "C_arr": C_arr,
        "kkt_res_arr": kkt_res_arr,
        "kkt_it_arr": kkt_it_arr,
    }
    print(f"  A = {action:.6e}, L = {length:.6e}, ΔS = {delta_S:.6e}")
    print(f"  KKT: max_res = {meta['max_residual']:.3e}, "
          f"med_res = {meta['median_residual']:.3e}, "
          f"max_it = {meta['max_iters']}, "
          f"med_it = {meta['median_iters']}")
    print("-" * 60)
    return meta, data


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("============================================================")
    print("23_speedlimit_flat_demo.py")
    print("Two-mode H^{-1}_{rho}(G) speed–limit check (flat patch)")
    print("------------------------------------------------------------")
    print(f"L = {L:.3f}, N = {N}, Nt = {Nt}")
    print(f"rho0_value = {rho0_value:.6f}, mode1 = {mode1}")
    print(f"(a1_start, a2_start) = ({a1_start:.4f}, {a2_start:.4f})")
    print(f"(a1_end,   a2_end)   = ({a1_end:.4f},   {a2_end:.4f})")
    print(f"T = {T:.3f}")
    print(f"a_clip = {a_clip:.3f}")
    print(f"KKT tol/maxit = {kkt_tol:.1e}, {kkt_maxit}")
    print("============================================================")

    # Evaluate all four protocols
    meta_linear, data_linear = compute_protocol_metrics("linear", protocol_linear, x, k, mask)
    meta_wait,   data_wait   = compute_protocol_metrics("wait_then_jump", protocol_wait_then_jump, x, k, mask)
    meta_jump,   data_jump   = compute_protocol_metrics("jump_then_wait", protocol_jump_then_wait, x, k, mask)
    meta_over,   data_over   = compute_protocol_metrics("overshoot_then_return", protocol_overshoot_then_return, x, k, mask)

    # Summary table
    metas = [meta_linear, meta_wait, meta_jump, meta_over]
    df_meta = pd.DataFrame(metas)
    meta_path = os.path.join(outdir, "protocol_speedlimit_flat_meta.csv")
    df_meta.to_csv(meta_path, index=False)

    print("Summary (actions and entropy changes):")
    for m in metas:
        print(f"{m['name']:>24}: "
              f"A = {m['action']:.6e}, "
              f"L = {m['length']:.6e}, "
              f"ΔS = {m['delta_S']:.6e}")
    print(f"Saved {meta_path}")

    # Simple figure: C(t) for all protocols
    plt.figure(figsize=(7, 4))
    plt.plot(data_linear["t_grid"][:-1], data_linear["C_arr"], label="linear")
    plt.plot(data_wait["t_grid"][:-1],   data_wait["C_arr"],   label="wait_then_jump")
    plt.plot(data_jump["t_grid"][:-1],   data_jump["C_arr"],   label="jump_then_wait")
    plt.plot(data_over["t_grid"][:-1],   data_over["C_arr"],   label="overshoot_then_return")
    plt.xlabel("t")
    plt.ylabel("C(t) [H^{-1}_{rho}(G) cost]")
    plt.title("Protocol metric cost C(t)")
    plt.legend(loc="best")
    plt.tight_layout()
    fig_path = os.path.join(outdir, "protocol_speedlimit_flat_Ct.png")
    plt.savefig(fig_path, dpi=180)
    print(f"Saved {fig_path}")

    t1 = time.time()
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
