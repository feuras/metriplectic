"""
24_speed_limit_curved.py

Two-mode H^{-1}_{rho}(G(rho)) speed–limit optimisation in a curved patch.

We restrict to a 1D periodic domain and a two-mode density family
    rho(x; a1, a2) = rho0 * [1 + a1 cos(k1 x) + a2 cos(2 k1 x)],
with amplitudes chosen so that rho stays positive after projection.

We fix start and end points in (a1, a2) space and compare:

  - a "linear" protocol: straight line in (a1, a2) at constant speed,
  - an optimised protocol in a low-dimensional sine basis that keeps
    endpoints fixed but can bend in time.

Key difference from the flat demo:
  The mobility is now density-dependent,
      G(rho) = exp( gamma_G * (rho / rho0 - 1) ),
  which induces a genuinely position-dependent H^{-1}_{rho}(G(rho)) metric
  on the (a1, a2) manifold. In such a curved patch, the linear protocol
  need not be geodesic, and a bent protocol can have strictly lower
  action A for the same endpoints and duration T.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import Grid, proj_field, dx, kkt_solve


# =============================================================================
# Configuration (curved metric; quick but rigorous)
# =============================================================================

L = 40.0
N = 1024
rho0_value = 1.0 / L

# Base mode wavenumber (k1 = 2π*mode1/L, k2 = 2*k1)
mode1 = 3

# Start and end in (a1, a2): larger amplitudes than flat case,
# but still with |a1| + |a2| < 0.9 to keep rho positive.
a1_start, a2_start = 0.25, 0.00
a1_end,   a2_end   = 0.00, 0.35

# Time discretisation
T = 1.0
Nt = 160

# KKT solver settings
kkt_tol = 1e-10
kkt_maxit = 5000

# Mobility nonlinearity parameter:
#   G(rho) = exp( gamma_G * (rho/rho0 - 1) )
# With |a1|+|a2| <= 0.9, (rho/rho0 - 1) in [-0.9, 0.9],
# so exponent in [-1.8, 1.8] and G ∈ [~0.16, ~6.1].
gamma_G = 2.0

# Control ansatz:
#   a_j(t) = a_j_start + (a_j_end - a_j_start) *
#            ( s + sum_{m=1..n_ctrl} c_{j,m} sin(m*pi*s) ),  s = t/T,
# j = 1,2. Sine basis fixes endpoints.
n_ctrl = 3
a_clip = 0.45   # keep amplitudes safely below 0.45 (|a1|+|a2| < 0.9)

# Optimisation settings: batch random search
n_batches = 16
batch_size = 14
step_scale_init = 0.25
step_scale_final = 0.05
rng_seed = 123456

# Parallelism: up to 20 workers, fall back to 1
max_workers_config = 20
cpu_count = mp.cpu_count() or 1
N_WORKERS = max(1, min(max_workers_config, cpu_count))

# Output directory
outdir = "out"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Grid and globals for workers
# =============================================================================

grid = Grid(L=L, N=N)
x, k, mask = grid.build()

k1 = 2.0 * np.pi * mode1 / L
k2 = 2.0 * k1


# =============================================================================
# Helpers: rho and entropy
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


def mobility_G_from_rho(rho):
    """
    Density-dependent mobility:
        G(rho) = exp( gamma_G * (rho / rho0 - 1) ),
    then projected into the spectral subspace.
    """
    theta = rho / rho0_value - 1.0  # dimensionless deviation from mean
    G = np.exp(gamma_G * theta)
    G = proj_field(G, mask)
    return G


# =============================================================================
# Protocols and control ansatz
# =============================================================================

def protocol_linear(t):
    """Baseline linear protocol in (a1, a2) at constant speed."""
    s = t / T
    a1 = a1_start + (a1_end - a1_start) * s
    a2 = a2_start + (a2_end - a2_start) * s
    return a1, a2


def make_protocol_from_coeffs(coeffs):
    """
    Build (a1(t), a2(t)) from coefficient vector coeffs of length 2*n_ctrl.

    For j in {1,2}:
      a_j(t) = a_j_start + (a_j_end - a_j_start) *
               ( s + sum_m c_{j,m} sin(m*pi*s) ),  s = t/T.

    Sine modes vanish at s = 0 and s = 1, so endpoints are fixed.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    assert coeffs.size == 2 * n_ctrl
    c1 = coeffs[:n_ctrl]
    c2 = coeffs[n_ctrl:]

    def a_of_t(t):
        s = t / T
        base1 = s
        base2 = s
        for m in range(1, n_ctrl + 1):
            sin_term = np.sin(np.pi * m * s)
            base1 = base1 + c1[m - 1] * sin_term
            base2 = base2 + c2[m - 1] * sin_term
        a1 = a1_start + (a1_end - a1_start) * base1
        a2 = a2_start + (a2_end - a2_start) * base2
        # Clip amplitudes conservatively to keep rho safe
        a1 = float(np.clip(a1, -a_clip, a_clip))
        a2 = float(np.clip(a2, -a_clip, a_clip))
        return a1, a2

    return a_of_t


# =============================================================================
# Core metric computation (curved G(rho))
# =============================================================================

def compute_protocol_metrics(a_of_t, x, k, mask):
    """
    For a given protocol (a1(t), a2(t)), compute:

      - S(t_n): Shannon entropy at Nt+1 time points,
      - C_n: H^{-1}_{rho}(G(rho)) metric cost on [t_n, t_{n+1}],
      - action A = ∑ C_n dt,
      - length L = ∑ sqrt(max(C_n, 0)) dt,
      - ΔS = S(T) - S(0),

    along with KKT residual and iteration diagnostics.

    Returns (meta, data_dict).
    """
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

    # Metric cost per interval
    C_arr = np.zeros(Nt, dtype=float)
    kkt_res_arr = np.zeros(Nt, dtype=float)
    kkt_it_arr = np.zeros(Nt, dtype=int)

    for n in range(Nt):
        rho_n = rho_arr[n]
        rho_np1 = rho_arr[n + 1]
        rho_mid = 0.5 * (rho_n + rho_np1)

        v = (rho_np1 - rho_n) / dt

        # Density-dependent mobility for this interval
        G_mid = mobility_G_from_rho(rho_mid)

        phi, it, res = kkt_solve(
            rho_mid, G_mid, v, k, mask, tol=kkt_tol, maxit=kkt_maxit
        )
        phi_x = dx(phi, k, mask)

        # H^{-1}_{rho}(G) cost: ∫ rho_mid G_mid |∂x φ|^2 dx
        C_n = np.trapezoid(rho_mid * G_mid * phi_x * phi_x, x)

        C_arr[n] = C_n
        kkt_res_arr[n] = res
        kkt_it_arr[n] = it

    C_pos = np.maximum(C_arr, 0.0)
    action = float(np.sum(C_arr * dt))
    length = float(np.sum(np.sqrt(C_pos) * dt))
    delta_S = float(S_arr[-1] - S_arr[0])

    meta = {
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
    return meta, data


# =============================================================================
# Worker wrapper for random search
# =============================================================================

def evaluate_coeffs(coeffs):
    """
    Worker wrapper:

    - Build (a1(t), a2(t)) from coeffs.
    - Compute metrics with curved G(rho).
    - Return (A, L, ΔS, coeffs, meta).
    """
    a_of_t = make_protocol_from_coeffs(coeffs)
    meta, _ = compute_protocol_metrics(a_of_t, x, k, mask)
    return meta["action"], meta["length"], meta["delta_S"], coeffs, meta


# =============================================================================
# Main: baseline linear + curved optimisation
# =============================================================================

def main():
    t0 = time.time()
    print("============================================================")
    print("24_speed_limit_curved.py")
    print("Two-mode H^{-1}_{rho}(G(rho)) speed–limit optimisation (curved patch)")
    print("------------------------------------------------------------")
    print(f"L = {L:.3f}, N = {N}, Nt = {Nt}")
    print(f"rho0_value = {rho0_value:.6f}, mode1 = {mode1}")
    print(f"(a1_start, a2_start) = ({a1_start:.4f}, {a2_start:.4f})")
    print(f"(a1_end,   a2_end)   = ({a1_end:.4f},   {a2_end:.4f})")
    print(f"T = {T:.3f}")
    print(f"a_clip = {a_clip:.3f}, gamma_G = {gamma_G:.3f}")
    print(f"KKT tol/maxit = {kkt_tol:.1e}, {kkt_maxit}")
    print(f"Parallel workers requested = {max_workers_config}, using N_WORKERS = {N_WORKERS}")
    print("============================================================")

    # Baseline: linear protocol in (a1, a2)
    print("Evaluating baseline linear protocol...")
    lin_meta, lin_data = compute_protocol_metrics(protocol_linear, x, k, mask)
    print(f"Linear: A = {lin_meta['action']:.6e}, "
          f"L = {lin_meta['length']:.6e}, "
          f"ΔS = {lin_meta['delta_S']:.6e}")
    print(f"Linear KKT: max_res = {lin_meta['max_residual']:.3e}, "
          f"med_res = {lin_meta['median_residual']:.3e}, "
          f"max_it = {lin_meta['max_iters']}, "
          f"med_it = {lin_meta['median_iters']}")
    print("------------------------------------------------------------")

    rng = np.random.default_rng(rng_seed)

    # Start from linear: coeffs = 0 in 2*n_ctrl dimensional space
    best_coeffs = np.zeros(2 * n_ctrl, dtype=float)
    best_action = lin_meta["action"]
    best_length = lin_meta["length"]
    best_delta_S = lin_meta["delta_S"]
    best_meta = lin_meta
    best_data = lin_data

    total_evals = 0

    for b in range(n_batches):
        frac = b / max(n_batches - 1, 1)
        step_scale = step_scale_init * (1.0 - frac) + step_scale_final * frac

        batch_coeffs = [best_coeffs.copy()]
        for _ in range(batch_size - 1):
            noise = rng.normal(scale=step_scale, size=2 * n_ctrl)
            cand = best_coeffs + noise
            batch_coeffs.append(cand)

        results = []

        if N_WORKERS > 1:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                fut_to_coeff = {ex.submit(evaluate_coeffs, c): c for c in batch_coeffs}
                for fut in as_completed(fut_to_coeff):
                    action, length, delta_S, coeffs_out, meta = fut.result()
                    results.append((action, length, delta_S, coeffs_out, meta))
        else:
            for c in batch_coeffs:
                action, length, delta_S, coeffs_out, meta = evaluate_coeffs(c)
                results.append((action, length, delta_S, coeffs_out, meta))

        total_evals += len(results)

        for action, length, delta_S, coeffs_out, meta in results:
            if not np.isfinite(action):
                continue
            if action < best_action:
                best_action = action
                best_length = length
                best_delta_S = delta_S
                best_coeffs = np.array(coeffs_out, copy=True)
                best_meta = meta

        print(f"[batch {b+1:02d}/{n_batches:02d}] "
              f"step_scale = {step_scale:.3f}, "
              f"best A = {best_action:.6e}, "
              f"best L = {best_length:.6e}, "
              f"best ΔS = {best_delta_S:.6e}, "
              f"evals so far = {total_evals}")

    print("============================================================")
    print("Final best protocol (curved G(rho) ansatz search):")
    print(f"  coeffs = {best_coeffs}")
    print(f"  A_best = {best_action:.6e}")
    print(f"  L_best = {best_length:.6e}")
    print(f"  ΔS_best = {best_delta_S:.6e}")
    print("Comparison vs linear:")
    print(f"  A_best / A_linear = {best_action / lin_meta['action']:.6f}")
    print(f"  L_best / L_linear = {best_length / lin_meta['length']:.6f}")
    print(f"  ΔS_best - ΔS_linear = {best_delta_S - lin_meta['delta_S']:.6e}")

    # Recompute best protocol for clean arrays
    best_a_of_t = make_protocol_from_coeffs(best_coeffs)
    best_meta2, best_data2 = compute_protocol_metrics(best_a_of_t, x, k, mask)
    best_data = best_data2
    best_meta = best_meta2

    # Save data
    np.savez(
        os.path.join(outdir, "protocol_speedlimit_curved_data.npz"),
        L=L,
        N=N,
        rho0_value=rho0_value,
        mode1=mode1,
        a1_start=a1_start,
        a2_start=a2_start,
        a1_end=a1_end,
        a2_end=a2_end,
        T=T,
        Nt=Nt,
        n_ctrl=n_ctrl,
        a_clip=a_clip,
        gamma_G=gamma_G,
        t_grid_linear=lin_data["t_grid"],
        a1_vals_linear=lin_data["a1_vals"],
        a2_vals_linear=lin_data["a2_vals"],
        S_arr_linear=lin_data["S_arr"],
        C_arr_linear=lin_data["C_arr"],
        t_grid_best=best_data["t_grid"],
        a1_vals_best=best_data["a1_vals"],
        a2_vals_best=best_data["a2_vals"],
        S_arr_best=best_data["S_arr"],
        C_arr_best=best_data["C_arr"],
    )

    meta_rows = [
        {
            "protocol": "linear",
            "action": lin_meta["action"],
            "length": lin_meta["length"],
            "delta_S": lin_meta["delta_S"],
            "max_residual": lin_meta["max_residual"],
            "median_residual": lin_meta["median_residual"],
            "max_iters": lin_meta["max_iters"],
            "median_iters": lin_meta["median_iters"],
        },
        {
            "protocol": "ansatz_best_curved",
            "action": best_meta["action"],
            "length": best_meta["length"],
            "delta_S": best_meta["delta_S"],
            "max_residual": best_meta["max_residual"],
            "median_residual": best_meta["median_residual"],
            "max_iters": best_meta["max_iters"],
            "median_iters": best_meta["median_iters"],
        },
    ]
    df_meta = pd.DataFrame(meta_rows)
    meta_path = os.path.join(outdir, "protocol_speedlimit_curved_meta.csv")
    df_meta.to_csv(meta_path, index=False)

    # Quick figures: control paths and cost
    plt.figure(figsize=(7, 4))
    plt.subplot(2, 1, 1)
    plt.plot(lin_data["t_grid"], lin_data["a1_vals"], label="a1 linear")
    plt.plot(lin_data["t_grid"], lin_data["a2_vals"], label="a2 linear")
    plt.plot(best_data["t_grid"], best_data["a1_vals"], "--", label="a1 best")
    plt.plot(best_data["t_grid"], best_data["a2_vals"], "--", label="a2 best")
    plt.ylabel("a_j(t)")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(lin_data["t_grid"][:-1], lin_data["C_arr"], label="C(t) linear")
    plt.plot(best_data["t_grid"][:-1], best_data["C_arr"], label="C(t) best")
    plt.xlabel("t")
    plt.ylabel("C(t)")
    plt.legend(loc="best")

    plt.tight_layout()
    fig_path = os.path.join(outdir, "protocol_speedlimit_curved_plot.png")
    plt.savefig(fig_path, dpi=180)

    t1 = time.time()
    print(f"Saved {meta_path}, protocol_speedlimit_curved_data.npz, and {fig_path}")
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
