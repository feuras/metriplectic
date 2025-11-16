"""
22_holonomy_synthetic_monopole.py

Synthetic monopole testbench for Berry curvature machinery.

We define a complex field on a 2D control plane (lambda1, lambda2):

    Z(lambda1, lambda2) = exp(i * atan2(lambda2, lambda1)) * exp(-alpha * r^2),
    where r^2 = lambda1^2 + lambda2^2.

Key properties:
  - Phase arg(Z) = atan2(lambda2, lambda1) winds once (2π) around the origin.
  - |Z| = exp(-alpha r^2) > 0 everywhere (no true zeros).

We then:
  * Discretise a rectangular domain in (lambda1, lambda2).
  * Evaluate Z on a uniform grid (N1 x N2).
  * Compute a discrete "Berry curvature" F[i,j] per plaquette using the same
    phase-sum plaquette rule as the metriplectic scripts:
        (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1) -> (i,j).
  * Sum F to estimate total flux and Chern number:
        flux_total = sum_ij F[i,j],
        C ≈ flux_total / (2π).
  * Construct loops around the origin and a far loop, compute windings.

Expected:
  - Chern ≈ +1 (single vortex).
  - Loops enclosing origin: winding ≈ +1.
  - Loop not enclosing: winding ≈ 0.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

L1_min, L1_max = -1.5, 1.5
L2_min, L2_max = -1.5, 1.5

N1 = 201
N2 = 201

alpha = 0.5

r_inner = 0.3
r_outer = 1.0
r_far = 0.5
N_loop = 400

outdir = "out_synth"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Synthetic Z and Berry curvature machinery
# =============================================================================

def synthetic_Z(l1, l2):
    """
    Synthetic complex field with a single phase winding:

        Z = exp(i * atan2(l2, l1)) * exp(-alpha * r^2).

    |Z| > 0 everywhere, but the phase has a 2π winding around the origin.
    """
    theta = np.arctan2(l2, l1)
    r2 = l1 * l1 + l2 * l2
    return np.exp(1j * theta) * np.exp(-alpha * r2)


def plaquette_phase(z00, z10, z11, z01):
    """
    Phase sum around a plaquette:

        (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0),

    using log-ratios of complex values. We explicitly avoid exact zeros
    in denominators by nudging magnitudes below eps.
    """
    eps = 1e-30

    def safe(z):
        if abs(z) < eps:
            # Replace with a tiny nonzero complex number preserving phase (if any)
            return complex(eps, 0.0)
        return z

    z00 = safe(z00)
    z10 = safe(z10)
    z11 = safe(z11)
    z01 = safe(z01)

    d1 = np.angle(z10 / z00)
    d2 = np.angle(z11 / z10)
    d3 = np.angle(z01 / z11)
    d4 = np.angle(z00 / z01)
    return d1 + d2 + d3 + d4


def loop_winding(Z_values):
    """
    Given complex values Z[k] sampled around a closed loop in control space,
    compute:

        total_phase = ∑_k Δ arg(Z),
        n = total_phase / (2π),

    where Δ arg(Z) is computed via angle(z_{k+1}/z_k).
    """
    total_phase = 0.0
    eps = 1e-30
    M = len(Z_values)
    for k in range(M):
        z0 = Z_values[k]
        z1 = Z_values[(k + 1) % M]

        if abs(z0) < eps:
            z0 = complex(eps, 0.0)
        if abs(z1) < eps:
            z1 = complex(eps, 0.0)

        total_phase += np.angle(z1 / z0)

    return total_phase / (2.0 * np.pi), total_phase


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("============================================================")
    print("22_holonomy_synthetic_monopole.py")
    print("Synthetic monopole testbench for Berry curvature machinery")
    print("------------------------------------------------------------")
    print(f"Control-space rectangle:")
    print(f"  lambda1 ∈ [{L1_min:.3f}, {L1_max:.3f}] with N1 = {N1}")
    print(f"  lambda2 ∈ [{L2_min:.3f}, {L2_max:.3f}] with N2 = {N2}")
    print(f"Synthetic field: Z = exp(i atan2(l2,l1)) * exp(-alpha r^2)")
    print(f"alpha = {alpha:.3f}")
    print(f"Loop radii (inner, outer, far) = ({r_inner:.3f}, {r_outer:.3f}, {r_far:.3f})")
    print(f"Loop samples N_loop = {N_loop}")
    print(f"Output dir: {outdir}")
    print("============================================================")

    lambda1_grid = np.linspace(L1_min, L1_max, N1)
    lambda2_grid = np.linspace(L2_min, L2_max, N2)

    dL1 = lambda1_grid[1] - lambda1_grid[0]
    dL2 = lambda2_grid[1] - lambda2_grid[0]
    print(f"Grid spacings: dL1 = {dL1:.4f}, dL2 = {dL2:.4f}")

    print("Evaluating synthetic Z over control-space grid...")
    L1_vals, L2_vals = np.meshgrid(lambda1_grid, lambda2_grid, indexing="ij")
    Z_grid = synthetic_Z(L1_vals, L2_vals)

    Z_abs_min = np.abs(Z_grid).min()
    Z_abs_max = np.abs(Z_grid).max()
    print(f"|Z| min / max over grid = {Z_abs_min:.3e} / {Z_abs_max:.3e}")

    print("Computing discrete Berry curvature on plaquettes...")
    F = np.zeros((N1 - 1, N2 - 1), dtype=float)
    for i in range(N1 - 1):
        for j in range(N2 - 1):
            z00 = Z_grid[i, j]
            z10 = Z_grid[i + 1, j]
            z11 = Z_grid[i + 1, j + 1]
            z01 = Z_grid[i, j + 1]
            F[i, j] = plaquette_phase(z00, z10, z11, z01)

    if np.isnan(F).any() or np.isinf(F).any():
        print("WARNING: NaN or inf detected in curvature array; cleaning...")
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    flux_total = F.sum()
    C_est = flux_total / (2.0 * np.pi)
    C_rounded = int(np.rint(C_est))

    print("------------------------------------------------------------")
    print("Synthetic Berry curvature / Chern diagnostics")
    print(f"  total flux ∑ F = {flux_total:.6f} rad")
    print(f"  Chern estimate C ≈ {C_est:.6f}")
    print(f"  Chern rounded      C = {C_rounded}")
    print("Expected: C ≈ +1 for a single phase vortex at the origin.")
    print("------------------------------------------------------------")

    print("Computing loop windings around the origin...")

    def sample_loop(cx, cy, radius):
        t = np.linspace(0.0, 2.0 * np.pi, N_loop, endpoint=False)
        l1 = cx + radius * np.cos(t)
        l2 = cy + radius * np.sin(t)
        Z_loop = synthetic_Z(l1, l2)
        n_est, phase_tot = loop_winding(Z_loop)
        return {
            "centre_x": cx,
            "centre_y": cy,
            "radius": radius,
            "n_est": float(n_est),
            "phase_total": float(phase_tot),
        }

    loop_results = []

    loop_inner = sample_loop(0.0, 0.0, r_inner)
    print(f"  Inner loop (r = {r_inner:.3f}):")
    print(f"    total phase = {loop_inner['phase_total']:.6f} rad")
    print(f"    winding n ≈ {loop_inner['n_est']:.6f}")
    loop_results.append({"label": "inner", **loop_inner})

    loop_outer = sample_loop(0.0, 0.0, r_outer)
    print(f"  Outer loop (r = {r_outer:.3f}):")
    print(f"    total phase = {loop_outer['phase_total']:.6f} rad")
    print(f"    winding n ≈ {loop_outer['n_est']:.6f}")
    loop_results.append({"label": "outer", **loop_outer})

    cx_far, cy_far = 1.0, 0.0
    loop_far = sample_loop(cx_far, cy_far, r_far)
    print(f"  Far loop (centre=({cx_far:.3f},{cy_far:.3f}), r = {r_far:.3f}):")
    print(f"    total phase = {loop_far['phase_total']:.6f} rad")
    print(f"    winding n ≈ {loop_far['n_est']:.6f}")
    loop_results.append({"label": "far", **loop_far})

    print("Expected: inner/outer loops n ≈ +1, far loop n ≈ 0.")
    print("------------------------------------------------------------")

    np.savez(
        os.path.join(outdir, "monopole_synthetic_data.npz"),
        lambda1_grid=lambda1_grid,
        lambda2_grid=lambda2_grid,
        Z_real=Z_grid.real,
        Z_imag=Z_grid.imag,
        F=F,
        flux_total=flux_total,
        C_est=C_est,
        C_rounded=C_rounded,
        alpha=alpha,
    )

    meta_df = pd.DataFrame([{
        "L1_min": L1_min,
        "L1_max": L1_max,
        "L2_min": L2_min,
        "L2_max": L2_max,
        "N1": N1,
        "N2": N2,
        "alpha": alpha,
        "flux_total": float(flux_total),
        "C_est": float(C_est),
        "C_rounded": int(C_rounded),
    }])
    meta_path = os.path.join(outdir, "monopole_synthetic_meta.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path}")

    loops_df = pd.DataFrame(loop_results)
    loops_path = os.path.join(outdir, "monopole_synthetic_loops.csv")
    loops_df.to_csv(loops_path, index=False)
    print(f"Wrote {loops_path}")

    plt.figure(figsize=(5, 4))
    L1_mid = 0.5 * (lambda1_grid[:-1] + lambda1_grid[1:])
    L2_mid = 0.5 * (lambda2_grid[:-1] + lambda2_grid[1:])
    extent = [L2_mid[0], L2_mid[-1], L1_mid[0], L1_mid[-1]]
    im = plt.imshow(F, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(im, label="Flux per plaquette [rad]")
    plt.xlabel("lambda2")
    plt.ylabel("lambda1")
    plt.title("Synthetic Berry curvature (phase vortex)")
    plt.tight_layout()
    fig_path = os.path.join(outdir, "monopole_synthetic_curvature.png")
    plt.savefig(fig_path, dpi=180)
    print(f"Wrote {fig_path}")

    t1 = time.time()
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
