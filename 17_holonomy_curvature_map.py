"""
17_holonomy_curvature_map.py

Compute a 2D control-space Berry curvature map for the complex pairing (reader)
built on top of the 1D metriplectic KKT machinery.

This script generalises 07_holonomy_loop.py from a single closed loop in control
space to a full rectangular grid in (a, theta), where:
  - a controls the amplitude of a small density modulation,
  - theta controls the mixing of the reversible quadrature in the
    two-plane { d_x mu, H d_x mu }.

For each control point (a, theta) we:
  1. Build a density rho(x) satisfying the mass constraint.
  2. Construct the entropic chemical potential mu and its plane derivatives.
  3. Build the irreversible velocity v_G and Liouville-correct reversible
     velocity v_J as in 07_holonomy_loop.py.
  4. Solve the KKT problem L_{rho,G} phi = v.
  5. Form the complex pairing Z(a, theta) = Re + i Im via
        Re = ∫ rho phi_x mu_x dx,
        Im = ∫ rho phi_x H(mu_x) dx.
  6. Store Z and diagnostics.

We then estimate the Berry curvature on each elementary plaquette in control
space using discrete phase differences and compute an approximate Chern number.

Outputs:
  out/holonomy_curvature_meta.csv     summary statistics
  out/holonomy_curvature_data.npz     arrays for Z and curvature
  out/holonomy_curvature_map.png      heatmap of curvature
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import Grid, proj_field, mu_entropic, dx, hilbert, kkt_solve


# =============================================================================
# Configuration
# =============================================================================

# Spatial grid
L = 40.0
N = 512

# Base density rho0 = 1 / L so that ∫ rho0 dx = 1
rho0_value = 1.0 / L

# Entropic bias for mu = log rho - rho_bias * dxx rho
rho_bias = 0.10

# Reversible structure parameters
cJ = 0.6          # overall reversible amplitude (dimensionless)
mode_rho = 3      # spatial mode of the density modulation

# Control-space grid in (a, theta)
a_min, a_max = 0.05, 0.15
theta_min, theta_max = 0.0, np.pi / 2.0
Na, Ntheta = 16, 32

# KKT solver tolerances
kkt_tol = 1e-10
kkt_maxit = 6000

# Liouville diagnostic tolerance (on sup |d_x( rho * a(x) )|)
tol_liouville = 1e-12

# Reader mixing between v_G and v_J
mix_G = 0.5
mix_J = 0.5

# Parallelism
max_workers = min(8, os.cpu_count() or 1)

# Output directory
outdir = "out"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def build_state(a, x, k, mask):
    """
    Build rho(x), mu(x), and the derivative plane {mu_x, H mu_x} for a given
    modulation amplitude a.

    Returns:
      rho, mu_x, Hmu_x
    """
    # Base density with small cosine modulation
    rho = rho0_value * (1.0 + a * np.cos(2.0 * np.pi * mode_rho * x / L))
    # Project to dealiased band and restore mean so that ∫ rho dx = 1
    rho = proj_field(rho, mask)
    rho = rho - rho.mean() + rho0_value

    # Entropic chemical potential and its x derivative
    mu = mu_entropic(rho, rho_bias, k, mask)
    mu_x = dx(mu, k, mask)
    Hmu_x = hilbert(mu_x, k, mask)

    return rho, mu_x, Hmu_x


def build_velocities(rho, mu_x, Hmu_x, k, mask):
    """
    Build irreversible and reversible velocities (v_G, v_J) as in
    07_holonomy_loop.py but without fixing the rotation angle theta yet.

    Returns:
      v_G, helper dict with fields used in v_J construction.
    """
    # Scalar mobility G = 1
    G = np.ones_like(rho)

    # Irreversible velocity: v_G = - d_x( rho G mu_x )
    v_G = -dx(rho * G * mu_x, k, mask)

    # Liouville correcting amplitude a(x) = cJ / rho(x)
    # so that d_x( rho a(x) ) = 0 identically (up to spectral truncation).
    a_x = cJ / rho

    # Pack helper fields so that we can cheaply build v_J(θ) later
    helpers = {
        "G": G,
        "a_x": a_x,
        "mu_x": mu_x,
        "Hmu_x": Hmu_x,
    }

    return v_G, helpers


def build_vJ(theta, helpers, k, mask):
    """
    Build reversible velocity v_J at a given rotation angle theta using the
    two-plane {H mu_x, mu_x} and Liouville amplitude a_x.

    The rotation is:
      rot(θ) = cos θ H mu_x + sin θ mu_x,
      v_J(θ) = d_x( rho a_x rot ) but since rho a_x = cJ we implement
               v_J = d_x( cJ rot ) directly as in 07_holonomy_loop.py.
    """
    mu_x = helpers["mu_x"]
    Hmu_x = helpers["Hmu_x"]
    # Rotation in the diagnostic two-plane
    rot = np.cos(theta) * Hmu_x + np.sin(theta) * mu_x
    # Reversible velocity
    v_J = dx(cJ * rot, k, mask)
    return v_J


def complex_pairing(rho, mu_x, Hmu_x, v_G, v_J, k, mask):
    """
    Solve the KKT problem for v = mix_G v_G + mix_J v_J and form the complex
    pairing Z = Re + i Im with
      Re = ∫ rho phi_x mu_x dx,
      Im = ∫ rho phi_x Hmu_x dx.

    Returns:
      Z (complex), kkt_it, kkt_res, liouville_sup
    """
    # Total test velocity
    v = mix_G * v_G + mix_J * v_J

    # Mobility G = 1 for the KKT solve
    G = np.ones_like(rho)

    # Solve L_{rho,G} phi = v by CG
    phi, it, res = kkt_solve(rho, G, v, k, mask, tol=kkt_tol, maxit=kkt_maxit)

    # Spatial derivatives of phi
    phi_x = dx(phi, k, mask)

    # Real and imaginary parts of the pairing
    Re_pair = np.trapz(rho * phi_x * mu_x, x)
    Im_pair = np.trapz(rho * phi_x * Hmu_x, x)
    Z = Re_pair + 1j * Im_pair

    # Liouville diagnostic: sup | d_x( rho a_x ) | with a_x = cJ / rho
    a_x = cJ / rho
    liouville_res = np.max(np.abs(dx(rho * a_x, k, mask)))

    return Z, it, res, liouville_res


def plaquette_phase(z00, z10, z11, z01):
    """
    Compute the net phase accumulated around an elementary plaquette with
    vertices (in order)
      (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
    using logarithmic phase differences.

    The result is a real number in radians; an integer multiple of 2π
    corresponds to an integer Berry flux quantum.
    """
    # Protect against division by zero; this should not happen in a regular
    # run but we guard and report it in the meta output.
    eps = 1e-30
    z00 = z00 if abs(z00) > eps else eps
    z10 = z10 if abs(z10) > eps else eps
    z11 = z11 if abs(z11) > eps else eps
    z01 = z01 if abs(z01) > eps else eps

    d1 = np.angle(z10 / z00)
    d2 = np.angle(z11 / z10)
    d3 = np.angle(z01 / z11)
    d4 = np.angle(z00 / z01)

    return d1 + d2 + d3 + d4


# =============================================================================
# Main computation
# =============================================================================

def main():
    t0 = time.time()
    print("============================================================")
    print("17_holonomy_curvature_map.py")
    print("Computing control-space Berry curvature for complex pairing")
    print("------------------------------------------------------------")
    print(f"Spatial grid:    L = {L:.3f}, N = {N}")
    print(f"Base density:    rho0 = {rho0_value:.6f} (∫ rho0 dx = 1)")
    print(f"Entropic bias:   rho_bias = {rho_bias:.3f}")
    print(f"Reversible amp:  cJ = {cJ:.3f}, mode_rho = {mode_rho:d}")
    print(f"Control grid:    a ∈ [{a_min:.3f}, {a_max:.3f}] (Na={Na}), "
          f"theta ∈ [{theta_min:.3f}, {theta_max:.3f}] (Nθ={Ntheta})")
    print(f"KKT tol/maxit:   tol = {kkt_tol:.1e}, maxit = {kkt_maxit}")
    print(f"Mixing:          v = {mix_G:.2f} v_G + {mix_J:.2f} v_J")
    print(f"Threads:         max_workers = {max_workers}")
    print(f"Output dir:      {outdir}")
    print("============================================================")

    # Build spatial grid and FFT frequencies
    g = Grid(L=L, N=N)
    global x  # used inside complex_pairing for trapz
    x, k, mask = g.build()

    # Build control-space grids
    a_grid = np.linspace(a_min, a_max, Na)
    theta_grid = np.linspace(theta_min, theta_max, Ntheta)

    # Containers for Z and diagnostics
    Z = np.zeros((Na, Ntheta), dtype=np.complex128)
    kkt_it = np.zeros((Na, Ntheta), dtype=int)
    kkt_res = np.zeros((Na, Ntheta), dtype=float)
    liouville_sup = np.zeros((Na, Ntheta), dtype=float)

    # Precompute rho, mu_x, Hmu_x and v_G for each a (shared across theta)
    print("Precomputing state and irreversible velocity for each a...")
    precomputed = []
    for ia, a in enumerate(a_grid):
        rho, mu_x, Hmu_x = build_state(a, x, k, mask)
        v_G, helpers = build_velocities(rho, mu_x, Hmu_x, k, mask)
        precomputed.append({
            "a": a,
            "rho": rho,
            "mu_x": mu_x,
            "Hmu_x": Hmu_x,
            "v_G": v_G,
            "helpers": helpers,
        })
    print("Done precomputing base states.")
    print("Launching threaded KKT solves for all (a, theta) control points...")

    # Worker for a single control point
    def worker(ia, itheta):
        a = a_grid[ia]
        theta = theta_grid[itheta]
        data = precomputed[ia]
        rho = data["rho"]
        mu_x = data["mu_x"]
        Hmu_x = data["Hmu_x"]
        v_G = data["v_G"]
        helpers = data["helpers"]
        v_J = build_vJ(theta, helpers, k, mask)
        Z_loc, it_loc, res_loc, liou_loc = complex_pairing(
            rho, mu_x, Hmu_x, v_G, v_J, k, mask
        )
        return ia, itheta, Z_loc, it_loc, res_loc, liou_loc

    # Submit jobs
    total_jobs = Na * Ntheta
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ia in range(Na):
            for itheta in range(Ntheta):
                futures.append(pool.submit(worker, ia, itheta))

        # Collect results with progress
        completed = 0
        next_report = max(1, total_jobs // 10)
        for fut in as_completed(futures):
            ia, itheta, Z_loc, it_loc, res_loc, liou_loc = fut.result()
            Z[ia, itheta] = Z_loc
            kkt_it[ia, itheta] = it_loc
            kkt_res[ia, itheta] = res_loc
            liouville_sup[ia, itheta] = liou_loc

            completed += 1
            if completed % next_report == 0 or completed == total_jobs:
                pct = 100.0 * completed / total_jobs
                print(f"  progress: {completed:5d} / {total_jobs:5d} "
                      f"({pct:5.1f} %)")

    # Basic diagnostics on Z
    Z_abs = np.abs(Z)
    Z_min = Z_abs.min()
    Z_max = Z_abs.max()
    print("------------------------------------------------------------")
    print("Complex pairing diagnostics over control grid")
    print(f"  |Z| min / max = {Z_min:.3e} / {Z_max:.3e}")
    print(f"  KKT residual:  max = {kkt_res.max():.3e}, "
          f"median = {np.median(kkt_res):.3e}")
    print(f"  KKT iterations: max = {kkt_it.max()}, "
          f"median = {int(np.median(kkt_it))}")
    print(f"  Liouville sup: max = {liouville_sup.max():.3e}, "
          f"median = {np.median(liouville_sup):.3e}")
    print(f"  Liouville OK (<= 10 * tol_liouville={10*tol_liouville:.1e}):",
          liouville_sup.max() <= 10.0 * tol_liouville)
    print("------------------------------------------------------------")

    # Compute discrete Berry curvature on plaquettes
    print("Computing discrete Berry curvature on control-space plaquettes...")
    F = np.zeros((Na - 1, Ntheta - 1), dtype=float)
    for ia in range(Na - 1):
        for itheta in range(Ntheta - 1):
            z00 = Z[ia, itheta]
            z10 = Z[ia + 1, itheta]
            z11 = Z[ia + 1, itheta + 1]
            z01 = Z[ia, itheta + 1]
            F[ia, itheta] = plaquette_phase(z00, z10, z11, z01)

    total_flux = F.sum()
    chern_est = total_flux / (2.0 * np.pi)
    chern_rounded = int(np.rint(chern_est))

    print("------------------------------------------------------------")
    print("Berry curvature and Chern number estimate")
    print(f"  total flux ∑_plaquettes F = {total_flux:.6f} rad")
    print(f"  Chern estimate           C ≈ {chern_est:.6f}")
    print(f"  Chern rounded            C = {chern_rounded:d}")
    print("------------------------------------------------------------")

    # Save data
    np.savez(
        os.path.join(outdir, "holonomy_curvature_data.npz"),
        a_grid=a_grid,
        theta_grid=theta_grid,
        Z=Z,
        F=F,
        kkt_it=kkt_it,
        kkt_res=kkt_res,
        liouville_sup=liouville_sup,
        L=L,
        N=N,
        rho0_value=rho0_value,
        rho_bias=rho_bias,
        cJ=cJ,
        mode_rho=mode_rho,
    )
    meta = pd.DataFrame([{
        "L": L,
        "N": N,
        "rho0": rho0_value,
        "rho_bias": rho_bias,
        "cJ": cJ,
        "mode_rho": mode_rho,
        "a_min": a_min,
        "a_max": a_max,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "Na": Na,
        "Ntheta": Ntheta,
        "kkt_tol": kkt_tol,
        "kkt_maxit": kkt_maxit,
        "mix_G": mix_G,
        "mix_J": mix_J,
        "max_workers": max_workers,
        "Z_min_abs": Z_min,
        "Z_max_abs": Z_max,
        "kkt_res_max": float(kkt_res.max()),
        "kkt_res_median": float(np.median(kkt_res)),
        "liouville_sup_max": float(liouville_sup.max()),
        "liouville_sup_median": float(np.median(liouville_sup)),
        "tol_liouville": tol_liouville,
        "chern_est": float(chern_est),
        "chern_rounded": chern_rounded,
    }])
    meta_path = os.path.join(outdir, "holonomy_curvature_meta.csv")
    meta.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path}")
    print(f"Wrote {os.path.join(outdir, 'holonomy_curvature_data.npz')}")

    # Plot curvature map
    plt.figure(figsize=(6, 4))
    A_mid = 0.5 * (a_grid[:-1] + a_grid[1:])
    Theta_mid = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    extent = [Theta_mid[0], Theta_mid[-1], A_mid[0], A_mid[-1]]
    im = plt.imshow(
        F,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.colorbar(im, label="Berry flux per plaquette [rad]")
    plt.xlabel("theta")
    plt.ylabel("a (density modulation amplitude)")
    plt.title("Control-space Berry curvature of complex pairing")
    plt.tight_layout()
    fig_path = os.path.join(outdir, "holonomy_curvature_map.png")
    plt.savefig(fig_path, dpi=180)
    print(f"Wrote {fig_path}")

    t1 = time.time()
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
