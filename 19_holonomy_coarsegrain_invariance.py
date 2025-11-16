"""
19_holonomy_coarsegrain_invariance.py

Test the invariance of the complex pairing Z(a, theta) and the associated
Berry curvature under spatial coarse graining of the density.

We:
  1. Work on a 1D periodic domain with L, N.
  2. Use a density ansatz
        rho(x; a) = rho0 * [ 1 + a cos(k x) ]
     with |a| small enough that rho > 0.
  3. For each control point (a, theta) we:
        - Build rho(x; a), mu, mu_x, Hmu_x.
        - Build irreversible v_G and reversible v_J(theta) as in the
          holonomy scripts.
        - Solve L_{rho,G} phi = v for v = mix_G v_G + mix_J v_J.
        - Compute complex pairing
              Z_fine = Re + i Im
          from phi_x with the reader quadratures.
  4. Coarse grain rho(x; a) by applying a Gaussian filter in Fourier space
     at scale ell_cg, renormalise to preserve mass 1, and repeat the steps
     above to obtain Z_coarse.
  5. Build discrete Berry curvature maps F_fine and F_coarse on the control
     grid using plaquette phase sums.
  6. Compare:
        - phase differences arg(Z_coarse / Z_fine),
        - modulus ratios |Z_coarse| / |Z_fine|,
        - total flux and Chern estimates from F_fine, F_coarse.

Outputs:
  out/holonomy_cg_data.npz
  out/holonomy_cg_meta.csv
  out/holonomy_cg_phase_diff.png
  out/holonomy_cg_curvature_diff.png
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
N = 1024

rho0_value = 1.0 / L      # base density so that ∫ rho dx = 1
rho_bias = 0.10           # entropic curvature

# Reversible structure
cJ = 0.6
mode_rho = 3              # modulation mode in rho(x; a)

# Control grid in (a, theta)
a_min, a_max = 0.01, 0.20
theta_min, theta_max = 0.0, np.pi / 2.0
Na, Ntheta = 24, 32

# Coarse graining scale (physical length)
# Gaussian filter exp( -0.5 * (ell_cg * k)^2 ) in Fourier space
ell_cg = L / 20.0

# KKT solver settings
kkt_tol = 1e-10
kkt_maxit = 6000

tol_liouville = 1e-11

# Mixing of irreversible and reversible velocities
mix_G = 0.5
mix_J = 0.5

# Parallelism
max_workers = min(20, os.cpu_count() or 1)

# Output directory
outdir = "out"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def coarse_grain_gaussian(rho, k, ell):
    """
    Coarse grain rho by applying a Gaussian filter in Fourier space:

        rho_c(k) = exp( -0.5 * (ell * k)^2 ) * rho(k),

    then inverse FFT back to x space and renormalise so that ∫ rho_c dx = 1.

    Assumes rho has mean 1 / L initially.
    """
    # Forward FFT (consistent with common.Grid conventions)
    rho_hat = np.fft.rfft(rho)
    k_rfft = k[:rho_hat.size]  # match half spectrum

    filt = np.exp(-0.5 * (ell * k_rfft)**2)
    rho_hat_c = rho_hat * filt
    rho_c = np.fft.irfft(rho_hat_c, n=rho.size)

    # Renormalise total mass to 1
    dx_phys = L / rho.size
    mass_c = rho_c.sum() * dx_phys
    if mass_c != 0.0:
        rho_c *= (1.0 / mass_c)
    return rho_c


def build_state(a, x, k, mask):
    """
    Build rho(x; a), mu_x, Hmu_x for a given amplitude a.
    """
    rho = rho0_value * (1.0 + a * np.cos(2.0 * np.pi * mode_rho * x / L))
    rho = proj_field(rho, mask)
    rho = rho - rho.mean() + rho0_value

    mu = mu_entropic(rho, rho_bias, k, mask)
    mu_x = dx(mu, k, mask)
    Hmu_x = hilbert(mu_x, k, mask)
    return rho, mu_x, Hmu_x


def build_velocities(rho, mu_x, Hmu_x, k, mask):
    """
    Build irreversible velocity v_G and helpers for reversible v_J.
    """
    G = np.ones_like(rho)
    v_G = -dx(rho * G * mu_x, k, mask)

    a_x = cJ / rho
    helpers = {
        "G": G,
        "a_x": a_x,
        "mu_x": mu_x,
        "Hmu_x": Hmu_x,
    }
    return v_G, helpers


def build_vJ(theta, helpers, k, mask):
    """
    Build reversible velocity v_J(theta) via rotated quadrature:

        rot(theta) = cos theta Hmu_x + sin theta mu_x,
        v_J(theta) = d_x [ cJ * rot(theta) ].
    """
    mu_x = helpers["mu_x"]
    Hmu_x = helpers["Hmu_x"]
    rot = np.cos(theta) * Hmu_x + np.sin(theta) * mu_x
    v_J = dx(cJ * rot, k, mask)
    return v_J


def complex_pairing(rho, mu_x, Hmu_x, v_G, v_J, x, k, mask):
    """
    Solve L_{rho,G} phi = v and form complex pairing

        Re(Z) = ∫ rho phi_x mu_x dx,
        Im(Z) = ∫ rho phi_x Hmu_x dx.
    """
    v = mix_G * v_G + mix_J * v_J
    G = np.ones_like(rho)

    phi, it, res = kkt_solve(rho, G, v, k, mask, tol=kkt_tol, maxit=kkt_maxit)
    phi_x = dx(phi, k, mask)

    Re_pair = np.trapezoid(rho * phi_x * mu_x, x)
    Im_pair = np.trapezoid(rho * phi_x * Hmu_x, x)
    Z = Re_pair + 1j * Im_pair

    a_x = cJ / rho
    liouville_res = np.max(np.abs(dx(rho * a_x, k, mask)))

    return Z, it, res, liouville_res


def plaquette_phase(z00, z10, z11, z01):
    """
    Phase sum around an elementary plaquette:

        (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0).

    Returns total phase in radians.
    """
    eps = 1e-30
    if abs(z00) < eps:
        z00 = eps
    if abs(z10) < eps:
        z10 = eps
    if abs(z11) < eps:
        z11 = eps
    if abs(z01) < eps:
        z01 = eps

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
    print("19_holonomy_coarsegrain_invariance.py")
    print("Testing coarse graining invariance of complex pairing and curvature")
    print("------------------------------------------------------------")
    print(f"Spatial grid:    L = {L:.3f}, N = {N}")
    print(f"Base density:    rho0 = {rho0_value:.6f}")
    print(f"rho_bias:        {rho_bias:.3f}")
    print(f"Reversible amp:  cJ = {cJ:.3f}, mode_rho = {mode_rho:d}")
    print(f"Control grid:    a ∈ [{a_min:.3f}, {a_max:.3f}] (Na={Na}), "
          f"theta ∈ [{theta_min:.3f}, {theta_max:.3f}] (Nθ={Ntheta})")
    print(f"Coarse graining: ell_cg = {ell_cg:.3f}")
    print(f"Mixing:          v = {mix_G:.2f} v_G + {mix_J:.2f} v_J")
    print(f"KKT tol/maxit:   tol = {kkt_tol:.1e}, maxit = {kkt_maxit}")
    print(f"Threads:         max_workers = {max_workers}")
    print(f"Output dir:      {outdir}")
    print("============================================================")

    # Build spatial grid
    g = Grid(L=L, N=N)
    x, k, mask = g.build()

    # Control grids
    a_grid = np.linspace(a_min, a_max, Na)
    theta_grid = np.linspace(theta_min, theta_max, Ntheta)

    # Precompute fine states and v_G
    print("Precomputing fine states and irreversible velocities for each a...")
    pre_fine = []
    for ia, a in enumerate(a_grid):
        rho_f, mu_x_f, Hmu_x_f = build_state(a, x, k, mask)
        v_G_f, helpers_f = build_velocities(rho_f, mu_x_f, Hmu_x_f, k, mask)
        pre_fine.append({
            "a": a,
            "rho": rho_f,
            "mu_x": mu_x_f,
            "Hmu_x": Hmu_x_f,
            "v_G": v_G_f,
            "helpers": helpers_f,
        })
    print("Done precomputing fine states.")

    # Precompute coarse states and v_G
    print("Precomputing coarse grained states and irreversible velocities...")
    pre_coarse = []
    for ia, data in enumerate(pre_fine):
        rho_f = data["rho"]
        rho_c = coarse_grain_gaussian(rho_f, k, ell_cg)
        rho_c = proj_field(rho_c, mask)
        rho_c = rho_c - rho_c.mean() + rho0_value  # enforce consistent mean

        mu_c = mu_entropic(rho_c, rho_bias, k, mask)
        mu_x_c = dx(mu_c, k, mask)
        Hmu_x_c = hilbert(mu_x_c, k, mask)
        v_G_c, helpers_c = build_velocities(rho_c, mu_x_c, Hmu_x_c, k, mask)
        pre_coarse.append({
            "a": data["a"],
            "rho": rho_c,
            "mu_x": mu_x_c,
            "Hmu_x": Hmu_x_c,
            "v_G": v_G_c,
            "helpers": helpers_c,
        })
    print("Done precomputing coarse states.")
    print("------------------------------------------------------------")

    # Containers
    Z_fine = np.zeros((Na, Ntheta), dtype=np.complex128)
    Z_coarse = np.zeros_like(Z_fine)
    kkt_it_fine = np.zeros((Na, Ntheta), dtype=int)
    kkt_it_coarse = np.zeros_like(kkt_it_fine)
    kkt_res_fine = np.zeros((Na, Ntheta), dtype=float)
    kkt_res_coarse = np.zeros_like(kkt_res_fine)
    liou_fine = np.zeros((Na, Ntheta), dtype=float)
    liou_coarse = np.zeros_like(liou_fine)

    # Worker
    def worker(ia, itheta):
        theta = theta_grid[itheta]

        # Fine
        df = pre_fine[ia]
        rho_f = df["rho"]
        mu_x_f = df["mu_x"]
        Hmu_x_f = df["Hmu_x"]
        v_G_f = df["v_G"]
        helpers_f = df["helpers"]
        v_J_f = build_vJ(theta, helpers_f, k, mask)
        Zf, it_f, res_f, liou_f = complex_pairing(
            rho_f, mu_x_f, Hmu_x_f, v_G_f, v_J_f, x, k, mask
        )

        # Coarse
        dc = pre_coarse[ia]
        rho_c = dc["rho"]
        mu_x_c = dc["mu_x"]
        Hmu_x_c = dc["Hmu_x"]
        v_G_c = dc["v_G"]
        helpers_c = dc["helpers"]
        v_J_c = build_vJ(theta, helpers_c, k, mask)
        Zc, it_c, res_c, liou_c = complex_pairing(
            rho_c, mu_x_c, Hmu_x_c, v_G_c, v_J_c, x, k, mask
        )

        return ia, itheta, Zf, Zc, it_f, it_c, res_f, res_c, liou_f, liou_c

    # Run in parallel
    print("Computing Z_fine and Z_coarse over control grid...")
    total_jobs = Na * Ntheta
    completed = 0
    next_report = max(1, total_jobs // 10)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
          pool.submit(worker, ia, itheta)
          for ia in range(Na)
          for itheta in range(Ntheta)
        ]
        for fut in as_completed(futures):
            (ia, itheta,
             Zf, Zc,
             it_f, it_c,
             res_f, res_c,
             liou_f, liou_c) = fut.result()

            Z_fine[ia, itheta] = Zf
            Z_coarse[ia, itheta] = Zc
            kkt_it_fine[ia, itheta] = it_f
            kkt_it_coarse[ia, itheta] = it_c
            kkt_res_fine[ia, itheta] = res_f
            kkt_res_coarse[ia, itheta] = res_c
            liou_fine[ia, itheta] = liou_f
            liou_coarse[ia, itheta] = liou_c

            completed += 1
            if completed % next_report == 0 or completed == total_jobs:
                pct = 100.0 * completed / total_jobs
                print(f"  progress: {completed:5d} / {total_jobs:5d} "
                      f"({pct:5.1f} %)")

    print("------------------------------------------------------------")
    print("Diagnostics for fine and coarse runs")
    print(f"|Z_fine| min / max   = {np.abs(Z_fine).min():.3e} / "
          f"{np.abs(Z_fine).max():.3e}")
    print(f"|Z_coarse| min / max = {np.abs(Z_coarse).min():.3e} / "
          f"{np.abs(Z_coarse).max():.3e}")
    print(f"KKT residual fine    : max = {kkt_res_fine.max():.3e}, "
          f"median = {np.median(kkt_res_fine):.3e}")
    print(f"KKT residual coarse  : max = {kkt_res_coarse.max():.3e}, "
          f"median = {np.median(kkt_res_coarse):.3e}")
    print(f"Liouville fine       : max = {liou_fine.max():.3e}, "
          f"median = {np.median(liou_fine):.3e}")
    print(f"Liouville coarse     : max = {liou_coarse.max():.3e}, "
          f"median = {np.median(liou_coarse):.3e}")
    print("------------------------------------------------------------")

    # Phase and modulus comparison
    ratio = Z_coarse / Z_fine
    phase_diff = np.angle(ratio)
    mod_ratio = np.abs(Z_coarse) / (np.abs(Z_fine) + 1e-30)

    print("Phase difference statistics (coarse vs fine)")
    print(f"  max |Δ phase| = {np.max(np.abs(phase_diff)):.3e} rad")
    print(f"  RMS Δ phase   = {np.sqrt(np.mean(phase_diff**2)):.3e} rad")
    print("Modulus ratio statistics (coarse vs fine)")
    print(f"  min |Z_c|/|Z_f| = {mod_ratio.min():.3e}")
    print(f"  max |Z_c|/|Z_f| = {mod_ratio.max():.3e}")
    print(f"  RMS(|Z_c|/|Z_f| - 1) = "
          f"{np.sqrt(np.mean((mod_ratio - 1.0)**2)):.3e}")
    print("------------------------------------------------------------")

    # Curvature maps
    print("Computing discrete Berry curvature for fine and coarse grids...")
    F_fine = np.zeros((Na - 1, Ntheta - 1), dtype=float)
    F_coarse = np.zeros_like(F_fine)

    for ia in range(Na - 1):
        for itheta in range(Ntheta - 1):
            z00_f = Z_fine[ia, itheta]
            z10_f = Z_fine[ia + 1, itheta]
            z11_f = Z_fine[ia + 1, itheta + 1]
            z01_f = Z_fine[ia, itheta + 1]
            F_fine[ia, itheta] = plaquette_phase(z00_f, z10_f, z11_f, z01_f)

            z00_c = Z_coarse[ia, itheta]
            z10_c = Z_coarse[ia + 1, itheta]
            z11_c = Z_coarse[ia + 1, itheta + 1]
            z01_c = Z_coarse[ia, itheta + 1]
            F_coarse[ia, itheta] = plaquette_phase(z00_c, z10_c, z11_c, z01_c)

    flux_fine = F_fine.sum()
    flux_coarse = F_coarse.sum()
    chern_fine = flux_fine / (2.0 * np.pi)
    chern_coarse = flux_coarse / (2.0 * np.pi)

    print("Berry curvature and Chern estimates")
    print(f"  fine:   total flux = {flux_fine:.6f} rad, "
          f"C ≈ {chern_fine:.6f}")
    print(f"  coarse: total flux = {flux_coarse:.6f} rad, "
          f"C ≈ {chern_coarse:.6f}")
    print(f"  Δ flux = {flux_coarse - flux_fine:.3e} rad")
    print(f"  Δ Chern = {chern_coarse - chern_fine:.3e}")
    print("------------------------------------------------------------")

    # Save data
    np.savez(
        os.path.join(outdir, "holonomy_cg_data.npz"),
        a_grid=a_grid,
        theta_grid=theta_grid,
        Z_fine=Z_fine,
        Z_coarse=Z_coarse,
        phase_diff=phase_diff,
        mod_ratio=mod_ratio,
        F_fine=F_fine,
        F_coarse=F_coarse,
        kkt_it_fine=kkt_it_fine,
        kkt_it_coarse=kkt_it_coarse,
        kkt_res_fine=kkt_res_fine,
        kkt_res_coarse=kkt_res_coarse,
        liou_fine=liou_fine,
        liou_coarse=liou_coarse,
        L=L,
        N=N,
        rho0_value=rho0_value,
        rho_bias=rho_bias,
        cJ=cJ,
        mode_rho=mode_rho,
        ell_cg=ell_cg,
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
        "ell_cg": ell_cg,
        "kkt_tol": kkt_tol,
        "kkt_maxit": kkt_maxit,
        "mix_G": mix_G,
        "mix_J": mix_J,
        "max_workers": max_workers,
        "Z_fine_min_abs": float(np.abs(Z_fine).min()),
        "Z_fine_max_abs": float(np.abs(Z_fine).max()),
        "Z_coarse_min_abs": float(np.abs(Z_coarse).min()),
        "Z_coarse_max_abs": float(np.abs(Z_coarse).max()),
        "phase_diff_max_abs": float(np.max(np.abs(phase_diff))),
        "phase_diff_rms": float(np.sqrt(np.mean(phase_diff**2))),
        "mod_ratio_min": float(mod_ratio.min()),
        "mod_ratio_max": float(mod_ratio.max()),
        "mod_ratio_rms_minus1": float(
            np.sqrt(np.mean((mod_ratio - 1.0)**2))
        ),
        "flux_fine": float(flux_fine),
        "flux_coarse": float(flux_coarse),
        "chern_fine": float(chern_fine),
        "chern_coarse": float(chern_coarse),
    }])
    meta_path = os.path.join(outdir, "holonomy_cg_meta.csv")
    meta.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path}")
    print(f"Wrote {os.path.join(outdir, 'holonomy_cg_data.npz')}")

    # Plots
    A_mid = 0.5 * (a_grid[:-1] + a_grid[1:])
    Theta_mid = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    extent_F = [Theta_mid[0], Theta_mid[-1], A_mid[0], A_mid[-1]]

    # Phase difference map
    plt.figure(figsize=(6, 4))
    extent = [theta_grid[0], theta_grid[-1], a_grid[0], a_grid[-1]]
    im = plt.imshow(
        phase_diff,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.colorbar(im, label="Δ phase (coarse - fine) [rad]")
    plt.xlabel("theta")
    plt.ylabel("a")
    plt.title("Phase difference of complex pairing under coarse graining")
    plt.tight_layout()
    fig_phase = os.path.join(outdir, "holonomy_cg_phase_diff.png")
    plt.savefig(fig_phase, dpi=180)
    print(f"Wrote {fig_phase}")

    # Curvature difference map
    plt.figure(figsize=(6, 4))
    im2 = plt.imshow(
        F_coarse - F_fine,
        origin="lower",
        extent=extent_F,
        aspect="auto",
    )
    plt.colorbar(im2, label="Δ Berry flux per plaquette [rad]")
    plt.xlabel("theta")
    plt.ylabel("a")
    plt.title("Curvature difference (coarse - fine)")
    plt.tight_layout()
    fig_curv = os.path.join(outdir, "holonomy_cg_curvature_diff.png")
    plt.savefig(fig_curv, dpi=180)
    print(f"Wrote {fig_curv}")

    t1 = time.time()
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
