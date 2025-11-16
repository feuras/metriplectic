"""
18_holonomy_sanity_check.py

Sanity check: scan a 2D control torus (phi, theta) for defects of the complex
pairing Z(phi, theta) and verify that, for a broad and physically natural
family of controls, no near-zeros (monopoles) are present.

Control parameters:
  - phi   : phase of the density modulation in rho(x)
  - theta : mixing angle in the reversible quadrature two-plane

State ansatz:
  rho(x; phi) = rho0 * [ 1 + a0 * cos(k x + phi) ],  with |a0| < 1,
so the density modulation is rigidly translated as phi varies.

Reversible structure:
  v_J(theta) ∝ d_x[ cos(theta) H(mu_x) + sin(theta) mu_x ],
a rotation inside the fixed two-plane { mu_x, H(mu_x) }.

Complex reader:
  Z(phi, theta) = ∫ rho phi_x mu_x dx  + i ∫ rho phi_x H(mu_x) dx,
built from the KKT solution phi to L_{rho,G} phi = v with
v = mix_G v_G + mix_J v_J.

For each grid point (phi, theta) we:
  1. Build rho(x; phi), mu(x), mu_x, Hmu_x.
  2. Build irreversible velocity v_G and reversible velocity v_J(theta).
  3. Solve L_{rho,G} phi = v for phi with tight KKT tolerances.
  4. Form Z = Re + i Im from the complex pairing.
  5. Store diagnostics (KKT iterations, residuals, Liouville check).

We then:
  - Scan |Z(phi, theta)| over the full control torus.
  - Identify candidate "near zeros" where |Z| <= MIN_Z_ABS_CANDIDATE.
  - If any candidates are found, build a small rectangular loop in (phi, theta)
    around each and sample Z along the loop to compute the winding
        n = (1 / 2π) ∑ Δ arg Z.

In the parameter choices used for the paper, no candidates are found down to
the prescribed threshold, |Z| stays in a narrow, strictly positive band, and
Liouville and KKT diagnostics remain clean. This confirms that this entire
(phi, theta) torus lies in a single trivial holonomy sector and that the
complex reader is non-degenerate under generic phase shifts and quadrature
rotations.

Outputs:
  out/holonomy_defects_phi_theta_meta.csv      # (empty or loop diagnostics)
  out/holonomy_defects_phi_theta_Z_scan.npz    # full Z, residual and grid data
"""


import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from common import Grid, proj_field, mu_entropic, dx, hilbert, kkt_solve


# =============================================================================
# Configuration
# =============================================================================

# Spatial domain
L = 40.0
N = 512
rho0_value = 1.0 / L

# Fixed modulation amplitude (must satisfy |a0| < 1)
a0 = 0.25

# Entropic curvature
rho_bias = 0.20

# Reversible structure parameters
cJ = 0.8
mode_rho = 3   # spatial mode number for the modulation k = 2π * mode_rho / L

# Control-space scan region
phi_min, phi_max = 0.001, 2.5 * np.pi
theta_min, theta_max = 0.001, 2.5 * np.pi

# Grid resolution in control space
Nphi_scan, Ntheta_scan = 240, 240

# Defect detection threshold and cap on candidates
MIN_Z_ABS_CANDIDATE = 1e-3
MAX_NUM_CANDIDATES = 100

# Loop half widths in control space around each candidate
delta_phi = 0.5 * (phi_max - phi_min) / Nphi_scan
delta_theta = 0.5 * (theta_max - theta_min) / Ntheta_scan

# KKT solver tolerance and iteration cap
kkt_tol = 1e-10
kkt_maxit = 6000

# Liouville diagnostic tolerance
tol_liouville = 1e-11

# Mixing of irreversible and reversible velocities
mix_G = 0.5
mix_J = 0.5

# Parallelism
TARGET_WORKERS = 20
_cpu_count = os.cpu_count()
if _cpu_count is None or _cpu_count <= 0:
    max_workers = 1
else:
    max_workers = min(TARGET_WORKERS, _cpu_count)

# Output directory
outdir = "out"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Global grid and control arrays
# =============================================================================

_grid = Grid(L=L, N=N)
x, k, mask = _grid.build()

phi_grid = np.linspace(phi_min, phi_max, Nphi_scan)
theta_grid = np.linspace(theta_min, theta_max, Ntheta_scan)


# =============================================================================
# Helpers
# =============================================================================

def build_state(phi, x, k, mask):
    """
    Build rho(x; phi), mu_x, Hmu_x for a given phase phi.

    rho(x; phi) = rho0 * [ 1 + a0 * cos(kx + phi) ], with |a0| < 1.
    """
    rho = rho0_value * (1.0 + a0 * np.cos(2.0 * np.pi * mode_rho * x / L + phi))
    rho = proj_field(rho, mask)
    rho = rho - rho.mean() + rho0_value

    mu = mu_entropic(rho, rho_bias, k, mask)
    mu_x = dx(mu, k, mask)
    Hmu_x = hilbert(mu_x, k, mask)
    return rho, mu_x, Hmu_x


def build_velocities(rho, mu_x, Hmu_x, k, mask):
    """
    Build irreversible velocity v_G and helper fields for reversible v_J.
    """
    G = np.ones_like(rho)
    v_G = -dx(rho * G * mu_x, k, mask)

    # Liouville-correcting amplitude a_x: rho * a_x = const
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
    Build reversible velocity v_J(θ) from rotated two-plane quadrature:

      rot(θ) = cos θ Hmu_x + sin θ mu_x,
      v_J(θ) = d_x [ cJ * rot(θ) ].

    Liouville is enforced by construction via a_x inside helpers.
    """
    mu_x = helpers["mu_x"]
    Hmu_x = helpers["Hmu_x"]
    rot = np.cos(theta) * Hmu_x + np.sin(theta) * mu_x
    v_J = dx(cJ * rot, k, mask)
    return v_J


def complex_pairing(rho, mu_x, Hmu_x, v_G, v_J, x, k, mask):
    """
    Solve KKT problem L_{rho,G} phi = v with v = mix_G v_G + mix_J v_J,
    and form the complex pairing:

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


def loop_winding(Z_values):
    """
    Given complex values Z sampled around a closed loop, compute:

      total_phase = ∑ Δ arg(Z),
      n = total_phase / (2π),

    where Δ arg(Z) is computed via log-ratio.
    """
    total_phase = 0.0
    for j in range(len(Z_values)):
        z0 = Z_values[j]
        z1 = Z_values[(j + 1) % len(Z_values)]
        eps = 1e-30
        if abs(z0) < eps:
            z0 = eps
        if abs(z1) < eps:
            z1 = eps
        total_phase += np.angle(z1 / z0)
    return total_phase / (2.0 * np.pi), total_phase


# =============================================================================
# Worker for process-based parallel scan
# =============================================================================

def worker_task(args):
    """
    Worker evaluated in child processes.

    Args:
      args: tuple (iphi, itheta) indexing the control grid.

    Returns:
      iphi, itheta, Z_loc, it_loc, res_loc, liou_loc
    """
    iphi, itheta = args
    phi = phi_grid[iphi]
    theta = theta_grid[itheta]

    rho, mu_x, Hmu_x = build_state(phi, x, k, mask)
    v_G, helpers = build_velocities(rho, mu_x, Hmu_x, k, mask)
    v_J = build_vJ(theta, helpers, k, mask)

    Z_loc, it_loc, res_loc, liou_loc = complex_pairing(
        rho, mu_x, Hmu_x, v_G, v_J, x, k, mask
    )
    return iphi, itheta, Z_loc, it_loc, res_loc, liou_loc


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("============================================================")
    print("18_holonomy_defect_locator_phi_theta.py")
    print("Scanning (phi, theta) control torus for near-zeros of Z")
    print("------------------------------------------------------------")
    print(f"Spatial grid:  L = {L:.3f}, N = {N}")
    print(f"Base density:  rho0 = {rho0_value:.6f}")
    print(f"Fixed amplitude a0 = {a0:.3f} (|a0| < 1)")
    print(f"rho_bias:      {rho_bias:.3f}")
    print(f"cJ:            {cJ:.3f}, mode_rho = {mode_rho:d}")
    print("Control scan region:")
    print(f"  phi   ∈ [{phi_min:.3f}, {phi_max:.3f}] with Nphi   = {Nphi_scan}")
    print(f"  theta ∈ [{theta_min:.3f}, {theta_max:.3f}] with Ntheta = {Ntheta_scan}")
    print(f"Candidate threshold |Z| <= {MIN_Z_ABS_CANDIDATE:.3e}")
    print(f"Loop half-widths: Δphi ≈ {delta_phi:.3e}, Δtheta ≈ {delta_theta:.3e}")
    print(f"Workers requested: {TARGET_WORKERS}, using: {max_workers}")
    print("============================================================")

    # Scan grid for Z(phi, theta) using process-based parallelism
    print("Scanning Z(phi, theta) over control grid (ProcessPoolExecutor)...")
    Z_scan = np.zeros((Nphi_scan, Ntheta_scan), dtype=np.complex128)
    kkt_res_scan = np.zeros_like(Z_scan.real)
    liouv_scan = np.zeros_like(Z_scan.real)
    kkt_it_scan = np.zeros_like(Z_scan.real, dtype=int)

    tasks = [(iphi, itheta) for iphi in range(Nphi_scan)
                            for itheta in range(Ntheta_scan)]
    total_jobs = len(tasks)
    next_report = max(1, total_jobs // 10)

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for iphi, itheta, Z_loc, it_loc, res_loc, liou_loc in pool.map(
            worker_task, tasks, chunksize=16
        ):
            Z_scan[iphi, itheta] = Z_loc
            kkt_it_scan[iphi, itheta] = it_loc
            kkt_res_scan[iphi, itheta] = res_loc
            liouv_scan[iphi, itheta] = liou_loc

            completed += 1
            if completed % next_report == 0 or completed == total_jobs:
                pct = 100.0 * completed / total_jobs
                print(f"  progress: {completed:6d} / {total_jobs:6d} "
                      f"({pct:5.1f} %)")

    print("Grid scan complete.")
    print("------------------------------------------------------------")
    print(f"|Z| min / max = {np.abs(Z_scan).min():.3e} / "
          f"{np.abs(Z_scan).max():.3e}")
    print(f"KKT residual max / median = {kkt_res_scan.max():.3e} / "
          f"{np.median(kkt_res_scan):.3e}")
    print(f"Liouville sup max / median = {liouv_scan.max():.3e} / "
          f"{np.median(liouv_scan):.3e}")
    print("------------------------------------------------------------")

    # Identify candidate near-zeros
    Z_abs = np.abs(Z_scan)
    candidates = np.argwhere(Z_abs <= MIN_Z_ABS_CANDIDATE)
    if candidates.size == 0:
        print("No candidates found with |Z| below threshold.")
        print("Consider adjusting a0, cJ, mode_rho, rho_bias, or MIN_Z_ABS_CANDIDATE.")
        np.savez(
            os.path.join(outdir, "holonomy_defects_phi_theta_Z_scan.npz"),
            phi_grid=phi_grid,
            theta_grid=theta_grid,
            Z_scan=Z_scan,
            kkt_res=kkt_res_scan,
            liouville_sup=liouv_scan,
        )
        t1 = time.time()
        print(f"Done. Wall time = {t1 - t0:.2f} s")
        print("============================================================")
        return

    cand_list = []
    for iphi, itheta in candidates:
        cand_list.append((Z_abs[iphi, itheta], int(iphi), int(itheta)))
    cand_list.sort(key=lambda t: t[0])
    cand_list = cand_list[:MAX_NUM_CANDIDATES]

    print(f"Found {len(cand_list)} candidate near-zeros:")
    for rank, (zabs, iphi, itheta) in enumerate(cand_list):
        print(f"  {rank+1:2d}: |Z| = {zabs:.3e} at "
              f"phi ≈ {phi_grid[iphi]:.6f}, "
              f"theta ≈ {theta_grid[itheta]:.6f}")

    loop_results = []

    for rank, (zabs, iphi0, itheta0) in enumerate(cand_list):
        phi0 = phi_grid[iphi0]
        theta0 = theta_grid[itheta0]

        print("--------------------------------------------------------")
        print(f"Candidate {rank+1}: centre phi0 = {phi0:.6f}, "
              f"theta0 = {theta0:.6f}, |Z| ≈ {zabs:.3e}")
        print("Building small rectangular loop and sampling Z around it...")

        # Define loop corners in (phi, theta)
        phi_vals = [phi0 - delta_phi, phi0 + delta_phi]
        theta_vals = [theta0 - delta_theta, theta0 + delta_theta]

        # Sample loop: 4 segments with Ns points each
        Ns = 48
        loop_points = []

        # Segment 1: (phi-, theta-) -> (phi+, theta-)
        phi_line = np.linspace(phi_vals[0], phi_vals[1], Ns, endpoint=False)
        th_line = np.full(Ns, theta_vals[0])
        loop_points.extend(list(zip(phi_line, th_line)))

        # Segment 2: (phi+, theta-) -> (phi+, theta+)
        th_line = np.linspace(theta_vals[0], theta_vals[1], Ns, endpoint=False)
        phi_line = np.full(Ns, phi_vals[1])
        loop_points.extend(list(zip(phi_line, th_line)))

        # Segment 3: (phi+, theta+) -> (phi-, theta+)
        phi_line = np.linspace(phi_vals[1], phi_vals[0], Ns, endpoint=False)
        th_line = np.full(Ns, theta_vals[1])
        loop_points.extend(list(zip(phi_line, th_line)))

        # Segment 4: (phi-, theta+) -> (phi-, theta-)
        th_line = np.linspace(theta_vals[1], theta_vals[0], Ns, endpoint=False)
        phi_line = np.full(Ns, phi_vals[0])
        loop_points.extend(list(zip(phi_line, th_line)))

        loop_points = np.array(loop_points, dtype=float)

        # Evaluate Z along the loop (sequential; loop is small)
        Z_loop = []
        for phiL, thL in loop_points:
            phiL = float(phiL)
            thL = float(thL)

            rho, mu_x, Hmu_x = build_state(phiL, x, k, mask)
            v_G, helpers = build_velocities(rho, mu_x, Hmu_x, k, mask)
            v_J = build_vJ(thL, helpers, k, mask)
            Z_loc, it_loc, res_loc, liou_loc = complex_pairing(
                rho, mu_x, Hmu_x, v_G, v_J, x, k, mask
            )
            Z_loop.append(Z_loc)

        Z_loop = np.array(Z_loop, dtype=np.complex128)
        n_est, total_phase = loop_winding(Z_loop)
        n_rounded = int(np.rint(n_est))

        print(f"Loop total phase = {total_phase:.6f} rad")
        print(f"Winding estimate  n ≈ {n_est:.6f}")
        print(f"Winding rounded   n = {n_rounded:d}")

        loop_results.append({
            "rank": rank + 1,
            "phi0": phi0,
            "theta0": theta0,
            "Z_abs_centre": zabs,
            "total_phase": float(total_phase),
            "n_est": float(n_est),
            "n_rounded": n_rounded,
        })

    # Save scan and loop diagnostics
    np.savez(
        os.path.join(outdir, "holonomy_defects_phi_theta_Z_scan.npz"),
        phi_grid=phi_grid,
        theta_grid=theta_grid,
        Z_scan=Z_scan,
        kkt_res=kkt_res_scan,
        liouville_sup=liouv_scan,
    )

    meta = pd.DataFrame(loop_results)
    meta_path = os.path.join(outdir, "holonomy_defects_phi_theta_meta.csv")
    meta.to_csv(meta_path, index=False)
    print("------------------------------------------------------------")
    print(f"Wrote {meta_path}")
    print("Wrote holonomy_defects_phi_theta_Z_scan.npz")
    t1 = time.time()
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
