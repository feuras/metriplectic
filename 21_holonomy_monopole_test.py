"""
21_holonomy_monopole_test.py

Synthetic monopole / vortex test for the holonomy machinery.

This script defines a *known* complex field Z(λ1, λ2) on a 2D control grid:

    Z(λ1,λ2) = (λ1 + i λ2) * exp(-alpha * (λ1^2 + λ2^2)),

which has:
  - a single zero at (λ1, λ2) = (0, 0),
  - a phase that winds by +2π once around that point.

We then:
  1. Compute |Z| and its phase on a regular grid in (λ1, λ2).
  2. Build link variables along λ1 and λ2 directions:
         U_x(i,j) = Z_{i+1,j} / Z_{i,j} / |Z_{i+1,j} / Z_{i,j}|
         U_y(i,j) = Z_{i,j+1} / Z_{i,j} / |Z_{i,j+1} / Z_{i,j}|
  3. Compute discrete Berry curvature on plaquettes:
         F(i,j) = arg( U_x(i,j) * U_y(i+1,j)
                        * conj(U_x(i,j+1)) * conj(U_y(i,j)) )
  4. Sum F over all plaquettes to obtain a total flux Φ and
         Chern estimate C = Φ / (2π).
  5. Compute a simple loop winding number by sampling Z along a
     circle around the origin and summing phase differences.

We expect:
  - A Chern number C ≈ +1,
  - Loop winding ≈ +1,
  - Clear curvature concentrated near the origin in the heatmap.

This acts as a synthetic "unit test" for the holonomy / Chern machinery
used in the metriplectic scripts (17–20): it shows that the algorithm
correctly detects a known, clean monopole when one is present.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Control-space grid limits (λ1, λ2)
L_ctrl = 1.0         # half-width of the square domain [-L_ctrl, L_ctrl]
N1 = 128             # number of points in λ1
N2 = 128             # number of points in λ2

# Gaussian damping for Z to keep |Z| localised and well behaved
alpha = 1.0

# Loop radius for winding test (must be < L_ctrl)
loop_radius = 0.7 * L_ctrl
loop_points = 512    # number of points on the loop

# Output directory
outdir = "out"
os.makedirs(outdir, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def build_control_grid():
    """
    Build a regular 2D control grid (lambda1, lambda2) over a square domain.
    """
    lam1 = np.linspace(-L_ctrl, L_ctrl, N1)
    lam2 = np.linspace(-L_ctrl, L_ctrl, N2)
    L1, L2 = np.meshgrid(lam1, lam2, indexing="ij")
    return lam1, lam2, L1, L2


def synthetic_Z(L1, L2, alpha):
    """
    Build the synthetic vortex / monopole:
        Z = (λ1 + i λ2) * exp(-alpha * (λ1^2 + λ2^2)).
    """
    r2 = L1**2 + L2**2
    Z = (L1 + 1j * L2) * np.exp(-alpha * r2)
    return Z


def compute_link_variables(Z):
    """
    Compute link variables U_x and U_y on the control grid.

    U_x(i,j) = Z_{i+1,j} / Z_{i,j} / |Z_{i+1,j} / Z_{i,j}|
    U_y(i,j) = Z_{i,j+1} / Z_{i,j} / |Z_{i,j+1} / Z_{i,j}|

    We avoid division by zero by flooring very small |Z|.
    """
    # Small floor to avoid division by zero
    eps = 1e-30
    Z_safe = np.where(np.abs(Z) < eps, eps, Z)

    # Links in λ1 direction: shape (N1-1, N2)
    ratio_x = Z_safe[1:, :] / Z_safe[:-1, :]
    U_x = ratio_x / np.abs(ratio_x)

    # Links in λ2 direction: shape (N1, N2-1)
    ratio_y = Z_safe[:, 1:] / Z_safe[:, :-1]
    U_y = ratio_y / np.abs(ratio_y)

    return U_x, U_y


def compute_plaquette_curvature(U_x, U_y):
    """
    Compute discrete Berry curvature on each plaquette using link variables.

    U_x has shape (N1-1, N2)  : links in +λ1 direction
    U_y has shape (N1,   N2-1): links in +λ2 direction

    We define plaquettes on the (N1-1) x (N2-1) grid:

        F(i,j) = arg( U_x(i,j) * U_y(i+1,j)
                      * conj(U_x(i,j+1)) * conj(U_y(i,j)) )

    with i = 0,...,N1-2, j = 0,...,N2-2.
    """
    # U_x: (N1-1, N2)
    # U_y: (N1,   N2-1)

    # All these slices have shape (N1-1, N2-1)
    U_x_ij   = U_x[:, :-1]       # link along +λ1 at (i,j)
    U_x_i_j1 = U_x[:, 1:]        # link along +λ1 at (i,j+1)

    U_y_ij   = U_y[:-1, :]       # link along +λ2 at (i,j)
    U_y_i1_j = U_y[1:, :]        # link along +λ2 at (i+1,j)

    plaquette = U_x_ij * U_y_i1_j * np.conjugate(U_x_i_j1) * np.conjugate(U_y_ij)
    F = np.angle(plaquette)      # shape (N1-1, N2-1)

    return F



def loop_winding(Z_loop):
    """
    Compute the winding number around a closed loop from complex samples Z_loop.

    Uses the sum of phase differences:
        n = (1 / 2π) * sum_j arg(Z_{j+1} / Z_j).
    """
    total_phase = 0.0
    eps = 1e-30
    for j in range(len(Z_loop)):
        z0 = Z_loop[j]
        z1 = Z_loop[(j + 1) % len(Z_loop)]
        if abs(z0) < eps:
            z0 = eps
        if abs(z1) < eps:
            z1 = eps
        total_phase += np.angle(z1 / z0)
    n_est = total_phase / (2.0 * np.pi)
    return n_est, total_phase


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("============================================================")
    print("21_holonomy_monopole_test.py")
    print("Synthetic monopole / vortex test for Berry curvature and Chern")
    print("------------------------------------------------------------")
    print(f"Control domain: λ1, λ2 ∈ [-{L_ctrl:.3f}, {L_ctrl:.3f}]")
    print(f"Grid size: N1 = {N1}, N2 = {N2}")
    print(f"Gaussian damping alpha = {alpha:.3f}")
    print("============================================================")

    # Build grid and synthetic Z
    lam1, lam2, L1, L2 = build_control_grid()
    Z = synthetic_Z(L1, L2, alpha)
    Z_abs = np.abs(Z)
    print("Basic diagnostics for synthetic Z:")
    print(f"  |Z| min / max = {Z_abs.min():.3e} / {Z_abs.max():.3e}")
    print("  Zero of Z is at (λ1, λ2) = (0, 0) analytically;")
    print("  grid samples approach this zero near the centre.")
    print("------------------------------------------------------------")

    # Link variables and plaquette curvature
    print("Computing link variables and plaquette curvature...")
    U_x, U_y = compute_link_variables(Z)
    F = compute_plaquette_curvature(U_x, U_y)

    # Total flux and Chern estimate
    flux_total = np.sum(F)
    Chern_est = flux_total / (2.0 * np.pi)
    Chern_rounded = int(np.rint(Chern_est))

    print("Berry curvature and Chern number estimate:")
    print(f"  total flux ∑_plaquettes F = {flux_total:.6f} rad")
    print(f"  Chern estimate           C ≈ {Chern_est:.6f}")
    print(f"  Chern rounded            C = {Chern_rounded:d}")
    print("------------------------------------------------------------")

    # Loop winding test around a circle enclosing the origin
    print("Computing loop winding around a circle enclosing the origin...")
    theta_loop = np.linspace(0.0, 2.0 * np.pi, loop_points, endpoint=False)
    lam1_loop = loop_radius * np.cos(theta_loop)
    lam2_loop = loop_radius * np.sin(theta_loop)

    # Interpolate Z on the loop via nearest-neighbour indexing
    # (sufficient for a topological test)
    idx1 = np.clip(((lam1_loop - lam1[0]) / (lam1[1] - lam1[0])).astype(int), 0, N1 - 1)
    idx2 = np.clip(((lam2_loop - lam2[0]) / (lam2[1] - lam2[0])).astype(int), 0, N2 - 1)
    Z_loop = Z[idx1, idx2]

    n_est, total_phase_loop = loop_winding(Z_loop)
    n_rounded = int(np.rint(n_est))
    print(f"  Loop radius                = {loop_radius:.3f}")
    print(f"  Loop total phase           = {total_phase_loop:.6f} rad")
    print(f"  Winding estimate  n ≈      = {n_est:.6f}")
    print(f"  Winding rounded   n =      = {n_rounded:d}")
    print("------------------------------------------------------------")

    # Save data
    npz_path = os.path.join(outdir, "monopole_test_data.npz")
    np.savez(
        npz_path,
        lam1=lam1,
        lam2=lam2,
        Z=Z,
        F=F,
        flux_total=flux_total,
        Chern_est=Chern_est,
        Chern_rounded=Chern_rounded,
        loop_radius=loop_radius,
        theta_loop=theta_loop,
        Z_loop=Z_loop,
        total_phase_loop=total_phase_loop,
        n_est=n_est,
        n_rounded=n_rounded,
    )

    meta = pd.DataFrame(
        [
            {
                "L_ctrl": L_ctrl,
                "N1": N1,
                "N2": N2,
                "alpha": alpha,
                "flux_total": float(flux_total),
                "Chern_est": float(Chern_est),
                "Chern_rounded": int(Chern_rounded),
                "loop_radius": float(loop_radius),
                "total_phase_loop": float(total_phase_loop),
                "n_est": float(n_est),
                "n_rounded": int(n_rounded),
            }
        ]
    )
    csv_path = os.path.join(outdir, "monopole_test_meta.csv")
    meta.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")

    # Simple curvature heatmap
    plt.figure(figsize=(5, 4))
    # Plaquette centres for plotting
    lam1_c = 0.5 * (lam1[:-1] + lam1[1:])
    lam2_c = 0.5 * (lam2[:-1] + lam2[1:])
    L1_c, L2_c = np.meshgrid(lam1_c, lam2_c, indexing="ij")
    im = plt.pcolormesh(L1_c, L2_c, F, shading="auto")
    plt.colorbar(im, label="Berry curvature F (rad)")
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel(r"$\lambda_2$")
    plt.title("Synthetic monopole curvature map")
    plt.tight_layout()
    png_path = os.path.join(outdir, "monopole_test_curvature_map.png")
    plt.savefig(png_path, dpi=180)
    plt.close()
    print(f"Wrote {png_path}")

    t1 = time.time()
    print("============================================================")
    print(f"Done. Total wall time: {t1 - t0:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()
