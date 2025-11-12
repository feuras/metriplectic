"""
Structure falsifier (multithreaded + PCG):
pulling G outside the divergence must break the equality dial.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import Grid, proj_field, mu_entropic, dx, Hm1G_energy, entropy_production

# -------------------------
# Config (matches your failing run)
# -------------------------
L = 40.0
N = 4096
lam2 = 0.10
alphas = [0.0, 0.2, 0.4, 0.6, 0.8]
max_iter = 2000
solver_tol = 1e-12
report_tol = 2e-4
workers = min(20, os.cpu_count() or 4)

# Numerical safeguards
eps_prec = 1e-14       # for preconditioner division safety
eps_mass = 0.0         # tiny mass stabiliser; set 1e-12 if your FFT implementation needs it

os.makedirs("out", exist_ok=True)

# -------------------------
# Grid and base state
# -------------------------
g = Grid(L=L, N=N)
x, k, mask = g.build()
rho0 = 1.0 / g.L

# smooth positive density, projected to the working subspace and renormalised
rho = rho0 * (1 + 0.2 * np.cos(2*np.pi*x/g.L))
rho = proj_field(rho, mask)
rho = rho - rho.mean() + rho0

# -------------------------
# Helpers: projection and integrations
# -------------------------
def mean_zero(u):
    return u - np.mean(u)

def trapz1d(y):
    return np.trapezoid(y, x)

# -------------------------
# Operators
# -------------------------
def L_correct(rho, G, phi, k, mask):
    """-∂x ( rho * G * ∂x phi ) + eps_mass * rho*G*phi (optional)"""
    core = -dx(rho * G * dx(phi, k, mask), k, mask)
    if eps_mass != 0.0:
        core = core + eps_mass * (rho * G * phi)
    return core

def L_wrong(rho, G, phi, k, mask):
    """- G * ∂x ( rho * ∂x phi )  [intentionally wrong]  + same mass term if enabled"""
    core = -G * dx(rho * dx(phi, k, mask), k, mask)
    if eps_mass != 0.0:
        core = core + eps_mass * (rho * G * phi)
    return core

# -------------------------
# PCG solver (symmetric SPD on mean-zero subspace)
# -------------------------
def pcg(op, b, rho, G, k, mask, tol=1e-12, itmax=2000):
    """
    Preconditioned Conjugate Gradient:
    - Projects RHS and iterates to mean-zero each step to remove nullspace.
    - Jacobi-like preconditioner M ≈ rho*G (positive, simple, cheap).
    """
    # Ensure RHS has zero mean (it should if it is a divergence)
    b = mean_zero(b)

    def M_inv(z):
        return z / (rho * G + eps_prec)

    phi = np.zeros_like(b)
    r = b - mean_zero(op(rho, G, phi, k, mask))
    z = M_inv(r)
    p = z.copy()
    rz_old = np.dot(r, z)
    it = 0

    # early exit if already tiny
    if np.linalg.norm(r) <= tol:
        return phi, it, float(np.linalg.norm(r))

    for it in range(1, itmax + 1):
        Ap = mean_zero(op(rho, G, p, k, mask))
        denom = np.dot(p, Ap) + 1e-30
        alpha = rz_old / denom
        phi = mean_zero(phi + alpha * p)
        r = mean_zero(r - alpha * Ap)

        nr = np.linalg.norm(r)
        if nr <= tol:
            break

        z = M_inv(r)
        rz_new = np.dot(r, z)
        beta = rz_new / (rz_old + 1e-30)
        p = mean_zero(z + beta * p)
        rz_old = rz_new

    res = float(np.linalg.norm(r))
    return phi, it, res

# -------------------------
# Worker
# -------------------------
def run_alpha(a: float):
    """Compute R_true and R_wrong for a single alpha using PCG. Returns a dict row and a console line."""
    # metric field, projected
    G = 1.0 + a * np.cos(2*np.pi*x/g.L)
    G = proj_field(G, mask)

    # chemical potential and irreversible velocity
    mu = mu_entropic(rho, lam2, k, mask)
    vG = -dx(rho * G * dx(mu, k, mask), k, mask)
    vG = mean_zero(vG)  # guard against any tiny DC leakage

    # solve KKT with correct operator
    phi_true, it_true, res_true = pcg(L_correct, vG, rho, G, k, mask,
                                      tol=solver_tol, itmax=max_iter)

    # solve KKT with wrong operator
    phi_wrong, it_wrong, res_wrong = pcg(L_wrong, vG, rho, G, k, mask,
                                         tol=solver_tol, itmax=max_iter)

    # dial components for correct operator
    twoC_true = Hm1G_energy(rho, G, phi_true, k, mask, x)
    sig_true  = entropy_production(rho, G, mu, k, mask, x)
    num_true  = trapz1d(rho * G * dx(phi_true, k, mask) * dx(mu, k, mask))
    R_true    = (num_true**2) / (twoC_true * sig_true + 1e-30)

    # dial components for wrong operator
    twoC_wrong = Hm1G_energy(rho, G, phi_wrong, k, mask, x)
    num_wrong  = trapz1d(rho * G * dx(phi_wrong, k, mask) * dx(mu, k, mask))
    R_wrong    = (num_wrong**2) / (twoC_wrong * sig_true + 1e-30)

    row = {
        "alpha": a,
        "iters_true": it_true,
        "res_true": res_true,
        "R_true": R_true,
        "iters_wrong": it_wrong,
        "res_wrong": res_wrong,
        "R_wrong": R_wrong
    }
    line = f"{a:7.2f} | {it_true:10d} | {res_true:9.2e} | {R_true:10.8f} || {it_wrong:11d} | {res_wrong:9.2e} | {R_wrong:10.8f}"
    return a, row, line

# -------------------------
# Run (multithreaded sweep)
# -------------------------
print("Structure falsifier configuration")
print(f"L={L}, N={N}, rho0={rho0}")
print(f"alphas={alphas}")
print(f"max_iter={max_iter}, solver_tol={solver_tol}, report_tol={report_tol}, threads={workers}")
print(f"eps_prec={eps_prec}, eps_mass={eps_mass}")
print("All derivatives and fields are projected with the same mask; RHS and iterates are mean-zero.")
print("--------------------------------------------------------------------------")
print(f"{'alpha':>7} | {'iters_true':>10} | {'res_true':>9} | {'R_true':>10} || {'iters_wrong':>11} | {'res_wrong':>9} | {'R_wrong':>10}")

rows, lines = [], {}

with ThreadPoolExecutor(max_workers=workers) as ex:
    futures = {ex.submit(run_alpha, a): a for a in alphas}
    for fut in as_completed(futures):
        a, row, line = fut.result()
        rows.append(row)
        lines[a] = line

# Deterministic console readout in ascending alpha
rows = sorted(rows, key=lambda r: r["alpha"])
for a in sorted(lines.keys()):
    print(lines[a])

# -------------------------
# Save, plot, and assertions
# -------------------------
df = pd.DataFrame(rows)
df.to_csv("out/structure_falsifier.csv", index=False)

# Assertions: equality on the irreversible ray for correct operator (R_true ~ 1)
max_eq_gap = float(np.max(np.abs(df["R_true"].values - 1.0)))
assert max_eq_gap <= report_tol, f"Equality dial failed for L_correct: max |1-R_true| = {max_eq_gap:.3e} > {report_tol:.1e}"

# Assertions for the wrong operator:
# 1) strictly decreasing in alpha (allowing tiny jitter)
# 2) strictly below the alpha=0 baseline for all alpha>0
# 3) achieves a meaningful drop by the largest alpha (e.g. ≥5%)
rw = df["R_wrong"].values
a  = df["alpha"].values
decreasing = np.all(np.diff(rw) <= 1e-12)  # monotone non-increasing
assert decreasing, "R_wrong is not monotonically decreasing in alpha."

r0 = float(rw[a == 0.0][0]) if np.any(a == 0.0) else 1.0
strict_drop = np.all(rw[a > 0.0] < r0 - 1e-6)
assert strict_drop, "R_wrong(α>0) is not strictly below the α=0 baseline."

min_required_drop = 0.05  # 5% by the largest alpha; adjust if you widen the sweep
achieves_drop = (r0 - float(rw[a.argmax()])) >= min_required_drop
assert achieves_drop, f"R_wrong did not drop by at least {min_required_drop*100:.1f}% at largest α."


# Plots
plt.figure()
plt.plot(df["alpha"], df["R_true"], marker="o", label="R true")
plt.plot(df["alpha"], df["R_wrong"], marker="x", linestyle="--", label="R wrong")
plt.xlabel("alpha")
plt.ylabel("R")
plt.title("Equality dial: correct vs wrong operator (PCG)")
plt.legend()
plt.tight_layout()
plt.savefig("out/fig_structure_falsifier.png", dpi=180)

print("OK. Wrote out/structure_falsifier.csv and out/fig_structure_falsifier.png")
print(f"Max |1-R_true| = {max_eq_gap:.3e}")
