"""
04_diffeo_slice.py — Diffeomorphic slice covariance (stable Euclidean-CG path)
+ Family sweep (k=1 and k=2 diffeos)
+ Refinement sweep (N in {256, 512, 1024} at eps=0.15, k=1)

Design (matches your original stable code):
- Pull rho(y)/jac and G(y) to the x grid, then renormalise rho to have x-mean = rho0
- Solve A phi = vG with standard CG in the Euclidean inner product
- Evaluate all diagnostics in the y-measure (weight = jac)
- Use the same de-aliased mask for all derivatives

Outputs
- out/diffeo_slice.csv           : main suite across eps and both diffeo families
- out/diffeo_slice_gaps_family.png
- out/diffeo_slice_refine.csv    : refinement at eps=0.15, N={256,512,1024}, family k=1
- out/diffeo_slice_refine.png

Asserts
- CG residual ≤ tol_res
- R → 1 within tol_R
- M → 1 within tol_M (Hilbert-proxy floor at N=512)
- Energy equalities (Euclidean-solve vs y-measure diagnostics): ≤ tol_energy
- Gradient proxy at floor ≤ tol_grad
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import Grid, proj_field, dx, dxx, hilbert

# -------------------------
# Global config
# -------------------------
L = 40.0
N = 512
rho0 = 1.0 / L
lam2 = 0.10
eps_list = [0.0, 0.10, 0.15, 0.20]
families = [
    {"name": "k1", "k": 1},   # y = x + eps * sin(2πx/L)
    {"name": "k2", "k": 2},   # y = x + eps * sin(4πx/L)
]

# Tolerances aligned to your successful run (Windows FFT/BLAS)
tol_R = 1e-12
tol_M = 1e-7
tol_res = 6e-10
tol_energy = 4e-4
tol_grad = 1e-10

maxit = 16000

os.makedirs("out", exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def trap_int(f, x):
    return np.trapezoid(f, x)

def inner_w(a, b, jac, x):
    return float(trap_int(a * b * jac, x))

def cg_euclidean(Aop, rhs, maxit=4000, rtol=1e-12):
    """Standard CG in Euclidean inner product."""
    phi = np.zeros_like(rhs)
    r = rhs - Aop(phi)
    p = r.copy()
    rz = float(np.dot(r, r))
    rhsn = np.linalg.norm(rhs) + 1e-30
    res = np.linalg.norm(r) / rhsn
    it = 0
    while it < maxit and res > rtol:
        Ap = Aop(p)
        denom = float(np.dot(p, Ap)) + 1e-30
        alpha = rz / denom
        phi = phi + alpha * p
        r = r - alpha * Ap
        rn = float(np.dot(r, r))
        res = np.sqrt(rn) / rhsn
        beta = rn / (rz + 1e-30)
        p = r + beta * p
        rz = rn
        it += 1
    # final residual
    true_res = np.linalg.norm(rhs - Aop(phi)) / (np.linalg.norm(rhs) + 1e-30)
    return phi, it, true_res

def diffeo_y(x, L, eps, kfam):
    """y(x) = x + eps * sin(2π kfam x / L); jac = dy/dx."""
    arg = 2.0 * np.pi * kfam * x / L
    y = x + eps * np.sin(arg)
    jac = 1.0 + eps * (2.0 * np.pi * kfam / L) * np.cos(arg)
    invj = 1.0 / np.maximum(jac, 1e-12)
    return y, jac, invj

def run_case(L, N, rho0, lam2, eps, kfam):
    """Single run for given grid, epsilon, diffeo family."""
    g = Grid(L=L, N=N)
    x, k, mask = g.build()

    # Diffeo and Jacobian
    y, jac, invj = diffeo_y(x, L, eps, kfam)

    # Smooth fields in y, pull to x, project
    rho_y = lambda yy: rho0 * (1 + 0.2 * np.cos(2 * np.pi * yy / L))
    G_y   = lambda yy: 1.0 + 0.3 * np.cos(2 * np.pi * yy / L)

    rho = proj_field(rho_y(y) / jac, mask)
    # Match original stable path: set x-mean to rho0 (not y-mass = 1)
    rho = rho - rho.mean() + rho0

    G = proj_field(G_y(y), mask)

    # y-derivatives expressed on x grid
    def dY(f):  return invj * dx(f, k, mask)
    def dYY(f): return invj * dx(invj * dx(f, k, mask), k, mask)

    # Chemical potential and irreversible velocity
    mu  = np.log(rho) - lam2 * dYY(rho)
    vG  = -invj * dx(rho * G * invj * dx(mu, k, mask), k, mask)

    # Linear operator
    def A(phi):
        return -invj * dx(rho * G * invj * dx(phi, k, mask), k, mask)

    # Solve A phi = vG with Euclidean CG
    phi, it, res = cg_euclidean(A, vG, maxit=maxit, rtol=tol_res)

    # Diagnostics in y measure
    dphi = dY(phi)
    dmu  = dY(mu)

    twoC = inner_w(rho * G * dphi * dphi, 1.0, jac, x)
    sig  = inner_w(rho * G * dmu  * dmu , 1.0, jac, x)
    Re   = inner_w(rho * G * dphi * dmu, 1.0, jac, x)
    Im   = inner_w(rho * dphi * hilbert(dmu, k, mask), 1.0, jac, x)

    R  = (Re**2) / (twoC * sig + 1e-30)
    MR = (Re**2 + Im**2) / (twoC * sig + 1e-30)

    # Energy equalities and gradient proxy
    v_mu = inner_w(vG, mu, jac, x)
    gap_2C = abs(v_mu - twoC)
    gap_sig = abs(v_mu - sig)
    grad_err = np.sqrt(inner_w((dphi - dmu) * (dphi - dmu), 1.0, jac, x)) / (
        np.sqrt(inner_w(dmu * dmu, 1.0, jac, x)) + 1e-16
    )

    # Console readout
    print(f"eps={eps:.3f} | family={kfam} | N={N:4d} | iters={it:4d} | CG_res={res:.2e}")
    print(f"          R={R:.12f}  M={MR:.12f}  |  gaps: dR={abs(1.0-R):.2e}, dM={abs(1.0-MR):.2e}")
    print(f"          energy: |<vG,mu>-2C|={gap_2C:.2e}  |<vG,mu>-sigma|={gap_sig:.2e}  | grad_rel_err={grad_err:.2e}")

    # Assertions
    assert res <= tol_res,       f"CG residual too high: {res:.3e} > {tol_res:.1e}"
    assert abs(1.0 - R)  <= tol_R,  f"R departed from 1: {R}"
    assert abs(1.0 - MR) <= tol_M,  f"M departed from 1: {MR}"
    assert gap_2C        <= tol_energy, f"Energy gap <vG,mu>-2C too large: {gap_2C:.3e}"
    assert gap_sig       <= tol_energy, f"Energy gap <vG,mu>-sigma too large: {gap_sig:.3e}"
    assert grad_err      <= tol_grad,   f"Gradient proxy not at floor: {grad_err:.3e}"

    return {
        "epsilon": eps,
        "family": f"k{kfam}",
        "N": N,
        "R": R,
        "M": MR,
        "iters": it,
        "residual": res,
        "energy_gap_vmu_2C": gap_2C,
        "energy_gap_vmu_sigma": gap_sig,
        "grad_rel_err": grad_err
    }

# -------------------------
# Main suite: families x eps (at N=512)
# -------------------------
print("Diffeomorphic slice covariance check (families sweep)")
print(f"L={L}, N={N}, rho0={rho0}")
print(f"lam2={lam2}, eps_list={eps_list}, families={[f['name'] for f in families]}")
print("All derivative ops use the same de-aliased mask.")
print("----------------------------------------------------")

rows = []
for fam in families:
    kfam = fam["k"]
    for eps in eps_list:
        rows.append(run_case(L, N, rho0, lam2, eps, kfam))

df = pd.DataFrame(rows)
df.to_csv("out/diffeo_slice.csv", index=False)

# Plot gaps vs epsilon for both families
plt.figure()
for fam in families:
    fam_name = fam["name"]
    sub = df[df["family"] == fam_name]
    plt.plot(sub["epsilon"], 1.0 - sub["R"], marker="o", label=f"1 - R ({fam_name})")
    plt.plot(sub["epsilon"], 1.0 - sub["M"], marker="x", linestyle="--", label=f"1 - M ({fam_name})")
plt.xlabel("epsilon")
plt.ylabel("gap to 1")
plt.title("Diffeomorphic slice invariance: dial gaps (families k=1,k=2)")
plt.legend()
plt.tight_layout()
plt.savefig("out/diffeo_slice_gaps_family.png", dpi=180)

# -------------------------
# Refinement sweep at eps=0.15, family k=1
# -------------------------
print("Refinement sweep at eps=0.15, family k=1")
ref_eps = 0.15
ref_Ns = [256, 512, 1024]
ref_rows = []
for Nref in ref_Ns:
    ref_rows.append(run_case(L, Nref, rho0, lam2, ref_eps, kfam=1))

df_ref = pd.DataFrame(ref_rows)
df_ref.to_csv("out/diffeo_slice_refine.csv", index=False)

# Plot refinement: 1 - M vs N (log-y), R is at machine floor so not informative
plt.figure()
plt.semilogy(df_ref["N"], 1.0 - df_ref["M"], marker="o", label="1 - M (refine)")
plt.xlabel("N")
plt.ylabel("1 - M")
plt.title("Refinement at eps=0.15 (family k=1)")
plt.grid(True, which="both", axis="y")
plt.tight_layout()
plt.savefig("out/diffeo_slice_refine.png", dpi=180)

print("OK. CSVs and figures written in out/")
