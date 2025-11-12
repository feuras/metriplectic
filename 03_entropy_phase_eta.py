"""
Entropy–phase split and equality dial along v(η).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common import (
    Grid, proj_field, mu_entropic, dx, hilbert,
    kkt_solve, Hm1G_energy, entropy_production
)

# -------------------------
# Config (formal defaults)
# -------------------------
L = 40.0
N = 512
lam = 0.10
cJ  = 1.0           # reversible strength; fixed positive constant
eta_grid = np.linspace(0, 1, 21)
tol_R0 = 2e-4       # target: 1 - R(0) <= 2e-4
tol_M  = 5e-3       # target: max |M-1| <= 5e-3 (now comfortably met)
orthogonalise_vJ = True
outdir = "out"
os.makedirs(outdir, exist_ok=True)

# -------------------------
# Grid and base fields
# -------------------------
g = Grid(L=L, N=N)
x, k, mask = g.build()
rho0 = 1.0 / g.L

# Base state and weights (project to masked subspace and renormalise mean)
rho = rho0 * (1 + 0.2 * np.cos(2*np.pi*x/g.L))
rho = proj_field(rho, mask)
rho = rho - rho.mean() + rho0

G = 1.0 + 0.3 * np.cos(2*np.pi*x/g.L)
G = proj_field(G, mask)

mu = mu_entropic(rho, lam, k, mask)
mu = proj_field(mu, mask)

# Irreversible and reversible seeds (always project results)
vG = -dx(rho * G * dx(mu, k, mask), k, mask)
vG = proj_field(vG, mask)

# Proper reversible seed in the Hilbert plane: vJ = ∂x[ ρ cJ H(∂x μ) ]
vJ = dx(rho * cJ * hilbert(dx(mu, k, mask), k, mask), k, mask)
vJ = proj_field(vJ - vJ.mean(), mask)

# Optionally orthogonalise vJ against vG in H^{-1}_ρ(G)
def trapint(arr):
    return np.trapezoid(arr, x)

def ip_deriv(a, b, w):
    # 〈a,b〉_{ρG} with consistent masking on derivative fields
    return trapint(w * a * b)

phiG, _, _ = kkt_solve(rho, G, vG, k, mask)
phiJ, _, _ = kkt_solve(rho, G, vJ, k, mask)

dphiG = proj_field(dx(phiG, k, mask), mask)
dphiJ = proj_field(dx(phiJ, k, mask), mask)
w = rho * G

if orthogonalise_vJ:
    # Orthogonalise vJ so that {dphiG, dphiJ} is orthogonal in the ρG metric
    alpha = ip_deriv(dphiJ, dphiG, w) / (ip_deriv(dphiG, dphiG, w) + 1e-30)
    vJ = proj_field(vJ - alpha * vG, mask)
    phiJ, _, _ = kkt_solve(rho, G, vJ, k, mask)
    dphiJ = proj_field(dx(phiJ, k, mask), mask)

# Build a metric-orthonormal two-plane basis from {dphiG, dphiJ}
def build_two_plane_from_kkt(dphiG, dphiJ):
    # e1 ∝ dphiG
    n1 = np.sqrt(max(ip_deriv(dphiG, dphiG, w), 1e-30))
    e1 = dphiG / n1
    # e2 from Gram–Schmidt on dphiJ
    b = dphiJ - ip_deriv(dphiJ, e1, w) * e1
    n2 = np.sqrt(max(ip_deriv(b, b, w), 1e-30))
    e2 = b / n2
    return e1, e2, n1, n2

e1, e2, n1, n2 = build_two_plane_from_kkt(dphiG, dphiJ)

# Precompute sigma = 〈dmu,dmu〉_{ρG} once (state fixed)
dmu = proj_field(dx(mu, k, mask), mask)
sigma = ip_deriv(dmu, dmu, w)

# -------------------------
# Sweep
# -------------------------
rows = []

print("Entropy–phase and equality dial sweep (formal run)")
print(f"L = {L}, N = {N}, rho0 = {rho0:.6g}, lambda = {lam}, cJ = {cJ}")
print("Reader: R from raw pairing; M from KKT-based metric two-plane {dphiG, dphiJ}; vJ orthogonalised:",
      orthogonalise_vJ)
print("Using identical de-aliased subspace for all operators.")
print(f"η grid: {eta_grid}")
print("-"*64)

for eta in eta_grid:
    v = proj_field((1 - eta) * vG + eta * vJ, mask)
    phi, it, res = kkt_solve(rho, G, v, k, mask)
    phi = proj_field(phi, mask)

    twoC = Hm1G_energy(rho, G, phi, k, mask, x)       # 2*C_min = 〈dphi,dphi〉_{ρG}
    dphi = proj_field(dx(phi, k, mask), mask)

    # Equality dial: raw pairing with dmu
    Re_raw = ip_deriv(dphi, dmu, w)
    denom  = max(twoC * sigma, 1e-30)
    R = (Re_raw**2) / denom

    # Complex modulus via KKT-based metric two-plane:
    # Represent dphi on {e1, e2}, scale by sqrt(sigma), normalise by sqrt(twoC*sigma).
    a1 = ip_deriv(dphi, e1, w)
    a2 = ip_deriv(dphi, e2, w)
    Re_tp = a1 * np.sqrt(sigma)
    Im_tp = a2 * np.sqrt(sigma)
    M = (Re_tp**2 + Im_tp**2) / denom

    # Fractions for plotting based on two-plane components
    mod2_tp = Re_tp**2 + Im_tp**2
    real_frac = 0.0 if mod2_tp == 0.0 else Re_tp**2 / mod2_tp
    imag_frac = 1.0 - real_frac

    rows.append({
        "eta": float(eta),
        "real_frac": float(real_frac),
        "imag_frac": float(imag_frac),
        "R": float(R),
        "mod_ratio": float(M),
        "iters": int(it),
        "residual": float(res),
        "twoC": float(twoC),
        "sigma": float(sigma),
        "a1": float(a1),
        "a2": float(a2)
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(outdir, "entropy_phase_eta.csv")
df.to_csv(csv_path, index=False)

# -------------------------
# Console summary and checks
# -------------------------
def row_by_eta(dframe, val):
    return dframe.iloc[(dframe["eta"] - val).abs().argsort().iloc[0]]

r0  = row_by_eta(df, 0.0)
r50 = row_by_eta(df, 0.5)
r1  = row_by_eta(df, 1.0)

print("Summary at key η:")
print(f"  η=0.00: real_frac={r0['real_frac']:.12f}, R={r0['R']:.12e}, M={r0['mod_ratio']:.12e}, "
      f"iters={int(r0['iters'])}, res={r0['residual']:.3e}")
print(f"  η=0.50: real_frac={r50['real_frac']:.12f}, R={r50['R']:.12e}, M={r50['mod_ratio']:.12e}, "
      f"iters={int(r50['iters'])}, res={r50['residual']:.3e}")
print(f"  η=1.00: imag_frac={r1['imag_frac']:.12f}, R={r1['R']:.12e}, M={r1['mod_ratio']:.12e}, "
      f"iters={int(r1['iters'])}, res={r1['residual']:.3e}")

gap0 = abs(1.0 - r0["R"])
max_M_dev = float(np.max(np.abs(df["mod_ratio"] - 1.0)))

print(f"  Check: 1 - R(η=0) = {gap0:.3e}  (target <= {tol_R0:.1e})")
print(f"  Max |mod_ratio - 1| over η: {max_M_dev:.3e}  (target <= {tol_M:.1e})")
print("-"*64)

# Hard assertions
assert gap0 <= tol_R0, f"Equality dial too loose at η=0: 1-R = {gap0:.3e} > {tol_R0:.1e}"
assert max_M_dev <= tol_M, f"Two-plane modulus deviates: max |M-1| = {max_M_dev:.3e} > {tol_M:.1e}"

# -------------------------
# Plots
# -------------------------
plt.figure()
plt.plot(df["eta"], df["real_frac"], marker="o", label="real fraction (KKT two-plane)")
plt.plot(df["eta"], df["imag_frac"], marker="x", linestyle="--", label="imag fraction (KKT two-plane)")
plt.xlabel("η"); plt.ylabel("fraction"); plt.title("Entropy–phase split along v(η)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_entropy_phase_eta.png"), dpi=180)

plt.figure()
plt.plot(df["eta"], df["R"], marker="o")
plt.xlabel("η"); plt.ylabel("R"); plt.title("Equality dial along v(η)")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_R_eta.png"), dpi=180)

plt.figure()
plt.plot(df["eta"], df["mod_ratio"], marker="o")
plt.axhline(1.0, lw=1, alpha=0.6)
plt.xlabel("η"); plt.ylabel("modulus ratio")
plt.title("Complex modulus via KKT two-plane")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_mod_eta.png"), dpi=180)

print(f"CSV written to {csv_path}")
print("Figures written to out/fig_entropy_phase_eta.png, out/fig_R_eta.png, out/fig_mod_eta.png")
print("Done.")
