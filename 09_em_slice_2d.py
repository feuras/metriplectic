"""
EM-slice equivalence check (weighted Liouville vs Faraday-form source)

Verifies in 2D (periodic box) that:
  src        = -(∂x(B * Ey) - ∂y(B * Ex))
  div_rho_uJ = -(∂x(ρ uJx) + ∂y(ρ uJy))
with  E = -∇μ,  uJ = c0 * (Ey, -Ex),  B = c0 * ρ

Both fields should agree to de-aliased spectral accuracy when the same subspace is used.
Outputs:
  - CSV with detailed norms and errors
  - PNG heatmap of residual field
  - Console readout with explicit tolerances and pass/fail
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Config
# -------------------------
L  = 40.0
N  = 256
c0 = 0.8
amp_rho = 0.2           # density modulation amplitude
tol_L2_rel = 1e-12      # relative L2 tolerance
tol_Linf   = 1e-10      # absolute Linf tolerance

os.makedirs("out", exist_ok=True)

# -------------------------
# Grid and spectral helpers
# -------------------------
x = np.linspace(0.0, L, N, endpoint=False)
y = np.linspace(0.0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L/N)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=L/N)
KX, KY = np.meshgrid(kx, ky, indexing="ij")

# 2/3 de-alias mask (square)
cut = N // 3
freq = np.fft.fftfreq(N)
mask1d = (np.abs(freq) * N <= cut).astype(float)
MASK = (mask1d[:, None] * mask1d[None, :]).astype(float)

def fft2(f):  return np.fft.fft2(f)
def ifft2(F): return np.real(np.fft.ifft2(F * MASK))

def dx(f):
    return ifft2(1j * KX * fft2(f))

def dy(f):
    return ifft2(1j * KY * fft2(f))

# -------------------------
# Fields
# -------------------------
rho0 = 1.0 / L
rho  = rho0 * (1.0 + amp_rho * np.cos(2.0 * np.pi * X / L) * np.cos(2.0 * np.pi * Y / L))

# A mixed-mode μ to avoid accidental cancellations
mu = (np.cos(2.0 * np.pi * 2 * X / L) * np.cos(2.0 * np.pi * 1 * Y / L)
    + 0.5 * np.cos(2.0 * np.pi * 1 * X / L) * np.cos(2.0 * np.pi * 2 * Y / L))

# Electric field E = -∇μ
Ex, Ey = -dx(mu), -dy(mu)

# Reversible velocity u_J = c0 * (Ey, -Ex)
uJx, uJy = c0 * Ey, -c0 * Ex

# Magnetic-like weight B = c0 * rho
B = c0 * rho

# Faraday-form source and weighted-Liouville divergence
src         = -(dx(B * Ey) - dy(B * Ex))
div_rho_uJ  = -(dx(rho * uJx) + dy(rho * uJy))

# Residuals and norms
residual     = src - div_rho_uJ
L2_src       = float(np.linalg.norm(src))
L2_div       = float(np.linalg.norm(div_rho_uJ))
L2_resid     = float(np.linalg.norm(residual))
Linf_resid   = float(np.max(np.abs(residual)))
rel_L2_error = float(L2_resid / (L2_src + 1e-30))

# -------------------------
# Console readout
# -------------------------
print("EM-slice equivalence check (2D periodic, de-aliased 2/3)")
print(f"L = {L}, N = {N}, rho0 = {rho0:.12g}, c0 = {c0}")
print(f"De-alias cut = N//3 = {cut}  -> square mask applied to all ops")
print(f"Density modulation amplitude = {amp_rho}")
print("---- Norms ----")
print(f"||src||_2          = {L2_src:.12e}")
print(f"||div_rho_uJ||_2   = {L2_div:.12e}")
print(f"||residual||_2     = {L2_resid:.12e}")
print(f"||residual||_inf   = {Linf_resid:.12e}")
print("---- Errors ----")
print(f"Relative L2 error  = {rel_L2_error:.12e}")
print(f"Tolerances: rel L2 <= {tol_L2_rel:.1e}, Linf <= {tol_Linf:.1e}")

# Assertions
assert rel_L2_error <= tol_L2_rel, f"Relative L2 error too large: {rel_L2_error:.3e}"
assert Linf_resid   <= tol_Linf,   f"Linf residual too large: {Linf_resid:.3e}"

print("PASS: Faraday-form source matches weighted-Liouville divergence within tolerances.")

# -------------------------
# CSV + figure outputs
# -------------------------
df = pd.DataFrame([{
    "L": L, "N": N, "rho0": rho0, "c0": c0,
    "amp_rho": amp_rho,
    "norm_Faraday_src": L2_src,
    "norm_div_rho_uJ": L2_div,
    "norm_residual_L2": L2_resid,
    "norm_residual_Linf": Linf_resid,
    "relative_L2_error": rel_L2_error
}])
df.to_csv("out/em_slice_compare.csv", index=False)

plt.figure(figsize=(6,5))
plt.imshow(residual, origin="lower", extent=[0, L, 0, L], aspect="equal")
plt.colorbar(label="residual")
plt.title("Residual: src - div(ρ u_J)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("out/em_slice_residual.png", dpi=180)
print("Wrote out/em_slice_compare.csv and out/em_slice_residual.png")
