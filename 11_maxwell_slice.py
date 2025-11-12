"""
Maxwell-slice fit (static law + Leray projection) with a controlled transverse E seed.

Identity on the de-aliased slice (projected to solenoidal fields):
    curl B  =  alpha * (∂t E)_disp  +  beta * j_J

Key idea:
- Keep j_J from the scalar-potential field (so no displacement leakage).
- Add a small divergence-free electric seed E_T = ∇^⊥ ψ to build a *transverse* displacement predictor:
      (∂t E)_disp := c^2 ∇^⊥ Δ ψ
- Project target and predictors with the Helmholtz–Leray projector before fitting.
"""

import os
import numpy as np
import pandas as pd

# -------------------------
# Grid, spectra, ops
# -------------------------
L = 40.0
N = 256
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

kx = 2*np.pi*np.fft.fftfreq(N, d=L/N)
ky = 2*np.pi*np.fft.fftfreq(N, d=L/N)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2

mask_1d = (np.abs(np.fft.fftfreq(N)) * N <= N//3).astype(float)
mask = (mask_1d[:, None] * mask_1d[None, :]).astype(float)

def fft2(f):  return np.fft.fft2(f)
def ifft2(F): return np.real(np.fft.ifft2(F * mask))
def dx(f):    return ifft2(1j * KX * fft2(f))
def dy(f):    return ifft2(1j * KY * fft2(f))
def lap(f):   return ifft2(-(K2) * fft2(f))

def proj_divfree(vx, vy):
    Fx, Fy = fft2(vx), fft2(vy)
    denom = K2.copy(); denom[0,0] = 1.0
    kdotF = KX*Fx + KY*Fy
    Px = Fx - KX * kdotF / denom
    Py = Fy - KY * kdotF / denom
    Px[0,0] = 0.0; Py[0,0] = 0.0
    return ifft2(Px), ifft2(Py)

def vstack(vx, vy): return np.concatenate([vx.ravel(), vy.ravel()], axis=0)
def corr(a,b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a,b)/((na*nb)+1e-30))

# -------------------------
# Fields
# -------------------------
rho0 = 1.0 / L
rho  = rho0 * (1 + 0.2*np.cos(2*np.pi*X/L)*np.cos(2*np.pi*Y/L))
c    = 0.8 + 0.1*np.cos(2*np.pi*X/L) + 0.07*np.cos(2*np.pi*Y/L)
mu   = np.cos(2*np.pi*2*X/L)*np.cos(2*np.pi*1*Y/L) + 0.25*np.cos(2*np.pi*3*Y/L)

# Longitudinal electric from scalar potential (used for j_J only)
Ex_L, Ey_L = -dx(mu), -dy(mu)

# Reversible J-velocity/current (from longitudinal field)
uJx, uJy = c*Ey_L, -c*Ex_L
jJx, jJy = rho*uJx, rho*uJy

# Target: curl B with B = c*rho (slice)
B = c * rho
RBx, RBy = dy(B), -dx(B)

# -------------------------
# Transverse electric seed and displacement predictor
# -------------------------
# Small streamfunction ψ to generate E_T = ∇^⊥ ψ with low-k content
eps_T = 0.05  # keep small: linear-response regime
psi = eps_T * ( np.cos(2*np.pi*X/L) + 0.7*np.cos(2*np.pi*Y/L) + 0.5*np.cos(2*np.pi*(X+Y)/L) )

# Divergence-free electric seed
ETx =  dy(psi)      # ∂y ψ
ETy = -dx(psi)      # -∂x ψ

# Displacement predictor from the transverse seed:
# mu_t^T is replaced by a transverse surrogate with the same wave operator structure:
# (∂t E)_disp := c^2 ∇^⊥ Δ ψ
lap_psi = lap(psi)
dExdt_T =  dy(c**2 * lap_psi)
dEydt_T = -dx(c**2 * lap_psi)

# -------------------------
# Project to solenoidal subspace and fit
# -------------------------
RBx_p,   RBy_p   = proj_divfree(RBx, RBy)
dExdt_p, dEydt_p = proj_divfree(dExdt_T, dEydt_T)
jJx_p,   jJy_p   = proj_divfree(jJx, jJy)

b    = vstack(RBx_p, RBy_p)
col0 = vstack(dExdt_p, dEydt_p)      # displacement (transverse) predictor
col1 = vstack(jJx_p,   jJy_p)        # reversible current predictor

A = np.stack([col0, col1], axis=1)
coef, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
alpha, beta = coef
Ax = A @ coef
r  = b - Ax

# Diagnostics
b_norm  = np.linalg.norm(b)
c0_norm = np.linalg.norm(col0)
c1_norm = np.linalg.norm(col1)
r_norm  = np.linalg.norm(r)
rel_resid = r_norm / (b_norm + 1e-30)
R2        = 1.0 - (r_norm**2) / (b_norm**2 + 1e-30)
condA     = (svals[0] / svals[-1]) if svals.size >= 2 and svals[-1] > 0 else np.inf

corr_b_c0 = corr(b, col0)
corr_b_c1 = corr(b, col1)
corr_cols = corr(col0, col1)

divJ_rel = np.linalg.norm(dx(jJx) + dy(jJy)) / (np.linalg.norm(np.sqrt(jJx**2 + jJy**2)) + 1e-30)

# Gauge-like shift test (acts only on longitudinal μ; the transverse predictor is ψ-based and unchanged)
chi = 0.3*np.cos(2*np.pi*3*X/L + 0.17) * np.cos(2*np.pi*2*Y/L + 0.53)
mug = mu + chi
Exg_L, Eyg_L = -dx(mug), -dy(mug)
uJxg, uJyg = c*Eyg_L, -c*Exg_L
jJxg, jJyg = rho*uJxg, rho*uJyg
jJx_gp, jJygp = proj_divfree(jJxg, jJyg)

bg   = b  # target unchanged (depends on c,rho only)
Ag = np.stack([col0, vstack(jJx_gp, jJygp)], axis=1)
coef_g, *_ = np.linalg.lstsq(Ag, bg, rcond=None)
alpha_g, beta_g = coef_g

# -------------------------
# Console report
# -------------------------
print("\n=== Maxwell-slice fit (projected) with transverse E seed ===")
print(f"L={L}, N={N}, rho0={rho0:.6g}, eps_T={eps_T}")
print("Operators: spectral, 2/3 mask, Leray projection")
print("\n-- Norms (projected) --")
print(f"||b||2 = {b_norm:.6e}")
print(f"|| (∂t E)_disp(T) ||2 = {c0_norm:.6e}   || j_J ||2 = {c1_norm:.6e}")
print("\n-- Correlations (projected) --")
print(f"corr(b, (∂t E)_disp(T)) = {corr_b_c0:.6f}")
print(f"corr(b, j_J)            = {corr_b_c1:.6f}")
print(f"corr(columns)           = {corr_cols:.6f}")
print("\n-- Coefficients (base) --")
print(f"alpha = {alpha:.12g}")
print(f"beta  = {beta:.12g}")
print("\n-- Diagnostics (base) --")
print(f"||resid||2 = {r_norm:.6e}, rel_resid = {rel_resid:.3e}, R^2 = {R2:.6f}, rank(A) = {rank}, cond(A) ~ {condA:.3e}")
print(f"Liouville div check (unprojected): ||div(rho*uJ)|| / ||j_J|| = {divJ_rel:.3e}")
print("\n-- Coefficients (after mu -> mu + chi) --")
print(f"alpha_g = {alpha_g:.12g}")
print(f"beta_g  = {beta_g:.12g}")
print("======================================\n")

# -------------------------
# CSV
# -------------------------
os.makedirs("out", exist_ok=True)
row = {
    "L": L, "N": N, "rho0": float(rho0), "eps_T": float(eps_T),
    "alpha": float(alpha), "beta": float(beta),
    "rel_resid": float(rel_resid), "R2": float(R2), "rank": int(rank), "condA": float(condA),
    "corr_b_col0": float(corr_b_c0), "corr_b_col1": float(corr_b_c1), "corr_cols": float(corr_cols),
    "alpha_g": float(alpha_g), "beta_g": float(beta_g),
    "liouville_div_rel": float(divJ_rel),
}
pd.DataFrame([row]).to_csv("out/maxwell_slice_fit.csv", index=False)
print("Wrote out/maxwell_slice_fit.csv")
