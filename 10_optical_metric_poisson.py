"""
Weighted optical Poisson profile (periodic, 1D, mean-zero slice)

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Config
# -------------------------
L = 40.0
N = 512
dx_phys = L / N
x = np.linspace(0.0, L, N, endpoint=False)

# Spectral grid
k = np.fft.fftfreq(N, d=dx_phys)          # cycles per unit length
kx = 2.0 * np.pi * k                       # angular wavenumbers

# 2/3 de-aliased projector P: keep |n| <= N/3
mask = (np.abs(np.fft.fftfreq(N)) * N <= N // 3).astype(float)
mask[0] = 1.0  # P keeps the mean for scalars; we enforce mean-zero explicitly where needed

def F(f):
    return np.fft.fft(f)

def Finv(Fhat):
    return np.real(np.fft.ifft(Fhat))

def P(f):
    """Symmetric spectral projector."""
    return Finv(mask * F(f))

def D(f):
    """Skew-adjoint spectral derivative."""
    return Finv(1j * kx * F(f))

def DP(f):
    """Derivative on the projected subspace."""
    return P(D(P(f)))

def mean_zero(f):
    return f - np.mean(f)

# -------------------------
# Fields (smooth, positive)
# -------------------------
rho0 = 1.0 / L
rho  = rho0 * (1.0 + 0.3 * np.exp(-((x - 20.0)**2) / (2.0 * 3.0**2)))
G    = 1.0 + 0.4 * np.exp(-((x - 26.0)**2) / (2.0 * 4.0**2))

# Source sigma: difference of two bumps, then projected to mean-zero
sigma_raw = np.exp(-((x - 12.0)**2) / (2.0 * 2.0**2)) - np.exp(-((x - 30.0)**2) / (2.0 * 2.5**2))
sigma = mean_zero(sigma_raw)

# -------------------------
# SPD operator: A = P[ - d/dx ( rho G d/dx ) ]P
# -------------------------
def apply_A(phi):
    grad = DP(phi)                 # P D P
    flux = rho * G * grad          # pointwise product in physical space
    return P(-D(flux))             # final P to stay in subspace

# Right hand side b = -sigma, then project and enforce mean-zero
b = P(-sigma)
b = mean_zero(b)

# -------------------------
# Spectral preconditioner
# A0 ≈ P[ -(rho0*G0) dxx ]P
# In Fourier: diagA0 = (rho0*G0) * kx^2 on retained modes, zero at k=0
# -------------------------
G0 = float(np.mean(G))
rho0G0 = rho0 * G0
kx2 = kx**2
diagA0 = rho0G0 * kx2
diagA0[0] = np.inf  # remove mean component

def M_apply(r):
    # project r to subspace first
    r_hat = F(P(r))
    z_hat = np.zeros_like(r_hat)
    finite = np.isfinite(diagA0)
    z_hat[finite] = r_hat[finite] / diagA0[finite]
    z_hat[~finite] = 0.0
    z = Finv(z_hat)
    return P(mean_zero(z))

# -------------------------
# PCG
# -------------------------
def pcg(A_apply, b, M_apply=None, x0=None, maxit=50000, tol=1e-12, log_every=25):
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return np.zeros_like(b), {"iters": 0, "relres": 0.0, "absres": 0.0}

    x = np.zeros_like(b) if x0 is None else P(mean_zero(x0))
    r = b - A_apply(x)
    z = M_apply(r) if M_apply is not None else r
    p = z.copy()
    rz = float(np.dot(r, z))

    info = {}
    for it in range(1, maxit + 1):
        Ap = A_apply(p)
        denom = float(np.dot(p, Ap))
        if denom <= 0 or not np.isfinite(denom):
            raise RuntimeError(f"PCG breakdown with <p,Ap> = {denom}")
        alpha = rz / denom
        x = P(mean_zero(x + alpha * p))
        r = b - A_apply(x)

        absres = float(np.linalg.norm(r))
        relres = absres / nb
        if (it == 1) or (it % log_every == 0) or (relres <= tol):
            print(f"[PCG] it={it:4d}  |r|={absres:.3e}  |r|/|b|={relres:.3e}")

        if relres <= tol:
            info = {"iters": it, "relres": relres, "absres": absres}
            break

        z_new = M_apply(r) if M_apply is not None else r
        rz_new = float(np.dot(r, z_new))
        beta = rz_new / rz
        p = z_new + beta * p
        z = z_new
        rz = rz_new
    else:
        info = {"iters": maxit, "relres": relres, "absres": absres}

    return x, info

# -------------------------
# Solve with automatic restarts
# -------------------------
target_tol = 6e-9
max_restarts = 5

print("=== Weighted optical Poisson profile ===")
print(f"L={L}, N={N}, dx={dx_phys:.6f}")
print(f"rho0={rho0:.12f}, G0={G0:.12f}")
print("Operator: A = P[ - d/dx ( rho G d/dx ) ]P on mean-zero subspace")
print("Projection: symmetric 2/3 rule applied consistently")
print("Preconditioner: constant-coefficient spectral A0 with k=0 removed")
print("----------------------------------------")
print(f"||sigma||_2 = {np.linalg.norm(sigma):.12e}, mean(sigma) = {np.mean(sigma):.3e}")

mu = np.zeros_like(x)
relres = np.inf
total_its = 0

for rcount in range(max_restarts + 1):
    mu, info = pcg(apply_A, b, M_apply=M_apply, x0=mu, maxit=2000, tol=1e-12, log_every=50)
    total_its += info["iters"]

    # residual in the projected subspace
    res = b - apply_A(mu)
    absres = float(np.linalg.norm(res))
    relres = absres / (np.linalg.norm(b) + 1e-30)
    print(f"[restart {rcount}] total iters = {total_its},  relres = {relres:.3e}")

    if relres <= target_tol:
        break

print("----------------------------------------")
print(f"PCG finished in {total_its} iterations across {rcount+1} run(s)")
print(f"Final residual: ||r||_2 = {absres:.3e},  ||r||/||b|| = {relres:.3e}")
mu_mean = float(np.mean(mu))
print(f"mean(mu) = {mu_mean:.3e}  (enforced ~ 0)")
recon = apply_A(mu) + sigma
print(f"||A mu + sigma||_2 (projected) = {np.linalg.norm(P(recon)):.3e}")

# Assertions for review
assert relres <= target_tol, f"Relative residual too high: {relres:.3e}"
assert abs(mu_mean) <= 1e-12, "mu not mean-zero to numerical floor"
assert np.all(rho > 0.0), "rho must be strictly positive"
assert np.all(G > 0.0), "G must be strictly positive"

# -------------------------
# Outputs
# -------------------------
os.makedirs("out", exist_ok=True)
df = pd.DataFrame({"x": x, "mu": mu, "rho": rho, "G": G})
df.to_csv("out/optical_poisson_profile.csv", index=False)

plt.figure()
plt.plot(x, mu, label="mu (solution)")
plt.plot(x, P(sigma), label="sigma (mean-zero source, projected)")
plt.plot(x, rho / np.max(rho) * np.max(mu), label="rho (scaled)")
plt.plot(x, G / np.max(G) * np.max(mu), label="G (scaled)")
plt.xlabel("x")
plt.ylabel("value")
plt.title("Weighted optical Poisson profile")
plt.legend()
plt.tight_layout()
plt.savefig("out/optical_poisson_profile.png", dpi=180)

print("OK. Wrote out/optical_poisson_profile.csv and out/optical_poisson_profile.png")
