"""
Kramers-Kronig susceptibility check at fixed k with symmetric-grid FFT Hilbert transform.

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
rho0 = 1.0 / L
G0 = 1.0
lam2 = 0.10
lam4 = 0.00   # set to 0.02 if you want k^6 in the abscissa
m = 4         # mode index

# Symmetric frequency window and resolution
W = 6.0                 # half-width of symmetric window [-W, W]
n_sym = 2048            # number of points on the symmetric grid (even)
w_min_pos, w_max_pos = 0.05, 3.00   # positive-range slice to report (must be within (0, W))
w_pts_pos = 90

tol_rmse = 2e-3

os.makedirs("out", exist_ok=True)

# -------------------------
# Abscissa and prefactor at fixed k
# -------------------------
kphys = 2.0 * np.pi * m / L
a = G0 * (kphys**2) + G0 * rho0 * lam2 * (kphys**4) + G0 * rho0 * lam4 * (kphys**6)
S = rho0 * G0 * (kphys**2)

# -------------------------
# Ground truth χ(ω) helpers
# -------------------------
def re_chi(om):
    return S * a / (a*a + om*om)

def im_chi(om):
    return S * om / (a*a + om*om)

# -------------------------
# Periodic FFT-based Hilbert transform on symmetric grid
# For a periodic real signal f(ω), H_per[f] = ifft( -i * sign(n) * fft(f) )
# On a symmetric grid with f even, H_per[f] is approximately the PV Hilbert transform
# when the window is wide enough and resolution is sufficiently high.
# -------------------------
def periodic_hilbert(f):
    F = np.fft.fft(f)
    n = f.size
    h = np.zeros(n)
    # frequency index sign convention for DFT-based Hilbert filter
    if n % 2 == 0:
        # n even: indices 1..n/2-1 are positive, n/2 is Nyquist, n/2+1..n-1 negative
        h[1:n//2] = 2.0
        h[n//2] = 1.0  # Nyquist
        # h[0] stays 0 for DC
    else:
        # n odd: indices 1..(n-1)/2 positive, the rest negative
        h[1:(n+1)//2] = 2.0
    # Apply the analytic signal construction: Hilbert = imag part of ifft(F * h) with a minus sign
    analytic = np.fft.ifft(F * h)
    return np.imag(analytic)

# -------------------------
# Build symmetric grid and evaluate Re χ
# -------------------------
omega_sym = np.linspace(-W, W, n_sym, endpoint=False)  # periodic grid
Re_sym = re_chi(omega_sym)

# Remove tiny DC bias to stabilise the periodic transform
Re_sym = Re_sym - np.mean(Re_sym)

# Hilbert estimate on symmetric grid
Im_est_sym = periodic_hilbert(Re_sym)

# Extract the positive branch we will report and compare on
omega_pos = np.linspace(w_min_pos, w_max_pos, w_pts_pos)
Re_pos = re_chi(omega_pos)
Im_pos = im_chi(omega_pos)

# Interpolate Im_est_sym to positive omega sample points
Im_est_pos = np.interp(omega_pos, omega_sym, Im_est_sym)

# Through-origin scale fit
scale = float(np.dot(Im_pos, Im_est_pos) / np.dot(Im_est_pos, Im_est_pos))
Im_fit_pos = scale * Im_est_pos

# Unconstrained LS diagnostics
A = np.vstack([Im_est_pos, np.ones_like(Im_est_pos)]).T
coef_ls, residuals, _, _ = np.linalg.lstsq(A, Im_pos, rcond=None)
scale_ls, intercept_ls = float(coef_ls[0]), float(coef_ls[1])

# Errors and diagnostics
res = Im_pos - Im_fit_pos
rmse = float(np.sqrt(np.mean(res**2)))
max_abs = float(np.max(np.abs(res)))
corr = float(np.corrcoef(Im_pos, Im_fit_pos)[0, 1])

# -------------------------
# Console readout
# -------------------------
print("KK susceptibility check (symmetric-grid FFT Hilbert)")
print("----------------------------------------------------")
print(f"L = {L}, N = {N}, rho0 = {rho0}")
print(f"k = {kphys:.12f}, mode m = {m}")
print(f"a (abscissa) = {a:.12f}, S = {S:.12f}")
print(f"Symmetric window [-W, W] with W = {W}, n_sym = {n_sym}, Δω_sym = {2*W/n_sym:.6f}")
print(f"Report window: [{w_min_pos}, {w_max_pos}] with {w_pts_pos} points, Δω_pos = {(w_max_pos-w_min_pos)/(w_pts_pos-1):.6f}")
print("Re χ(ω) = S a / (a^2 + ω^2)")
print("Im χ(ω) = S ω / (a^2 + ω^2)")
print("Hilbert estimator: periodic FFT on symmetric grid, mean removed")
print(f"Through-origin scale = {scale:.12e}")
print(f"Diagnostics (unconstrained LS): scale = {scale_ls:.12e}, intercept = {intercept_ls:.12e}")
print(f"RMSE(Im - scale*H[Re]) = {rmse:.3e},  max|.| = {max_abs:.3e},  corr = {corr:.6f}")
print("Tolerance:", tol_rmse)
print("----------------------------------------------------")

assert rmse <= tol_rmse, f"KK fit RMSE {rmse:.3e} exceeds tolerance {tol_rmse:.3e}"

# -------------------------
# Save CSVs
# -------------------------
df = pd.DataFrame({
    "omega": omega_pos,
    "Re_chi": Re_pos,
    "Im_chi": Im_pos,
    "Im_from_H_Re": Im_est_pos,
    "Im_fit": Im_fit_pos,
})
meta = {
    "k": kphys,
    "a": a,
    "S": S,
    "scale": scale,
    "scale_ls": scale_ls,
    "intercept_ls": intercept_ls,
    "rmse": rmse,
    "max_abs_err": max_abs,
    "corrcoef": corr,
    "lam2": lam2,
    "lam4": lam4,
    "rho0": rho0,
    "G0": G0,
    "W": W,
    "n_sym": n_sym,
    "w_min_pos": w_min_pos,
    "w_max_pos": w_max_pos,
    "w_pts_pos": w_pts_pos,
}
df.to_csv("out/kk_susceptibility.csv", index=False)
pd.DataFrame([meta]).to_csv("out/kk_meta.csv", index=False)

# -------------------------
# Plot
# -------------------------
plt.figure()
plt.plot(omega_pos, Im_pos, marker="o", label="Im χ (ground truth)")
plt.plot(omega_pos, Im_fit_pos, marker="x", linestyle="--", label="scaled Hilbert[Re χ]")
plt.xlabel("ω")
plt.ylabel("Im χ")
plt.title("Kramers-Kronig consistency at fixed k")
plt.legend()
plt.tight_layout()
plt.savefig("out/fig_kk.png", dpi=180)

print("OK. Wrote out/kk_susceptibility.csv, out/kk_meta.csv, and out/fig_kk.png")
