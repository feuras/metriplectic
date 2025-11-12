"""
What it verifies
- Measures the complex symbol ω(k) from the discrete operators used in the code.
- Compares Re ω against the sectorial analytic form: Re ω = -k^2 - rho0*lam2*k^4 - rho0*lam4*k^6.
- Compares Im ω against the reversible analytic form from the J term: Im ω = cJ*k^2 + rho0*cJ*lam2*k^4 + rho0*cJ*lam4*k^6.
- Uses the same de-aliased subspace for every operation to avoid pairing drift.
- Emits CSV with absolute errors and plots for Re and Im branches.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.fft import fft, ifft

from common import Grid, proj_field, mu_highorder, dx, dxx, hilbert  # keep imports aligned with project

# -------------------------
# Config
# -------------------------
L = 40.0
N = 1024
lam2 = 0.10
lam4 = 0.02
cJ = 0.6
modes = [1, 2, 3, 4, 6, 8]
tol = 1e-12

os.makedirs("out", exist_ok=True)

# -------------------------
# Grid and base state
# -------------------------
g = Grid(L=L, N=N)
x, k, mask = g.build()
rho0 = 1.0 / g.L

print("Dispersion check configuration")
print(f"L = {L}, N = {N}, rho0 = {rho0}")
print("Dealiasing mask used in all derivative operators")
print(f"lam2 = {lam2}, lam4 = {lam4}, cJ = {cJ}")
print(f"tolerance = {tol}")
print("----------------------------------------------------")

rows = []

for m in modes:
    # Pure mode perturbation
    kphys = 2.0 * np.pi * m / g.L
    dr = np.cos(kphys * x)  # amplitude 1

    # Chemical potential mu consistent with paper's high order form
    # Use identical de-aliased subspace for all derivative operations
    mu = (
        dr / rho0
        - lam2 * dxx(dr, k, mask)
        + np.real(ifft((k**4) * fft(dr) * mask)) * lam4  # apply mask here too
    )

    # Irreversible branch: v_G operator
    irr = dx(rho0 * dx(mu, k, mask), k, mask)

    # Reversible branch: J with weighted Liouville and Hilbert transform
    rev = dx(rho0 * cJ * hilbert(dx(mu, k, mask), k, mask), k, mask)

    # Complex growth rate from ratio in Fourier space at the excited mode
    Fdr = fft(dr)
    idx = np.argmax(np.abs(Fdr))  # one of the two symmetric peaks
    num_symbol = fft(irr + rev)[idx] / (Fdr[idx] + 1e-30)

    Re_num = float(np.real(num_symbol))
    Im_num = float(np.imag(num_symbol))

    # Analytic targets
    Re_an = -(kphys**2) - rho0 * lam2 * (kphys**4) - rho0 * lam4 * (kphys**6)
    Im_an = cJ * (kphys**2) + rho0 * cJ * lam2 * (kphys**4) + rho0 * cJ * lam4 * (kphys**6)

    rows.append({
        "m": m,
        "k": kphys,
        "Re_num": Re_num,
        "Im_num": Im_num,
        "Re_analytic": Re_an,
        "Im_analytic": Im_an,
        "Re_abs_err": abs(Re_num - Re_an),
        "Im_abs_err": abs(Im_num - Im_an),
    })

# -------------------------
# Outputs
# -------------------------
df = pd.DataFrame(rows)
df.to_csv("out/dispersion_table.csv", index=False)

# Hard assertions at the stated tolerance
max_re_err = df["Re_abs_err"].max()
max_im_err = df["Im_abs_err"].max()
print(f"Max Re abs error: {max_re_err:.3e}")
print(f"Max Im abs error: {max_im_err:.3e}")

assert max_re_err <= tol, f"Re branch exceeds tolerance: {max_re_err:.3e} > {tol:.1e}"
assert max_im_err <= tol, f"Im branch exceeds tolerance: {max_im_err:.3e} > {tol:.1e}"

# Plots
plt.figure()
plt.plot(df["k"], df["Re_num"], marker="o", label="Re ω num")
plt.plot(df["k"], df["Re_analytic"], marker="x", linestyle="--", label="Re ω analytic")
plt.xlabel("k")
plt.ylabel("Re ω")
plt.title("Dispersion: dissipative branch")
plt.legend()
plt.tight_layout()
plt.savefig("out/fig_dispersion_re.png", dpi=180)

plt.figure()
plt.plot(df["k"], df["Im_num"], marker="o", label="Im ω num")
plt.plot(df["k"], df["Im_analytic"], marker="x", linestyle="--", label="Im ω analytic")
plt.xlabel("k")
plt.ylabel("Im ω")
plt.title("Dispersion: reversible branch")
plt.legend()
plt.tight_layout()
plt.savefig("out/fig_dispersion_im.png", dpi=180)

print("OK. CSV and figures written in out/")
