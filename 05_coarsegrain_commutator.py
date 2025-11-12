"""
Coarse-graining commutator check for 'Further assembly, emergence and unification'.

It measures, for filters C_ell,
    rel(ell) = || C_ell[ Q(rho) ] - Q( C_ell[rho] ) || / || Q(rho) ||
with Q(rho) = d_x( rho * G * d_x(mu) ), using the same de-aliased subspace everywhere.

Expectations (per paper):
- Smooth Gaussian coarse-graining: small commutator that grows gently with ell.
- Non-smooth box (moving-average) coarse-graining: large O(1) commutator (aliasing + non-smooth kernel).
The script prints a detailed table and asserts these behaviours.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

from common import Grid, proj_field, mu_entropic, dx

# -------------------------
# Config
# -------------------------
L = 40.0
N = 512
lam2 = 0.10          # parameter used by mu_entropic
ell_list = np.array([0.2, 0.3, 0.4, 0.6, 0.8])
tol_gauss_max = 2e-2  # Gaussian commutator should stay below this across tested ells
tol_box_min = 1.5     # Box commutator should stay above this (order-one violation)

os.makedirs("out", exist_ok=True)

# -------------------------
# Grid and base fields
# -------------------------
g = Grid(L=L, N=N)
x, k, mask = g.build()
rho0 = 1.0 / g.L

# mildly inhomogeneous density and elliptic weight, projected to de-aliased subspace
rho = rho0 * (1.0 + 0.2 * np.cos(2.0 * np.pi * x / g.L))
rho = proj_field(rho, mask)
rho = rho - rho.mean() + rho0

G = 1.0 + 0.3 * np.cos(2.0 * np.pi * x / g.L)
G = proj_field(G, mask)

# -------------------------
# Filters
# -------------------------
def gaussian_filter(f: np.ndarray, ell: float) -> np.ndarray:
    """Periodic Gaussian filter in Fourier space, reprojected to masked subspace."""
    F = fft(f)
    Gf = np.exp(-0.5 * (ell * np.abs(k))**2)
    out = np.real(ifft(F * Gf))
    return proj_field(out, mask)

def box_filter(f: np.ndarray, ell: float) -> np.ndarray:
    """Periodic moving-average box filter of half-width ~ell, then project to masked subspace.
    Non-smooth kernel by design (to trigger the large commutator as per paper)."""
    w = x[1] - x[0]
    half = int(max(1, np.round(ell / w)))
    size = 2 * half + 1
    ker = np.zeros_like(f)
    ker[:size] = 1.0 / size

    # circular convolution via padding-two-copies trick to avoid boundary bias
    ff = np.r_[f, f, f]
    conv = np.convolve(ff, ker, mode="same")[f.size:2 * f.size]
    return proj_field(conv, mask)

# -------------------------
# Commutator measurement
# -------------------------
def comm_rel(ell: float, kind: str = "gauss") -> float:
    mu = mu_entropic(rho, lam2, k, mask)
    Q  = dx(rho * G * dx(mu, k, mask), k, mask)

    if kind == "gauss":
        C_rho = gaussian_filter(rho, ell)
        r1 = gaussian_filter(Q, ell)
    else:
        C_rho = box_filter(rho, ell)
        r1 = box_filter(Q, ell)

    r2 = dx(C_rho * G * dx(mu, k, mask), k, mask)
    num = np.linalg.norm(r1 - r2)
    den = np.linalg.norm(Q) + 1e-16
    return float(num / den)

# -------------------------
# Run sweep and report
# -------------------------
rows = []
print("Coarse-graining commutator check")
print(f"L={L}, N={N}, rho0={rho0:.6f}, lam2={lam2}")
print("All operators and filters projected to the same de-aliased subspace.")
print("ell    rel_gauss         rel_box")
for ell in ell_list:
    rg = comm_rel(ell, "gauss")
    rb = comm_rel(ell, "box")
    rows.append({"ell": ell, "rel_gauss": rg, "rel_box": rb})
    print(f"{ell:<4.1f}  {rg:>1.12f}   {rb:>1.12f}")

df = pd.DataFrame(rows)
df.to_csv("out/coarsegrain.csv", index=False)

# Assertions in line with the paper's stated behaviour
max_gauss = df["rel_gauss"].max()
min_box = df["rel_box"].min()
print("\nSummary:")
print(f"max rel_gauss = {max_gauss:.6e}  (target <= {tol_gauss_max:.2e})")
print(f"min rel_box   = {min_box:.6e}  (target >= {tol_box_min:.2e})")

assert max_gauss <= tol_gauss_max, (
    f"Gaussian commutator too large: {max_gauss:.3e} > {tol_gauss_max:.3e}"
)
assert min_box >= tol_box_min, (
    f"Box commutator too small (should be O(1)): {min_box:.3e} < {tol_box_min:.3e}"
)

# Optional quick plot for visual sanity
plt.figure()
plt.plot(df["ell"], df["rel_gauss"], marker="o", label="Gaussian")
plt.plot(df["ell"], df["rel_box"], marker="x", label="Box")
plt.xlabel("ell")
plt.ylabel("relative commutator norm")
plt.title("Coarse-graining commutator")
plt.legend()
plt.tight_layout()
plt.savefig("out/coarsegrain.png", dpi=160)

print("\nOK. Wrote out/coarsegrain.csv and out/coarsegrain.png")
