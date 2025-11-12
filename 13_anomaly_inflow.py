"""
Anomaly injection and repair (weighted Liouville) — 1D periodic

"""

import os
import numpy as np
import pandas as pd

# -------------------------
# Grid and operators
# -------------------------
L = 40.0
N = 1024
x = np.linspace(0.0, L, N, endpoint=False)

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)  # physical wavenumbers
mask = (np.abs(np.fft.fftfreq(N)) * N <= N // 3).astype(float)  # 2/3 dealias

def fft(f):  # forward FFT
    return np.fft.fft(f)

def ifft(F):  # inverse FFT with consistent mask
    return np.real(np.fft.ifft(F * mask))

def dx(f):  # spectral derivative with mask
    return ifft(1j * kx * fft(f))

def invdx(f):  # mean-zero spectral anti-derivative: g_x = f, mean(g)=0
    F = fft(f)
    G = np.zeros_like(F, dtype=complex)
    nonzero = kx != 0.0
    G[nonzero] = F[nonzero] / (1j * kx[nonzero])
    G[~nonzero] = 0.0  # enforce zero mean
    return ifft(G)

# -------------------------
# Fields and source
# -------------------------
rho0 = 1.0 / L
rho = rho0 * (1.0 + 0.2 * np.cos(2.0 * np.pi * 2 * x / L))  # smooth, strictly positive

# Prescribed anomaly source s with zero mean
s = 0.05 * np.cos(2.0 * np.pi * 1 * x / L)

# -------------------------
# Build the defective and compensating currents
# -------------------------
# Q is a mean-zero antiderivative of s: Q_x = s
Q = invdx(s)

# Anomaly-injecting current: rho * J_defect = Q  =>  ∂x(rho J_defect) = s
J_defect = Q / rho

# Optional background reversible current J0 = C / rho, divergence-free by construction
C_bg = 0.8
J0 = C_bg / rho  # ∂x(ρ J0) = ∂x(C_bg) = 0

# Compensator that cancels the defect exactly in the continuum limit
J_comp = -Q / rho  # so rho * (J_defect + J_comp) = 0 pointwise

# -------------------------
# Diagnostics
# -------------------------
def L1(f):
    return float(np.trapz(np.abs(f), x))

def L2(f):
    return float(np.sqrt(np.trapz(np.abs(f) ** 2, x)))

def Linf(f):
    return float(np.max(np.abs(f)))

# Divergences
div_defect = dx(rho * J_defect)          # should match s
div_total_before = dx(rho * (J_defect + J0))      # should equal s
div_total_after  = dx(rho * (J_defect + J_comp + J0))  # should be ~ 0

# Errors
err_before = div_total_before - s
err_after = div_total_after  # target is 0

# Norms
report = {
    "L": L, "N": N,
    "rho_min": float(np.min(rho)),
    "rho_max": float(np.max(rho)),
    "anomaly_L1_before": L1(err_before),
    "anomaly_L2_before": L2(err_before),
    "anomaly_Linf_before": Linf(err_before),
    "anomaly_L1_after": L1(err_after),
    "anomaly_L2_after": L2(err_after),
    "anomaly_Linf_after": Linf(err_after),
}

# -------------------------
# Output and assertions
# -------------------------
os.makedirs("out", exist_ok=True)
pd.DataFrame([report]).to_csv("out/anomaly_inflow.csv", index=False)

print("Weighted Liouville anomaly injection and repair")
print(f"L={L}, N={N}, rho0={rho0:.6g}, dealias=2/3")
print(f"Background reversible current J0 = C_bg / rho with C_bg={C_bg}")
print("Diagnostics relative to the intended target:")
print("  Before compensation: target divergence is s")
print("  After  compensation: target divergence is 0")
print("----------------------------------------------------")
print(f"rho min/max: {report['rho_min']:.6e} {report['rho_max']:.6e}")
print("Errors BEFORE compensation:")
print(f"  L1  = {report['anomaly_L1_before']:.6e}")
print(f"  L2  = {report['anomaly_L2_before']:.6e}")
print(f"  Linf= {report['anomaly_Linf_before']:.6e}")
print("Errors AFTER compensation:")
print(f"  L1  = {report['anomaly_L1_after']:.6e}")
print(f"  L2  = {report['anomaly_L2_after']:.6e}")
print(f"  Linf= {report['anomaly_Linf_after']:.6e}")

# Tight tolerances for this pure-mode test on a spectral grid
tol_before = 1e-12  # div_defect should equal s exactly in this construction, up to numerical floor
tol_after = 1e-12

# Assert both that the injection matches s and that compensation removes it
assert L1(div_defect - s) <= tol_before, f"Injected anomaly does not match s within tolerance: L1={L1(div_defect - s):.3e}"
assert report["anomaly_L1_after"] <= tol_after, f"Residual anomaly after compensation exceeds tolerance: L1={report['anomaly_L1_after']:.3e}"

print("OK. CSV written to out/anomaly_inflow.csv")
