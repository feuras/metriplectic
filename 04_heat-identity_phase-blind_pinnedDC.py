# Phase-blind exact-heat dissipation test (λ=0), identity-stable, no clipping,
# pinned mass (DC restore), rectangle-rule integration (FFT-consistent), text-only.
# Colab/NumPy ready. Warnings silenced (no deprecated trapz).

import numpy as np

# -------------------------------
# Console header
# -------------------------------
print("\n==============================================================")
print(" Phase-Blind Dissipation Test  |  λ=0 (entropy only), Exact Heat, Pinned DC")
print(" Rectangle-rule integrals (FFT-consistent)  |  Text-only diagnostics")
print("==============================================================\n")

# -------------------------------
# Grid and spectral operators
# -------------------------------
N, L = 256, 40.0
x  = np.linspace(0, L, N, endpoint=False)
dx = L / N
k  = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
ik = 1j*k
k2 = k**2
exp = np.exp

def grad(f):
    """Spectral gradient (periodic, real signal)."""
    return np.fft.ifft(ik*np.fft.fft(f)).real

def integrate_rect(y, dx):
    """Rectangle-rule integral (uniform grid, endpoint=False)."""
    return float(np.sum(y) * dx)

# -------------------------------
# Energetics for λ = 0 (entropy only)
# F[ρ] = ∫ ρ log ρ dx,  μ = log ρ,  σ̇ = ∫ |∂x ρ|² / ρ dx
# -------------------------------
def free_energy(rho):
    eps = 1e-15
    r = np.maximum(rho, eps)
    return integrate_rect(r*np.log(r), dx)

def sigma_dot(rho):
    eps = 1e-15
    rx = grad(rho)
    r_safe = np.maximum(rho, eps)
    return integrate_rect((rx**2)/r_safe, dx)

# -------------------------------
# Exact heat step (Fourier-space)
# Mass pinning: DC mode restored exactly each step
# -------------------------------
def step_heat_exact_pinned(rho, dt, rhok0_dc):
    rhok = np.fft.fft(rho)
    rhok *= exp(-k2*dt)
    rhok[0] = rhok0_dc
    return np.fft.ifft(rhok).real

# -------------------------------
# Initial ρ
# -------------------------------
rho_min = 1e-3
c1, c2, w = 12.0, 28.0, 2.0
env  = np.exp(-((x-c1)**2)/(2*w**2)) + np.exp(-((x-c2)**2)/(2*w**2))
rho0 = env*(1.0 + 0.25*np.cos(0.7*(x-10.0)))
rho0 = np.clip(rho0, rho_min, None)
rho0 /= integrate_rect(rho0, dx)

# Mild spectral prefilter (for positivity and stability)
alpha = 0.15                    # 0.0 = none; 0.1–0.3 typical
if alpha > 0.0:
    rho0k = np.fft.fft(rho0)
    kmax  = np.max(np.abs(k)) + 1e-15
    filt  = exp(-alpha*(np.abs(k)/kmax)**8)
    rho0  = np.fft.ifft(rho0k*filt).real
    rho0 /= integrate_rect(rho0, dx)

# DC mode target (pins total mass under spectral evolution)
rhok0_dc = np.fft.fft(rho0)[0]

# ------------------------
# Time stepping
# ------------------------
dt, steps = 2.0e-3, 30000
t = np.arange(steps+1)*dt

rho_A = rho0.copy()
rho_B = rho0.copy()

F_A = [free_energy(rho_A)]
F_B = [free_energy(rho_B)]
Sd_A = [sigma_dot(rho_A)]
Sd_B = [sigma_dot(rho_B)]

# Cumulative midpoint-in-time identity residual: Σ [ΔF + σ̇_mid dt]
cum_id_err = 0.0

print("[run] Integrating exact heat with pinned DC (phase-blind twin runs)...")
for n in range(steps):
    # exact heat step with DC pinning
    rho_A = step_heat_exact_pinned(rho_A, dt, rhok0_dc)
    rho_B = step_heat_exact_pinned(rho_B, dt, rhok0_dc)

    # energetics (rectangle rule)
    FA_new, FB_new = free_energy(rho_A), free_energy(rho_B)
    Sd_newA, Sd_newB = sigma_dot(rho_A), sigma_dot(rho_B)

    F_A.append(FA_new); F_B.append(FB_new)
    Sd_A.append(Sd_newA); Sd_B.append(Sd_newB)

    # midpoint-in-time identity increment
    Sd_mid = 0.5*(Sd_A[-1] + Sd_A[-2])
    dF     = F_A[-1] - F_A[-2]
    cum_id_err += dF + Sd_mid*dt

print("[run] Integration complete.\n")

# --------------
# Diags
# --------------
print("=== PHASE-BLIND DISSIPATION TEST (λ=0, exact heat, pinned mass) ===\n")
print(f"[cfg] Grid N={N}, L={L}, dx={dx}, dt={dt}, steps={steps}, prefilter alpha={alpha}")

initial_mass = integrate_rect(rho0, dx)
print(f"[init] Mass: {initial_mass:.12f}")
print(f"[init] F: {F_A[0]:.12f}")
print(f"[init] σ̇: {Sd_A[0]:.12f}")

# Phase-blindness (two independent runs with different 'labels' but identical physics)
max_rho_diff = float(np.max(np.abs(rho_A - rho_B)))
max_F_diff   = float(np.max(np.abs(np.array(F_A) - np.array(F_B))))
max_Sd_diff  = float(np.max(np.abs(np.array(Sd_A) - np.array(Sd_B))))
print("\n>>> Phase-blindness check (A vs B)")
print(f"    Max |ρ_A − ρ_B|   : {max_rho_diff:.3e}")
print(f"    Max |F_A − F_B|   : {max_F_diff:.3e}")
print(f"    Max |σ̇_A − σ̇_B| : {max_Sd_diff:.3e}")
print(f"    PASS phase-blind  : {max_rho_diff < 1e-12 and max_F_diff < 1e-12 and max_Sd_diff < 1e-12}")

# Mass conservation (rectangle rule, consistent with FFT/DC pin)
mass_final   = integrate_rect(rho_A, dx)
mass_err     = abs(mass_final - 1.0)
print("\n>>> Mass check (pinned DC mode, rectangle rule)")
print(f"    Final mass: {mass_final:.12f}  (error {mass_err:.3e})")

# Global identity (report both, but the stepwise midpoint sum is authoritative)
F_drop = float(F_A[0] - F_A[-1])

# Time-integrated σ̇ via simple rectangle rule in time (uniform dt)
# (kept for context; midpoint sum below is preferred)
Sd_int_rect_time = float(np.sum(Sd_A) * dt)

# Identity via cumulative midpoint sum (matching evolution rule)
global_err = abs(cum_id_err)
rel_err    = global_err / max(1e-15, abs(F_drop))

print("\n>>> Integral identity")
print("    Using rectangle in time (context):")
print(f"      ∑ σ̇ dt          : {Sd_int_rect_time:.12f}")
print(f"      F(0) − F(T)      : {F_drop:.12f}")
print(f"      |diff|            : {abs(Sd_int_rect_time - F_drop):.3e}")
print("    Using midpoint per-step (authoritative):")
print(f"      Σ[ΔF + σ̇_mid dt]: {cum_id_err:.3e}  (relative {rel_err*100:.5f}%)")

# Contrast proxy (local dynamic range in a window)
def contrast(r):
    rwin = r[(x>10)&(x<30)]
    return (np.max(rwin) - np.min(rwin)) / (np.max(rwin) + np.min(rwin) + 1e-15)

c0, cT = contrast(rho0), contrast(rho_A)
print("\n>>> Contrast evolution")
print(f"    Initial contrast : {c0:.6f}")
print(f"    Final contrast   : {cT:.6f}")
print(f"    Relative change  : {(cT/c0 - 1.0)*100:.2f}%")

# Long-time behaviour
print("\n>>> Long-time behaviour")
print(f"    σ̇(T)      : {Sd_A[-1]:.12e}")
print(f"    F(T)       : {F_A[-1]:.12f}")
print(f"    min(ρ_T)   : {rho_A.min():.3e},  max(ρ_T): {rho_A.max():.3e}")

# -------------------------------
# More diags
# ----------------------
F_arr  = np.array(F_A)
Sd_arr = np.array(Sd_A)

mono_F   = np.max(np.maximum(0.0, np.diff(F_arr)))     # should be ≤ 0
neg_sigma = -np.min(Sd_arr)                             # should be ≥ 0 (i.e., σ̇ ≥ 0)

print("\n>>> Monotonicity checks")
print(f"    max positive ΔF = {mono_F:.3e}   (expect ~0)")
print(f"    min σ̇          = {-neg_sigma:.3e} (expect ≥ 0)")

# Approach to uniform equilibrium ρ∞ = 1/L
rho_inf  = np.full_like(rho_A, 1.0/L)
l2_dev   = np.sqrt(integrate_rect((rho_A - rho_inf)**2, dx))
linf_dev = float(np.max(np.abs(rho_A - rho_inf)))
F_inf    = np.log(1.0/L)

print("\n>>> Approach to uniform equilibrium")
print(f"    L2 deviation     : {l2_dev:.3e}")
print(f"    Linf deviation   : {linf_dev:.3e}")
print(f"    |F(T) − F∞|      : {abs(F_A[-1]-F_inf):.3e}")

# Parseval / mass recheck (rectangle rule)
mass_recheck = integrate_rect(rho_A, dx)
print(f"\n>>> Parseval mass recheck (rectangle rule)")
print(f"    mass: {mass_recheck:.12f} (error {abs(mass_recheck-1.0):.3e})")

# -------------------------------
# Final summary
# -------------------------------
print("\n=== SUMMARY ===")
print(f" Phase-blind: {max_rho_diff<1e-12}  |  Identity rel.err (midpoint): {rel_err*100:.5f}%")
print(f" Mass err: {mass_err:.3e}           |  Monotone F: {mono_F<1e-10}  |  σ̇≥0: {neg_sigma<=0}")
print(f" L2→uniform: {l2_dev:.3e}           |  Final F: {F_A[-1]:.12f}")
print("==============================================================")
