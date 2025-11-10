# ============================
# Metriplectic Phase-Blind / Identity-Exact Mini (tight, rectangle rule)
# Experiments:
#   1) Heat-only (λ=0) with per-step DC pinning (exact mass), Simpson time-integration
#   2) Reversible (shift) stirring + heat: σ̇ integrated only on G-steps (Simpson)
# Text-only output; no plots.
# ============================

import numpy as np

# -------------------------------
# Grid and spectral operators
# -------------------------------
N, L = 1024, 40.0
x  = np.linspace(0, L, N, endpoint=False)
dx = L / N
k  = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
ik = 1j*k
k2 = k**2
exp = np.exp

def grad(f):
    return np.fft.ifft(ik*np.fft.fft(f)).real

def mass_rect(r):
    return dx * np.sum(r)

# -------------------------------
# Energetics for λ = 0 (entropy only)
# F[ρ] = ∫ ρ log ρ dx
# σ̇(ρ) = ∫ |∂x ρ|² / ρ dx
# (rectangle rule, consistent with DC pin)
# -------------------------------
def free_energy(rho):
    eps = 1e-15
    r = np.clip(rho, eps, None)
    return dx * np.sum(r * np.log(r))

def sigma_dot(rho):
    eps = 1e-15
    r  = np.clip(rho, eps, None)
    rx = grad(r)
    return dx * np.sum((rx**2) / r)

# -------------------------------
# Exact heat semigroup with per-step DC pin (mass = 1 exactly)
# -------------------------------
M_target_sum = 1.0/dx  # ensures ∫ρ dx = 1
def heat_step_exact_pinned(rho, dt):
    rhok = np.fft.fft(rho)
    rhok *= exp(-k2*dt)
    rhok[0] = M_target_sum
    return np.fft.ifft(rhok).real

def heat_halfstep_exact(rho, dt):
    rhok = np.fft.fft(rho)
    rhok *= exp(-k2*(0.5*dt))
    return np.fft.ifft(rhok).real

# -------------------------------
# Reversible stirring (shift): ρ(x) → ρ(x − v dt)
# -------------------------------
def shift_exact(rho, v, dt):
    return np.fft.ifft(np.fft.fft(rho)*np.exp(-1j*k*v*dt)).real

# -------------------------------
# Initial condition setup (rectangle-rule normalisation)
# -------------------------------
rho_min = 1e-3
c1, c2, w = 12.0, 28.0, 2.0
env  = np.exp(-((x-c1)**2)/(2*w**2)) + np.exp(-((x-c2)**2)/(2*w**2))
rho0 = env*(1.0 + 0.25*np.cos(0.7*(x-10.0)))
rho0 = np.clip(rho0, rho_min, None)

# Gentle spectral prefilter (optional)
alpha = 0.0
if alpha > 0.0:
    Fk   = np.fft.fft(rho0)
    kmax = np.max(np.abs(k)) + 1e-15
    filt = exp(-alpha*(np.abs(k)/kmax)**8)
    rho0 = np.fft.ifft(Fk*filt).real

rho0 /= mass_rect(rho0)  # exact normalisation

# Distinct phases (for phase-blindness checks)
S1 = np.zeros_like(x)
S2 = 0.8*np.sin(0.5*x) + 0.3*np.sin(1.7*x) + 0.9

# -------------------------------
# Phase-blindness spot-check (optional)
# -------------------------------
print("\n=== PHASE-BLINDNESS SPOT-CHECK (G-only evolution, A vs B) ===")
rhoA = rho0.copy()
rhoB = rho0.copy()
steps_pb, dt_pb = 100, 1.0e-3
for _ in range(steps_pb):
    rhoA = heat_step_exact_pinned(rhoA, dt_pb)
    rhoB = heat_step_exact_pinned(rhoB, dt_pb)
print(f"Max |ρA−ρB| after {steps_pb} steps: {np.max(np.abs(rhoA - rhoB)):.3e}")

# -------------------------------
# Experiment 1: Heat-only (λ=0), Simpson quadrature of σ̇
# -------------------------------
dt1, steps1 = 0.2e-3, 30000
rho_A = rho0.copy()
F_A0  = free_energy(rho_A)

Sd_int_A = 0.0
for _ in range(steps1):
    Sd_n   = sigma_dot(rho_A)
    rho_mid = heat_halfstep_exact(rho_A, dt1)
    Sd_mid = sigma_dot(rho_mid)
    rho_A  = heat_step_exact_pinned(rho_A, dt1)
    Sd_np1 = sigma_dot(rho_A)
    Sd_int_A += (dt1/6.0)*(Sd_n + 4.0*Sd_mid + Sd_np1)

F_A1 = free_energy(rho_A)
mass_final_A = mass_rect(rho_A)
F_drop_A = F_A0 - F_A1
err_A = abs(Sd_int_A - F_drop_A)
rel_A = err_A / max(1e-15, abs(F_drop_A))

print("\n=== EXPERIMENT 1: Heat-only (λ=0), exact semigroup, DC pin, Simpson ===")
print(f"N={N}, L={L}, dt={dt1}, steps={steps1}")
print(f"Initial mass: {mass_rect(rho0):.12f}")
print(f"Initial F:    {F_A0:.12f}")
print(f"Final mass:   {mass_final_A:.12f}")
print(f"σ̇(T):        {sigma_dot(rho_A):.12e}")
print(f"F(T):         {F_A1:.12f}")
print(f"∫ σ̇ dt : {Sd_int_A:.12f}")
print(f"F(0)-F(T): {F_drop_A:.12f}")
print(f"|diff| : {err_A:.3e} (relative {rel_A*100:.5f}%)")
print("PASS ∫σ̇=ΔF :", rel_A < 1e-3)

# -------------------------------
# Experiment 2: Reversible stirring + heat (σ̇ only on G steps)
# -------------------------------
v0 = 0.75
nJ = 5
dtJ = 0.1e-3
dtG = 0.1e-3
macro = 6000
T_total = macro*(nJ*dtJ + dtG)

rho_B = rho0.copy()
F_B0 = free_energy(rho_B)
Sd_int_B = 0.0

for _ in range(macro):
    for _ in range(nJ):
        rho_B = shift_exact(rho_B, v0, dtJ)
    Sd_n = sigma_dot(rho_B)
    rho_mid = heat_halfstep_exact(rho_B, dtG)
    Sd_mid = sigma_dot(rho_mid)
    rho_B = heat_step_exact_pinned(rho_B, dtG)
    Sd_np1 = sigma_dot(rho_B)
    Sd_int_B += (dtG/6.0)*(Sd_n + 4.0*Sd_mid + Sd_np1)

F_B1 = free_energy(rho_B)
mass_final_B = mass_rect(rho_B)
F_drop_B = F_B0 - F_B1
err_B = abs(Sd_int_B - F_drop_B)
rel_B = err_B / max(1e-15, abs(F_drop_B))

print("\n=== EXPERIMENT 2: Reversible (shift) stirring + heat (astonishing precheck, Simpson) ===")
print(f"v0={v0}, nJ={nJ}, dtJ={dtJ}, dtG={dtG}, macro={macro}, T={T_total}")
print(f"Final mass: {mass_final_B:.12f}")
print(f"σ̇@last-G:  {sigma_dot(rho_B):.12e}")
print(f"F(T):       {F_B1:.12f}")
print(f"∫ σ̇_G dt : {Sd_int_B:.12f}")
print(f"F(0)-F(T): {F_drop_B:.12f}")
print(f"|diff| : {err_B:.3e} (relative {rel_B*100:.5f}%)")
print("PASS ∫σ̇=ΔF with J:", rel_B < 1e-3)

# -------------------------------
# Summary
# -------------------------------
print("\n=== SUMMARY ===")
print(f"Heat-only: Identity rel.err {rel_A*100:.4f}%  | Mass err {abs(mass_rect(rho_A)-1.0):.3e}")
print(f"J+Heat:    Identity rel.err {rel_B*100:.4f}%  | Mass err {abs(mass_rect(rho_B)-1.0):.3e}")
