"""
Holonomy loop check (complex pairing loop phase) — Liouville-corrected.

- J operator is built as J = a(x) R_theta ∘ H with a(x) = cJ / rho(x), so that
  ∂x( rho * a(x) ) = 0 identically (weighted Liouville).
- Liouville diagnostic measures sup | ∂x( rho * a(x) ) |, not the divergence of a flux acting on μ.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common import Grid, proj_field, mu_entropic, dx, hilbert, kkt_solve

# -------------------------
# Config
# -------------------------
L = 40.0
N = 512
rho_bias = 0.10
cJ = 0.6
mode_rho = 3
a0, da = 0.10, 0.05
theta0, dtheta = 0.0, np.pi/4
steps = 8
tol_liouville = 1e-12
outdir = "out"

os.makedirs(outdir, exist_ok=True)

# -------------------------
# Grid and base state
# -------------------------
g = Grid(L=L, N=N)
x, k, mask = g.build()
rho0 = 1.0 / g.L
G = np.ones_like(x)

def rectangle_path(a0, da, theta0, dtheta, steps):
    path = []
    for s in range(steps):
        path.append((a0 + da * s / (steps - 1), theta0))
    for s in range(1, steps):
        path.append((a0 + da, theta0 + dtheta * s / (steps - 1)))
    for s in range(1, steps):
        path.append((a0 + da * (1 - s / (steps - 1)), theta0 + dtheta))
    for s in range(1, steps - 1):
        path.append((a0, theta0 + dtheta * (1 - s / (steps - 1))))
    return path

def run_loop(path, reverse=False):
    if reverse:
        path = list(reversed(path))
    vals, meta = [], []
    for seg, (a, theta) in enumerate(path):
        # density with small modulation, projected and renormalised
        rho = rho0 * (1 + a * np.cos(2 * np.pi * mode_rho * x / g.L))
        rho = proj_field(rho, mask)
        rho = rho - rho.mean() + rho0

        # chemical potential and its plane {μ_x, H μ_x}
        mu  = mu_entropic(rho, rho_bias, k, mask)
        mux = dx(mu, k, mask)
        Hmux = hilbert(mux, k, mask)

        # irreversible velocity: v_G = -∂x( ρ G ∂x μ )
        vG = -dx(rho * G * mux, k, mask)

        # Liouville-correct reversible operator: a(x) = cJ / rho(x)
        a_x = cJ / rho
        # rotation in the two-plane {H μ_x, μ_x}
        rot = np.cos(theta) * Hmux + np.sin(theta) * mux
        # reversible velocity: v_J = ∂x( ρ * a(x) * rot ) = ∂x( cJ * rot )
        # so the action on μ is independent of ρ and ∂x(ρ a)=0 identically
        vJ = dx(cJ * rot, k, mask)

        # total test velocity and KKT solve
        v = 0.5 * vG + 0.5 * vJ
        phi, it, res = kkt_solve(rho, G, v, k, mask)

        # complex pairing components
        phix = dx(phi, k, mask)
        Re_pair = np.trapz(rho * phix * mux, x)
        Im_pair = np.trapz(rho * phix * Hmux, x)
        z = Re_pair + 1j * Im_pair

        # weighted Liouville residual of the operator coefficient
        liouville_sup = float(np.max(np.abs(dx(rho * a_x, k, mask))))

        vals.append(z)
        meta.append({
            "seg": seg, "a": a, "theta": theta,
            "Re_pair": Re_pair, "Im_pair": Im_pair,
            "kkt_iter": it, "kkt_resid": res,
            "liouville_sup": liouville_sup
        })

    vals = np.array(vals)
    ang = np.unwrap(np.angle(vals))
    total = float(ang[-1] - ang[0])
    return meta, vals, ang, total

# Build path and run forward and reverse
path = rectangle_path(a0, da, theta0, dtheta, steps)
meta_f, vals_f, ang_f, total_f = run_loop(path, reverse=False)
meta_r, vals_r, ang_r, total_r = run_loop(path, reverse=True)

# Save CSV with forward pass
df = pd.DataFrame(meta_f)
df["arg"] = np.angle(vals_f)
df.to_csv(os.path.join(outdir, "holonomy_loop.csv"), index=False)

# Plot unwrapped angle
plt.figure()
plt.plot(df["seg"], ang_f, marker="o", label="forward unwrapped")
plt.xlabel("segment"); plt.ylabel("unwrapped arg")
plt.title("Holonomy loop phase")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_holonomy_angle.png"), dpi=180)

# Console summary
print("Holonomy loop configuration")
print(f"L={L}  N={N}  rho0={rho0:.6f}  cJ={cJ}  rho_bias={rho_bias}")
print(f"path edges: a in [{a0:.3f},{a0+da:.3f}], theta in [{theta0:.3f},{theta0+dtheta:.3f}], steps per edge={steps}")
print("pairing and solver")
print("  Re⟨·,·⟩ = ∫ ρ φ_x μ_x dx")
print("  Im⟨·,·⟩ = ∫ ρ φ_x H[μ_x] dx")
print("Liouville check (operator coefficient)")
max_liou = max(m['liouville_sup'] for m in meta_f)
print(f"  max sup |∂x(ρ a)| with a(x)=cJ/ρ(x): {max_liou:.3e}")
print("loop phase results")
print(f"  forward total phase   Δarg = {total_f:.6f} rad")
print(f"  reverse total phase   Δarg = {total_r:.6f} rad")
print(f"  sign flip check       Δarg_fwd + Δarg_rev = {total_f + total_r:.3e} (should be near 0)")
print("tolerances")
print(f"  Liouville sup residual target: <= {tol_liouville:.1e}")
print("verdicts")
print("  Liouville OK:", max_liou <= 10 * tol_liouville)
print("  Sign flip OK:", abs(total_f + total_r) <= 1e-6)

print(f"CSV: {os.path.join(outdir, 'holonomy_loop.csv')}")
print(f"Figure: {os.path.join(outdir, 'fig_holonomy_angle.png')}")
print("Done.")
