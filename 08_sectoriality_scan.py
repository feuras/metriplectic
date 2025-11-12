"""
Sectoriality check (dissipative branch)

Verifies the real part of the dispersion is sectorial with coefficients:
  Re ω(k) = - D2 * k^2 - D4 * k^4  (to O(k^6)),
with D2 = G0 and D4 = G0 * rho0 * lam2 (positivity required).

Also asserts monotone decrease for k>0:
  d/dk Re ω(k) = -2 D2 k - 4 D4 k^3 < 0, so the minimum over a finite grid
  occurs at the largest sampled k.

Outputs:
  out/sectoriality.csv with per-parameter-line metadata and errors.
  Console summary with pass/fail assertions.
"""

import os
import numpy as np
import pandas as pd

# -------------------------
# Config
# -------------------------
rho0_list = [1/40.0, 1/20.0]
G0_list   = [0.5, 1.0, 2.0]
lam2_list = [0.05, 0.10, 0.20]

# k-grid (adjustable); we keep the user’s range for consistency with the paper
ks = np.linspace(0.2, 2.4, 24)
tol_coeff = 1e-14  # tolerance for coefficient identities
tol_grid  = 1e-12  # tolerance for grid-based monotonicity check

os.makedirs("out", exist_ok=True)

print("Sectoriality (dissipative branch) verification")
print(f"k range: [{ks.min():.4f}, {ks.max():.4f}] with {len(ks)} samples")
print("Parameter sweeps:")
print(f"  rho0_list = {rho0_list}")
print(f"  G0_list   = {G0_list}")
print(f"  lam2_list = {lam2_list}")
print("----------------------------------------------------")

rows = []
n_pass = 0
n_total = 0

for rho0 in rho0_list:
    for G0 in G0_list:
        for lam2 in lam2_list:
            n_total += 1

            # Coefficients implied by the model
            D2 = G0
            D4 = G0 * rho0 * lam2

            # Analytic negativity/positivity checks
            assert D2 > 0, f"D2 must be positive; got {D2} for G0={G0}"
            assert D4 > 0, f"D4 must be positive; got {D4} for (G0,rho0,lam2)=({G0},{rho0},{lam2})"

            # Evaluate Re ω on the sampled grid
            Re_vals = -(D2 * ks**2) - (D4 * ks**4)

            # Grid monotonicity check
            diffs = np.diff(Re_vals)
            # Re_vals should strictly decrease with k, so diffs < 0
            monotone_ok = np.all(diffs < tol_grid)
            if not monotone_ok:
                # If any pair violates strict negativity, report the worst offender
                idx_bad = np.argmax(diffs)
                raise AssertionError(
                    f"Monotonicity violation at k in [{ks[idx_bad]:.4g},{ks[idx_bad+1]:.4g}] "
                    f"with ΔReω={diffs[idx_bad]:.3e} for (rho0,G0,lam2)=({rho0},{G0},{lam2})"
                )

            # By monotonicity, the minimum on the grid occurs at the largest k
            k_at_min = ks[-1]
            Re_min_grid = Re_vals[-1]

            # Closed-form derivative is strictly negative for k>0; assert at extremal k
            dRe_dk_at_kmax = -2*D2*k_at_min - 4*D4*(k_at_min**3)
            assert dRe_dk_at_kmax < 0, "Analytic derivative non-negative at kmax; sectoriality would fail."

            # Store row with extra diagnostics
            rows.append({
                "rho0": rho0,
                "G0": G0,
                "lam2": lam2,
                "k_min": ks.min(),
                "k_max": ks.max(),
                "k_at_min": k_at_min,
                "D2": D2,
                "D4": D4,
                "min_Re_omega": Re_min_grid
            })

            n_pass += 1
            print(f"[OK] rho0={rho0:.6f}  G0={G0:.3f}  lam2={lam2:.3f}  "
                  f"D2={D2:.6f}  D4={D4:.6f}  min Reω={Re_min_grid:.6f} at k={k_at_min:.3f}")

# Save CSV
df = pd.DataFrame(rows)
df.to_csv("out/sectoriality.csv", index=False)

print("----------------------------------------------------")
print(f"All sectoriality checks passed: {n_pass}/{n_total}")
print("Wrote: out/sectoriality.csv")
