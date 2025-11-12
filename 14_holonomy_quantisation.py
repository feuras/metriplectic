# 14_holonomy_quantisation — mean-zero pairing with quantised holonomy
# Computes the loop holonomy of the complex pairing z = Re + i Im,
# demonstrating a single winding when the control loop encloses a branch point.
# Outputs: out/holonomy_index.csv and out/holonomy_path.npz

import os
import numpy as np
import pandas as pd

# ----- Grid and operators -----
L = 40.0
N = 1024
x = np.linspace(0.0, L, N, endpoint=False)
kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)

# Two-thirds de-aliasing mask
mask = (np.abs(np.fft.fftfreq(N)) * N <= N // 3).astype(float)

def fft(f):
    return np.fft.fft(f)

def ifft(F):
    return np.real(np.fft.ifft(F * mask))

def dx(f):
    return ifft(1j * kx * fft(f))

def hilb(f):
    return ifft(-1j * np.sign(kx) * fft(f))

# ----- Parameters -----
rho0 = 1.0 / L
k = 5                 # base mode for mu, so u has mode k
k2 = 2 * k            # second harmonic couples to u and H[u]

# ----- Complex pairing on the mean-zero subspace -----
def pair(a1, a2, theta):
    # Density carries 2k content. Pairing weight is mean-zero to match KKT subspace.
    rho = rho0 * (1.0 + a1 * np.cos(2.0 * np.pi * k2 * x / L) + a2 * np.sin(2.0 * np.pi * k2 * x / L))
    w = rho - rho0  # mean-zero weight
    mu = np.cos(2.0 * np.pi * k * x / L + theta)
    u = -dx(mu)     # sign consistent with paper convention
    Re = np.trapz(w * u * u, x)
    Im = np.trapz(w * u * hilb(u), x)
    return Re + 1j * Im

# ----- Control loop over (a1, a2) on a circle, with optional centre and orientation -----
def loop(n=144, r=0.10, theta=0.0, clockwise=False, a1c=0.0, a2c=0.0, return_path=False):
    vals = []
    a1s, a2s = [], []
    for j in range(n + 1):
        ang = 2.0 * np.pi * ((-j) if clockwise else j) / n
        a1 = a1c + r * np.cos(ang)
        a2 = a2c + r * np.sin(ang)
        z = pair(a1, a2, theta)  # theta fixed to avoid double winding
        vals.append(z)
        a1s.append(a1)
        a2s.append(a2)
    z = np.array(vals)
    arg = np.unwrap(np.angle(z))
    total = float(arg[-1] - arg[0])
    idx = int(np.round(total / (2.0 * np.pi)))
    if return_path:
        return total, idx, z, np.array(a1s), np.array(a2s)
    return total, idx

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)

    # Main loop: should enclose one branch point and wind once
    total, idx, z, a1s, a2s = loop(n=144, r=0.10, theta=0.0, clockwise=False, return_path=True)
    total_rev, idx_rev = loop(n=144, r=0.10, theta=0.0, clockwise=True, return_path=False)

    # Control loop: small radius, offset centre so it does not enclose the origin
    total0, idx0, z0, *_ = loop(n=144, r=0.01, theta=0.0, clockwise=False, a1c=0.04, a2c=0.0, return_path=True)

    # Assertions for reviewer robustness
    assert abs(abs(total) - 2.0 * np.pi) < 5e-3, "Total phase not approx 2π for enclosing loop"
    assert idx in (+1, -1), "Winding index not ±1 for enclosing loop"
    assert np.sign(total) == np.sign(idx), "Phase sign and index disagree for enclosing loop"

    # Guard: small offset loop should not wind and should stay away from the origin in z
    guard_min_abs = np.min(np.abs(z0))
    assert guard_min_abs > 1e-6, f"Small loop path approaches origin (min|z|={guard_min_abs:.2e})"
    assert abs(total0) < 5e-3 and idx0 == 0, "Holonomy should vanish for small non-enclosing loop"

    assert idx_rev == -idx, "Reversed orientation should flip the winding index"

    # Save CSV summary
    df = pd.DataFrame([{
        "total_phase": total,
        "index": idx,
        "total_phase_reversed": total_rev,
        "index_reversed": idx_rev,
        "total_phase_small_loop": total0,
        "index_small_loop": idx0,
        "L": L,
        "N": N,
        "r_main": 0.10,
        "r_small": 0.01,
        "center_small": [0.04, 0.0],
        "k": k,
        "k2": k2,
        "pairing": "mean-zero",
        "mask": "two_thirds",
        "operator": "v=-dx(mu)"
    }])
    df.to_csv("out/holonomy_index.csv", index=False)

    # Save the complex path for audit and plotting
    np.savez(
        "out/holonomy_path.npz",
        z=z,
        Re=z.real,
        Im=z.imag,
        a1=a1s,
        a2=a2s,
        z_small=z0,
        L=L,
        N=N,
        r_main=0.10,
        r_small=0.01,
        center_small=np.array([0.04, 0.0]),
        k=k,
        k2=k2,
        pairing="mean-zero",
        mask="two_thirds",
        operator="v=-dx(mu)"
    )

    # Human readable printout
    print("phase, index:", total, idx)
    print("phase (reversed), index (reversed):", total_rev, idx_rev)
    print("phase (small loop), index (small loop):", total0, idx0)
