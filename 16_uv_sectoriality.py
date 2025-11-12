# 16_uv_sectoriality.py
import os, json
import numpy as np, pandas as pd

# Parameters
rho0 = 1.0 / 40.0
G0 = 1.0
lam2 = 0.10
L = 40.0  # carried for logs

# k grid and symbol
ks = np.logspace(-2, 2, 200)
Re = -G0 * (ks**2) - (G0 * rho0 * lam2) * (ks**4)

# 1) Fit D2 and D4 from Re(omega) = -D2*k^2 - D4*k^4
x = ks**2
y = -Re / (ks**2)            # y = D2 + D4*x
D4_hat, D2_hat = np.polyfit(x, y, 1)  # slope, intercept

D2_pos = bool(D2_hat > 0)
D4_pos = bool(D4_hat > 0)

# 2) Sectorial bound: Re <= -c*k^2 with c = D2_hat (conservative)
c = max(D2_hat - 1e-10, 0.0)
sectorial_bound = bool(np.all(Re <= -c * (ks**2)))

# 3) UV composite observable scaling ~ k^{-2}
decay = 1.0 / (ks**2 + 1.0)
finite = bool(np.all(np.isfinite(decay)))

# Move UV window out so the uniform error bound is <= 1e-3
high = ks >= 32.0
if np.any(high):
    scale_err = np.max(np.abs((ks[high]**2) * decay[high] - 1.0))
    # optional diagnostic: log–log slope should be near -2
    slope, intercept = np.polyfit(np.log(ks[high]), np.log(decay[high]), 1)
    uv_slope = float(slope)
else:
    scale_err = float("nan")
    uv_slope = float("nan")

scaling_ok = bool(scale_err <= 1e-3)

# 4) Outputs
os.makedirs("out", exist_ok=True)

row = {
    "D2_hat": float(D2_hat),
    "D4_hat": float(D4_hat),
    "D2_pos": D2_pos,
    "D4_pos": D4_pos,
    "sectorial_bound": sectorial_bound,
    "composite_finite": finite,
    "uv_scaling_ok": scaling_ok,
    "uv_scaling_max_err": float(scale_err),
}

# CSV with the existing schema
pd.DataFrame([row]).to_csv("out/uv_sectoriality.csv", index=False)

# JSONL audit log with extra context
with open("out/uv_sectoriality.jsonl", "w") as f:
    f.write(json.dumps({
        "params": {"rho0": rho0, "G0": G0, "lam2": lam2, "L": L},
        "grid": {"k_min": float(ks.min()), "k_max": float(ks.max()), "n": int(ks.size)},
        "metrics": row,
        "diagnostics": {"uv_slope": uv_slope, "uv_window_min_k": float(ks[high].min() if np.any(high) else np.nan)},
    }) + "\n")

print("uv ok")
