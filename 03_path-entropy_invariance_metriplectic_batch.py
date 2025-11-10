# Path-integrated entropy invariance: high-rigor batch (single cell)
# Console-only, no files. Runs multiple J amplitudes and replicates, with stronger compute.

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

# --------------------------------
# Spectral operators, periodic 2D
# --------------------------------

@dataclass
class Spectral2D:
    Nx: int
    Ny: int
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi
    dealias: bool = True

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        kx = np.fft.fftfreq(self.Nx, d=self.dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(self.Ny, d=self.dy) * 2.0 * np.pi
        self.kx = kx.reshape(self.Nx, 1) * np.ones((1, self.Ny))
        self.ky = ky.reshape(1, self.Ny) * np.ones((self.Nx, 1))
        self.k2 = self.kx**2 + self.ky**2

        if self.dealias:
            kx_cut = (2.0 / 3.0) * np.max(np.abs(kx))
            ky_cut = (2.0 / 3.0) * np.max(np.abs(ky))
            self.dealias_mask = ((np.abs(self.kx) <= kx_cut) & (np.abs(self.ky) <= ky_cut)).astype(float)
        else:
            self.dealias_mask = np.ones_like(self.k2)

        x = np.linspace(0.0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0.0, self.Ly, self.Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

    def _fft(self, f: np.ndarray) -> np.ndarray:
        return np.fft.fft2(f)

    def _ifft(self, F: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(F).real

    def grad(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        F = self._fft(f)
        Fx = 1j * self.kx * F * self.dealias_mask
        Fy = 1j * self.ky * F * self.dealias_mask
        return self._ifft(Fx), self._ifft(Fy)

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        F = self._fft(f)
        return self._ifft(- self.k2 * F * self.dealias_mask)

    def divergence(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        Fx = self._fft(fx)
        Fy = self._fft(fy)
        Div = 1j * self.kx * Fx + 1j * self.ky * Fy
        Div *= self.dealias_mask
        return self._ifft(Div)

    def spectral_filter(self, f: np.ndarray) -> np.ndarray:
        return self._ifft(self._fft(f) * self.dealias_mask)


# ----------------------------
# Free energy, mu, sigma-dot
# ----------------------------

def chemical_potential(rho: np.ndarray, spec: Spectral2D, lam: float) -> np.ndarray:
    rho_safe = np.clip(rho, 1e-16, None)
    mu = np.log(rho_safe)
    if lam != 0.0:
        mu -= lam * spec.laplacian(rho_safe)
    return mu

def free_energy(rho: np.ndarray, spec: Spectral2D, lam: float) -> float:
    rho_safe = np.clip(rho, 1e-16, None)
    ent = rho_safe * np.log(rho_safe)
    if lam != 0.0:
        gx, gy = spec.grad(rho_safe)
        fisher = 0.5 * lam * (gx*gx + gy*gy)
        return float(np.mean(ent + fisher))
    return float(np.mean(ent))

def dot_sigma(rho: np.ndarray, mu: np.ndarray, spec: Spectral2D, G: np.ndarray) -> float:
    mux, muy = spec.grad(mu)
    gx = G[0, 0] * mux + G[0, 1] * muy
    gy = G[1, 0] * mux + G[1, 1] * muy
    return float(np.mean(rho * (mux*gx + muy*gy)))


# ----------------------------
# Reversible channel choices
# ----------------------------

def J_apply(mux: np.ndarray, muy: np.ndarray, spec: Spectral2D, amp: float, mode: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    if amp == 0.0:
        return np.zeros_like(mux), np.zeros_like(muy)
    X, Y = spec.X, spec.Y
    if mode == 1:
        s = np.sin(X) * np.cos(Y)
    elif mode == 2:
        s = np.sin(2*X + 0.3*np.cos(Y)) * np.cos(3*Y + 0.2*np.sin(X))
    else:
        s = np.cos(3*X) * np.cos(2*Y) + 0.5*np.sin(4*X + Y)
    jmux = - s * muy
    jmuy =   s * mux
    return amp * jmux, amp * jmuy


# ----------------------------
# Initial state
# ----------------------------

def make_initial_state(spec: Spectral2D, rng: np.random.Generator, rho_min: float, smooth_sigma: float = 3.0) -> np.ndarray:
    theta = rng.normal(0.0, 1.0, size=(spec.Nx, spec.Ny))
    F = np.fft.fft2(theta)
    kx, ky = spec.kx, spec.ky
    filt = np.exp(-(kx**2 + ky**2) / (2.0 * smooth_sigma**2))
    rho = np.exp(np.fft.ifft2(F * filt).real)
    rho = rho_min + rho
    rho /= np.mean(rho)
    return rho


# ----------------------------
# Time step: Heun RK2, positivity floor, mass 1
# Sign convention: dissipative flux uses + rho * G ∇mu so that dF/dt = - dot_sigma
# ----------------------------

def step_rk2(rho: np.ndarray, dt: float, spec: Spectral2D, G: np.ndarray, lam: float,
             J_amp: float, J_mode: int, rho_min: float) -> np.ndarray:
    def rhs(cur_rho: np.ndarray) -> np.ndarray:
        mu = chemical_potential(cur_rho, spec, lam)
        mux, muy = spec.grad(mu)
        jx, jy = J_apply(mux, muy, spec, J_amp, J_mode)
        gx = G[0, 0] * mux + G[0, 1] * muy
        gy = G[1, 0] * mux + G[1, 1] * muy
        fx = cur_rho * (jx + gx)
        fy = cur_rho * (jy + gy)
        fx = spec.spectral_filter(fx); fy = spec.spectral_filter(fy)
        return spec.divergence(fx, fy)

    k1 = rhs(rho)
    rho1 = np.clip(rho + dt*k1, rho_min, None); rho1 /= np.mean(rho1)
    k2 = rhs(rho1)
    rho_new = np.clip(rho + 0.5*dt*(k1 + k2), rho_min, None); rho_new /= np.mean(rho_new)
    return rho_new


# ----------------------------
# Runner, energy-balance, batch
# ----------------------------

def energy_balance_max_err(t: np.ndarray, F: np.ndarray, ds: np.ndarray) -> Tuple[float, float]:
    if len(t) < 3:
        return float("nan"), float("nan")
    dFdt = np.gradient(F, t)
    inf = float(np.max(np.abs(dFdt + ds)))
    rel = inf / max(np.max(np.abs(ds)) + 1e-30, 1e-30)
    return inf, rel

def run_path(spec: Spectral2D, G: np.ndarray, lam: float, dt: float, steps: int,
             target_drop: float, min_steps: int, rho0: np.ndarray,
             J_amp: float, J_mode: int, rho_min: float,
             tag: str, progress_every: int) -> Dict[str, Any]:

    rho = rho0.copy()
    F0 = free_energy(rho, spec, lam)
    target_F = (1.0 - target_drop) * F0
    t_series = []; F_series = []; ds_series = []
    t = 0.0

    print(f"[{tag}] start F0={F0:.8e}, target F={target_F:.8e}")
    for n in range(steps + 1):
        mu = chemical_potential(rho, spec, lam)
        ds = dot_sigma(rho, mu, spec, G)
        Fcur = free_energy(rho, spec, lam)
        t_series.append(t); F_series.append(Fcur); ds_series.append(ds)

        if (n % progress_every == 0) or (n == steps) or (Fcur <= target_F and n >= min_steps):
            drop = (F0 - Fcur) / max(F0, 1e-16)
            print(f"[{tag}] step {n:6d}  t={t:.6f}  F={Fcur:.8e}  σ̇={ds:.8e}  drop={100*drop:.3f}%")

        if (Fcur <= target_F) and (n >= min_steps):
            print(f"[{tag}] hit target at step {n}, t={t:.6f}")
            break

        rho = step_rk2(rho, dt, spec, G, lam, J_amp, J_mode, rho_min)
        t += dt

    t = np.asarray(t_series); F = np.asarray(F_series); ds = np.asarray(ds_series)
    S_total = float(np.trapezoid(ds, x=t))
    inf, rel = energy_balance_max_err(t, F, ds)
    return {
        "t": t, "F": F, "dot_sigma": ds,
        "S_total": S_total, "F0": float(F0), "F_end": float(F[-1]),
        "arrive_t": float(t[-1]),
        "steps": int(len(t) - 1),
        "bal_inf": inf, "bal_rel": rel
    }


@dataclass
class Params:
    Nx: int = 192
    Ny: int = 192
    lam: float = 0.0          # set small positive, e.g. 0.05, and reduce dt if req
    dt: float = 1e-4
    steps: int = 200000
    target_drop: float = 0.20
    min_steps: int = 300
    rho_min: float = 1e-6
    progress_every: int = 200
    J_modes: Tuple[int, ...] = (1, 2)       # two spatial patterns to avoid trivial alignment
    J_amps: Tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)
    replicates: int = 3
    seed0: int = 12345

def batch_main(p: Params) -> None:
    print("Params:", p)
    spec = Spectral2D(p.Nx, p.Ny, dealias=True)
    G = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    rows = []
    rep = 0
    for r in range(p.replicates):
        rng = np.random.default_rng(p.seed0 + r)
        rho0 = make_initial_state(spec, rng, rho_min=p.rho_min, smooth_sigma=3.5)

        for mode in p.J_modes:
            # Baseline J=0
            res0 = run_path(spec, G, p.lam, p.dt, p.steps, p.target_drop, p.min_steps,
                            rho0, J_amp=0.0, J_mode=mode, rho_min=p.rho_min,
                            tag=f"rep{r}-mode{mode}-J0", progress_every=p.progress_every)
            Fdrop0 = res0["F0"] - res0["F_end"]

            for Jamp in p.J_amps[1:]:
                tag = f"rep{r}-mode{mode}-J{Jamp:g}"
                resJ = run_path(spec, G, p.lam, p.dt, p.steps, p.target_drop, p.min_steps,
                                rho0, J_amp=Jamp, J_mode=mode, rho_min=p.rho_min,
                                tag=tag, progress_every=p.progress_every)
                FdropJ = resJ["F0"] - resJ["F_end"]

                row = {
                    "rep": r, "mode": mode, "J": Jamp,
                    "S_J0": res0["S_total"], "S_J": resJ["S_total"],
                    "dS_abs": abs(res0["S_total"] - resJ["S_total"]),
                    "Fdrop_J0": Fdrop0, "Fdrop_J": FdropJ,
                    "SminusF_J0": abs(res0["S_total"] - Fdrop0),
                    "SminusF_J": abs(resJ["S_total"] - FdropJ),
                    "bal_inf_J0": res0["bal_inf"], "bal_rel_J0": res0["bal_rel"],
                    "bal_inf_J": resJ["bal_inf"], "bal_rel_J": resJ["bal_rel"],
                    "t_J0": res0["arrive_t"], "t_J": resJ["arrive_t"],
                    "steps_J0": res0["steps"], "steps_J": resJ["steps"],
                }
                rows.append(row)
                rep += 1
                print(f"[SUMMARY] rep={r} mode={mode} J={Jamp:g}  |ΔS|={row['dS_abs']:.3e}  "
                      f"|S-F|J0={row['SminusF_J0']:.3e} |S-F|J={row['SminusF_J']:.3e}  "
                      f"t(J0)={row['t_J0']:.4f}  t(J)={row['t_J']:.4f}")

    # Aggregate
    if not rows:
        print("No rows produced.")
        return

    dS = np.array([r["dS_abs"] for r in rows])
    e0 = np.array([r["SminusF_J0"] for r in rows])
    eJ = np.array([r["SminusF_J"] for r in rows])
    tdiff = np.array([abs(r["t_J"] - r["t_J0"]) for r in rows])

    print("\n=== Batch summary across all runs ===")
    print(f"|ΔS_total|:   min={dS.min():.3e}  med={np.median(dS):.3e}  max={dS.max():.3e}")
    print(f"|S-Fdrop| J0: min={e0.min():.3e}  med={np.median(e0):.3e}  max={e0.max():.3e}")
    print(f"|S-Fdrop| J>0:min={eJ.min():.3e}  med={np.median(eJ):.3e}  max={eJ.max():.3e}")
    print(f"|Δt| between paths: min={tdiff.min():.3e}  med={np.median(tdiff):.3e}  max={tdiff.max():.3e}")

# ---------------- Run with strong defaults ----------------
if __name__ == "__main__":
    params = Params(
        Nx=192, Ny=192,   # increase to 256x256+ for heavier runs
        lam=0.0,          # maybe try lam=0.05 with dt=5e-5 for Fisher
        dt=1e-4,
        steps=200000,
        target_drop=0.20,
        min_steps=300,
        rho_min=1e-6,
        progress_every=200,
        J_modes=(1, 2),
        J_amps=(0.0, 2.0, 5.0, 10.0),
        replicates=3,
        seed0=12345
    )
    batch_main(params)
