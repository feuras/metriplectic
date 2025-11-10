# 01_wave-dispersion_probe-fft_suite.py
# Probe-FFT dispersion, anisotropy, damping, time-reversal, with CSV logs.

import numpy as np, pandas as pd, json, math
from numpy.fft import fftn, ifftn
from dataclasses import dataclass
from time import perf_counter

# Progress bars
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x  # no-op fallback

# Reproducibility
RNG_SEED = 12345
np.random.seed(RNG_SEED)

@dataclass
class SpectralGrid:
    Nx:int; Ny:int; Lx:float; Ly:float
    kx:np.ndarray; ky:np.ndarray; K2:np.ndarray
    X:np.ndarray; Y:np.ndarray

def make_spectral_grid(Nx=128, Ny=128, Lx=2*np.pi, Ly=2*np.pi):
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2*np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    return SpectralGrid(Nx, Ny, Lx, Ly, KX, KY, K2, X, Y)

def leapfrog_init(u0, pi0, grid:SpectralGrid, dt, c=1.0, anisotropy=None, gamma=0.0):
    if anisotropy is None:
        Lop = - (c**2) * grid.K2
    else:
        cx, cy = anisotropy
        Lop = - (cx**2 * grid.kx**2 + cy**2 * grid.ky**2)
    uhat = fftn(u0.copy()); pihat = fftn(pi0.copy())
    pihat_half = pihat + 0.5 * dt * (Lop * uhat - gamma * pihat)
    return uhat, pihat_half, Lop

def leapfrog_step(uhat, pihat_half, Lop, dt, gamma=0.0):
    uhat = uhat + dt * pihat_half
    pihat_half = pihat_half + dt * (Lop * uhat - gamma * pihat_half)
    return uhat, pihat_half

def energy_from_spectral(uhat, pihat_half, grid:SpectralGrid, c=1.0):
    dxdy = (grid.Lx/grid.Nx) * (grid.Ly/grid.Ny)
    pi_energy = 0.5 * np.sum(np.abs(pihat_half)**2) / (grid.Nx*grid.Ny) * dxdy
    grad_energy = 0.5 * (c**2) * np.sum(grid.K2 * np.abs(uhat)**2) / (grid.Nx*grid.Ny) * dxdy
    return pi_energy + grad_energy

# Robust sub-bin interpolation for FFT peaks
def quad_peak_freq(freqs, S_abs):
    k = int(np.argmax(S_abs))
    if 1 <= k <= len(S_abs)-2:
        eps = 1e-12
        y0, y1, y2 = np.log(S_abs[k-1] + eps), np.log(S_abs[k] + eps), np.log(S_abs[k+1] + eps)
        denom = 2*(y0 - 2*y1 + y2)
        if abs(denom) > 1e-14:
            delta = (y0 - y2)/denom
            return freqs[k] + delta*(freqs[1]-freqs[0])
    return freqs[k]

# Fast frequency estimator for single-mode initial data
def estimate_omega_mode_probe(kx_i, ky_i, grid:SpectralGrid, dt, nsteps, c=1.0,
                              anisotropy=None, gamma=0.0):
    # Dispersion eigenvalue for this mode
    if anisotropy is None:
        lam = - (c**2) * (kx_i**2 + ky_i**2)
    else:
        cx, cy = anisotropy
        lam = - (cx**2 * kx_i**2 + cy**2 * ky_i**2)

    # FFT normalisation: u0 = cos(k·x) gives ±k spikes of amplitude 0.5*Ntot
    Ntot = grid.Nx * grid.Ny
    uhat_plus = 0.5 * Ntot + 0j
    uhat_minus = 0.5 * Ntot + 0j
    pihat_plus = 0.0 + 0j
    pihat_minus = 0.0 + 0j

    # Leapfrog half-step for pihat
    pihat_plus += 0.5 * dt * (lam * uhat_plus - gamma * pihat_plus)
    pihat_minus += 0.5 * dt * (lam * uhat_minus - gamma * pihat_minus)

    # Probe location
    x0_idx = max(0, grid.Nx // 17)
    y0_idx = max(0, grid.Ny // 13)
    x0 = grid.X[x0_idx, y0_idx]
    y0 = grid.Y[y0_idx, x0_idx] if False else grid.Y[x0_idx, y0_idx]  # clarity

    phase_plus = np.exp(1j * (kx_i * x0 + ky_i * y0))
    phase_minus = np.conj(phase_plus)

    sig = np.empty(nsteps, dtype=float)

    for n in tqdm(range(nsteps), desc="probe evolution", leave=False):
        # update the two spectral coefficients
        uhat_plus += dt * pihat_plus
        uhat_minus += dt * pihat_minus
        pihat_plus += dt * (lam * uhat_plus - gamma * pihat_plus)
        pihat_minus += dt * (lam * uhat_minus - gamma * pihat_minus)
        # reconstruct real probe value
        u_val = (uhat_plus * phase_plus + uhat_minus * phase_minus) / Ntot
        sig[n] = np.real(u_val)

    # detrend, window, FFT
    sig -= sig.mean()
    window = np.hanning(len(sig))
    aw = sig * window
    padlen = 2 * len(aw)
    S = np.fft.rfft(aw, n=padlen)
    freqs = np.fft.rfftfreq(padlen, d=dt)
    Sabs = np.abs(S)
    fhat = quad_peak_freq(freqs, Sabs)
    return 2*np.pi * float(fhat)

def fit_origin_jackknife(k2, w2):
    x = np.asarray(k2, dtype=float); y = np.asarray(w2, dtype=float)
    slope = (x*y).sum()/(x*x).sum()
    yhat = slope*x
    R2 = 1 - ((y - yhat)**2).sum()/(y**2).sum()
    N = len(x); slopes=[]
    for i in range(N):
        m = np.ones(N, dtype=bool); m[i]=False
        xi, yi = x[m], y[m]
        slopes.append((xi*yi).sum()/(xi*xi).sum())
    slopes=np.array(slopes, dtype=float)
    slope_std = math.sqrt((N-1)/N * np.sum((slopes - slopes.mean())**2))
    return slope, slope_std, R2

# Core test routines
def run_isotropic_convergence():
    configs = [
        {'Nx':128,'Ny':128,'dt':0.0020,'nsteps':12000},
        {'Nx':256,'Ny':256,'dt':0.0020,'nsteps':12000}
    ]
    modes = [(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2),(3,0),(0,3),(3,1),(1,3)]
    rows=[]
    for cfg in tqdm(configs, desc="isotropic_convergence"):
        grid = make_spectral_grid(cfg['Nx'], cfg['Ny'])
        k2, w2 = [], []
        for kx_i, ky_i in tqdm(modes, desc="modes", leave=False):
            w = estimate_omega_mode_probe(kx_i, ky_i, grid, cfg['dt'], cfg['nsteps'])
            k2.append(kx_i**2 + ky_i**2); w2.append(w*w)
        slope, slope_std, R2 = fit_origin_jackknife(k2, w2)

        # Energy conservation test
        u0 = np.cos(2*grid.X + 1*grid.Y); pi0=np.zeros_like(u0)
        uhat, pihat_half, Lop = leapfrog_init(u0, pi0, grid, cfg['dt'])
        samples=200; E=np.empty(samples)
        steps_per_sample = max(1, cfg['nsteps']//samples)
        for s in tqdm(range(samples), desc="energy samples", leave=False):
            for _ in range(steps_per_sample):
                uhat, pihat_half = leapfrog_step(uhat, pihat_half, Lop, cfg['dt'])
            E[s] = energy_from_spectral(uhat, pihat_half, grid)
        # tiny polish: cover any remainder steps to reach exactly nsteps
        rem = cfg['nsteps'] - steps_per_sample*samples
        for _ in range(rem):
            uhat, pihat_half = leapfrog_step(uhat, pihat_half, Lop, cfg['dt'])
        if rem > 0:
            E[-1] = energy_from_spectral(uhat, pihat_half, grid)

        drift = (E.max()-E.min())/E.mean()
        rows.append({'Nx':cfg['Nx'],'Ny':cfg['Ny'],'dt':cfg['dt'],'nsteps':cfg['nsteps'],
                     'c2_est':slope,'c2_std_jk':slope_std,'R2':R2,'energy_rel_drift':drift})
    df = pd.DataFrame(rows); df.to_csv('isotropic_convergence.csv', index=False)
    return df

def run_isotropic_scatter(Nx=256, Ny=256, dt=0.0020, nsteps=12000):
    grid = make_spectral_grid(Nx, Ny)
    modes=[(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2),(3,0),(0,3),(3,1),(1,3),(4,0),(0,4),(4,1),(1,4)]
    k2, w2 = [], []
    for kx_i, ky_i in tqdm(modes, desc="isotropic_scatter modes"):
        w = estimate_omega_mode_probe(kx_i, ky_i, grid, dt, nsteps)
        k2.append(kx_i**2 + ky_i**2); w2.append(w*w)
    slope, slope_std, R2 = fit_origin_jackknife(k2, w2)
    pd.DataFrame({'k2':k2,'omega2':w2}).to_csv('isotropic_dispersion_scatter.csv', index=False)
    return slope, slope_std, R2

def run_anisotropy(Nx=256, Ny=256, dt=0.0020, nsteps=12000, cx=1.0, cy=0.8):
    grid = make_spectral_grid(Nx, Ny)
    modes=[(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2),(3,0),(0,3),(3,1),(1,3),(4,0),(0,4),(4,1),(1,4)]
    k2, w2, kx2, ky2 = [], [], [], []
    for kx_i, ky_i in tqdm(modes, desc="anisotropy modes"):
        w = estimate_omega_mode_probe(kx_i, ky_i, grid, dt, nsteps, anisotropy=(cx, cy))
        k2.append(kx_i**2 + ky_i**2); w2.append(w*w); kx2.append(kx_i**2); ky2.append(ky_i**2)
    slope, slope_std, R2 = fit_origin_jackknife(k2, w2)
    A = np.vstack([kx2, ky2]).T; coeffs, *_ = np.linalg.lstsq(A, w2, rcond=None)
    a_est, b_est = coeffs
    pd.DataFrame({'kx2':kx2,'ky2':ky2,'omega2':w2}).to_csv('anisotropy_ellipse_fit.csv', index=False)
    return slope, slope_std, R2, a_est, b_est

def run_damping(Nx=256, Ny=256, dt=0.0020, nsteps=12000, gamma=0.1):
    grid = make_spectral_grid(Nx, Ny)
    u0 = np.cos(2*grid.X + 1*grid.Y); pi0 = np.zeros_like(u0)
    uhat, pihat_half, Lop = leapfrog_init(u0, pi0, grid, dt, gamma=gamma)
    samples=400; E=np.empty(samples); T=np.empty(samples)
    steps_per_sample = max(1, nsteps//samples)
    for s in tqdm(range(samples), desc="damping samples"):
        for _ in range(steps_per_sample):
            uhat, pihat_half = leapfrog_step(uhat, pihat_half, Lop, dt, gamma=gamma)
        E[s] = energy_from_spectral(uhat, pihat_half, grid); T[s] = (s+1)*steps_per_sample*dt
    # tiny polish: cover any remainder to hit exactly nsteps
    rem = nsteps - steps_per_sample*samples
    for _ in range(rem):
        uhat, pihat_half = leapfrog_step(uhat, pihat_half, Lop, dt, gamma=gamma)
    if rem > 0:
        E[-1] = energy_from_spectral(uhat, pihat_half, grid)
        T[-1] = nsteps*dt
    pd.DataFrame({'t':T,'E':E}).to_csv('energy_decay.csv', index=False)
    return (E[0]-E[-1])/E[0]

def run_time_reversal(Nx=128, Ny=128, dt=0.0020, nsteps=12000):
    grid = make_spectral_grid(Nx, Ny)
    u0 = np.zeros((Nx,Ny)); pi0 = np.zeros_like(u0)
    for i in range(1,4):
        for j in range(0,4):
            phase = np.random.rand()*2*np.pi
            u0 += np.cos(i*grid.X + j*grid.Y + phase)
    uhat, pihat_half, Lop = leapfrog_init(u0, pi0, grid, dt, gamma=0.0)
    for _ in tqdm(range(nsteps), desc="time-forward", leave=False):
        uhat, pihat_half = leapfrog_step(uhat, pihat_half, Lop, dt)
    # reverse momentum half-step
    pihat_half = -pihat_half
    for _ in tqdm(range(nsteps), desc="time-reverse", leave=False):
        uhat, pihat_half = leapfrog_step(uhat, pihat_half, Lop, dt)
    u_final = np.real(ifftn(uhat))
    return np.linalg.norm(u_final - u0)/np.linalg.norm(u0)

# Run suite
if __name__ == "__main__":
    t_all = perf_counter()

    print("\n[section] Isotropic convergence: probe-FFT dispersion")
    t0 = perf_counter()
    conv_df = run_isotropic_convergence()
    t1 = perf_counter()
    for row in conv_df.to_dict(orient='records'):
        print(f"  grid {row['Nx']}x{row['Ny']}: c^2={row['c2_est']:.6f} ± {row['c2_std_jk']:.2e}, "
              f"R^2={row['R2']:.9f}, energy drift={100*row['energy_rel_drift']:.2f}%")

    print("\n[section] Isotropic scatter: k2 vs omega2 fit")
    c2_est, c2_std, R2 = run_isotropic_scatter()
    t2 = perf_counter()
    print(f"  fit: c^2={c2_est:.6f} ± {c2_std:.2e}, R^2={R2:.9f} (CSV saved)")

    print("\n[section] Anisotropy: ellipse fit vs wrong isotropic model")
    c2_wrong, c2_wrong_std, R2_wrong, a_est, b_est = run_anisotropy()
    t3 = perf_counter()
    cx_est = math.sqrt(max(a_est, 0.0)); cy_est = math.sqrt(max(b_est, 0.0))
    print(f"  wrong isotropic model: c^2={c2_wrong:.6f} ± {c2_wrong_std:.2e}, R^2={R2_wrong:.6f}")
    print(f"  ellipse fit: cx={cx_est:.6f}, cy={cy_est:.6f} (CSV saved)")

    print("\n[section] Damping: energy decay with gamma")
    decay = run_damping()
    t4 = perf_counter()
    print(f"  decayed fraction over window: {decay:.6f} (CSV saved)")

    print("\n[section] Time reversal: round trip error")
    rev_err = run_time_reversal()
    t5 = perf_counter()
    print(f"  relative L2 error after reverse: {rev_err:.6f}")

    result = {
        'rng_seed': RNG_SEED,
        'timings_seconds': {
            'isotropic_convergence': round(t1 - t0, 3),
            'isotropic_scatter': round(t2 - t1, 3),
            'anisotropy': round(t3 - t2, 3),
            'damping': round(t4 - t3, 3),
            'time_reversal': round(t5 - t4, 3),
            'total': round(t5 - t_all, 3),
        },
        'isotropic_convergence': conv_df.to_dict(orient='records'),
        'isotropic_fit': {'c2_est': float(c2_est), 'c2_std': float(c2_std), 'R2': float(R2)},
        'anisotropy_wrong_model': {'c2_est': float(c2_wrong), 'c2_std': float(c2_wrong_std), 'R2': float(R2_wrong)},
        'anisotropy_ellipse_fit': {'cx_est': float(cx_est), 'cy_est': float(cy_est)},
        'damping_decayed_fraction': float(decay),
        'time_reversal_L2_error': float(rev_err)
    }

    print("\n[summary] JSON summary below")
    print(json.dumps(result, indent=2))
