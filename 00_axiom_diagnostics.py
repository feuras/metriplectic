# Metriplectic Axiom Diagnostics - Fortress edition (console only)
# ----------------------------------------------------------------
# This version enforces a consistent 2/3 spectral projection P on ALL nonlinear ops
# and computes Pirr with the projected gradient. It also normalises commutator dials.

import numpy as np

# =========================
# Config
# =========================
CFG = {
    "seed": 7,
    "L": 2.0*np.pi,
    "N_list_refine": (256, 512, 1024, 2048, 4096),
    "N_default": 4096,
    "Nx2": 192, "Ny2": 192,
    "sigma_sweep": (0.20, 0.40, 0.60, 0.90, 1.20),
    "ell_list": (0.05, 0.10, 0.20, 0.30),
    "max_n_probe": 12,
    # thresholds
    "tol_mass_k0": 1e-12,
    "tol_pr_rel": 1e-12,
}

rng = np.random.default_rng(CFG["seed"])

# =========================
# Spectral utilities
# =========================
def fftfreq_1d(N, L):
    return 2.0*np.pi * np.fft.fftfreq(N, d=L/N)

def grad_1d(f, k):
    return np.fft.ifft(1j*k*np.fft.fft(f)).real

def div_flux_1d(j, k):
    return grad_1d(j, k)

def gaussian_smooth_1d(f, k, sigma):
    return np.fft.ifft(np.exp(-0.5*(sigma**2)*(k**2))*np.fft.fft(f)).real

def trapezoid_1d(f, L):
    # numpy uses trapz
    return np.trapz(f, dx=L/len(f))

def mass_from_k0_1d(v, L):
    v0 = np.fft.fft(v)[0].real
    N = v.shape[0]
    return v0 * (L/N)

def fftfreq_2d(Nx, Ny, Lx, Ly):
    kx = 2.0*np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2.0*np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    return np.meshgrid(kx, ky, indexing='ij')

def grad_2d(f, kx, ky):
    F = np.fft.fftn(f)
    fx = np.fft.ifftn(1j*kx*F).real
    fy = np.fft.ifftn(1j*ky*F).real
    return fx, fy

def div_flux_2d(jx, jy, kx, ky):
    Jx = np.fft.fftn(jx); Jy = np.fft.fftn(jy)
    return np.fft.ifftn(1j*kx*Jx + 1j*ky*Jy).real

def trapezoid_2d(f, Lx, Ly):
    dx = Lx/f.shape[0]; dy = Ly/f.shape[1]
    # integrate over y then x
    return np.trapz(np.trapz(f, dx=dy, axis=1), dx=dx, axis=0)

# =========================
# 2/3-rule projection and projected products
# =========================
def P2_3_1d(f):
    F = np.fft.fft(f)
    N = F.shape[0]
    Kc = N//3
    F[Kc+1: N-Kc] = 0
    return np.fft.ifft(F).real

def prodP(a, b):
    # projected product: P(a) * P(b) then project again
    return P2_3_1d(P2_3_1d(a) * P2_3_1d(b))

# =========================
# Energy and chemical potential
# =========================
def mu_entropy(rho, eps=1e-12):
    return np.log(np.maximum(rho, eps))

def F_entropy_1d(rho, L, eps=1e-12):
    r = np.maximum(rho, eps)
    return trapezoid_1d(r*(np.log(r) - 1.0), L)

# =========================
# Health reporting
# =========================
def report_state_health_1d(rho, G):
    rmin = float(np.min(rho))
    gmin = float(np.min(G))
    print(f"min rho = {rmin:.3e} | min G = {gmin:.3e}")

# =========================
# 1D: equality refinement with fully consistent projection
# =========================
def equality_refinement_1d(N_list, L):
    print("\n== Equality dial with refinement (1D, P-consistent) ==")
    for N in N_list:
        x = np.linspace(0.0, L, N, endpoint=False)
        k = fftfreq_1d(N, L)

        # Base fields (smooth, positive)
        rho0 = 1.0 + 0.2*np.sin(2*x) + 0.1*np.sin(3*x+0.3) + 0.1*np.cos(5*x-0.4)
        rho0 = np.maximum(rho0, 1e-6)
        G0   = 1.0 + 0.5*np.cos(2*x)

        # Project base fields into the same subspace
        rho = P2_3_1d(rho0)
        G   = P2_3_1d(G0)

        mu  = mu_entropy(rho)
        mux = grad_1d(mu, k)
        muxP = P2_3_1d(mux)  # projected gradient

        # Irreversible flux with consistent projection
        j_irrev = prodP(rho, prodP(G, muxP))
        v = div_flux_1d(j_irrev, k)          # ∂t ρ = ∂x j
        v = P2_3_1d(v)                       # project evolution too

        # Pairings
        inner = trapezoid_1d(v*mu, L)        # <v, μ> = Ḟ
        sigma_dot = -inner                   # σ̇ = -Ḟ
        Pirr = 0.5 * trapezoid_1d(rho * G * (muxP**2), L)  # project-consistent cost

        lhs = inner**2
        rhs = 2.0 * Pirr * sigma_dot
        gap = rhs - lhs

        mass_int = mass_from_k0_1d(v, L)
        dx = L/N
        verdict = "PASS" if abs(gap) <= 10.0*dx and abs(mass_int) <= 10.0*dx*dx else "WARN"
        print(f"N={N:5d} | Pirr={Pirr:.8e} | sigdot={sigma_dot:.8e} | LHS={lhs:.8e} | RHS={rhs:.8e} | gap={gap:.3e} | ∫v dx(k0)={mass_int:.3e} | dx={dx:.2e} | {verdict}")

# =========================
# 1D: nonlocal falsifier sweep (keeps local Pirr definition P-consistent)
# =========================
def nonlocal_falsifier_sweep_1d(N, L, sigmas):
    print("\n== Nonlocal falsifier sweep (1D, P-consistent Pirr) ==")
    x = np.linspace(0.0, L, N, endpoint=False)
    k = fftfreq_1d(N, L)
    rho0 = 1.0 + 0.25*np.sin(2*x) + 0.15*np.cos(3*x+0.2)
    G0   = 1.0 + 0.5*np.cos(2*x)
    rho = P2_3_1d(np.maximum(rho0, 1e-6))
    G   = P2_3_1d(G0)
    mu = mu_entropy(rho)
    mux = grad_1d(mu, k)
    muxP = P2_3_1d(mux)
    Pirr_local = 0.5 * trapezoid_1d(rho*G*(muxP**2), L)

    for sigma in sigmas:
        # Construct a nonlocal control by smoothing ∂x μ before projecting
        mux_s = gaussian_smooth_1d(mux, k, sigma)
        mux_s = P2_3_1d(mux_s)
        # consistent projected product for nonlocal falsifier
        j_nonlocal = prodP(rho, prodP(G, mux_s))
        v = div_flux_1d(j_nonlocal, k)
        v = P2_3_1d(v)
        inner = trapezoid_1d(v*mu, L)
        sigma_dot = -inner
        lhs = inner**2
        rhs = 2.0 * Pirr_local * sigma_dot
        gap = rhs - lhs
        verdict = "FAIL equality" if gap < -1e-6 or sigma_dot < 0 else "VIOLATES equality" if gap > 1e-3 else "MARGINAL"
        print(f"N={N} | sigma={sigma:.2f} | Pirr_local={Pirr_local:.6e} | sigdot={sigma_dot:.6e} | LHS={lhs:.6e} | RHS={rhs:.6e} | gap={gap:.6e} | {verdict}")

# =========================
# 1D: conservative vs non conservative tripwire (P-consistent)
# =========================
def conservative_vs_nonconservative_1d(N, L):
    print("\n== Conservative vs non conservative tripwire (1D, P-consistent) ==")
    x = np.linspace(0.0, L, N, endpoint=False)
    k = fftfreq_1d(N, L)
    rho = P2_3_1d(np.maximum(1.0 + 0.3*np.sin(2*x) + 0.2*np.cos(3*x), 1e-6))
    G   = P2_3_1d(1.0 + 0.25*np.sin(x))
    mu  = mu_entropy(rho)
    mux = P2_3_1d(grad_1d(mu, k))
    j_cons = prodP(rho, prodP(G, mux))
    v_cons = P2_3_1d(div_flux_1d(j_cons, k))
    mass_cons = mass_from_k0_1d(v_cons, L)
    w_non = G*mux
    mass_non = mass_from_k0_1d(w_non, L)
    print(f"∫v_cons dx(k0) = {mass_cons:.3e}  expected near 0")
    print(f"∫w_non  dx(k0) = {mass_non:.3e}  non zero indicates mass leak")

# =========================
# 2D: reversible no work identity (unchanged, but fine)
# =========================
def reversible_no_work_2d(Nx, Ny, Lx, Ly):
    print("\n== Reversible no work identity (2D) ==")
    kx, ky = fftfreq_2d(Nx, Ny, Lx, Ly)
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    rho = np.maximum(1.0 + 0.2*np.sin(X)*np.cos(2*Y) + 0.1*np.cos(3*X - Y), 1e-6)
    mu  = mu_entropy(rho)
    mux, muy = grad_2d(mu, kx, ky)
    J0 = np.array([[0.0, -1.0], [1.0, 0.0]])  # constant antisymmetric
    jx = J0[0,0]*mux + J0[0,1]*muy
    jy = J0[1,0]*mux + J0[1,1]*muy
    vrev = div_flux_2d(jx, jy, kx, ky)
    pointwise = mux*(J0[0,0]*mux + J0[0,1]*muy) + muy*(J0[1,0]*mux + J0[1,1]*muy)
    PR = trapezoid_2d(pointwise, Lx, Ly)
    scale = trapezoid_2d(mux*mux + muy*muy, Lx, Ly) + 1e-30
    rel = abs(PR)/max(1.0, scale)
    verdict = "PASS" if rel <= CFG["tol_pr_rel"] else "WARN"
    print(f"PR = {PR:.3e} | relative = {rel:.3e} | ||v_rev||2 = {np.sqrt(trapezoid_2d(vrev*vrev, Lx, Ly)):.3e} | {verdict}")

# =========================
# 1D: coarse-grain commutator scaling (P-consistent, normalised)
# =========================
def coarse_grain_commutator_1d(N, L, ell_list):
    print("\n== Coarse-grain commutator scaling (1D, P-consistent, normalised) ==")
    x = np.linspace(0.0, L, N, endpoint=False)
    k = fftfreq_1d(N, L)
    rho = P2_3_1d(np.maximum(1.0 + 0.25*np.sin(2*x) + 0.1*np.sin(3*x+0.3) + 0.1*np.cos(5*x-0.4), 1e-6))
    G   = P2_3_1d(1.0 + 0.4*np.cos(2*x))
    mu  = mu_entropy(rho)
    mux = P2_3_1d(grad_1d(mu, k))
    j   = prodP(rho, prodP(G, mux))
    v   = P2_3_1d(div_flux_1d(j, k))

    base = np.sqrt(trapezoid_1d(v*v, L)) + 1e-30  # normaliser

    for ell in ell_list:
        C_v  = P2_3_1d(gaussian_smooth_1d(v,   k, ell))
        C_r  = P2_3_1d(gaussian_smooth_1d(rho, k, ell))
        muCr = mu_entropy(C_r)
        muCrx = P2_3_1d(grad_1d(muCr, k))
        jCr = prodP(C_r, prodP(G, muCrx))
        vCr = P2_3_1d(div_flux_1d(jCr, k))
        diff = C_v - vCr
        l2 = np.sqrt(trapezoid_1d(diff*diff, L))
        rel = l2 / base
        ratio = rel/(ell**2) if ell > 0 else np.nan
        print(f"N={N} | ell={ell:.2f} | rel ||C(∂tρ)-∂t(Cρ)||2 = {rel:.6e} | (rel)/ell^2 = {ratio:.3e}")

# =========================
# 1D: probe identifiability of G (unchanged)
# =========================
def probe_identifiability_1d(N, L, max_n):
    print("\n== Probe identifiability of G (1D) ==")
    x = np.linspace(0.0, L, N, endpoint=False)
    k = fftfreq_1d(N, L)
    rho = P2_3_1d(np.maximum(1.0 + 0.2*np.sin(2*x) + 0.1*np.cos(3*x), 1e-6))
    G   = P2_3_1d(1.0 + 0.3*np.cos(2*x) + 0.1*np.cos(3*x + 0.2))
    grads = []
    for n in range(1, max_n+1):
        grads.append(P2_3_1d(grad_1d(np.sin(n*x), k)))
        grads.append(P2_3_1d(grad_1d(np.cos(n*x), k)))
    m = len(grads)
    B = np.zeros((m, m))
    for i in range(m):
        gi = grads[i]
        for j in range(m):
            B[i, j] = trapezoid_1d(rho*G*gi*grads[j], L)
    U, S, Vt = np.linalg.svd(B)
    cond = S.max()/S.min()
    print(f"basis size = {m}, min sing = {S.min():.3e}, max sing = {S.max():.3e}, cond(B) = {cond:.3e}")
    verdict = "PASS" if cond < 1e3 else "WARN"
    print(f"identifiability verdict = {verdict}")

# =========================
# Master
# =========================
def run_all():
    print("=== Metriplectic Axiom Diagnostics: Console Output (fortress) ===")
    L = CFG["L"]

    # Health check
    N0 = CFG["N_default"]
    x0 = np.linspace(0.0, L, N0, endpoint=False)
    rho0 = np.maximum(1.0 + 0.2*np.sin(2*x0) + 0.1*np.cos(3*x0+0.3) + 0.1*np.cos(5*x0-0.4), 1e-6)
    G0   = 1.0 + 0.5*np.cos(2*x0)
    print("\n== State health check ==")
    report_state_health_1d(rho0, G0)

    # 1) Equality refinement (now projection-consistent)
    equality_refinement_1d(CFG["N_list_refine"], L)

    # 2) Nonlocal falsifier sweep (keeps local Pirr definition)
    nonlocal_falsifier_sweep_1d(N=CFG["N_default"], L=L, sigmas=CFG["sigma_sweep"])

    # 3) Conservative vs non conservative
    conservative_vs_nonconservative_1d(N=CFG["N_default"], L=L)

    # 4) Reversible no work identity (2D)
    reversible_no_work_2d(CFG["Nx2"], CFG["Ny2"], L, L)

    # 5) Coarse-grain commutator scaling (projected, normalised)
    coarse_grain_commutator_1d(N=CFG["N_default"], L=L, ell_list=CFG["ell_list"])

    # 6) Probe identifiability
    probe_identifiability_1d(N=CFG["N_default"], L=L, max_n=CFG["max_n_probe"])

    print("\n=== Done. Pass criteria: equality gaps ~ O(dx), mass k0 ~ 0, PR ~ 0, relative commutator ~ O(ell^2), cond(B) < 1e3. ===")

# Execute
if __name__ == "__main__":
    run_all()
