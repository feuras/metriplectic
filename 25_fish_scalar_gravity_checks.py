#!/usr/bin/env python3
"""
fish_scalar_gravity_checks.py

Reproducibility script for the Fisher scalar sector.

Tests included:

1. Operator identity:
   L_rho phi = - div(rho grad phi) ≈ - Δ rho for phi = log(rho / rho0).

2. Fisher Laplacian identity:
   For Phi_eff = - (c^2 / 2) log(rho / rho0),
   check numerically that

     Δ Phi_eff ≈ - (c^2 / 2) [ Δ rho / rho - |grad rho|^2 / rho^2 ].

3. Self-sourced Fisher star:
   Solve rho'' + (2 / r) rho' + lambda kappa rho = 0 with regular data,
   compare to analytic n=1 polytrope rho(r) = rho_c sin(k r)/(k r),
   and check radius, mass and the dimensionless compactness GM/(R c^2).

Author: J. R. Dunkley et al.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from dataclasses import dataclass

from scipy.integrate import solve_ivp, simpson  # simpson is the modern API


@dataclass
class Params:
    c: float = 2.99792e8           # speed of light (m/s)
    G: float = 6.674e-11       # gravitational constant (SI)
    rho0: float = 1.0          # reference density (arbitrary units)
    kappa: float = 1.0         # Fisher coupling (dimensionless here)


def spectral_derivative_1d(f, L):
    """
    Periodic spectral first and second derivatives of f(x) on [0, L).
    Returns f_x, f_xx.
    """
    N = f.size
    k = 2.0 * np.pi * fftfreq(N, d=L / N)
    f_hat = fft(f)
    f_x_hat = 1j * k * f_hat
    f_xx_hat = - (k ** 2) * f_hat
    f_x = np.real(ifft(f_x_hat))
    f_xx = np.real(ifft(f_xx_hat))
    return f_x, f_xx


def make_smooth_positive_rho(N=1024, L=2.0 * np.pi, seed=1234):
    """
    Build a smooth positive rho(x) on a periodic domain by filtering random modes.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L, N, endpoint=False)

    modes = 10
    coeffs = rng.normal(size=modes) * 0.2
    rho = np.ones_like(x)
    for n in range(1, modes + 1):
        rho += coeffs[n - 1] * np.cos(2.0 * np.pi * n * x / L)

    rho_min = rho.min()
    if rho_min <= 0.0:
        rho += 1.1 * abs(rho_min)

    return x, rho


def test_operator_identity_1d(N=1024, L=2.0 * np.pi):
    """
    Test that L_rho phi = - div(rho grad phi) agrees with - Δ rho
    for random smooth positive rho on a periodic 1D domain.
    """
    x, rho = make_smooth_positive_rho(N=N, L=L)
    rho0 = rho.mean()
    phi = np.log(rho / rho0)

    phi_x, _ = spectral_derivative_1d(phi, L)
    _, rho_xx = spectral_derivative_1d(rho, L)

    div_term = np.gradient(rho * phi_x, x, edge_order=2)
    L_rho_phi = -div_term

    minus_lap_rho = -rho_xx

    num = np.sqrt(np.mean((L_rho_phi - minus_lap_rho) ** 2))
    den = np.sqrt(np.mean(minus_lap_rho ** 2))
    rel_err = num / den if den > 0 else num

    print("=== Test 1: operator identity L_rho phi ≈ -Δ rho ===")
    print(f"Grid: N = {N}, L = {L}")
    print(f"rho_min = {rho.min():.6e}, rho_max = {rho.max():.6e}")
    print(f"Relative L2 error: {rel_err:.3e}")
    print("Expected: error limited by spectral resolution and aliasing.\n")


def test_fisher_laplacian_identity(N=1024, L=2.0 * np.pi, p=None):
    """
    Test the exact Fisher Laplacian identity:

      Phi_eff = - (c^2 / 2) log(rho / rho0)

      Δ Phi_eff = - (c^2 / 2) [ Δ rho / rho - |grad rho|^2 / rho^2 ].

    Numerically evaluate both sides on a smooth positive rho(x)
    and report the relative L2 error.
    """
    if p is None:
        p = Params()

    x, rho = make_smooth_positive_rho(N=N, L=L)
    rho0 = rho.mean()

    phi = np.log(rho / rho0)
    Phi_eff = -0.5 * (p.c ** 2) * phi

    Phi_x, Phi_xx = spectral_derivative_1d(Phi_eff, L)
    Delta_Phi = Phi_xx

    rho_x, rho_xx = spectral_derivative_1d(rho, L)
    rhs = -0.5 * (p.c ** 2) * (rho_xx / rho - (rho_x ** 2) / (rho ** 2))

    num = np.sqrt(np.mean((Delta_Phi - rhs) ** 2))
    den = np.sqrt(np.mean(rhs ** 2))
    rel_err = num / den if den > 0 else num

    print("=== Test 2: Fisher Laplacian identity for Phi_eff ===")
    print("Checking Δ Phi_eff ≈ - (c^2 / 2) [ Δ rho / rho - |grad rho|^2 / rho^2 ]")
    print(f"Relative L2 error: {rel_err:.3e}")
    print("This directly tests the core identity used in the weak-field analysis.\n")


def helmholtz_rhs(r, y, k2):
    """
    Right hand side for the radial Helmholtz equation:

      rho'' + (2/r) rho' + k^2 rho = 0

    written as first order system:

      y[0] = rho
      y[1] = rho'
    """
    rho, drho = y
    if r == 0.0:
        return [drho, -k2 * rho]
    return [drho, - (2.0 / r) * drho - k2 * rho]


def test_self_sourced_fisher_star(
    rho_c=1.0,
    lam=1.0,
    p=None,
    r_max_factor=1.1,
    n_points=2000,
):
    """
    Solve the radial Helmholtz equation

      rho'' + (2/r) rho' + lambda kappa rho = 0

    with regular data and compare to the analytic
    n=1 polytrope rho(r) = rho_c sin(k r)/(k r).

    Also compute:

      numerical first zero R_num
      analytic R_ana = pi / k
      numerical mass M_num ~ 4 pi ∫ rho_m(r) r^2 dr
      analytic M_ana for n=1 profile
      dimensionless compactness GM/(R c^2) (diagnostic only)
    """
    if p is None:
        p = Params()

    k2 = lam * p.kappa
    k = np.sqrt(k2)

    r_min = 1.0e-6 / k
    r_max = r_max_factor * np.pi / k

    y0 = [rho_c, 0.0]

    sol = solve_ivp(
        fun=lambda r, y: helmholtz_rhs(r, y, k2),
        t_span=(r_min, r_max),
        y0=y0,
        dense_output=True,
        rtol=1.0e-9,
        atol=1.0e-12,
    )

    r = np.linspace(r_min, r_max, n_points)
    rho_num = sol.sol(r)[0]

    sign_changes = np.where(np.sign(rho_num[:-1]) * np.sign(rho_num[1:]) < 0)[0]
    if sign_changes.size == 0:
        print("No zero crossing found for numerical rho(r).")
        return
    idx0 = sign_changes[0]
    r1, r2 = r[idx0], r[idx0 + 1]
    rho1, rho2 = rho_num[idx0], rho_num[idx0 + 1]
    R_num = r1 - rho1 * (r2 - r1) / (rho2 - rho1)

    R_ana = np.pi / k

    rho_ana = rho_c * np.sin(k * r) / (k * r)
    rho_ana[0] = rho_c  # limit r -> 0

    mask = r <= R_ana
    num = np.sqrt(np.trapz((rho_num[mask] - rho_ana[mask]) ** 2, r[mask]))
    den = np.sqrt(np.trapz(rho_ana[mask] ** 2, r[mask]))
    rel_profile_err = num / den if den > 0 else num

    R_int = min(R_num, R_ana)
    mask_int = r <= R_int
    rho_m_num = lam * rho_num[mask_int]
    M_num = 4.0 * np.pi * simpson(rho_m_num * (r[mask_int] ** 2), r[mask_int])

    # Analytic mass for n=1 profile:
    # M_ana = 4 pi^2 rho_c / (sqrt(lambda) kappa^(3/2))
    M_ana = 4.0 * (np.pi ** 2) * rho_c / (np.sqrt(lam) * (p.kappa ** 1.5))

    C_num = p.G * M_num / (R_num * (p.c ** 2))
    C_ana = p.G * M_ana / (R_ana * (p.c ** 2))

    print("=== Test 3: self-sourced Fisher star (n=1 polytrope) ===")
    print(f"lambda = {lam:.3e}, kappa = {p.kappa:.3e}, k = {k:.3e}")
    print(f"R_num = {R_num:.6e}, R_ana = {R_ana:.6e}")
    print(f"Relative difference in R: {(R_num - R_ana) / R_ana:.3e}")
    print(f"Relative L2 error in profile on [0, R_ana]: {rel_profile_err:.3e}")
    print(f"M_num = {M_num:.6e}, M_ana = {M_ana:.6e}")
    print(f"Relative difference in M: {(M_num - M_ana) / M_ana:.3e}")
    print(f"Compactness C_num = GM/(R c^2) = {C_num:.3e}")
    print(f"Compactness C_ana = GM/(R c^2) = {C_ana:.3e}")
    print("C is used here purely as a diagnostic combination.\n")


def main():
    p = Params()  # keep everything dimensionless/diagnostic by default

    print("Fisher scalar gravity reproducibility checks")
    print("-------------------------------------------")
    print(f"c = {p.c:.3e}, G = {p.G:.3e}, rho0 = {p.rho0:.3e}")
    print(f"kappa (dimensionless, used as scale) = {p.kappa:.3e}\n")

    test_operator_identity_1d()
    test_fisher_laplacian_identity(p=p)
    test_self_sourced_fisher_star(p=p)


if __name__ == "__main__":
    main()
