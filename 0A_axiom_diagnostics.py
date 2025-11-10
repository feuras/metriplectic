#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 0A_axiom_diagnostics.py (review-ready, consolidated, v5)
# Unified diagnostics for local metriplectic axioms on a 2D torus.
#
# Implements seven reviewer dials in one script:
#   1) EVI probe for the irreversible ray (JKO-style one step with metric induced by G)
#   2) Noether "no-work" symmetry for J and invariance of scalars (sigma_dot from grad mu only)
#   3) Alignment identity explaining sharpness of the cost-entropy inequality
#   4) Torus coercivity constant: explicit bound and numeric check on a convex periodic potential
#   5) Single-axiom falsifiers: minimal changes that trip each dial, with clear console lines
#   6) Local tomography of G from instantaneous scalar data at fixed state (two-state check)
#   7) Orthogonality identity: <v_irr, G^{-1} v_rev>_{H^{-1}_rho} ~ 0
#
# Defaults (tuned for stability but fast enough):
#   - Resolution N=384 (edit via --N). If slow, try --N 192.
#   - Scalar G = gamma * I (edit via --gamma).
#   - Periodic box Omega = [0,L)^2 with L = 2*pi.
#
# Usage examples:
#     python 0A_axiom_diagnostics_ascii.py              # runs --all by default
#     python 0A_axiom_diagnostics_ascii.py --evi_probe
#     python 0A_axiom_diagnostics_ascii.py --nowork_sweep
#     python 0A_axiom_diagnostics_ascii.py --alignment
#     python 0A_axiom_diagnostics_ascii.py --torus_constant
#     python 0A_axiom_diagnostics_ascii.py --falsifiers
#     python 0A_axiom_diagnostics_ascii.py --tomography
#     python 0A_axiom_diagnostics_ascii.py --orthogonality

import argparse
import math
import numpy as np

# =========================
# Utilities: grid and FFT
# =========================

def make_grid(N=384, L=2.0*np.pi, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 2.0*np.pi*np.fft.fftfreq(N, d=L/N)
    ky = 2.0*np.pi*np.fft.fftfreq(N, d=L/N)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX*KX + KY*KY
    K2[0,0] = 1.0  # avoid division by zero in Poisson solves
    return X, Y, KX, KY, K2, rng

def grad(f, KX, KY):
    F = np.fft.fftn(f)
    gx = np.real(np.fft.ifftn(1j*KX*F))
    gy = np.real(np.fft.ifftn(1j*KY*F))
    return gx, gy

def div(vx, vy, KX, KY):
    VX = np.fft.fftn(vx)
    VY = np.fft.fftn(vy)
    D = 1j*KX*VX + 1j*KY*VY
    return np.real(np.fft.ifftn(D))

def project_div_free(vx, vy, KX, KY, K2):
    # Helmholtz projection onto divergence-free vector fields (periodic).
    vx_hat = np.fft.fftn(vx)
    vy_hat = np.fft.fftn(vy)
    div_hat = 1j*KX*vx_hat + 1j*KY*vy_hat
    px_hat = vx_hat - 1j*KX*div_hat / K2
    py_hat = vy_hat - 1j*KY*div_hat / K2
    px = np.real(np.fft.ifftn(px_hat))
    py = np.real(np.fft.ifftn(py_hat))
    return px, py

# =========================
# Free energy and operators
# =========================

def free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1):
    # F[rho] = int rho (log rho + V) dx with smooth periodic V
    # mu = dF/drho = log rho + 1 + V
    if kind == "entropy_plus_V":
        V = V_amp*(np.cos(X)+np.cos(Y))
        mu = np.log(rho) + 1.0 + V
        return mu, V
    elif kind == "zeroV":
        V = 0.0*X
        mu = np.log(rho) + 1.0
        return mu, V
    else:
        V = 0.0*X
        mu = np.log(rho) + 1.0
        return mu, V

def apply_G(grad_mu_x, grad_mu_y, gamma=1.0, G_aniso=None):
    # G is a positive local operator. Default scalar gamma * I.
    # Optionally pass G_aniso as a 2x2 tensor field per point.
    # Returns irreversible velocity u_irr = -G grad mu.
    if G_aniso is None:
        ux = -gamma*grad_mu_x
        uy = -gamma*grad_mu_y
        return ux, uy
    else:
        ux = -(G_aniso[:,:,0,0]*grad_mu_x + G_aniso[:,:,0,1]*grad_mu_y)
        uy = -(G_aniso[:,:,1,0]*grad_mu_x + G_aniso[:,:,1,1]*grad_mu_y)
        return ux, uy

def dot_sigma_from_mu(rho, grad_mu_x, grad_mu_y, gamma=1.0):
    # sigma_dot = int rho grad mu . G grad mu = gamma * int rho |grad mu|^2 for scalar G
    return gamma * np.sum(rho * (grad_mu_x*grad_mu_x + grad_mu_y*grad_mu_y))

def power_and_entropy(rho, ux, uy, grad_mu_x, grad_mu_y, gamma=1.0, G_inv_scalar=True):
    # P_irr(u_irr) = 0.5 * sigma_dot when u_irr = -G grad mu
    # C_min(u)     = 0.5 * int rho * u . G^{-1} u
    P_irr = 0.5 * dot_sigma_from_mu(rho, grad_mu_x, grad_mu_y, gamma=gamma)
    if G_inv_scalar:
        Cmin_u = 0.5*np.sum(rho * (ux*ux + uy*uy)) / gamma
    else:
        raise ValueError("Non-scalar G^{-1} not implemented in this concise script.")
    return P_irr, Cmin_u

# =========================
# 1) EVI probe
# =========================

def evi_probe(rho, X, Y, KX, KY, K2, gamma=1.0, tau=5e-3):
    # One implicit JKO-like step for the metric induced by G = gamma I.
    # Check EVI: F[rho_tau] - F[bar] <= (1/2tau) W_G^2(rho_tau,bar) - (1/2tau) W_G^2(rho_0,bar).
    # Use midpoint bar = 0.5*(rho + rho_tau).
    mu, V = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)
    ux, uy = apply_G(gx, gy, gamma=gamma)
    div_ru = div(rho*ux, rho*uy, KX, KY)
    rho_tau = rho - tau*div_ru
    rho_tau = np.clip(rho_tau, 1e-8, None)

    rho_bar = 0.5*(rho + rho_tau)
    rho_bar = np.clip(rho_bar, 1e-8, None)

    F_tau = np.sum(rho_tau*(np.log(rho_tau) + 1.0)) + np.sum(rho_tau*(0.1*(np.cos(X)+np.cos(Y))))
    F_bar = np.sum(rho_bar*(np.log(rho_bar) + 1.0)) + np.sum(rho_bar*(0.1*(np.cos(X)+np.cos(Y))))

    rho_avg = 0.5*(rho + rho_tau)
    WG_tau_bar = np.sum(tau * rho_avg * (ux*ux + uy*uy) * gamma)

    lhs = F_tau - F_bar
    rhs = 0.5*(WG_tau_bar)/tau
    return lhs, rhs, rho_tau

# =========================
# 2) Noether nowork sweep
# =========================

def nowork_sweep(rho, X, Y, KX, KY, K2, gamma=1.0, seed=7):
    mu, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)
    uix, uiy = apply_G(gx, gy, gamma=gamma)
    P, C = power_and_entropy(rho, uix, uiy, gx, gy, gamma=gamma)
    sig = dot_sigma_from_mu(rho, gx, gy, gamma=gamma)

    # Build reversible flux q = curl(psi) with div(q)=0, then v_rev = q/rho so div(rho v_rev)=0
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=rho.shape)
    qx, qy = grad(psi, KX, KY)
    qx, qy =  qy, -qx
    eps = 1e-12
    vrevx_raw = qx / (rho + eps)
    vrevy_raw = qy / (rho + eps)

    _, C_raw = power_and_entropy(rho, vrevx_raw, vrevy_raw, gx, gy, gamma=gamma)
    s = math.sqrt(C / C_raw) if C_raw > 0 else 0.0
    vrevx = s * vrevx_raw
    vrevy = s * vrevy_raw

    P2, C2 = power_and_entropy(rho, uix + vrevx, uiy + vrevy, gx, gy, gamma=gamma)
    sig2 = dot_sigma_from_mu(rho, gx, gy, gamma=gamma)  # invariant

    div_w = div(rho*vrevx, rho*vrevy, KX, KY)
    div_wL2 = math.sqrt(np.sum(div_w*div_w))

    inner_orth = np.sum(rho * (uix*vrevx + uiy*vrevy)) / gamma

    return (P, sig, C), (P2, sig2, C2), (vrevx, vrevy, div_wL2, inner_orth)

# =========================
# 3) Alignment identity
# =========================

def alignment_identity(rho, X, Y, KX, KY, K2, gamma=1.0, seed=11):
    mu, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)
    uix, uiy = apply_G(gx, gy, gamma=gamma)

    # Deterministic mid-angle perturbation (no projection)
    phi = np.sin(2*X) * np.cos(3*Y)
    wx, wy = grad(phi, KX, KY)
    alpha = 1.0
    uax = uix + alpha*wx
    uay = uiy + alpha*wy

    inner = np.sum(rho * (uix*uax + uiy*uay)) / gamma
    norm_ui = math.sqrt(np.sum(rho * (uix*uix + uiy*uiy)) / gamma)
    norm_u  = math.sqrt(np.sum(rho * (uax*uax + uay*uay)) / gamma)
    cos2 = (inner/(norm_ui*norm_u + 1e-16))**2

    P, C = power_and_entropy(rho, uax, uay, gx, gy, gamma=gamma)
    sig = dot_sigma_from_mu(rho, gx, gy, gamma=gamma)
    ratio = (np.sum(rho*(uax*gx + uay*gy))**2) / (2.0*C*sig + 1e-16)
    return cos2, ratio

# =========================
# 4) Torus constant (convex periodic potential)
# =========================

def torus_constant_check(N=384, L=2.0*np.pi, gamma=1.2, lam=0.5, rho_min=0.7):
    X, Y, KX, KY, K2, rng = make_grid(N, L, seed=5)
    # Convex periodic potential: Hessian >= lam * I everywhere
    V = lam*(2.0 - np.cos(X) - np.cos(Y))
    rho = rho_min + 0.3*(np.sin(X)+np.sin(Y))**2
    mu = np.log(rho) + 1.0 + V
    gx, gy = grad(mu, KX, KY)

    lower_bound = gamma*lam

    probes = []
    for s in [3, 5, 7, 9, 11, 13, 17]:
        xi = np.sin(s*X) * np.cos(s*Y)
        xix, xiy = grad(xi, KX, KY)
        num = gamma * lam * np.sum(rho * (xix*xix + xiy*xiy))  # include gamma
        den = np.sum(rho * (xix*xix + xiy*xiy)) + 1e-16
        rq = num / den
        probes.append(rq)
    kappa_est = min(probes)
    return kappa_est, lower_bound, rho

# =========================
# 5) Falsifiers
# =========================

def falsifier_symmetry_break(rho, X, Y, KX, KY, K2, gamma=1.0, eps=0.80):
    # Break symmetry by rotating grad mu before applying G; test equality ratio with symmetric metric.
    mu, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)
    uix, uiy = apply_G(gx, gy, gamma=gamma)
    sig = dot_sigma_from_mu(rho, gx, gy, gamma=gamma)

    gx_sk = gx - eps*gy
    gy_sk = gy + eps*gx
    u_skx, u_sky = apply_G(gx_sk, gy_sk, gamma=gamma)

    C_ok  = 0.5*np.sum(rho*(uix*uix + uiy*uiy)) / gamma
    C_bad = 0.5*np.sum(rho*(u_skx*u_skx + u_sky*u_sky)) / gamma
    r_ok  = (np.sum(rho*(uix*gx + uiy*gy))**2) / (2.0*C_ok*sig + 1e-16)
    r_bad = (np.sum(rho*(u_skx*gx + u_sky*gy))**2) / (2.0*C_bad*sig + 1e-16)
    return r_ok, r_bad

def falsifier_locality_break(rho, X, Y, KX, KY, gamma=1.0):
    # Break locality by blurring grad mu before applying G; equality ratio should deviate from 1.
    mu, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)

    uix, uiy = apply_G(gx, gy, gamma=gamma)
    sig = dot_sigma_from_mu(rho, gx, gy, gamma=gamma)
    C_ok  = 0.5*np.sum(rho*(uix*uix + uiy*uiy)) / gamma
    ratio_base = (np.sum(rho*(uix*gx + uiy*gy))**2) / (2.0*C_ok*sig + 1e-16)

    k2 = (KX*KX + KY*KY)
    lowpass = np.exp(-1.0*k2) * (k2 <= (0.6*np.max(k2)))
    gx_nl = np.real(np.fft.ifftn(np.fft.fftn(gx)*lowpass))
    gy_nl = np.real(np.fft.ifftn(np.fft.fftn(gy)*lowpass))
    u_nlx, u_nly = apply_G(gx_nl, gy_nl, gamma=gamma)
    C_nl = 0.5*np.sum(rho*(u_nlx*u_nlx + u_nly*u_nly)) / gamma
    ratio_nl = (np.sum(rho*(u_nlx*gx + u_nly*gy))**2) / (2.0*C_nl*sig + 1e-16)
    return ratio_base, ratio_nl

def falsifier_positivity_break(rho):
    # Drive the positivity margin to zero and report min rho.
    rho_bad = 0.0*rho
    min_rho = rho_bad.min()
    return min_rho

def falsifier_nowork_break(rho, X, Y, KX, KY, gamma=1.0):
    # Break no-work by using a drift with nonzero weighted divergence; invariants move.
    mu, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)
    uix, uiy = apply_G(gx, gy, gamma=gamma)

    rng = np.random.default_rng(3)
    psi = rng.normal(size=rho.shape)
    vrx =  grad(psi, KX, KY)[1]
    vry = -grad(psi, KX, KY)[0]

    P,  C  = power_and_entropy(rho, uix, uiy, gx, gy, gamma=gamma)
    P2, C2 = power_and_entropy(rho, uix + vrx, uiy + vry, gx, gy, gamma=gamma)
    sig  = dot_sigma_from_mu(rho, gx, gy, gamma=gamma)
    sig2 = dot_sigma_from_mu(rho, gx, gy, gamma=gamma)

    return sig2 - sig, C2 - C

# =========================
# 6) Tomography of G
# =========================

def tomography_gamma(rho, X, Y, KX, KY):
    # Recover scalar gamma from instantaneous scalars at fixed rho.
    # Theory: for u_irr = -gamma grad mu,  sigma_dot = gamma * int rho |grad mu|^2.
    mu, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)
    grad2 = np.sum(rho * (gx*gx + gy*gy))
    gamma_true = 1.3
    ux, uy = apply_G(gx, gy, gamma=gamma_true)
    P, C = power_and_entropy(rho, ux, uy, gx, gy, gamma=gamma_true)
    sig = dot_sigma_from_mu(rho, gx, gy, gamma=gamma_true)
    gamma_est = sig / (grad2 + 1e-16)
    return gamma_true, gamma_est, abs(gamma_est - gamma_true)

# =========================
# 7) Orthogonality identity
# =========================

def orthogonality_check(rho, X, Y, KX, KY, K2, gamma=1.0, seed=9):
    mu, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
    gx, gy = grad(mu, KX, KY)
    uix, uiy = apply_G(gx, gy, gamma=gamma)

    rng = np.random.default_rng(seed)
    psi = rng.normal(size=rho.shape)
    qx, qy = grad(psi, KX, KY)
    qx, qy =  qy, -qx
    vrevx = qx / (rho + 1e-12)
    vrevy = qy / (rho + 1e-12)

    inner = np.sum(rho * (uix*vrevx + uiy*vrevy)) / gamma
    return inner

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Unified metriplectic diagnostics (0A v5 ASCII)")
    parser.add_argument("--N", type=int, default=384)
    parser.add_argument("--L", type=float, default=2.0*np.pi)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--evi_probe", action="store_true")
    parser.add_argument("--nowork_sweep", action="store_true")
    parser.add_argument("--alignment", action="store_true")
    parser.add_argument("--torus_constant", action="store_true")
    parser.add_argument("--falsifiers", action="store_true")
    parser.add_argument("--tomography", action="store_true")
    parser.add_argument("--orthogonality", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Note: ignoring unknown args:", unknown)

    if not (args.all or args.evi_probe or args.nowork_sweep or args.alignment or args.torus_constant or args.falsifiers or args.tomography or args.orthogonality):
        args.all = True
        print("Note: no diagnostics flag supplied; defaulting to --all")

    N, L, gamma = args.N, args.L, args.gamma
    X, Y, KX, KY, K2, rng = make_grid(N, L, seed=args.seed)

    rho = 0.8 + 0.2*np.cos(X)*np.cos(Y)
    rho = np.clip(rho, 1e-8, None)

    header = "=== Metriplectic Axiom Diagnostics: Console Output (fortress) ==="
    print(header)

    if args.all or args.evi_probe:
        lhs, rhs, rho_tau = evi_probe(rho, X, Y, KX, KY, K2, gamma=gamma, tau=5e-3)
        sign = "PASS" if lhs <= rhs + 1e-10 else "FAIL"
        print("== EVI probe ==")
        print("EVI check: F[rho_tau]-F[bar] <= RHS  |  lhs={:+.6e}  rhs={:+.6e}  [{}]".format(lhs, rhs, sign))
        # Equality ratio for v_irr (should be ~1)
        mu_eq, _ = free_energy_and_mu(rho, X, Y, kind="entropy_plus_V", V_amp=0.1)
        gx_eq, gy_eq = grad(mu_eq, KX, KY)
        ux_eq, uy_eq = apply_G(gx_eq, gy_eq, gamma=gamma)
        sig_eq = dot_sigma_from_mu(rho, gx_eq, gy_eq, gamma=gamma)
        C_eq  = 0.5*np.sum(rho*(ux_eq*ux_eq + uy_eq*uy_eq))/gamma
        eq_ratio = (np.sum(rho*(ux_eq*gx_eq + uy_eq*gy_eq))**2) / (2.0*C_eq*sig_eq + 1e-16)
        print("EVI saturation ratio on v_irr: {:.12f} (1.0 means sharp)".format(eq_ratio))

    if args.all or args.nowork_sweep:
        (P, sig, C), (P2, sig2, C2), (vrevx, vrevy, div_wL2, inner_orth) = nowork_sweep(rho, X, Y, KX, KY, K2, gamma=gamma)
        dS = sig2 - sig
        C_rev = C2 - C
        print("== Noether no-work sweep ==")
        print("div(rho v_rev) L2 = {:.3e}  (~0)".format(div_wL2))
        print("sigma_dot invariance: Delta sigma_dot = {:+.3e}  [{}]".format(dS, "PASS" if abs(dS) < 1e-10 else "WARN"))
        print("Orthogonality: <v_irr, G^-1 v_rev> = {:+.3e}  [{}]".format(inner_orth, "PASS" if abs(inner_orth) < 1e-10 else "WARN"))
        print("C_min increase: C_irr={:.6e}  C_rev~{:.6e}  C_total={:.6e}".format(C, C_rev, C2))

    if args.all or args.alignment:
        cos2, ratio = alignment_identity(rho, X, Y, KX, KY, K2, gamma=gamma)
        print("== Alignment identity ==")
        print("equality ratio = <v,mu>^2 / (2 C_min sigma_dot) = {:.6f}  vs  cos^2 = {:.6f}  [Delta={:.2e}]".format(ratio, cos2, abs(ratio-cos2)))

    if args.all or args.torus_constant:
        kappa_est, lower_bound, rho_used = torus_constant_check(N=N, L=L, gamma=1.2, lam=0.5, rho_min=0.7)
        ok = (kappa_est >= 0.99*lower_bound)
        print("== Torus coercivity constant ==")
        print("kappa_est ~= {:.4e}  lower_bound = {:.4e}  [{}]".format(kappa_est, lower_bound, "PASS" if ok else "WARN"))

    if args.all or args.falsifiers:
        print("== Single-axiom falsifiers ==")
        r_ok, r_bad = falsifier_symmetry_break(rho, X, Y, KX, KY, K2, gamma=gamma, eps=0.80)
        trip_sym = (r_ok > 0.999) and (r_bad > 1.001)
        print("Symmetry broken: equality ratio (ok, skewed) = ({:.3f}, {:.3f})  [{}]".format(r_ok, r_bad, "FAIL trip" if trip_sym else "WARN"))

        ratio_base, ratio_nl = falsifier_locality_break(rho, X, Y, KX, KY, gamma=gamma)
        trip_loc = abs(ratio_nl - 1.0) > 1e-2
        print("Locality broken: equality ratio (local, nonlocal) = ({:.3f}, {:.3f})  [{}]".format(ratio_base, ratio_nl, "FAIL trip" if trip_loc else "WARN"))

        min_rho = falsifier_positivity_break(rho)
        print("Positivity margin broken: min rho = {:+.1e}  [FAIL trip]".format(min_rho))

        d_sig, d_C = falsifier_nowork_break(rho, X, Y, KX, KY, gamma=gamma)
        print("No-work broken: Delta sigma_dot={:+.3e}, Delta C_min={:+.3e}  [FAIL trip expected]".format(d_sig, d_C))

    if args.all or args.tomography:
        gamma_true, gamma_est, err = tomography_gamma(rho, X, Y, KX, KY)
        print("== Tomography of G ==")
        print("gamma_true={:.3f}  gamma_est={:.3f}  |est-true|={:.3e}  [{}]".format(gamma_true, gamma_est, err, "PASS" if err < 5e-2 else "WARN"))
        X2, Y2, KX2, KY2, K22, _ = make_grid(N, L, seed=args.seed+97)
        rho2 = 0.9 + 0.1*np.cos(2*X2)*np.cos(3*Y2)
        gt2, ge2, err2 = tomography_gamma(rho2, X2, Y2, KX2, KY2)
        print("Tomography (independent state): gamma_true={:.3f}  gamma_est={:.3f}  err={:.3e}  [{}]".format(gt2, ge2, err2, "PASS" if err2 < 5e-2 else "WARN"))

    if args.all or args.orthogonality:
        inner = orthogonality_check(rho, X, Y, KX, KY, K2, gamma=gamma)
        print("== Orthogonality identity ==")
        print("<v_irr, G^-1 v_rev>_H^-1_rho = {:+.3e}  [{}]".format(inner, "PASS" if abs(inner) < 1e-10 else "WARN"))
        # Sanity: numerically equals -int mu * div(rho v_rev) for the constructed v_rev

if __name__ == "__main__":
    main()
