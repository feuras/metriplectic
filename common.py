
"""
common.py — shared utilities


Requirements: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
from dataclasses import dataclass

def dealias_mask(N: int):
    n = np.fft.fftfreq(N) * N
    cutoff = N//3
    return (np.abs(n) <= cutoff).astype(float)

def proj_band(F, mask):
    return F * mask

def proj_field(f, mask):
    return np.real(ifft(proj_band(fft(f), mask)))

def zero_mean(f):
    return f - f.mean()

def dx(f, k, mask):
    F = proj_band(fft(f), mask)
    return np.real(ifft(1j * k * F))

def dxx(f, k, mask):
    F = proj_band(fft(f), mask)
    return np.real(ifft(-(k**2) * F))

def dxxxx(f, k, mask):
    F = proj_band(fft(f), mask)
    return np.real(ifft((k**4) * F))

def hilbert(f, k, mask):
    signk = np.sign(k)
    F = proj_band(fft(f), mask)
    return np.real(ifft(-1j * signk * F))

def inner_L2(a, b, x):
    return np.trapz(a * b, x)

def L_rhoG_phi(rho, G, phi, k, mask):
    return -zero_mean(dx(rho * G * dx(phi, k, mask), k, mask))

def Hm1G_energy(rho, G, phi, k, mask, x):
    return inner_L2(rho * G * dx(phi, k, mask)**2, np.ones_like(rho), x)

def entropy_production(rho, G, mu, k, mask, x):
    return inner_L2(rho * G * dx(mu, k, mask)**2, np.ones_like(rho), x)

def kkt_solve(rho, G, v, k, mask, tol=1e-12, maxit=6000):
    def A(phi):
        return L_rhoG_phi(rho, G, phi, k, mask)
    phi = np.zeros_like(rho)
    r = v - A(phi)
    p = r.copy()
    rz_old = np.dot(r, r)
    it = 0
    res = np.linalg.norm(r) / (np.linalg.norm(v) + 1e-16)
    while it < maxit and res > tol:
        Ap = A(p)
        denom = np.dot(p, Ap) + 1e-30
        alpha = rz_old / denom
        phi = phi + alpha * p
        r = r - alpha * Ap
        res = np.linalg.norm(r) / (np.linalg.norm(v) + 1e-16)
        rz_new = np.dot(r, r)
        beta = rz_new / (rz_old + 1e-30)
        p = r + beta * p
        rz_old = rz_new
        it += 1
    return zero_mean(phi), it, res

def mu_entropic(rho, lam2, k, mask):
    return np.log(rho) - lam2 * dxx(rho, k, mask)

def mu_highorder(rho, lam2, lam4, k, mask):
    return np.log(rho) - lam2 * dxx(rho, k, mask) + lam4 * dxxxx(rho, k, mask)

@dataclass
class Grid:
    L: float = 40.0
    N: int = 512
    def build(self):
        x = np.linspace(0.0, self.L, self.N, endpoint=False)
        k = 2*np.pi*fftfreq(self.N, d=self.L/self.N)
        mask = dealias_mask(self.N)
        return x, k, mask

