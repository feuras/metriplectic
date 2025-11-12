import os, json, numpy as np, pandas as pd

# --- params (unchanged defaults) ---
L=60.0; N=512
ds=0.05; steps=4000
impacts=np.linspace(-8,8,17)

# --- grid & spectra ---
x = np.linspace(-L/2,L/2,N,endpoint=False)
y = np.linspace(-L/2,L/2,N,endpoint=False)
X,Y = np.meshgrid(x,y,indexing="ij")
kx = 2*np.pi*np.fft.fftfreq(N, d=L/N); ky = 2*np.pi*np.fft.fftfreq(N, d=L/N)
KX,KY = np.meshgrid(kx,ky,indexing="ij")

# 2/3 de-alias mask
oneD = (np.abs(np.fft.fftfreq(N))*N <= N//3).astype(float)
mask = np.outer(oneD, oneD)

def fft2(f): return np.fft.fft2(f)
def ifft2(F): return np.real(np.fft.ifft2(F*mask))
def dx(f): return ifft2(1j*KX*fft2(f))
def dy(f): return ifft2(1j*KY*fft2(f))

# --- fields ---
rho0 = 1.0/40.0
G = 1.0 + 0.5*np.exp(-(X**2+Y**2)/(2*3.0**2))
rho = rho0*(1 + 0.6*np.exp(-((X-5.0)**2+Y**2)/(2*2.0**2)))
n = 1.0/np.sqrt(np.maximum(rho*G,1e-12))

nx, ny = dx(n), dy(n)  # spectral gradients
# precompute grad log n (safer numerically)
glnx = nx/np.maximum(n,1e-12); glny = ny/np.maximum(n,1e-12)

# --- bilinear sampler on periodic grid ---
hx = L/N; hy = L/N
def wrap_idx(i): 
    # periodic wrap
    return i % N

def sample_bilinear(F, xq, yq):
    # map to [0,N)
    fx = (xq + L/2)/hx
    fy = (yq + L/2)/hy
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy))
    tx = fx - i0; ty = fy - j0
    i1 = i0 + 1; j1 = j0 + 1
    i0 = wrap_idx(i0); i1 = wrap_idx(i1)
    j0 = wrap_idx(j0); j1 = wrap_idx(j1)
    f00 = F[i0, j0]; f10 = F[i1, j0]; f01 = F[i0, j1]; f11 = F[i1, j1]
    return ( (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11 )

def grad_ln_n(xq, yq):
    return sample_bilinear(glnx, xq, yq), sample_bilinear(glny, xq, yq)

# --- projected RHS: v' = (I - v v^T) ∇ln n ---
def rhs_v(xq, yq, vx, vy):
    gx, gy = grad_ln_n(xq, yq)
    vnorm = np.hypot(vx, vy); vx/=vnorm; vy/=vnorm
    dot = gx*vx + gy*vy
    ax = gx - dot*vx
    ay = gy - dot*vy
    return ax, ay

# --- RK4 step for v (direction only), Euler for x with unit speed ---
def trace(b, ds=ds, steps=steps):
    x = -L/2 + 1.0; y = b
    vx, vy = 1.0, 0.0
    for _ in range(steps):
        if x > L/2 - 1.0: break
        if abs(y) > L/2 - 1.0: break
        # RK4 on v
        k1x, k1y = rhs_v(x, y, vx, vy)
        k2x, k2y = rhs_v(x + 0.5*ds*vx, y + 0.5*ds*vy, vx + 0.5*ds*k1x, vy + 0.5*ds*k1y)
        k3x, k3y = rhs_v(x + 0.5*ds*vx, y + 0.5*ds*vy, vx + 0.5*ds*k2x, vy + 0.5*ds*k2y)
        k4x, k4y = rhs_v(x + ds*vx, y + ds*vy, vx + ds*k3x, vy + ds*k3y)
        dvx = (k1x + 2*k2x + 2*k3x + k4x)/6.0
        dvy = (k1y + 2*k2y + 2*k3y + k4y)/6.0
        vx += ds*dvx; vy += ds*dvy
        vn = np.hypot(vx, vy); vx/=vn; vy/=vn
        x += ds*vx; y += ds*vy
    return float(np.arctan2(vy, vx))

# --- main run ---
angles = [(b, trace(b)) for b in impacts]

os.makedirs("out", exist_ok=True)
pd.DataFrame(angles, columns=["impact","deflection"]).to_csv("out/lensing_deflection.csv", index=False)
with open("out/lensing_meta.json","w") as f:
    json.dump({"L":L,"N":N,"ds":ds,"steps":steps,"rho0":rho0,"fields":"spectral+2/3","integrator":"RK4+bilinear","units":"radians"}, f)

print("lensing done")
