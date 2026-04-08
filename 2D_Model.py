import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

#Definição do Grid
nz = 301
nx = 301

#Receptores e fonte
fz = 1
fx = nx//2
rz = np.full(nx,1)
rx = np.arange(0,nx)

#Fonte Ricker
def ricker(f, t):
    t = t - 0.15
    return (1 - 2*(np.pi*f*t)**2) * np.exp(-(np.pi*f*t)**2)

f = 10
nt = 5000
time = np.linspace(0, 2, nt)
wavelet = ricker(f, time)

plt.plot(wavelet)
plt.show()

#Camadas
v = [1000, 2500]

camadas = np.zeros((nz, nx))
camadas[:nz//2, :] = v[0]
camadas[nz//2:nz, :] = v[1]

plt.figure()
plt.imshow(camadas)
plt.scatter(rx, rz, label = "receptor")
plt.scatter(fx, fz, label = "fonte", color='r')
plt.show()

#Parâmetros
dx = 10
dz = 10
dt = 0.001

# Snapshot no tempo escolhido
snapshot = np.zeros((nt,nz,nx))

u = np.zeros((nz, nx))
u_prev = np.zeros((nz, nx))
u_next = np.zeros((nz, nx))

#Diferenças Finitas
@njit(nopython=True, parallel=True)
def DF(u, nx, nz, dx, dz):
    laplacian = np.zeros_like(u)
    c0 = -14350 / 5040
    c1 = 8064 / 5040
    c2 = -1008.0 / 5040
    c3 = 128.0 / 5040
    c4 = -9.0 / 5040
    for i in prange(4,nx-4):
        for j in prange(4,nz-4):
            pxx = (c0 * u[j, i] + c1 * (u[j, i+1] + u[j, i-1]) + c2 * (u[j, i+2] + u[j, i-2]) + c3 * (u[j, i+3] + u[j, i-3]) +c4 * (u[j, i+4] + u[j, i-4])) / (dx * dx)
            pzz = (c0 * u[j, i] + c1 * (u[j+1, i] + u[j-1, i]) + c2 * (u[j+2, i] + u[j-2, i]) + c3 * (u[j+3, i] + u[j-3, i]) + c4 * (u[j+4, i] + u[j-4, i])) / (dz * dz)
            laplacian[j, i] = pxx + pzz

    return laplacian 

# Propagação a partir da fonte    
for t in range(nt):
    u[fz, fx] += wavelet[t]
    laplacian = DF(u, nx, nz, dx, dz)
    u_next = 2*u - u_prev + (camadas**2)*(dt**2)*laplacian
    snapshot[t, :, :] = u

    if t % 100 == 0:
        plt.imshow(u)
        plt.show()

    # Atualiza estados
    u_prev = u
    u = u_next