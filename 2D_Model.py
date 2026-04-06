import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from matplotlib.animation import FuncAnimation

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
nt = 4000
time = np.linspace(0, 2, nt)
wavelet = ricker(f, time)

plt.plot(wavelet)
plt.show()

#Camadas
v = [1500, 5000]

camadas = np.zeros((nz, nx))
camadas[:nz//2, :] = v[0]
camadas[nz//2:nz, :] = v[1]

plt.figure()
plt.imshow(camadas)
plt.scatter(rx, rz, label = "receptor")
plt.scatter(fx, fz, label = "fonte", color='r')
# plt.show()
#Parâmetros
dx = 10
dz = 10
dt = 0.0005

# Snapshot no tempo escolhido
snapshot = np.zeros((nt,nz,nx))

u = np.zeros((nz, nx))
u_prev = np.zeros((nz, nx))
u_next = np.zeros((nz, nx))

# fig, ax = plt.subplots()
# img = ax.imshow(u, cmap='viridis', vmin=-1, vmax=1)
# plt.colorbar(img)

#Diferenças Finitas
@jit(nopython=True, parallel=True)
def DF(u, nx, nz, dx, dz):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0
    for i in prange(4,nx-4):
        for j in prange(4,nz-4):
            pxx = (c0 * u[j, i] + c1 * (u[j, i+1] + u[j, i-1]) + c2 * (u[j, i+2] + u[j, i-2]) + c3 * (u[j, i+3] + u[j, i-3]) +c4 * (u[j, i+4] + u[j, i-4])) / (dx * dx)
            pzz = (c0 * u[j, i] + c1 * (u[j+1, i] + u[j-1, i]) + c2 * (u[j+2, i] + u[j-2, i]) + c3 * (u[j+3, i] + u[j-3, i]) + c4 * (u[j+4, i] + u[j-4, i])) / (dz * dz)
            
    # img.set_array(u)
    
    laplacian = pxx + pzz

    return laplacian #, [img]

for t in range(nt):
    # Propagação a partir da fonte    
    u[fz, fx] += wavelet[t]
    laplacian = DF(u, nx, nz, dx, dz)
    u_prev = (camadas ** 2) * (dt ** 2) * laplacian + 2 * u - u_next
    #snapshot[t, :, :] = u

    if t % 300 == 0:
        plt.imshow(u)
        plt.show()

    # Atualiza estados
    u_next = u
    u = u_prev

#Plot do snapshot

def plot_snapshot(snapshot, t, step):     
    # if t % step != 0:
    #     return

    plt.figure(figsize=(8,6))
    plt.imshow(snapshot[t, :, :], cmap='seismic', aspect='auto')
    plt.colorbar(label='Amplitude')
    plt.title(f"Snapshot no tempo t = {t}")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.gca().invert_yaxis()
    plt.show()

step = 200
t = 2000

plot_snapshot(snapshot, t, step)

# ani = FuncAnimation(fig, DF, frames=10000, interval=200)

@njit(parallel=True, fastmath=True)
def _forward_kernel(
  upas: np.ndarray,
  upre: np.ndarray,
  ufut: np.ndarray,
  laplacian: np.ndarray,
  inv_dh2: float,
  nzz: int,
  nxx: int,
  ricker: np.ndarray,
  ix: int,
  iz: int,
  dh2: float,
  arg: np.ndarray,
  t: int,
) -> None:

  
  upre[iz, ix] += ricker[t] / dh2

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2 = (
        -9.0   * upre[i-4, j] + 128.0   * upre[i-3, j] - 1008.0 * upre[i-2, j] +
        8064.0 * upre[i-1, j] - 14350.0 * upre[i,   j] + 8064.0 * upre[i+1, j] -
        1008.0 * upre[i+2, j] + 128.0   * upre[i+3, j] - 9.0    * upre[i+4, j]
      )

      d2u_dz2 = (
        -9.0   * upre[i, j-4] + 128.0   * upre[i, j-3] - 1008.0 * upre[i, j-2] +
        8064.0 * upre[i, j-1] - 14350.0 * upre[i, j]   + 8064.0 * upre[i, j+1] -
        1008.0 * upre[i, j+2] + 128.0   * upre[i, j+3] - 9.0    * upre[i, j+4]
      )

      laplacian[i, j] = (d2u_dx2 + d2u_dz2) * inv_dh2

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):

      upas[i, j] = arg[i, j] * laplacian[i, j] + 2.0 * upre[i, j] - ufut[i, j]

      ufut[i, j] = upre[i, j]
      upre[i, j] = upas[i, j]