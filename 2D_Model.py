import numpy as np
import matplotlib.pyplot as plt

#Definição do Grid
nz = 500
nx = 1000

#Receptores e fonte
fz,fx = (1, nx//2)
rz,rx = (1, nx)

#Fonte Ricker
from propagation import ricker

f = 10
nt = 1000
time = np.linspace(0, 0.1, nt)
wavelet = ricker(f, time)

#Camadas
v = [1500, 5000]

camadas = np.zeros((nz, nx))
camadas[:nz//2, :] = v[0]
camadas[nz//2:nz, :] = v[1]

#Parâmetros
dx = 1
dt = 0.001

#Diferenças Finitas
def DF(u, u_prev, u_next, c, dt, dx, fx, fz, wavelet):

    # Propagação a partir da fonte    
    u[fz, fx] += wavelet[t]

    # Atualização (vetorizada)
    u_next[1:-1, 1:-1] = (2*u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + ((c[1:-1, 1:-1]**2)**2 * dt**2 / dx**2) * (
                u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]))

    u_prev, u = u, u_next.copy()

    return u

# Snapshot no tempo escolhido
snapshot = np.zeros((nt,nz,nx))

u = np.zeros((nz, nx))
u_prev = np.zeros((nz, nx))
u_next = np.zeros((nz, nx))

for t in range(nt):
    
    u_next = DF(u, u_prev, u_next, camadas, dt, dx, fx, fz, wavelet)
    snapshot[t, :, :] = u_next

    # Atualiza estados
    u_prev, u = u, u_next.copy()

#Plot do snapshot

def plot_snapshot(snapshot, t, step):
    
    if t >= snapshot.shape[0]:
        print("Tempo fora do range")
        return
    
    if t % step != 0:
        return

    plt.figure(figsize=(8,6))
    plt.imshow(snapshot[t, :, :], cmap='seismic', aspect='auto')
    plt.colorbar(label='Amplitude')
    plt.title(f"Snapshot no tempo t = {t}")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.gca().invert_yaxis()
    plt.show()

step = 10
t = 600

plot_snapshot(snapshot, t, step)