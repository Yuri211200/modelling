import numpy as np
import matplotlib.pyplot as plt

#Definição do Grid
nz = 3000
nx = 1500

#Receptores e fonte
fz,fx = (1, len(nx)/2)
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
    u[fx, fz] += wavelet

    # Atualização (vetorizada)
    u_next[1:-1, 1:-1] = (2*u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (c**2 * dt**2 / dx**2) * (
                u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]))

    u_prev, u = u, u_next.copy()

    return u

# Snapshot no tempo escolhido
snapshot = np.zeros((nt,nx,nz))

u = np.zeros((nx, nz))
u_prev = np.zeros((nx, nz))
u_next = np.zeros((nx, nz))

for t in range(nt):
    
    u_next = DF(u, u_prev, u_next, camadas, dt, dx, fx, fz, wavelet)
    snapshot[t, :, :] = u_next

    # Atualiza estados
    u_prev, u = u, u_next.copy()

#Plot do snapshot

def save_snapshot(last,step, t):        
        
        if t > last:
             return
        if t % step != 0:
            return

        snapshot = snapshot[N_abc:nz_abc - N_abc, N_abc:nx_abc - N_abc]
        
        plt.figure(figsize=(8,8))
        plt.imshow(snapshot)
        plt.show()

last = range(nt)
step = 100