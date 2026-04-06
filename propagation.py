import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Tamanho da grade
nx, ny = 300, 300

# Parâmetros físicos
c = 250.0
dx = 1
dt = 0.001

# Estabilidade (CFL)
if c * dt / dx > 1/np.sqrt(2):
    raise ValueError("Instabilidade numérica!")

# Campos
u = np.zeros((nx, ny))
u_prev = np.zeros((nx, ny))
u_next = np.zeros((nx, ny))

# Fonte
cx, cy = 1, ny // 2

# Wavelet de Ricker
def ricker(f, t):
    return (1 - 2*(np.pi*f*t)**2) * np.exp(-(np.pi*f*t)**2)

f = 15
nt = 1000
time = np.linspace(0, 0.1, nt)
wavelet = ricker(f, time)

# Plot
fig, ax = plt.subplots()
img = ax.imshow(u, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(img)

def update(frame):
    global u, u_prev, u_next

    # Fonte
    if frame < nt:
        u[cx, cy] += wavelet[frame]

    # Atualização (vetorizada)
    u_next[1:-1, 1:-1] = (2*u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
        (c**2 * dt**2 / dx**2) * (u[2:, 1:-1] + u[:-2, 1:-1] +
            u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]))

    # Atualiza estados
    u_prev, u = u, u_next.copy()

    img.set_array(u)
    return [img]

ani = FuncAnimation(fig, update, frames=10000, interval=1)



plt.title("Onda 2D com fonte Ricker")
plt.show()