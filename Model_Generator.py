import numpy as np
import matplotlib.pyplot as plt

nz = 3000
nx = 1500

# Modelo de 3 camadas
camadas = np.zeros((nz, nx))
camadas[:1000, :] = 1500
camadas[500:1000, :] = 3000
camadas[1000:2500, :] = 4500
camadas[2500:, :] = 5500

# Parâmetros da falha
x0 = 500
m = 5
D = 300  # rejeito ao longo da falha

# Vetor unitário paralelo à falha
tx = 1 / np.sqrt(1 + m**2)
tz = m / np.sqrt(1 + m**2)

dx = int(round(D * tx))
dz = int(round(D * tz))

X, Z = np.meshgrid(np.arange(nx), np.arange(nz))

falha = Z - m*(X - x0) < 0

X[falha] -= dx
Z[falha] -= dz

mask = ((X >= 0) & (X < nx) &
        (Z >= 0) & (Z < nz))

camadas[mask] = camadas[Z[mask], X[mask]]

camadas[500:1000, :] = 3000

# Visualização
plt.figure(figsize=(12, 6))
img = plt.imshow(camadas[:, int(dx):],vmax=6000, aspect='auto', cmap='jet')
cbar = plt.colorbar(img)
cbar.ax.invert_yaxis()
cbar.ax.set_title('Vp (m/s)', pad=20)
cbar.set_ticks([1500,3000, 4500, 5500])
cbar.set_ticklabels(['Camada 1 = 1500m/s','Camada 2 = 3000m/s','Camada 3 = 4500m/s','Camada 4 = 5500m/s'])
plt.title('Modelo com Falha Normal')
plt.xlabel('distância (m)')
plt.ylabel('profundidade (m)')
plt.show()