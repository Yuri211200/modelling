import numpy as np
import matplotlib.pyplot as plt

nz = 3000
nx = 1500

# Modelo de 3 camadas
camadas = np.zeros((nz, nx))
camadas[:nz//3, :] = 1500
camadas[nz//3:2*nz//3, :] = 2250
camadas[2*nz//3:, :] = 2700

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

# Visualização
plt.figure(figsize=(10, 6))
img = plt.imshow(camadas[:, int(dx):], aspect='auto', cmap='jet')
plt.colorbar(img, label='Vp (m/s)')
plt.title('Modelo com Falha Normal')
plt.xlabel('distância (m)')
plt.ylabel('profundidade (m)')
plt.show()