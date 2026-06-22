import numpy as np
import matplotlib.pyplot as plt

#Definição do Grid
nz = 301
nx = 301

#Função para criar uma elipse
def create_ellipse(center, a, b):
    y, x = np.ogrid[-center[0]:nz-center[0], -center[1]:nx-center[1]]
    elipse = ((x**2)/a**2) + (y**2/b**2) <= 1
    return elipse

ellipse = create_ellipse((210, 150), 100, 10)

#Camadas
v = [1500, 2250, 2700, 2500]

camadas = np.zeros((nz, nx))
camadas[:nz//3, :] = v[0]
camadas[nz//3:2*nz//3, :] = v[1]
camadas[2*nz//3:nz, :] = v[2]

#Atribuindo valores anômalos na região da elipse
for i in range(nz):
    for j in range(nx):
        if ellipse[i, j]:
            camadas[i, j] = v[3]

plt.figure()
img = plt.imshow(camadas, aspect='auto', vmin=1000, vmax=3000, cmap='jet')
plt.colorbar(img, label='Vp (m/s)')
plt.title('Modelo Geológico')
plt.show()