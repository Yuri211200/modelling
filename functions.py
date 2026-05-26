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

ellipse = create_ellipse((170, 150), 100, 20)

#Camadas
v = [1000, 2500, 5000]

camadas = np.zeros((nz, nx))
camadas[:nz//2, :] = v[0]
camadas[nz//2:nz, :] = v[1]

#Atribuindo valores anomalos na região da elipse
for i in range(nz):
    for j in range(nx):
        if ellipse[i, j]:
            camadas[i, j] = v[2]

plt.figure()
plt.imshow(camadas)
plt.show()