import numpy as np
import matplotlib.pyplot as plt

#Número de amostras de cada dimensão
nz = 751
nx = 2301

#Leitura do valor de Vp do modelo
vp = np.fromfile("marmousi_vp.bin", dtype=np.float32)
vp = vp.reshape(nx,nz)
vp = vp.T

#Plot
plt.figure(figsize=(15,5))
plt.imshow(vp, aspect='auto', cmap='jet')
plt.title("VP")
plt.colorbar()
plt.show()

#Parametros
f = 10
alfa = 4
NC = 0.4
fmax = 3*f
vp_min = np.min(vp)
vp_max = np.max(vp)

#Grid e dt
dx = vp_min/(alfa * fmax)
dt = (NC * dx)/vp_max