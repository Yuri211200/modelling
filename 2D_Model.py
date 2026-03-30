import numpy as np

#Definição do Grid
nz,nx = np.zeros(1000), np.zeros(2000)

#Receptores e fonte
rz,rx = (1,len(nx))
fz,fx = (1, len(nx)/2)

#Fonte Ricker
from propagation import ricker

f = 10
nt = 1000
time = np.linspace(0, 0.1, nt)
wavelet = ricker(f, time)

#Parametros
v1 = 1500
v2 = 5000

camada_1:
for i in nz:
    for j in nx:
        

#Diferenças Finitas