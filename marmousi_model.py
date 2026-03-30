import numpy as np
import matplotlib.pyplot as plt

nz = 751
nx = 2301

vp = np.fromfile("marmousi_vp.bin", dtype=np.float32)
vp = vp.reshape(nx,nz)
vp = vp.T

plt.figure(figsize=(15,5))
plt.imshow(vp, aspect='auto', cmap='jet')
plt.title("VP")
plt.colorbar()
plt.show()