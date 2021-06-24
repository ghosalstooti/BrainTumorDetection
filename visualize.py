import numpy as np
import matplotlib.pyplot as plt
a=np.load("Data/Image/1.npy")
plt.imshow(a,'gray',origin='lower')
plt.show()
