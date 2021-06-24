import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

filepath="Datamat/Train/"
file_list=os.listdir(filepath)
print(file_list)

for i in file_list:
	path=filepath+i
	f=h5py.File(path)
	cjdata=f['cjdata']
	
	tumorMask = np.array(cjdata.get('tumorMask')).astype(np.float64)
	image=np.array(cjdata.get('image')).astype(np.float64)
	destination=("Data/Image/"+i).rstrip(".mat")+".npy"
	np.save(destination,image)
	destination=("Data/Mask/"+i).rstrip(".mat")+".npy"
	np.save(destination,tumorMask)
