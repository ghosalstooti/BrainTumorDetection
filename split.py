import os
from shutil import copy

path_image="Data/Data/Image/"
path_mask="Data/Data/Mask/"

images=os.listdir(path_image)
masks=os.listdir(path_mask)
images.sort()
masks.sort()

for i in range(len(images)):
	n=int(images[i].rstrip(".npy"))
	if(n%4==0):
		copy(path_image+images[i],"Data/Test/Image")
		copy(path_mask+images[i],"Data/Test/Mask")
	else:
		copy(path_image+images[i],"Data/Train/Image")
		copy(path_mask+images[i],"Data/Train/Mask")


