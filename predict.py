import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from unet import DataGenerator

path_img="Data/Train/Image/3.npy"
path_mask="Data/Train/Mask/3.npy"
img=np.load(path_mask)
w=len(img)
h=len(img[0])
plt.imshow(img,'gray')
plt.show()

img=np.load(path_img)
plt.imshow(img,'gray')
plt.show()

I=np.empty((1,128,128,1))
img=cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
img=tf.keras.utils.normalize(img)
img=np.expand_dims(img,axis=2)

I[0,]=img

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model=tf.keras.models.load_model("result.h5",custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss})
mask=model.predict(I)
print(mask)
try:
	mask=cv2.resize(mask[0],(w,h),interpolation=cv2.INTER_CUBIC)
except:
	print("hw")
plt.imshow(mask,'gray')
plt.show()

