import numpy as np
import tensorflow as tf
import cv2 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


input_size=(128,128,1)
inputs = Input(input_size)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1=BatchNormalization()(conv1)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
conv1=BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
pool1=BatchNormalization()(pool1)

conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2=BatchNormalization()(conv2)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
conv2=BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool2=BatchNormalization()(pool2)

conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3=BatchNormalization()(conv3)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
conv3=BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
pool3=BatchNormalization()(pool3)

conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4=BatchNormalization()(conv4)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
conv4=BatchNormalization()(conv4)
drop4 = Dropout(0.5)(conv4)
drop4=BatchNormalization()(drop4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
pool4=BatchNormalization()(pool4)


conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5=BatchNormalization()(conv5)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
conv5=BatchNormalization()(conv5)
drop5 = Dropout(0.5)(conv5)
drop5=BatchNormalization()(drop5)
pool5=MaxPooling2D(pool_size=(2, 2))(drop5)
pool5=BatchNormalization()(pool5)

conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
conv6=BatchNormalization()(conv6)
conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
conv6=BatchNormalization()(conv6)
drop6 = Dropout(0.5)(conv6)
drop6=BatchNormalization()(drop6)


up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
up7=BatchNormalization()(up7)
merge7 = concatenate([drop5,up7], axis = 3)
merge7=BatchNormalization()(merge7)
conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7=BatchNormalization()(conv7)
conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
conv7=BatchNormalization()(conv7)

up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
up8=BatchNormalization()(up8)
merge8 = concatenate([conv4,up8], axis = 3)
merge8=BatchNormalization()(merge8)
conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8=BatchNormalization()(conv8)
conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
conv8=BatchNormalization()(conv8)


up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
up9=BatchNormalization()(up9)
merge9 = concatenate([conv3,up9], axis = 3)
merge9=BatchNormalization()(merge9)
conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9=BatchNormalization()(conv9)
conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

conv9=BatchNormalization()(conv9)

up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
up10=BatchNormalization()(up10)
merge10 = concatenate([conv2,up10], axis = 3)
merge10=BatchNormalization()(merge10)
conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
conv10=BatchNormalization()(conv10)
conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)

conv10=BatchNormalization()(conv10)

up11 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
up11=BatchNormalization()(up11)
merge11 = concatenate([conv1,up11], axis = 3)
merge11=BatchNormalization()(merge11)
conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11)
conv11=BatchNormalization()(conv11)
conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
conv11=BatchNormalization()(conv11)


conv12 = Conv2D(1, 1, activation = 'sigmoid')(conv11)

image_path="Data/Train/Image/"
mask_path="Data/Train/Mask/"
images_list=os.listdir(image_path)
masks_list=os.listdir(mask_path)
#print(images_list)
#print(masks_list)


TRI=[]
TRM=[]
for i in images_list:
	TRI.append(image_path+i)
	TRM.append(mask_path+i)

image_path="Data/Test/Image/"
mask_path="Data/Test/Mask/"
images_list=os.listdir(image_path)
masks_list=os.listdir(mask_path)
TEI=[]
TEM=[]
for i in images_list:
	TEI.append(image_path+i)
	TEM.append(mask_path+i)
#print(TEI)
#print(TEM)

sgd=SGD(lr=1e-3,decay=1e-2,momentum=.7,nesterov=True)
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



class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=1, dim=(128,128), n_channels=1,
		         n_classes=2, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)
		#print(np.shape(y))
		#np.save("sample.npy",X)
		
		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		Y = np.empty((self.batch_size, *self.dim, self.n_channels))
		

		# Generate data
		for batch, ID in enumerate(list_IDs_temp):
            # Store sample
			kernel=np.ones((5,5))
			x = np.load( ID )
			x=cv2.resize(x, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
			x=x.astype(int)
			x=tf.keras.utils.normalize(x)
			#x=cv2.equalizeHist(x)
			
			
			x=np.expand_dims(x,axis=2)
			X[batch,] = x
			ID=ID.replace("Image","Mask")
			y = np.load(ID)
			y=cv2.resize(y, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
			y=np.expand_dims(y,axis=2)
			Y[batch,] = y
			

		return X,Y


model=Model(inputs,conv12)
model.compile(optimizer = 'adam', loss = dice_coef_loss,metrics=[dice_coef])


trainGen=DataGenerator(TRI,TRM)
testGen=DataGenerator(TEI,TEM)
epochs=50
model.fit(trainGen,
	  steps_per_epoch=int(2000/epochs),
          epochs=epochs,
	  validation_data=testGen,
	  validation_steps=int(1064/epochs))
model.save("result.h5")


