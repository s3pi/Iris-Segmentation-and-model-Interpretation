import tensorflow as tf
from keras.models import load_model
from builtins import range, map, zip, filter
from io import open
import six
import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import innvestigate
import innvestigate.utils
from PIL import Image
import numpy as np
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add
from keras.models import Model, Sequential
from contextlib import redirect_stdout
import os
from sklearn.model_selection import train_test_split
import csv
from keras.losses import binary_crossentropy
from keras.optimizers import SGD,RMSprop,adam,Adam
import math
import scipy
from keras_applications import imagenet_utils
import keras.applications.vgg16 as vgg16

def autoencoder(input_img):
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(512, (3, 3), activation='sigmoid', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	up6 = concatenate([conv5, conv4],axis=3)
	# up6 = merge([conv5, conv4], mode='concat', concat_axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	up7 = UpSampling2D((2,2))(conv6)
	up7 = concatenate([up7, conv3],axis=3)
	# up7 = merge([up7, conv3], mode='concat', concat_axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	up8 = UpSampling2D((2,2))(conv7)
	up8 = concatenate([up8, conv2],axis=3)
	# up8 = merge([up8, conv2], mode='concat', concat_axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	up9 = UpSampling2D((2,2))(conv8)
	up9 = concatenate([up9, conv1],axis=3)
	# up9 = merge([up9, conv1], mode='concat', concat_axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)	
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)
	decoded_2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9)
	return decoded_2

img_row_size = 224
img_col_size = 224
input_shape = Input(shape = (img_row_size, img_col_size, 3))
output = autoencoder(input_shape)
model = Model(input_shape, output)


base_dir = os.path.dirname(__file__)
img_name = 'my_img.tiff'
utils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))
image = utils.load_image(os.path.join(base_dir, img_name), 224)
# plt.imshow(image/255)
plt.axis('off')
plt.savefig('readme_example_input_image' + img_name)
our_model = model.load_weights('model_epoch_9.h5')
# model, preprocess = our_model, our_model.preprocess_input
# model = our_model.preprocess_input
# model = innvestigate.utils.model_wo_softmax(model)
analyzer = innvestigate.create_analyzer("guided_backprop", our_model)
# preprocess = vgg16.preprocess_input
x = image[None]
print(x)
# print(analyzer)
a = analyzer.analyze(x)
print(a)

a = a.sum(axis=np.argmax(np.asarray(a.shape)==3))
# print(np.max(np.abs(a)))
a /= np.max(np.abs(a))

plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
plt.axis('off')
plt.savefig("readme_example_analysis_our_img_2.png")