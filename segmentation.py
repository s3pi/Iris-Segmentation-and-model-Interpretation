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

def make_mat_of_paths(Data_path, Mask_path):
	Data_folder = sorted(os.listdir(Data_path))
	Mask_folder = sorted(os.listdir(Mask_path))

	Data_path_mat = []
	for f in Data_folder:
		Data_path_mat.append(Data_path + '/' + f)

	Mask_path_mat = []
	for f in Mask_folder:
		Mask_path_mat.append(Mask_path + '/' + f)

	return Data_path_mat, Mask_path_mat

def load_data_from_paths(path_mat_1, path_mat_2):
	data_mat_1 = []
	for f in path_mat_1:
		img = Image.open(f)
		img = np.array(img)
		img = img[2:-2, :, :]
		img_min = np.min(img)
		img_max = np.max(img)
		img = (img - img_min) / (img_max - img_min)
		data_mat_1.append(img) 

	data_mat_2 = []
	for f in path_mat_2:
		img = Image.open(f)
		img = np.array(img)
		img = img[2:-2, :]
		shape = list(img.shape)
		shape = np.append(shape, 1)
		img = np.reshape(img, shape)
		img_min = np.min(img)
		img_max = np.max(img)
		img = (img - img_min) / (img_max - img_min)
		data_mat_2.append(img)

	data_mat_1 = np.asarray(data_mat_1)
	data_mat_2 = np.asarray(data_mat_2)

	return data_mat_1, data_mat_2


def log(row_entry):
	with open('losses_list.csv', 'a') as csvFile:
	    writer = csv.writer(csvFile)
	    writer.writerow(row_entry)

	csvFile.close()

def compute_losses(Mask_val_mat, predicted_imgs):
	mse_img =  np.mean((Mask_val_mat[:,:,:,:] - predicted_imgs[:,:,:,:]) ** 2)
	check = math.sqrt(mse_img)
	psnr_img = 20 * math.log10( 1.0 / check)
	mae_img = np.mean(np.abs((Mask_val_mat[:,:,:,:] - predicted_imgs[:,:,:,:])))

	return mae_img, mse_img, psnr_img

def save_images(numEpochs, jj, img_row_size, img_col_size, Data_val_mat, Mask_val_mat, predicted_imgs):
	if jj % 2 == 0 or jj == numEpochs - 1:
		for i in range(0, len(predicted_imgs), 1000):
			temp = np.zeros([img_row_size, img_col_size*3, 3])
			temp[:img_row_size,:img_col_size, :] = Data_val_mat[i,:,:,:]
			temp[:img_row_size,img_col_size:img_col_size*2] = Mask_val_mat[i,:,:,:]
			temp[:img_row_size,img_col_size*2:] = predicted_imgs[i,:,:,:]
			temp = temp * 255
			#scipy.misc.imsave('results_sliced_4layers_custom_loss/' + str(jj) + '.jpg', temp)
			
			scipy.misc.imsave('Results/epoch_num' + str(jj) + 'img_no_' + str(i) + '.jpg', temp)

def perform_training(img_row_size, img_col_size, model, Data_train_path_mat, Data_val_path_mat, Mask_train_path_mat, Mask_val_path_mat):
	numEpochs = 50
	batch_size = 9
	losses_title = ['Epoch Number', 'batch_loss', 'mae_img', 'mse_img', 'psnr_img']
	log(losses_title)
	for jj in range(numEpochs):
		print("Running epoch : %d" % jj)
		batch_loss_file = open('Results/batch_loss_file.txt', 'a')
		batch_loss_per_epoch = 0.0
		num_batches = int(len(Data_train_path_mat)/batch_size)
		for batch in range(num_batches):
			batch_train_X_paths = Data_train_path_mat[batch*batch_size:min((batch+1)*batch_size,len(Data_train_path_mat))]
			batch_train_Y_paths = Mask_train_path_mat[batch*batch_size:min((batch+1)*batch_size,len(Mask_train_path_mat))]
			batch_train_X, batch_train_Y = load_data_from_paths(batch_train_X_paths, batch_train_Y_paths)
			print('while training')
			loss = model.train_on_batch(batch_train_X, batch_train_Y)
			print ('epoch_num: %d batch_num: %d loss: %f\n' % (jj,batch,loss))
			batch_loss_file.write("%d %d %f\n" % (jj, batch, loss))
			batch_loss_per_epoch += loss
		print('training over')
		batch_loss_per_epoch = batch_loss_per_epoch / num_batches

		model.save_weights("Model_weights/model_epoch_"+ str(jj % 10) +".h5")

		Data_val_mat, Mask_val_mat = load_data_from_paths(Data_val_path_mat, Mask_val_path_mat)
		predicted_imgs = model.predict(Data_val_mat)
		z,x,y,w=np.where(predicted_imgs <= 0.5)	
		predicted_imgs[z,x,y,w]=0
		z,x,y,w=np.where(predicted_imgs > 0.5)	
		predicted_imgs[z,x,y,w]=1
		mae_img, mse_img, psnr_img = compute_losses(Mask_val_mat, predicted_imgs)
		losses_list = [jj, batch_loss_per_epoch, mae_img, mse_img, psnr_img]
		log(losses_list)
		save_images(numEpochs, jj, img_row_size, img_col_size, Data_val_mat, Mask_val_mat, predicted_imgs)

	batch_loss_file.close()

opt = Adam(lr=0.00001)

def perform_test(model, Test_Data_path, Test_Mask_path, img_row_size, img_col_size):
	model.load_weights("Model_weights/model_epoch_9.h5")
	Test_Data_path_mat, Test_Mask_path_mat = make_mat_of_paths(Test_Data_path, Test_Mask_path)
	for each_test_case in range(len(Test_Data_path_mat)):
		Test_Data_mat, Test_Mask_mat = load_data_from_paths([Test_Data_path_mat[each_test_case]], [Test_Mask_path_mat[each_test_case]])
		predicted_imgs = model.predict(Test_Data_mat)
		for i in range(0, len(predicted_imgs)):
			temp = np.zeros([img_row_size, img_col_size*3, 3])
			temp[:img_row_size,:img_col_size, :] = Test_Data_mat[i,:,:,:]
			temp[:img_row_size,img_col_size:img_col_size*2] = Test_Mask_mat[i,:,:,:]
			temp[:img_row_size,img_col_size*2:] = predicted_imgs[i,:,:,:]
			temp = temp * 255
			#scipy.misc.imsave('results_sliced_4layers_custom_loss/' + str(jj) + '.jpg', temp)

			scipy.misc.imsave('predicted_mask/predited_img_no_' + str(each_test_case) + '.jpg', temp)

def main():
	Data_path = "/home/ada/Preethi/DL/Assignments/Ass_3/Q2/Data"
	Mask_path = "/home/ada/Preethi/DL/Assignments/Ass_3/Q2/Mask"
	Test_Data_path = "/home/ada/Preethi/DL/Assignments/Ass_3/Q2/Test_Data"
	Test_Mask_path = "/home/ada/Preethi/DL/Assignments/Ass_3/Q2/Test_Mask"

	''' Make model '''
	# input has to be a multiple of 2**3 because of the number of downsampling and upsampling layers = 3.
	img_row_size = 296
	img_col_size = 400
	input_shape = Input(shape = (img_row_size, img_col_size, 3))
	output = autoencoder(input_shape)
	model = Model(input_shape, output)

	# model.load_weights("Model_HGSR/model_epoch_295.h5") #for loading the saved weights, prefix this line before compiling
	model.compile(loss=binary_crossentropy, optimizer = opt)
	
	with open('autoencoder.txt', 'w') as f:
		with redirect_stdout(f):
			model.summary()
	
	Data_path_mat, Mask_path_mat = make_mat_of_paths(Data_path, Mask_path)
	Data_train_path_mat, Data_val_path_mat, Mask_train_path_mat, Mask_val_path_mat = train_test_split(Data_path_mat, Mask_path_mat)
	
	perform_training(img_row_size, img_col_size, model, Data_train_path_mat, Data_val_path_mat, Mask_train_path_mat, Mask_val_path_mat)

	perform_test(model, Test_Data_path, Test_Mask_path, img_row_size, img_col_size)
	

main()
