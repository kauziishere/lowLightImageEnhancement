import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Concatenate,Reshape, Conv2DTranspose,Conv2D, MaxPooling2D, LeakyReLU, UpSampling2D, Lambda
import keras
from preprocess import *

def function_to_depth_to_space(x):
	x = tf.depth_to_space(x, 2)
	return x

def function_to_remove_negs(x):
	return tf.minimum(tf.maximum(x, 0), 1)

def model(input_shape = (512, 512, 4)):
	activation_fn = LeakyReLU(0.2)
	inpu = Input(shape = input_shape)

	conv1 = Conv2D(32, (3, 3), name = "conv2d_1", padding = 'same')(inpu)
	conv1 = LeakyReLU(0.2)(conv1)
	conv1 = Conv2D(16, (1, 1), name = "conv2d_1.5", padding = 'same')(conv1)
	conv1 = Conv2D(32, (3,3),  name = "conv2d_2", padding = 'same')(conv1)
	conv1 = LeakyReLU(0.2)(conv1)
	pool1 = MaxPooling2D((2, 2), padding = 'same',  name = "pool_1")(conv1)

	conv2 =  Conv2D(64, (3,3),  name = "conv2d_3", padding = 'same')(pool1)
	conv2 = LeakyReLU(0.2)(conv2)
	conv2 = Conv2D(32, (1, 1), name = "conv2d_3.5", padding = 'same')(conv2)
	conv2 =  Conv2D(64, (3,3),  name = "conv2d_4", padding = 'same')(conv2)
	conv2 = LeakyReLU(0.2)(conv2)
	pool2 = MaxPooling2D((2, 2), padding = 'same',  name = "pool_2")(conv2)

	conv3 =  Conv2D(128, (3,3),  name = "conv2d_5", padding = 'same')(pool2)
	conv3 = LeakyReLU(0.2)(conv3)
	conv3 = Conv2D(32, (1, 1), name = 'conv2d_5.5', padding = 'same')(conv3)
	conv3 =  Conv2D(128, (3,3),  name = "conv2d_6", padding = 'same')(conv3)
	conv3 = LeakyReLU(0.2)(conv3)
	pool3 = MaxPooling2D((2, 2), padding = 'same',  name = "pool_3")(conv3)

	conv4 =  Conv2D(256, (3,3),  name = "conv2d_7", padding = 'same')(pool3)
	conv4 = LeakyReLU(0.2)(conv4)
	conv4 = Conv2D(32, (1, 1), name = 'conv2d_7.5', padding = 'same')(conv4)
	conv4 =  Conv2D(256, (3,3),  name = "conv2d_8", padding = 'same')(conv4)
	conv4 = LeakyReLU(0.2)(conv4)
	pool4 = MaxPooling2D((2, 2), padding = 'same',  name = "pool_4")(conv4)

	conv5 = Conv2D(512, (3, 3), name = "conv2d_9", padding = 'same')(pool4)
	conv5 = LeakyReLU(0.2)(conv5)
	conv5 = Conv2D(32, (1, 1), name = 'conv2d_9.5', padding = 'same')(conv5)
	conv5 =  Conv2D(512, (3,3), name = "conv2d_10", padding = 'same')(conv5)
	conv5 = LeakyReLU(0.2)(conv5)

	#up6 = UpSampling2D(size = (2, 2))(conv5)
	up6 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(conv5)
	up6 = Concatenate(axis=3)([conv4, up6])
	conv6 = Conv2D(256, (3, 3), name = "conv2d_11", padding = 'same')(up6)
	conv6 = LeakyReLU(0.2)(conv6)
	conv6 = Conv2D(256, (3, 3), name = "conv2d_12", padding = 'same')(conv6)
	conv6 = LeakyReLU(0.2)(conv6)

	up7 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(conv6)
	up7 = Concatenate(axis=3)([conv3, up7])
	conv7 = Conv2D(128, (3, 3), name = "conv2d_13", padding = 'same')(up7)
	conv7 = LeakyReLU(0.2)(conv7)
	conv7 = Conv2D(128, (3, 3), name = "conv2d_14", padding = 'same')(conv7)
	conv7 = LeakyReLU(0.2)(conv7)

	up8 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(conv7)
	up8 = Concatenate(axis=3)([conv2, up8])
	conv8 = Conv2D(64, (3, 3), name = "conv2d_15", padding = 'same')(up8)
	conv8 = LeakyReLU(0.2)(conv8)
	conv8 = Conv2D(64, (3, 3), name = "conv2d_16", padding = 'same')(conv8)
	conv8 = LeakyReLU(0.2)(conv8)

	up9 = Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = 'same')(conv8)
	up9 = Concatenate(axis=3)([conv1, up9])
	conv9 = Conv2D(32, (3, 3), name = "conv2d_17", padding = 'same')(up9)
	conv9 = LeakyReLU(0.2)(conv9)
	conv9 = Conv2D(32, (3, 3), name = "conv2d_18", padding = 'same')(conv9)
	conv9 = LeakyReLU(0.2)(conv9)

	conv10 = Conv2D(12, (1, 1), name = "conv2d_19", padding = 'same')(conv9)
	conv10 = LeakyReLU(0.2)(conv10)
	convfin = Lambda(function_to_depth_to_space)(conv10)
	convexp = Lambda(function_to_remove_negs)(convfin)
	model = Model(inputs = inpu, outputs = convexp)
	return model

if __name__ == "__main__":
	net = model()
	net.summary()
