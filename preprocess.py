import numpy as np
import keras, os, pickle, scipy, rawpy
from model import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from PIL import Image
filepath="weights.{epoch:03d}.hdf5"
np.random.seed(0)
def pre_process(filename):
	raw = rawpy.imread(filename)
	image = raw.raw_image_visible.astype(np.float32)
	#image = raw.black_level_per_channel
	image = np.maximum(image - 512, 0)/(16383 - 512)
	image = np.expand_dims(image, axis = 2)
	H = image.shape[0]
	W = image.shape[1]
	out = np.concatenate((image[0:H:2, 0:W:2, :], image[0:H:2, 1:W:2, :],image[1:H:2, 1:W:2, :],image[1:H:2, 0:W:2, :]), axis = 2)
	out = np.expand_dims(out, axis = 0)*100.0
	return  out

def create_files(train_images_file, test_images_file, val_images_file):
	train_dict = dict()
	test_dict = dict()
	val_dict = dict()
	with open(train_images_file, "r") as f:
		while True:
			line = f.readline()
			if not line:
				break
			input_file, output_file,_, _ = line.strip().split()
			train_dict[input_file] = output_file

	with open(test_images_file, "r") as f:
		while True:
			line = f.readline()
			if not line:
				break
			input_file, output_file, _, _ = line.strip().split()
			test_dict[input_file] = output_file

	with open(val_images_file, "r") as f:
		while True:
			line = f.readline()
			if not line:
				break
			input_file, output_file, _, _	 = line.strip().split()
			val_dict[input_file] = output_file
	return train_dict, test_dict, val_dict

def dump_dictionary(dictionary, filename):
	with open(filename+".pkl", "wb") as f:
		pickle.dump(dictionary, f)

def get_file_from_pickle(filename):
	with open(filename, 'rb') as f:
		d = pickle.load(f)
	return d

def create_dict_files():
	train_dict, test_dict, val_dict = create_files("Sony_train_list.txt", "Sony_test_list.txt", "Sony_val_list.txt")
	dump_dictionary(train_dict, "train_dictionary")
	dump_dictionary(test_dict, "test_dictionary")
	dump_dictionary(val_dict, "val_dictionary")

if __name__ == "__main__":
	pass
