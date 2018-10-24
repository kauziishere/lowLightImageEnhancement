import rawpy
import numpy as np
import keras
import os
import pickle
from model import model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from PIL import Image
filepath="weights.best.hdf5"
np.seed(0)
def pre_process(filename):
	raw = rawpy.imread(filename)
	image = raw.raw_image_visible.astype(np.float32)
	#image = raw.black_level_per_channel
	image = np.maximum(image - 512, 0)/(16383 - 512)
	image = np.expand_dims(image, axis = 2)
	H = image.shape[0]
	W = image.shape[1]
	out = np.concatenate((image[0:H:2, 0:W:2, :], image[0:H:2, 1:W:2, :],image[1:H:2, 1:W:2, :],image[1:H:2, 0:W:2, :]), axis = 2)
	out = np.expand_dims(out, axis = 0)
	return  out

class DataGenerator(keras.utils.Sequence):
	def __init__(self, input_ids, output_ids, in_map_op_dict, batch_size = 4, in_dim = (512, 512), out_dim = (1024, 1024), shuffle = True, in_channels = 4, on_channels = 3):
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.batch_size = batch_size
		self.input_ids = input_ids
		self.output_ids = output_ids
		self.shuffle = shuffle
		self.in_channels = in_channels
		self.on_channels = on_channels
		self.in_map_op_dict = in_map_op_dict
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.input_ids))/self.batch_size)

	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		list_id_temps = [self.input_ids[k] for k in indexes]
		X, y = self.__data_generation(list_id_temps)
		return X, y

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.input_ids))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_id_temps):
		X = np.empty((self.batch_size, *self.in_dim, self.in_channels))
		y = np.empty((self.batch_size, *self.out_dim, self.on_channels))

		for i, ID in enumerate(list_id_temps):
			raw_in_image = pre_process(ID)
			H = raw_in_image.shape[1]
			W = raw_in_image.shape[2]
			xx = np.random.randint(0, W-512)
			yy = np.random.randint(0, H-512)
			input_fin = raw_in_image[:, yy:yy+512, xx:xx+512, :]
			raw_out_image = rawpy.imread(self.in_map_op_dict[ID])
			out = raw_out_image.postprocess(use_camera_wb = True, half_size = False, no_auto_bright = True, output_bps = 16)
			out = np.expand_dims(np.float32(out / 65535.0), axis=0)
			output_fin = out[:, yy*2: yy*2+1024, xx*2: xx*2 + 1024, :]
			X[i,] = input_fin
			y[i,] = output_fin
		return X, y

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

def custom_loss(act_image, out_image):
	return tf.reduce_mean(tf.abs(out_image - act_image))

def train():
	train_dict = get_file_from_pickle("train_dictionary.pkl")
	input_id_list = [x for x,_ in train_dict.items()]
	output_id_list = [x for _, x in train_dict.items()]
	train_generator = DataGenerator(input_id_list, output_id_list, train_dict)
	val_dict = get_file_from_pickle("val_dictionary.pkl")
	input_id_list = [x for x,_ in val_dict.items()]
	output_id_list = [x for _, x in val_dict.items()]
	val_generator = DataGenerator(input_id_list, output_id_list, val_dict)
	model = model()
	model.compile(optimizer = 'adam', loss = custom_loss, metrics = ['accuracy'])
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	if os.path.exists(filepath):
		model.load_weights(filepath)
	model.fit_generator(generator = train_generator,
			validation_data = val_generator,
			epochs = 10,
			callbacks = callbacks_list,
			verbose = 1)

def custom_test(filename):
	mod = model()
	mod.load_weights(filepath)
	image = pre_process(filename)
	image = image[:,:512,:512,:]
	out_img = mod.predict(image)
	out_img = np.squeeze(out_img, axis=0)
	print('image: {}'.format(out_img.shape))
	img = Image.fromarray(out_img, 'RGB')
	img.show()

if __name__ == "__main__":
#	train_dict, test_dict, val_dict = create_files("Sony_train_list.txt", "Sony_test_list.txt", "Sony_val_list.txt")
#	dump_dictionary(train_dict, "train_dictionary")
#	dump_dictionary(test_dict, "test_dictionary")
#	dump_dictionary(val_dict, "val_dictionary")
#	train()
	custom_test('./Sony/short/10003_08_0.1s.ARW')

