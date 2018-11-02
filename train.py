import numpy as np
import keras, os, pickle, scipy, rawpy
from model import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from PIL import Image
from preprocess import *
filepath="weights.{epoch:03d}.hdf5"
np.random.seed(0)

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
			input_fin = np.minimum(input_fin, 1.0)
			raw_out_image = rawpy.imread(self.in_map_op_dict[ID])
			out = raw_out_image.postprocess(use_camera_wb = True, half_size = False, no_auto_bright = True, output_bps = 16)
			out = np.expand_dims(np.float32(out / 65535.0), axis=0)
			output_fin = out[:, yy*2: yy*2+1024, xx*2: xx*2 + 1024, :]
			X[i,] = input_fin
			y[i,] = output_fin
		return X, y

def custom_loss(act_image, out_image):
	return tf.reduce_mean(tf.abs(out_image - act_image))

def decrease_train_input(input_id_list, output_id_list):
	small_in_list = []
	small_out_list = []
	for i in range(0, len(input_id_list)):
		if(input_id_list[i].split('_')[1] == '00' and input_id_list[i].split('.')[2] == '1s' and output_id_list[i].split('.')[1].split('_')[2] == '10s'):
				small_in_list.append(input_id_list[i])
				small_out_list.append(output_id_list[i])
	return small_in_list, small_out_list

if __name__ == "__main__":
	train_dict = get_file_from_pickle("train_dictionary.pkl")
	input_id_list = [x for x,_ in train_dict.items()]
	output_id_list = [x for _, x in train_dict.items()]
	input_id_list, output_id_list = decrease_train_input(input_id_list, output_id_list)
	train_generator = DataGenerator(input_id_list, output_id_list, train_dict)
	print("Number of input files are: {}".format(len(input_id_list)))
	net = model()
	net.load_weights('./result_dir/weights.020.hdf5')
	sgd = SGD(lr = 0.003, nesterov = True)
	net.compile(optimizer = sgd, loss = custom_loss, metrics = ['accuracy'])
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
	callbacks_list = [checkpoint]
	net.fit_generator(generator = train_generator,
			epochs = 70,
			callbacks = callbacks_list,
			verbose = 1)
