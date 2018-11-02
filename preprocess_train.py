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

def decrease_train_input(input_id_list, output_id_list):
	small_in_list = []
	small_out_list = []
	for i in range(0, len(input_id_list)):
		if(input_id_list[i].split('_')[1] == '00' and input_id_list[i].split('.')[2] == '1s' and output_id_list[i].split('.')[1].split('_')[2] == '10s'):
				small_in_list.append(input_id_list[i])
				small_out_list.append(output_id_list[i])
	return small_in_list, small_out_list
	#exit()

class RestoreCkptCallback(keras.callbacks.Callback):
    def __init__(self, pretrained_file):
        self.pretrained_file = pretrained_file
        self.sess = keras.backend.get_session()
        self.saver = tf.train.Saver()
    def on_train_begin(self, logs=None):
        if self.pretrained_file:
            self.saver.restore(self.sess, self.pretrained_file)
            print('load weights: OK.')

def set_ckpt_weights(net):
		p1 = './g_conv'
		p2 = '_'
		m = 1
		s = 1
		names = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12', 'conv2d_13', 'conv2d_14', 'conv2d_15', 'conv2d_16', 'conv2d_17', 'conv2d_18', 'conv2d_19']
		for name in names[:-1]:
			val1 = np.load(p1+str(s)+p2+str(m)+'_weights.npy')
			val2 = np.load(p1+str(s)+p2+str(m)+'_biases.npy')
			layer = net.get_layer(name)
			if m == 1:
				m = 2
			else:
				m = 1
				s+=1
		val1 = np.load('g_conv10_weights.npy')
		val2 = np.load('g_conv10_biases.npy')
		net.get_layer('conv2d_19').set_weights([val1, val2])
		return net

def train():
	train_dict = get_file_from_pickle("train_dictionary.pkl")
	input_id_list = [x for x,_ in train_dict.items()]
	output_id_list = [x for _, x in train_dict.items()]
	input_id_list, output_id_list = decrease_train_input(input_id_list, output_id_list)
	train_generator = DataGenerator(input_id_list, output_id_list, train_dict)
	print("Number of input files are: {}".format(len(input_id_list)))
#	val_dict = get_file_from_pickle("val_dictionary.pkl")
#	input_id_list = [x for x,_ in val_dict.items()]
#	output_id_list = [x for _, x in val_dict.items()]
#	val_generator = DataGenerator(input_id_list, output_id_list, val_dict)
	net = model()
	net.load_weights('./result_dir/weights.020.hdf5')
	sgd = SGD(lr = 0.003, nesterov = True)
	net.compile(optimizer = sgd, loss = custom_loss, metrics = ['accuracy'])
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
	callbacks_list = [checkpoint]
	net.fit_generator(generator = train_generator,
			epochs = 20,
			callbacks = callbacks_list,
			verbose = 1)

rows = {0: (0, 512), 1:(512, 1024), 2:(912, 1424)}
cols = {0: (0, 512), 1:(512, 1024), 2:(1024, 1536), 3: (1536, 2048), 4:(1616, 2128)}

def custom_test(filename):
	mod = model()
#	mod = set_ckpt_weights(mod)
	mod.load_weights("./result_dir/weights.020.hdf5")
	image = pre_process(filename)
	print("image shape:{}".format(image.shape))
	image_outs = np.empty(0)
	for i, r in rows.items():
		temp = np.empty(0)
		for j, c in cols.items():
			img = image[:,r[0]:r[1], c[0]:c[1], :]
			out_img = mod.predict(img)
			out_img = np.squeeze(out_img, axis=0)
			out_img = np.minimum(np.maximum(out_img, 0), 1)
			out_img = out_img * 255
			if(j == 0):
				temp = out_img
			elif(j == 4):
				temp = np.concatenate([temp, out_img[:,864:,:]], axis = 1)
			else:
				temp = np.concatenate([temp, out_img], axis = 1)
		if(i == 0):
			image_outs = temp
		elif(i == 1):
			image_outs = np.concatenate([image_outs, temp], axis = 0)
		else:
			image_out = np.concatenate([image_outs, temp[216:,:,]], axis = 0)
	print("image_out fin shape: {}".format(image_out.shape))
	img = scipy.misc.toimage(image_outs, high=255, low=0, cmin=0, cmax=255)
	print(filename.split('/')[3][:5]+'_fin_1')
	img.save('./result_dir/'+filename.split('/')[3][:5]+'_fin_1.png')

def create_dict_files():
	train_dict, test_dict, val_dict = create_files("Sony_train_list.txt", "Sony_test_list.txt", "Sony_val_list.txt")
	dump_dictionary(train_dict, "train_dictionary")
	dump_dictionary(test_dict, "test_dictionary")
	dump_dictionary(val_dict, "val_dictionary")

if __name__ == "__main__":
#	train()
	custom_test('./Sony/short/00002_00_0.1s.ARW')
