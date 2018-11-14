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
