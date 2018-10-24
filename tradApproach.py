import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from PIL import Image
def wb(channel, perc = 0.05):
	mi, ma = (np.percentile(channel, perc), np.percentile(channel, 100-perc))
	channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
	return channel

def white_balance(img):
	#result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	avg_a  = np.average(result[:, :, 1])
	avg_b  = np.average(result[:, :, 2])
	result[:, :, 1] = result[:, : , 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) *1.1)
	result[:, :, 2] = result[:, : , 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) *1.1)
	#result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
	return result

if __name__ == "__main__":
	filename = "example.arw"
#	processed_image = cv2.balanceWhite(filename)
#	plt.imshow(processed_image)
#	plt.show()
#	image = cv2.imread(filename, 1)
#	inWB  = np.dstack([wb(channel, 0.05) for channel in cv2.split(image)])
'''
	raw = rawpy.imread("example.arw")
	rgb = raw.postprocess()
	result = white_balance(rgb)
	img = Image.fromarray(result, 'RGB')
	img.show()
'''
	image = cv2.imread(filename, 0)

