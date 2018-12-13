
# coding: utf-8

# In[1]:


import rawpy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy, rawpy
from scipy import misc
from PIL import Image
# In[20]:


def white_balance(img_arr):
    result=cv2.cvtColor(img_arr,cv2.COLOR_BGR2LAB)
    avg_a=np.average(result[:,:,1])
    avg_b=np.average(result[:,:,2])
    result[:,:,1]=result[:,:,1]-((avg_a-128)*(result[:,:,0]/255.0)*1.1)
    result[:,:,2]=result[:,:,2]-((avg_b-128)*(result[:,:,0]/255.0)*1.1)
    result=cv2.cvtColor(result,cv2.COLOR_LAB2BGR)
    return result

def demosaicing(img_arr,imrows,imcols):
    nans = np.isnan(img_arr)
    x = lambda z: z.nonzero()[0]
    img_arr[nans]= np.interp(x(nans), x(~nans), img_arr[~nans])
    return img_arr

def denoising_sharpening(img_arr):
    dst= cv2.fastNlMeansDenoisingColored(img_arr, None,10,10,7,21)
    kernel=np.array([[-1,-1,-1,-1,-1],
                     [-1,2,2,2,-1],
                     [-1,2,8,2,-1],
                     [-2,2,2,2,-1],
                     [-1,-1,-1,-1,-1]])/8.0
    dst = cv2.filter2D(dst,-1,kernel)
    return dst

def gamma_correction(img_arr, gamma=1.0):
    invGamma=1.0/ gamma
    table=np.array([((i/255.0)**invGamma)*255
                   for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(img_arr,table)

def traditional_approach(filename):
    raw=rawpy.imread(filename)
    img_array=raw.postprocess()
    whiteb_result=white_balance(img_array)
    width,height,channels=whiteb_result.shape
    #fig=plt.figure(figsize=(150,150))
    #fig.add_subplot(20,20,1)
    #plt.imshow(img_array)
    #plt.imshow(whiteb_result)

    denoise_sharpen_result = demosaicing(whiteb_result,width,height)
    #plt.imshow(demosaic_result)

    #denoise_sharpen_result=denoising_sharpening(demosaic_result)
    #fig=plt.figure(figsize=(150,150))
    #fig.add_subplot(20,20,1)
    #plt.imshow(denoise_sharpen_result)

    gamma_correct=gamma_correction(denoise_sharpen_result)
    #fig=plt.figure(figsize=(150,150))
    #plt.imshow(gamma_correct)
    #plt.show()
    img = misc.toimage(gamma_correct, high=255, low=0, cmin=0, cmax=255)
    img.save('./Outputs/'+filename.split('/')[3][:5]+'v1_trad.png')

if __name__ == "__main__":
	traditional_approach('./00001_00_0.1s.ARW')
