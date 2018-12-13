from math import *
from preprocess import *
from model import *
import numpy as np, scipy, rawpy, cv2, imageio
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import pickle

def display_variance(filename):
    data = get_file_from_pickle(filename)
    print(data)

def calculate_psnr(actual, output):
    mse = np.mean( (actual - output[:2848,:,:]) ** 2)
    if mse == 0:
        return 100
    R = 255.0
    return 20*log10((R**2)/sqrt(mse))

def ssim(actual, output):
    return compare_ssim(actual, output, multichannel = True)

def variation_values(train_file, weights_file):
    net = model()
    net.load_weights(weights_file)
    #noise_vals = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '']
    values_dictionary = dict()
    values_count = dict()
    train_dict = get_file_from_pickle(train_file)
    input_id_list = [x for x,_ in train_dict.items()]
    output_id_list = [x for _, x in train_dict.items()]
    print(input_id_list[0])
    for i, file in zip([i for i in range(0, len(input_id_list))], input_id_list):
        print('working on file no. {} i.e. {} part of {}'.format(i, file, file.strip().split('_')[1]))
        img = pre_process(file)
        img = img[:,:512,:512,:]
        out = net.predict(img)
        out = np.squeeze(out, axis = 0)
        out = np.minimum(np.maximum(out, 0), 1)*255
        act = rawpy.imread(output_id_list[i])
        act = act.postprocess(use_camera_wb = True, half_size = False, no_auto_bright = True, output_bps = 16)
        act = np.expand_dims(np.float32(act / 65535.0), axis=0)
        act = act[:, :1024, :1024, :]
        loss = np_custom_loss(act, out)
        try:
            values_dictionary[file.strip().split('_')[1]] += loss
            values_count[file.strip().split('_')[1]] += 1
        except:
            values_dictionary[file.strip().split('_')[1]] = loss
            values_count[file.strip().split('_')[1]] = 1
    for val, cnt in values_count.items():
        if(cnt >= 10):
            values_dictionary[val] = values_dictionary[val]/cnt
    print(values_dictionary)
    dump_dictionary(values_dictionary, 'values_variance')
    print(values_dictionary)

def get_psnr_values_avg(actual_lists, output_lists):
    val = 0.0
    for a, b in zip(actual_lists, output_lists):
        a = rawpy.imread(a)
        b = cv2.imread(b)
        a = a.postprocess(use_camera_wb = True, half_size = False, no_auto_bright = True, output_bps = 16)
        val += calculate_psnr(a, b)
    return val/(len(actual_lists))

def get_ssim_values(actual_lists, output_list):
    val = 0.0
    for a, b in zip(actual_lists, output_list):
        a = rawpy.imread(a)
        a = a.postprocess(use_camera_wb = True, half_size = False, no_auto_bright = True, output_bps = 8)
        imageio.imsave("temp.png", a)
        a = cv2.imread("temp.png")
        b = cv2.imread(b)
        val += ssim(a, b)
    return (val/(len(actual_lists)))*100.0

def check_post_processed_images(image):
	rawData = open(image, 'rb').read()
	imgSize = (2856, 4256)
	img = Image.frombytes('L', imgSize, rawData, 'raw', 'F;16')
	img.save("Post_processed_image.png")

if __name__ == "__main__":
#    variation_values("val_dictionary.pkl", "./result_dir/weights.020_2.hdf5")
#    display_variance('values_variance.pkl')
    #check_post_processed_images("./Sony/long/00113_00_30s.ARW")
    #exit()
    object_focused_images = ['./'+'00224_fin.png', './'+'00202_fin.png', './'+'00204_fin.png', './'+'00205_fin.png', './'+'00189_fin.png']
    object_trad_images = ['./'+'00224v1_trad.png', './'+'00202v1_trad.png', './'+'00204v1_trad.png', './'+'00205v1_trad.png', './'+'00189v1_trad.png']
    actual_outside_images = ['./Sony/long/'+'00113_00_30s.ARW', './Sony/long/'+'00114_00_30s.ARW', './Sony/long/'+'00119_00_30s.ARW', './Sony/long/'+'00121_00_30s.ARW', './Sony/long/'+'00122_00_30s.ARW']
    outside_predicted_images = ['./'+'00113_fin.png', './'+'00114_fin.png', './'+'00119_fin.png', './'+'00121_fin.png', './'+'00122_fin.png']
    actual_object_images = ['./Sony/long/'+'00204_00_10s.ARW', './Sony/long/'+'00202_00_10s.ARW', './Sony/long/'+'00204_00_10s.ARW', './Sony/long/'+'00205_00_10s.ARW', './Sony/long/'+'00189_00_10s.ARW']
    outside_trad_images = ['./'+'00113v1_trad.png', './'+'00114v1_trad.png', './'+'00119v1_trad.png', './'+'00121v1_trad.png', './'+'00122v1_trad.png']
    #print("\nOutput files psnr with ground truth of object images: {}".format(get_psnr_values_avg(actual_object_images, object_focused_images)
    #print("\nTraditional files psnr with ground truth of object images: {}".format(get_psnr_values_avg(actual_object_images, object_trad_images)))
    #print("\nOutput files psnr with ground truth of outside images: {}".format(get_psnr_values_avg(actual_outside_images, outside_predicted_images)))
    #print("\nTraditional files psnr with ground truth of outside images: {}".format(get_psnr_values_avg(actual_outside_images, outside_trad_images)))
    #print("\nOutput files ssim with ground truth of object images: {}".format(get_ssim_values(actual_object_images, object_focused_images)))
    #print("\nTraditional files ssim with ground truth of object images: {}".format(get_ssim_values(actual_object_images, object_trad_images)))
    #print("\nOutput files ssim with ground truth of outside images: {}".format(get_ssim_values(actual_outside_images, outside_predicted_images)))
   #print("\nTraditional files ssim with ground truth of outside images: {}".format(get_ssim_values(actual_outside_images, outside_trad_images)))
    """
    psnr_object_og = get_psnr_values_avg(actual_object_images, object_focused_images)
    psnr_object_tr   = get_psnr_values_avg(actual_object_images, object_trad_images)
    psnr_out_og      =  get_psnr_values_avg(actual_outside_images, outside_predicted_images)
    psnr_out_tr	      = get_psnr_values_avg(actual_outside_images, outside_trad_images)
    ssim_object_og	= get_ssim_values(actual_object_images, object_focused_images)
    ssim_object_tr	= get_ssim_values(actual_object_images, object_trad_images)
    ssim_out_og	= get_ssim_values(actual_outside_images, outside_predicted_images)
    ssim_out_tr		= get_ssim_values(actual_outside_images, outside_trad_images)
    ssim = dict(zip(["og", "tr"], [ [ssim_object_og, ssim_out_og], [ssim_object_tr, ssim_out_tr]]))
    with open("ssim_og_tr.p", 'wb') as p:
    	pickle.dump(ssim, p)
    """
    with open("ssim_og_tr_1.pkl", "rb") as p:
    	ssim = pickle.load(p)
    plt.ylabel("SSIM")
    plt.xticks([0.19, 1.19], ("Object Focused", "Outdoor"))
    #plt.set_title("Structural Similarity Index")
    bar1 = plt.bar(np.arange(2) + 0.2, ssim["og"], 0.2, align = 'center', color ="r", label = "Model")
    bar2 = plt.bar(range(2), ssim["tr"], 0.2, align = 'center', color = "b", label = "Traditional")
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()
    plt.show()
