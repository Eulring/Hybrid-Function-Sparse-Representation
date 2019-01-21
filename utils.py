import numpy as np
import os
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from PIL import Image

from config import opt



##################################################################################
#
#  image convert operation
#
##################################################################################
def rgb2hsv(image):
    return image.convert('HSV')

def rgb2ycbcr(image):
    return image.convert('YCbCr')

def imgcut(img, batch_size = (6,6)):
    w, h = img.shape
    w_ = w - (w - batch_size[0]) % (batch_size[0] - int(batch_size[0]/3))
    h_ = h - (h - batch_size[0]) % (batch_size[0] - int(batch_size[0]/3))
    return img[0:w_, 0:h_]
##################################################################################
#
#  image read/write operation
#
##################################################################################
def img_route(scale = "2", img_id = "001", method = "LR"):
    img_route = os.path.join(opt.dataset_route, "image_SRF_"+str(scale))
    img_name = '/img_' + img_id + '_SRF_' + scale + '_' + method + '.png'
    print(img_route + img_name)
    return img_route + img_name



#=================================================================================
#  turn the img_route to the numpy array of y channel directly 
def numpy_from_img_route(img_route):
    img_rgb = Image.open(img_route)
    img_ycbcr = rgb2ycbcr(img_rgb)
    img1 = np.array(img_ycbcr)
    y = img1[:, :, 0]
    return y











##################################################################################
#
#  Evalution
#
##################################################################################




#=================================================================================
# PSNR
# refer : https://github.com/aizvorski/video-quality/blob/master/psnr.py
def PSNR1(img1, img2):
    import math
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def PSNR(img1, img2):
    from skimage.measure import compare_psnr
    return compare_psnr(img1, img2, 255)

def PSNR2(img1, img2):
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    psnr = 10.0 * np.log10(255.0 * 255 / mse)
    return psnr

#=================================================================================
# SSIM
# refer : https://github.com/aizvorski/video-quality/blob/master/ssim.py
def SSIM(img1, img2):
    import skimage
    return skimage.measure.compare_ssim(img1, img2, data_range=255)
'''
    return compare_ssim(
        img1, img2,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        data_range=1.0,
        K1=0.01,
        K2=0.03,
        sigma=1.5)
'''