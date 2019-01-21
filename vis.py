import numpy as np
import os
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from PIL import Image

from config import opt


##################################################################################
#
#  simple visualization
#
##################################################################################

def vis_img(img, gray = True, v_max = 1.0):
    if gray == True:
        plt.rcParams['image.cmap'] = 'gray'
    plt.imshow(img, vmin = 0, vmax = v_max)
    plt.show()

def vis_dic(dic):
    for k, v in dic.items():
        print(k)
        
        
        
        
##################################################################################
#
#  batches visualization
#
##################################################################################


#=================================================================================
# visualize the patches
#=================================================================================
def vis_batches(dicts, size=(6, 6), H=12, batch_type='array'):
    # dict: N x 36
    # size: 6 x 6 
    plt.rcParams['figure.figsize'] = (size[0], size[1]) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    if batch_type == 'list': dicts = np.array(dicts).reshape(-1, size[0]*size[1])
    
    N = dicts.shape[0]
    
    W = int(np.floor(N/H)) + 1
    
    for i in range(N):
        x = int(np.floor(i/H)) + 1
        y = np.mod(i, H) + 1
        
        plt.subplot(W, H, i+1)
        
        img = dicts[i].reshape(size)
        
        v_min = 0
        v_max = 1.0
        if np.mean(img) > 10:
            v_min *= 255
            v_max *= 255
        plt.imshow(img, vmin = v_min, vmax = v_max)
        plt.axis('off')

    plt.show()


