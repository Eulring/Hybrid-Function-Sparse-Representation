import numpy as np
import os
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from PIL import Image


##################################################################################
#
#  simple visualization
#
##################################################################################

def vis_img(img, gray = True, v_max = 1.0):
    if np.max(img) > 10: v_max = 255.0
    if gray == True:
        plt.rcParams['image.cmap'] = 'gray'
    plt.imshow(img, vmin = 0, vmax = v_max)
    plt.show()
    

def vis_dic(dic):
    for k, v in dic.items():
        print(k)
     
    
def vis_imgs(imgs, gray = True, v_max = 1.0, num_row = 6):
    
    N = len(imgs)
    plt.rcParams['figure.figsize'] = (45, 15)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    num_col = int(N / num_row) + 1
    
    for i in range(N):
        plt.subplot(num_col, num_row, i+1)
        
        plt.imshow(imgs[i][0], vmin=0, vmax=v_max)
        plt.title(imgs[i][1])
        
    plt.show()
    
def vis_simgs(imgs, size=(6, 6), v_max = 1.0):
    plt.rcParams['figure.figsize'] = (size[0], size[1]) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
        
    N = len(imgs)
    
    for i in range(N):
        plt.subplot(1, 10, i+1)
        img = imgs[i]
        plt.imshow(img, vmin = 0, vmax = v_max)
        plt.axis('off')
        
    plt.show()
        
##################################################################################
#
#  batches visualization
#
##################################################################################


#=================================================================================
# visualize the patches
#=================================================================================
def vis_batches(dicts, size=(6, 6), H=12, batch_type='array', save='None'):
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
    fig = plt.figure()
    fig.tight_layout()
    
    if save != 'None':
        plt.savefig(save, format='eps', dpi=1000)



