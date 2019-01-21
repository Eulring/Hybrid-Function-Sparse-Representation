import sys
sys.path.append("..")

import numpy as np
from utils import *

#=================================================================================
#  approximate functions
#=================================================================================
def tanh1d(x, b, xi):
    xx = x + b
    return 0.5 + np.arctan(xx / xi)/np.pi 

# function of the Heav2d function
def tanh2d(x, y, theta, b, xi):
    xx = x * np.cos(theta) + y * np.sin(theta) + b
    return 0.5 + np.arctan(xx / xi)/np.pi

def gauss2d(x, y, mu, cov):
    return multivariate_normal.pdf([x, y], mu, cov)





################################################################
# dic:   { 'data', 'params' }
#     data: 36 x 100 numpy array
#     params: [list of N Thetas]
#         params[i]: contain three parameter: th, b, xi 
################################################################
# generate the AHF function dictionary
################################################################
def generate_dic_AHF(ths, bs, xis, batch_size = (6, 6), comb = False, fit = True):
# ths : list of theta
# bs  : list of b
# xis : list of xi
# batch_size : default is 6 x 6
# comb :    True: dic will be all the combination among ths/bs/xis
    w, h = batch_size
    dic = {}
    data = []
    params = []
    names = []
    
    if comb == False:
        assert len(ths) == len(bs) == len(xis)
        N = len(ths)
        for i in range(N):
            img = np.zeros((w, h))
            for x in range(w):
                for y in range(h):
                    img[x, y] = tanh2d(x, y, th, b, xi)
                    
            if fit == True:
                dis = np.max(img) - np.min(img)
                if dis < 0.3:
                    continue
                    
            data.append(img)
            names.append('2Dtanh')
            Theta = []
            Theta.append(th)
            Theta.append(b)
            Theta.append(xi)
            params.append(Theta) 
            
    elif comb == True:
        for xi in xis:
            for th in ths:
                for b in bs:

                    img = np.zeros((w, h))

                    for i in range(w):
                        for j in range(h):
                            img[i, j] = tanh2d(i, j, th, b, xi)
                    
                    if fit == True:
                        dis = np.max(img) - np.min(img)
                        if dis < 0.3:
                            continue
                    
                    data.append(img)
                    names.append('2Dtanh')
                    Theta = []
                    Theta.append(th)
                    Theta.append(b)
                    Theta.append(xi)
                    params.append(Theta)
                    
    N = len(data)
    data = np.array(data).reshape(N, -1)
    dic['data'] = data
    dic['params'] = params
    dic['names'] = names
    
    return dic
    


################################################################
# merge two dictionaries into ones
################################################################
def merge_dic(D1, D2):
    dic = {}
    data1 = D1['data']
    data2 = D2['data']
    param1 = D1['params']
    param2 = D2['params']
    name1 = D1['names']
    name2 = D2['names']
    data = np.concatenate((data1, data2))
    
    N1 = len(param1)
    N2 = len(param2)
    
    
    
    params = []
    names = []
    for i in range(N1):
        params.append(param1[i])
        names.append(name1[i])
    for i in range(N2):
        params.append(param2[i])
        names.append(name2[i])
    
    dic['data'] = data
    dic['params'] = params
    dic['names'] = names
    
    return dic




#=================================================================================
#  upscale the dictionary
#=================================================================================
def upscale_dic(dic_L, batch_size=(6,6), upscale=2):
    w, h = batch_size
    N = len(dic_L['names'])
    dic_H = {}
    data = []
    params = []
    names = []
    
    for i in range(N):
        name = dic_L['names'][i]
        Theta = dic_L['params'][i]
        if name == '2Dtanh':
            # th b xi
            Theta[1] *= upscale
            img = np.zeros((w*upscale, h*upscale))
            for w_ in range(w*upscale):
                for h_ in range(h*upscale):
                    img[i, j] = tanh2d(w_, h_, Theta[0], Theta[1], Theta[2])
        
        params.append(Theta)
        names.append(name)
        data.append(data)
                 
    
    
    data = np.array(data).reshape(N, -1)
    dic['data'] = data
    dic['params'] = params
    dic['names'] = names
    
    return dic



