import sys
sys.path.append("..")

import numpy as np
from utils import *


#=================================================================================
#  dictionary with a LR and corresponding HR
#=================================================================================
class Dictionary():
    def __init__(self, psize=6, dtype=1, upscale=2, rdtype=11):
        self.upscale = upscale
        self.psize = psize
        print(dtype)
        # AHF functions
        if dtype == 1: self.dic1()
        if dtype == 3: self.dic3()
        if dtype == 4: self.dic4()
        if dtype == 5: self.dic5()
        if dtype == 6: self.dic6()
        if dtype == 11: self.dic11()
        self.H = upscale_dic(self.L, psize=self.psize, upscale=self.upscale)
        self.size = len(self.L['params'])
        print('size of the dic is :', self.size)
        self.mask = get_mask(self.L)
        

    
    def dic11(self):
        ths1 = np.pi * np.arange(6) / 6
        bs1 = np.linspace(0, 6, 7)
        alphas1 = [2.5, 2, 2.25]
        L1 = generate_dic_Sin(ths1, bs1, alphas1, psize=self.psize)
        
        ths = 2 * np.pi * np.arange(16) / 16
        bs = np.linspace(-6, 6, 12)
        L2 = generate_dic_AHF(ths, bs, [1e-4], psize=self.psize, comb=True)
        
        
        
        aa4 = np.array([0, 1, 2, 3, 4, 5])
        bb4 = np.linspace(0, 5, 6)
        L4 = generate_dic_DCT(aa4, bb4, psize=self.psize)
        
        self.L = merge_dic(L4, L1)
        # self.L = merge_dic(self.L, L2)


    def dic1(self):
        
        aa0 = np.linspace(0, 5, 6)
        bb0 = np.linspace(0, 5, 6)
        L0 = generate_dic_DCT(aa0, bb0, psize=self.psize)
        
        ths = 2 * np.pi * np.arange(16) / 16
        bs = np.linspace(-6, 6, 12)
        L1 = generate_dic_AHF(ths, bs, [0.1, 1e-4], psize=self.psize, comb=True)
        
        #self.L = merge_dic(L1, L0)
        self.L = L0
        
    def dic3(self):
        ths1 = np.pi * np.arange(6) / 6
        bs1 = np.linspace(0, 6, 7)
        alphas1 = [2.5, 2, 2.25]
        L1 = generate_dic_Sin(ths1, bs1, alphas1, psize=self.psize)
        
        aa4 = np.array([0, 1, 2, 3, 4, 5])
        bb4 = np.linspace(0, 5, 6)
        L4 = generate_dic_DCT(aa4, bb4, psize=self.psize)
        self.L = merge_dic(L1, L4)
        
        
    def dic4(self):
        aa0 = np.linspace(0, 5, 6)
        bb0 = np.linspace(0, 5, 6)
        L0 = generate_dic_DCT(aa0, bb0, psize=self.psize)
        
        ths = 2 * np.pi * np.arange(16) / 16
        bs = np.linspace(-6, 6, 12)
        L1 = generate_dic_AHF(ths, bs, [0.1, 1e-4], psize=self.psize, comb=True)
        
        self.L = merge_dic(L1, L0)
        # self.L = L0
    def dic5(self):
        ths = 2 * np.pi * np.arange(16) / 16
        bs = np.linspace(-6, 6, 12)
        L1 = generate_dic_AHF(ths, bs, [0.1, 1e-4], psize=self.psize, comb=True)
        
        
        self.L = L1
        
    def dic6(self):
        
        ths1 = np.pi * np.arange(6) / 6
        bs1 = np.linspace(0, 6, 7)
        alphas1 = [2.5, 2, 2.25]
        L1 = generate_dic_Sin(ths1, bs1, alphas1, psize=self.psize)
        
        ths = 2 * np.pi * np.arange(16) / 16
        bs = np.linspace(-6, 6, 12)
        L2 = generate_dic_AHF(ths, bs, [0.1, 1e-4], psize=self.psize, comb=True)
        
        
        
        aa4 = np.array([0, 1, 2, 3, 4, 5])
        bb4 = np.linspace(0, 5, 6)
        L4 = generate_dic_DCT(aa4, bb4, psize=self.psize)
        
        self.L = merge_dic(L1, L4)
        self.L = merge_dic(self.L, L2)
        # self.L = L4

   #=================================================================================
#  approximate functions
#=================================================================================


def tanh2d(x, y, theta, b, xi):
    xx = x * np.cos(theta) + y * np.sin(theta) + b
    return 0.5 + np.arctan(xx / xi)/np.pi

def sin2d(x, y, theta, b, a):
    xx = x * np.cos(theta) + y * np.sin(theta) + b
    return 0.5 + np.sin(a * xx)


def cos2d(x, y, a, b, psize = 6):
    cx = np.cos(np.pi * a * (x + 0.5) / psize)
    cy = np.cos(np.pi * b * (y + 0.5) / psize)
    return cx * cy

################################################################
# dic:   { 'data', 'params' }
#     data: 36 x 100 numpy array
#     params: [list of N Thetas]
#         params[i]: contain three parameter: th, b, xi 
################################################################
# generate the AHF function dictionary
################################################################
def generate_dic_AHF(ths, bs, xis, psize = 6, comb = True, fit = True):
# ths : list of theta
# bs  : list of b
# xis : list of xi
# batch_size : default is 6 x 6
# comb :    True: dic will be all the combination among ths/bs/xis
    w, h = psize, psize
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
                    
            mean = np.mean(img)
            norm = np.sqrt(np.sum((img - mean)**2))
            if norm < 1.0: continue
                    
            data.append(img)
            names.append('tanh')
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
                    
                    mean = np.mean(img)
                    norm = np.sqrt(np.sum((img - mean)**2))
                    if norm < 1.0: continue
                    
                    data.append(img)
                    names.append('tanh')
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

    

def generate_dic_Sin(ths, bs, alphas, psize = 6, normalize = False):
    dic = {}
    data = []
    params = []
    names = []
    
    for th in ths:
        for b in bs:
            for alpha in alphas:
                img = np.zeros((psize, psize))
                
                for x in range(psize):
                    for y in range(psize):
                        img[x, y] = sin2d(x, y, th, b, alpha) 
                
                    mean = np.mean(img)
                    norm = np.sqrt(np.sum((img - mean)**2))
                    if norm < 1.0: continue
                
                data.append(img)
                names.append('sin')
                Theta = []
                Theta.append(th)
                Theta.append(b)
                Theta.append(alpha)
                params.append(Theta)
    
                
    N = len(data)
    data = np.array(data).reshape(N, -1)
    dic['data'] = data
    dic['params'] = params
    dic['names'] = names
    
    return dic


def generate_dic_DCT(aa, bb, psize = 6, normalize = False):
    dic = {}
    data = []
    params = []
    names = []
    
    for a in aa:
        for b in bb:
            img = np.zeros((psize, psize))
            for x in range(psize):
                for y in range(psize):
                    img[x, y] = cos2d(x, y, a, b)
            data.append(img)
            names.append('dct')
            Theta = []
            Theta.append(a)
            Theta.append(b)
            params.append(Theta)    
                
    N = len(data)
    data = np.array(data).reshape(N, -1)
    dic['data'] = data
    dic['params'] = params
    dic['names'] = names
    return dic

#=================================================================================
#  upscale the dictionary
#=================================================================================
def upscale_dic(dic_L, psize=6, upscale=2):
    w, h = psize, psize
    N = len(dic_L['names'])
    dic_H = {}
    data = []
    params = []
    names = []
    
    uw = int(w*upscale)
    uh = int(h*upscale)
    
    for i in range(N):
        
        name = dic_L['names'][i]
        Theta = dic_L['params'][i]
        patch = np.zeros((uw, uh))
        
        
        if name == 'tanh':
            for w_ in range(uw):
                for h_ in range(uh):
                    patch[w_, h_] = tanh2d(w_/upscale, h_/upscale, Theta[0], Theta[1], Theta[2])
        elif name == 'sin':
            for w_ in range(uw):
                for h_ in range(uh):
                    patch[w_, h_] = sin2d(w_/upscale, h_/upscale, Theta[0], Theta[1], Theta[2])
        
        elif name == 'dct':
            for w_ in range(uw):
                for h_ in range(uh):
                    patch[w_, h_] = cos2d(w_/upscale, h_/upscale, Theta[0], Theta[1])

        
        params.append(Theta)
        names.append(name)
        data.append(patch)
                 
    
    data = np.array(data).reshape(N, -1)
    dic_H['data'] = data
    dic_H['params'] = params
    dic_H['names'] = names
    
    return dic_H





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


def get_mask(dic):
    N = len(dic['names'])
    mask = np.zeros((N, 1))
    for i in range(N):
        name = dic['names'][i]
        Theta = dic['params'][i]
          
        if name == 'sin': mask[i] = 1
        if name == 'tanh' and Theta[2] <= 1e-3: mask[i] = 1
                    
    return mask

def filter_dic(dic, DH):
    from scipy.ndimage import gaussian_filter
    
    N = len(dic['names'])
    psize = 12
    mask = np.zeros((N, 1))
    for i in range(N):
        name = dic['names'][i]
        Theta = dic['params'][i]
        flag = 0
        if name == 'sin': flag = 1
        if name == 'tanh' and Theta[2] <= 1e-3: flag = 1
            
        if flag == 1:
            # ipdb.set_trace()
            img = DH[:, i].reshape(psize, psize)
            DH[:, i] = gaussian_filter(img, sigma=1).reshape(-1)
            
    return DH
            
            

