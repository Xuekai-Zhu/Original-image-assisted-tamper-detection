#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image,image2, mask):
        image = (image - self.mean)/self.std
        image2 = (image2 - self.mean)/self.std
        mask /= 255
        return image,image2

class RandomCrop(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :]

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:]
        else:
            return image

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        #mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image

class ToTensor(object):
    def __call__(self, image,image2, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image2 = torch.from_numpy(image2)
        image2 = image2.permute(2, 0, 1)
        #mask  = torch.from_numpy(mask)
        return image,image2


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name  = self.samples[idx]
        print('----------------------', self.cfg.datapath)
        print(name)
        image = cv2.imread(self.cfg.datapath+'/trainA/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        image2 = cv2.imread(self.cfg.datapath+'/trainB/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        shape = cv2.imread(self.cfg.datapath+'/trainA/'+name+'.jpg',0).shape
        #image = image+gasuss_noise(image)
        #mask  = cv2.imread(self.cfg.datapath+'/train_mask/' +name+'.png', 0).astype(np.float32)

        mask=1
        if self.cfg.mode=='train':
            image, mask = (image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            return image, mask
        else:
            image,image2= self.normalize(image,image2, mask)
            #image = self.resize(image, mask)
            image,image2= self.totensor(image,image2, mask)
            return image,image2, shape, name

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image= [list(item) for item in zip(*batch)]
            #image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            #mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        image2 = torch.from_numpy(np.stack(image2, axis=0)).permute(0,3,1,2)
        #mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image,image2

    def __len__(self):
        return len(self.samples)


########################### Testing Script ###########################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='../data/DUTS')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image       = image*cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
