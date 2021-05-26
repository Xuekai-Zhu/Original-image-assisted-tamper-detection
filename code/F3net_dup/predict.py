#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pdataset as dataset
from net  import F3Net


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load('../user_data/model_data/F3net-model-67-2', map_location=device))
    def show(self):
        with torch.no_grad():
            for image, image2,mask, shape, name in self.loader:
                image,image2, mask = image.cuda().float(),image2.cuda().float(), mask.cuda().float()

                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image)
                out = out2u

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()
    
    def save(self):
        '''
        with torch.no_grad():
            for image,image2, shape, name in self.loader:
                image = image
                print(image.shape)
                print(shape)

                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image.cuda().float(),image2.cuda().float(), shape)
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()


                head  = '../predict/'+ self.cfg.datapath.split('/')[-1]+'/predict'
                cv2.imwrite(head+'/'+name[0]+'.png', pred)
                print(head)
                if not os.path.exists(head):
                    os.makedirs(head)
                pred[pred>225*0.1]=255
		
                kernel = np.ones((10,10),np.uint8)
                pred[pred<255]=0
                if(pred.sum()/255<1000):
                    pred[pred<=255]=0

                #pred = cv2.erode(pred,kernel,1)

                #pred = cv2.dilate(pred,kernel,2)
                #cv2.imwrite(mmpath+str(i),groundtruth)
                #cv2.imwrite(head+'/'+name[0]+'.png', pred)
        '''
        with torch.no_grad():
            for image,image2, shape, name in self.loader:
                image,image2 = image.cuda().float(),image2.cuda().float()
                size=352

                if((image.shape[2]<size) or (image.shape[3]<size)):
                                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image,image2)
                                print(image.shape)
                                
                                out   = out2u
                                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                                head  = './predict/images/'+ self.cfg.datapath.split('/')[-1]
                                pred[pred>225*0.8]=255
                                if(pred.sum()/255<1000):
                                                pred[pred<=255]=0

                                pred[pred<255]=0
                                cv2.imwrite('../../prediction_result/images'+'/'+name[0]+'.png', pred)
                                continue

                p=np.zeros((image[0][0].shape[0],image[0][0].shape[1]))
                for j in range(p.shape[0]//size):
                                for i in range(p.shape[1]//size):

                                                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image[:,:,j*size:(j+1)*size,i*size:(i+1)*size],image2[:,:,j*size:(j+1)*size,i*size:(i+1)*size] )
                                                out   = out2u
                                                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                                                p[j*size:(j+1)*size,i*size:(i+1)*size]+=pred
                for i in range(p.shape[0]//size):
                                                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image[:,:,i*size:(i+1)*size,-size:],image2[:,:,i*size:(i+1)*size,-size:])
                                                out   = out2u
                                                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                                                p[i*size:(i+1)*size,-size:]+=pred
                for i in range(p.shape[1]//size):
                                                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image[:,:,-size:,i*size:(i+1)*size],image2[:,:,-size:,i*size:(i+1)*size])
                                                out   = out2u
                                                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                                                p[-size:,i*size:(i+1)*size]+=pred
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image[:,:,-size:,-size:],image2[:,:,-size:,-size:])
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                p[-size:,-size:]+=pred
                #p[-size:,-size]=p[-size:,-size]/2

                #p[-size:-(p.shape[0]%size),:-size]=p[-size:-(p.shape[0]%size),:-size]/2
                #p[:-size,-size:-(p.shape[1]%size)]=p[:-size,-size:-(p.shape[1]%size)]/2
                #p[-size:-(p.shape[0]%size),-size:-(p.shape[1]%size)]=p[-size:-(p.shape[0]%size),-size:-(p.shape[1]%size)]*2/3
                #head  = '../predict/'+ self.cfg.datapath.split('/')[-1]+'/predict'
                #if not os.path.exists(head):
                #    os.makedirs(head)
                #pred=p
                #out1u, out2u, out2r, out3r, out4r, out5r = self.net(image)

                                
                #out   = out2u
                #pred2  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                #pred=pred+pred2

                pred[pred>225*0.8]=255
		
                kernel = np.ones((10,10),np.uint8)
                pred[pred<255]=0

                #pred = cv2.erode(pred,kernel,3)
                #cv2.imwrite(mmpath+str(i),groundtruth)
                cv2.imwrite('../prediction_result/images'+'/'+name[0]+'.png', pred)


if __name__=='__main__':
    for path in ['../user_data/test_data/match']:
        t = Test(dataset, F3Net, path)
        t.save()
        # t.show()
