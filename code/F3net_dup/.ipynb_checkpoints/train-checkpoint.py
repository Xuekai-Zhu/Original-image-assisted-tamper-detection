#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net
from apex import amp

m = nn.ReplicationPad2d(2)
#ssim_loss = pytorch_ssim.SSIM(window_size=5,size_average=True)
def structure_loss(pred, mask):
    #weit  =(1+torch.abs(F.avg_pool2d(m(mask), kernel_size=5, stride=1)))
    #print(weit)
    #weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2))
    #wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    #wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)

    #ssim=1-ssim_loss(pred,mask)
    inter = ((pred*mask)).sum(dim=(2,3))
    union = ((pred+mask)).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wiou).mean()
def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='../user_data/train_data/match', savepath='./out', mode='train', batch=16, lr=0.06, momen=0.9, decay=5e-4, epoch=200)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True,num_workers=0)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #net.load_state_dict(torch.load('out/model-64', map_location=device))
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image,image2, mask) in enumerate(loader):
	#print(image.shape,mask.shape)
            image,image2, mask = image.cuda().float(),image2.cuda().float(),mask.cuda().float()
            out1u, out2u, out2r, out3r, out4r, out5r = net(image,image2)

            loss1u = structure_loss(out1u, mask)
            loss2u = structure_loss(out2u, mask)

            loss2r = structure_loss(out2r, mask)
            loss3r = structure_loss(out3r, mask)
            loss4r = structure_loss(out4r, mask)
            loss5r = structure_loss(out5r, mask)
            loss   = (loss1u+loss2u)/2+loss2r/2+loss3r/4+loss4r/8+loss5r/16

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1u':loss1u.item(), 'loss2u':loss2u.item(), 'loss2r':loss2r.item(), 'loss3r':loss3r.item(), 'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if epoch%3==0:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, F3Net)
