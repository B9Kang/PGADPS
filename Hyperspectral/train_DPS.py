#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:34:16 2019

@author: aneesh
"""

import os 
import os.path as osp

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim

from helpers.augmentations import RandomHorizontallyFlip, RandomVerticallyFlip, \
    RandomTranspose, Compose
from helpers.utils import AeroCLoader, AverageMeter, Metrics, parse_args
from helpers.lossfunctions import cross_entropy2d

from torchvision import transforms

from networks.resnet6 import ResnetGenerator
from networks.segnet import segnet, segnetm
from networks.myModel import DPSMask,ADPSMask,PGADPSMask,fixedMask
from networks.model_utils import init_weights
from trainModel import train,val

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')
    
    ### 0. Config file?
    parser.add_argument('-config-file', default = None, help = 'Path to configuration file')
    
    ### 1. Data Loading
    parser.add_argument('-bands', default = 5, help = 'Number of bands', type = int)
    parser.add_argument('-window_size', default = 64, help = 'Size of the input size (cropped window)', type = int)
    parser.add_argument('-use_augs', default = True, help = 'Use data augmentations?')
        
    ### b. ResNet config
    parser.add_argument('-resnet_blocks', default = 6, help = 'How many blocks if ResNet architecture?', type = int)
    ### Use GPU or not
    parser.add_argument('-gpu_index', default = 0, type = int, help='GPU Number')  # 문자열로 받기
    
    ### Hyperparameters
    parser.add_argument('-batch_size', default = 100, type = int, help = 'Number of images sampled per minibatch?')
    parser.add_argument('-epochs', default = 60, type = int, help = 'Maximum number of epochs?')
    parser.add_argument('-sampling', default = 'DPS', help = "Choose from: 'ADPS', 'DPS', 'PGADPS', 'Uniform', 'GSS','SLR', 'all'")

    
    parser.add_argument('-percentage_Ps',type =int,help='percentage of prior sampling',default=80)  
    parser.add_argument('-percentage_As',type =int,help='percentage of Blocks of Active sampling',default=10) 
    
    parser.add_argument('-temperature',type =float,help='temperature scaling for softmax in DPS and ADPS',default=2.0)
    parser.add_argument('-seed',type =str,help='which seed to use',default=0)
    parser.add_argument('-init_weights', default = 'kaiming', help = "Choose from: 'normal', 'xavier', 'kaiming'")
    parser.add_argument('-learning_rate',type =float,help='learning rate for network parameters',default=1e-4)
    parser.add_argument('-learning_rate_logits',type =float,help='learning rate for logit parameters for DPS and PA-DPS',default=1)


    args = parse_args(parser)
    
    if args.sampling == 'PGADPS':
        args.save_name = args.sampling +"_"+str(args.bands)+"bands_"+str(args.percentage_Ps)+"Ps_"+str(args.percentage_As)+"As_"+str(args.seed)+"seed"
    else:
        args.save_name = args.sampling +"_"+str(args.bands)+"bands_"+str(args.seed)+"seed"

    args.network_weights_path = 'savedmodels/'+args.save_name + '.pth'

    GPU_NUM = args.gpu_index # GPU number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # put a limit on cpu resources used by pytorch
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.use_augs:
        augs = []
        augs.append(RandomHorizontallyFlip(p = 0.5))
        augs.append(RandomVerticallyFlip(p = 0.5))
        augs.append(RandomTranspose(p = 1))
        augs_tx = Compose(augs)
    else:
        augs_tx = None
        
    tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])    

        
    trainset = AeroCLoader(set_loc = 'left', set_type = 'train', size = 'small', \
                        hsi_sign='rad', hsi_mode = 'all',transforms = tx, augs = augs_tx)
    valset = AeroCLoader(set_loc = 'mid', set_type = 'test', size = 'small', \
                        hsi_sign='rad', hsi_mode = 'all', transforms = tx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, shuffle = False)
            
    if args.sampling in ["Uniform", "GSS", "all","SLR"]:
        net = fixedMask(args)
    elif args.sampling == 'DPS':
        net = DPSMask(args)
    elif args.sampling == 'ADPS':
        net = ADPSMask(args)
    elif args.sampling == 'PGADPS':
        net = PGADPSMask(args)

    net.to(device)
    
    # optimizers
    if args.sampling == "DPS":
        optimizer = optim.Adam([
            {'params':net.SegNet.parameters(),'lr': args.learning_rate},
            {'params':net.logit_parameter,'lr': args.learning_rate_logits},
            ],betas = (0.9,0.999), eps= 1e-7)
    elif args.sampling == 'PGADPS':
        other_params = [p for p in net.parameters() if p is not net.logit_parameter]
        optimizer = optim.Adam([
            {'params': other_params, 'lr': args.learning_rate},
            {'params': [net.logit_parameter], 'lr': args.learning_rate_logits},
            ], betas=(0.9, 0.999), eps=1e-7)
    else:
        optimizer = optim.Adam(net.parameters(),lr = args.learning_rate,betas = (0.9,0.999), eps= 1e-7)
    
    bestmiou = 0
    
    for epoch in range(args.epochs):
        train(net,trainloader,optimizer,args,epoch)
        oa, mpca, mIOU, mDICE, _ = val(net,valloader,optimizer,args,epoch)
        print('MPCA = {:.3f}, mIOU = {:.3f}, mDICE  = {:.3f}'.format(mpca, mIOU,mDICE))
        if mIOU > bestmiou:
            bestmiou = mIOU
            val_model_mpca = mpca
            val_model_mDICE = mDICE
            torch.save(net.state_dict(), args.network_weights_path)
    print('###############Final model saved###############################')
    print('MPCA = {:.3f}, mIOU = {:.3f}, mDICE  = {:.3f}'.format(val_model_mpca, bestmiou,val_model_mDICE))
