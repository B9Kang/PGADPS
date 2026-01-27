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

from networks.myModel import DPSMask,ADPSMask,PGADPSMask,fixedMask
from networks.model_utils import init_weights, load_weights

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
    parser.add_argument('-batch_size', default = 60, type = int, help = 'Number of images sampled per minibatch?')
    parser.add_argument('-init_weights', default = 'kaiming', help = "Choose from: 'normal', 'xavier', 'kaiming'")
    parser.add_argument('-epochs', default = 100, type = int, help = 'Maximum number of epochs?')
    parser.add_argument('-sampling', default = 'DPS', help = "Choose from: 'ADPS', 'DPS', 'PGADPS', 'Uniform', 'GSS', 'all'")
    
    parser.add_argument('-percentage_Ps',type =int,help='percentage of Prior sampling',default=80)  
    parser.add_argument('-percentage_As',type =int,help='percentage of Blocks of active sampling',default=20) 
    
    parser.add_argument('-temperature',type =float,help='temperature scaling for softmax in DPS and ADPS',default=2.0)
    parser.add_argument('-seed',type =str,help='which seed to use',default=0)
    
    parser.add_argument('-learning_rate',type =float,help='learning rate for network parameters',default=2e-4)
    parser.add_argument('-learning_rate_logits',type =float,help='learning rate for logit parameters for DPS and LOUPE',default=2e-3)

    ### Pretrained representation present?
    parser.add_argument('-network_saved_name', type = str, default = 'PGADPS_5bands_80Ps_20As_2seed')
    
    args = parse_args(parser)
    
    
    args.network_weights_path = 'savedmodels/'+args.network_saved_name+'.pth'

    GPU_NUM = args.gpu_index # GPU number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # put a limit on cpu resources used by pytorch
    torch.set_num_threads(8)
    torch.random.manual_seed(args.seed)

    perf = Metrics()
            
    tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])    

        
    testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = 'rad', hsi_mode = 'all', transforms = tx)
    
    #Pre-computed weights using median frequency balancing    
    weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]
    weights = torch.FloatTensor(weights)
    
    if args.sampling in ["Uniform", "GSS", "all","SLR"]:
        net = fixedMask(args)
    elif args.sampling == 'DPS':
        net = DPSMask(args)
    elif args.sampling == 'ADPS':
        net = ADPSMask(args)
    elif args.sampling == 'PGADPS':
        net = PGADPSMask(args)
    

    net.load_state_dict(torch.load(args.network_weights_path,map_location=device))
    net.eval()
    net.to(device)
        
    print('Completed loading pretrained network weights...')
    
    print('Calculating prediction accuracy...')
    
    labels_gt = []
    labels_pred = []
    
    for img_idx in range(len(testset)):
        _, hsi, label = testset[img_idx]
        label = label.numpy()
        
        label_pred = net(hsi.unsqueeze(0).to(device))
        label_pred = label_pred[:,:,:,:,-1].max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        
        label = label.flatten()
        label_pred = label_pred.flatten()

        labels_gt = np.append(labels_gt, label)
        labels_pred = np.append(labels_pred, label_pred)
    
    scores = perf(labels_gt, labels_pred)
    print('Statistics on Test set:\n')
    print('MPCA = {:.2f}%\nMean IOU is {:.2f}\nMean DICE score is {:.2f}'.format(scores[1]*100, scores[2]*100, scores[3]*100))
