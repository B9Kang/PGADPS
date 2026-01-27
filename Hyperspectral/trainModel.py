"""
===================================================================================
    Source Name   : TrainModel.py
    Description   : This file specifies a training and evaluation loop for MRI
===================================================================================
"""
# %% import dependencies
import torch
import torch.nn.functional as F
import time
from helpers.utils import AeroCLoader, AverageMeter, Metrics, parse_args
import numpy as np

def train(net,trainloader,optimizer,args,epoch = 0):
    
    trainloss2 = AverageMeter()
   
    net.train()

    batch_loss = 0.0
    #Pre-computed weights using median frequency balancing    
    weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]
    weights = torch.FloatTensor(weights)

    for idx, (rgb_ip, hsi_ip, labels) in enumerate(trainloader, 0):
        N = hsi_ip.size(0)
        optimizer.zero_grad()
        
        outputs = net(hsi_ip.to(args.device))

        labels_all = labels.to(args.device).unsqueeze(3).repeat(1,1,1,outputs.size(4))
    
        loss = torch.nn.functional.cross_entropy(outputs, labels_all.to(args.device),weight=weights.to(args.device), ignore_index = 5)

        loss.backward()
        optimizer.step()
        
        batch_loss += loss.item()
        trainloss2.update(loss.item(), N)
        
    print('\nTrain Epoch %d, loss: %.3f' % (epoch + 1, batch_loss / (idx+1)))
            
    
def val(net,valloader,optimizer,args,epoch = 0):

    perf=Metrics()
    valloss2 = AverageMeter()
    truth = []
    pred = []
    net.eval()

    valloss_fx = 0.0
    #Pre-computed weights using median frequency balancing    
    weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]
    weights = torch.FloatTensor(weights)
    
    with torch.no_grad():
        for idx, (rgb_ip, hsi_ip, labels) in enumerate(valloader, 0):
            N = hsi_ip.size(0)
            
            outputs = net(hsi_ip.to(args.device))
            labels_all = labels.to(args.device).unsqueeze(3).repeat(1,1,1,outputs.size(4))
            loss = torch.nn.functional.cross_entropy(outputs, labels_all.to(args.device),weight=weights.to(args.device), ignore_index = 5)

            valloss_fx += loss.item()
            
            valloss2.update(loss.item(), N)
            
            truth = np.append(truth, labels.cpu().numpy())
            pred = np.append(pred, outputs[:,:,:,:,-1].max(1)[1].cpu().numpy())
                
    print('VAL: %d loss: %.3f' % (epoch + 1, valloss_fx / (idx+1)))
    
    return perf(truth, pred)