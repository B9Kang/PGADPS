"""
===================================================================================
    Source Name   : main.py
    Description   : Use this file to create and start a training procedure using 
                    DPS or ADPS for an MNIST classification task
===================================================================================
"""
# %% import dependencies
import torch
import torch.optim as optim
import numpy as np
import argparse

import loadData
import cudaDeviceChecker
import myModel
import trainModel
       
# %% go through the test and print the results
if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-use_gpu',type =bool,help='If true will try to use cuda, otherwise will do on cpu',default=True)
    parser.add_argument('-gpu_index',type =int,help='index of cuda device to use when use_gpu=True',default=0)
    
    parser.add_argument('-batch_size',type=int,help='Batch size to use',default=128)
    
    parser.add_argument('-sampling',type =str,help='choose from: "DPS" or "ADPS" or "PGADPS"',default='PGADPS')
    parser.add_argument('-percentage',type =int,help='percentage of samples to use',default=1)
    parser.add_argument('-percentage_DPS',type =int,help='percentage of DPS vs. ADPS',default=60)  
    parser.add_argument('-percentage_Block_ADPS',type =int,help='percentage of Blocks of ADPS',default=20) 
    
    parser.add_argument('-temperature',type =float,help='temperature scaling for softmax in DPS and ADPS',default=2.0)
    parser.add_argument('-seed',type =str,help='which seed to use',default=0)
    
    parser.add_argument('-learning_rate',type =float,help='learning rate for network parameters',default=2e-4)
    parser.add_argument('-learning_rate_logits',type =float,help='learning rate for logit parameters for DPS and LOUPE',default=2e-3)
    parser.add_argument('-no_epochs',type =int,help='number of epochs to train for',default=50)
    
    
    args = parser.parse_args()
    for percentage in range(10):
        args.percentage = percentage*2+2
        args.no_pixels = int(32**2*args.percentage/100)
        
        acc_list=[]
        for seed in range(6):
            # setup correct name
            if args.sampling == 'PGADPS':
                args.save_name = args.sampling +"_"+str(args.percentage)+"percent_"+str(args.percentage_DPS)+"Ps_"+str(args.percentage_Block_ADPS)+"As_"+str(seed)+"seed"
            else:
                args.save_name = args.sampling +"_"+str(args.percentage)+"percent_"+str(seed)+"seed"
            
            # set up the gpu
            args.device = cudaDeviceChecker.device(args)

            # put a limit on cpu resources used by pytorch
            torch.set_num_threads(8)
            torch.random.manual_seed(seed)
            
            # load data
            train_loader,val_loader,test_loader = loadData.load(args)
            
            # %% create network depending on the type of sampling performed
            Network = myModel.Network(args)
            Network = Network.to(args.device)
        
            # optimizers
            if args.sampling == 'DPS':
                optimizer = optim.Adam([
                    {'params':Network.f1.parameters(),'lr': args.learning_rate},
                    {'params':Network.f2.parameters(),'lr': args.learning_rate},
                    {'params':Network.logit_parameter,'lr': args.learning_rate_logits},
                    ],betas = (0.9,0.999), eps= 1e-7)
            elif args.sampling == 'ADPS':
                optimizer = optim.Adam(Network.parameters(),lr = args.learning_rate,betas = (0.9,0.999), eps= 1e-7)
            else:
                optimizer = optim.Adam([
                    {'params':Network.f1.parameters(),'lr': args.learning_rate},
                    {'params':Network.f2.parameters(),'lr': args.learning_rate},
                    {'params':Network.g.parameters(),'lr': args.learning_rate},
                    {'params':Network.lstm.parameters(),'lr': args.learning_rate},
                    {'params':Network.logit_parameter,'lr': args.learning_rate_logits},
                    ],betas = (0.9,0.999), eps= 1e-7)
            
            # %% train the network
            results = trainModel.execute(Network,optimizer,train_loader,val_loader,args)
            # save_name_result = "results/"+args.save_name+"_result.npy"
            # np.save(save_name_result,results)

            # %% test network
            loss,acc = trainModel.test(Network,test_loader,args)
            print(f"\n\n\n Accuracy on the test set = {100*acc:>.3}%")
            acc_list.append(acc)
        save_name_result_acc = "results/"+args.save_name+"_test_acc_total_seeds.npy"
        np.save(save_name_result_acc,acc_list)