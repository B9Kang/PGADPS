"""
===================================================================================
    Source Name   : main.py
    Description   : Use this file to create and start a training procedure using 
                    one of the specified sampling strategies for MRI
===================================================================================
"""
# %% import dependencies
import torch
import torch.optim as optim
import argparse
import numpy as np

import loadData
import cudaDeviceChecker
import myModel
import discriminatorModel
import testModel
import trainModel
       
# %% go through the test and print the results
if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-use_gpu',type =bool,help='If true will try to use cuda, otherwise will do on cpu',default=True)
    parser.add_argument('-gpu_index',type =int,help='index of cuda device to use when use_gpu=True',default=0)
    
    parser.add_argument('-batch_size',type=int,help='Batch size to use',default=1)
    parser.add_argument('-Pineda',type=bool,help='Fetch the bigger dataset according to Pineda et al.',default=True)
    
    parser.add_argument('-sampling',type =str,help='choose from: "random_uniform", "low_pass", "VDS", "GMS", "LOUPE", "DPS", "ADPS" ,"PGADPS"', default='PGADPS')
    parser.add_argument('-temperature',type =float,help='temperature scaling for softmax in DPS and ADPS',default=2.0)
    parser.add_argument('-no_lines',type =int,help='Number of lines to sample',default=26)
    parser.add_argument('-seed',type =str,help='which seed to use',default=0)
    parser.add_argument('-percentage_DPS',type =int,help='percentage of DPS vs. ADPS',default=30)  
    parser.add_argument('-percentage_Block_ADPS',type =int,help='percentage of Blocks of ADPS',default=30)     
    
    parser.add_argument('-learning_rate',type =float,help='learning rate for network parameters',default=2e-4)
    parser.add_argument('-learning_rate_logits',type =float,help='learning rate for logit parameters for DPS and LOUPE',default=2e-3)
    parser.add_argument('-no_epochs',type =int,help='number of epochs to train for',default=5)
    
    parser.add_argument('-weight_mse',type =float,help='weight for mse loss',default=1)
    parser.add_argument('-weight_disc',type =float,help='weight for discriminator loss',default=5e-6)
    parser.add_argument('-weight_disc_features',type =float,help='weight for discriminator feature loss',default=1e-7)
    
    args = parser.parse_args()
    
    # setup correct name
    if args.sampling == 'PGADPS':
        args.save_name = args.sampling +"_"+str(args.no_lines)+"lines_"+str(args.percentage_DPS)+"Ps_"+str(args.percentage_Block_ADPS)+"As_"+str(args.seed)+"seed_Pineda"
    else:
        args.save_name = args.sampling +"_"+str(args.no_lines)+"lines_"+str(args.seed)+"seed_Pineda"
    
    # set up the gpu
    args.device = cudaDeviceChecker.device(args)

    # put a limit on cpu resources used by pytorch
    torch.set_num_threads(8)
    torch.random.manual_seed(args.seed)
    
    # load data
    dataloader_train,dataloader_val,dataloader_test = loadData.load(args)
    
    # %% create network depending on the type of sampling performed
    if args.sampling in ["random_uniform", "low_pass", "VDS", "GMS"]:
        Network = myModel.fixedMask(args)
    if args.sampling == "LOUPE":
        Network = myModel.LOUPEMask(args)
    if args.sampling == "DPS":
        Network = myModel.DPSMask(args)
    if args.sampling == "ADPS":
        Network = myModel.ADPSMask_legacy(args)
    if args.sampling == "PGADPS":
        Network = myModel.PGADPSMask_legacy(args)
        
    Network = Network.to(args.device)
    
    # discriminator network
    Discriminator = discriminatorModel.Discriminator(args)
    Discriminator = Discriminator.to(args.device)
    
    # optimizers
    if args.sampling in ["LOUPE","DPS"]:
        optimizer_recon = optim.Adam([
            {'params':Network.proximal.parameters(),'lr': args.learning_rate},
            {'params':Network.logit_parameter,'lr': args.learning_rate_logits},
            ],betas = (0.9,0.999), eps= 1e-7)
    elif args.sampling == 'PGADPS':
        other_params = [p for p in Network.parameters() if p is not Network.logit_parameter]
        optimizer_recon = optim.Adam([
            {'params': other_params, 'lr': args.learning_rate},
            {'params': [Network.logit_parameter], 'lr': args.learning_rate_logits},
            ], betas=(0.9, 0.999), eps=1e-7)
    else:
        optimizer_recon = optim.Adam(Network.parameters(),lr = args.learning_rate,betas = (0.9,0.999), eps= 1e-7)

    optimizer_disc  = optim.Adam(Discriminator.parameters(),lr = args.learning_rate,betas = (0.9,0.999), eps= 1e-7)    
    
    # %% train the network
    results = trainModel.execute(Network, Discriminator, optimizer_recon, optimizer_disc, args, dataloader_train,dataloader_val)

    # %% test network
    mse_mean,PSNR_mean,SSIM_mean = testModel.Test(Network,dataloader_test,args)
