"""
===================================================================================
    Source Name   : myModel.py
    Description   : Specification of the models used
===================================================================================
"""

# %% import dependencies
import torch
import torch.nn as nn
from networks.DPSLayerMemory import DPS
from networks.resnet6 import ResnetGenerator

from networks.fixedMasks import createMask
    
class fixedMask(nn.Module):
    def __init__(self,args):
           # %% initialization
           super(fixedMask,self).__init__()
           print(args.device)
           self.device = args.device
           self.no_classes = 6
           args.mux_in = 51
       
           # create the layers as a module list
           self.SegNet = ResnetGenerator(args.mux_in, self.no_classes, n_blocks=args.resnet_blocks)    
           self.mask = createMask(args)

        
    # %% forward call    
    def forward(self, x):
           # get the batch size
           batch_size, C, H, W = x.size()

           mask = self.mask
           y = x*mask.repeat(batch_size,1,1,1)
           # Segmentation netowrk
           output_image = self.SegNet(y)
                  
           return output_image.unsqueeze(4)
    
     
# %% DPS Network       
        
class DPSMask(nn.Module):
    def __init__(self,args):
           # %% initialization
           super(DPSMask,self).__init__()
           print(args.device)
           self.device = args.device
           self.no_classes = 6
           args.mux_in = 51
           
           self.logit_parameter = nn.Parameter(torch.randn(args.mux_in)/4.)
           self.mux_out = args.bands
       
           # create the layers as a module list
           self.SegNet = ResnetGenerator(args.mux_in, self.no_classes, n_blocks=args.resnet_blocks)
           self.DPS = DPS(args)

        
    # %% forward call    
    def forward(self, x):
           # get the batch size
           batch_size, C, H, W = x.size()
           #initialize the DPS memory
           self.DPS.initialize_sample_memory(x)
                      
           # generate logits and corresponding mask
           logits = self.logit_parameter.unsqueeze(0).repeat(batch_size,1)
           mask = self.DPS(logits,self.mux_out)
           mask = self.expandSampleMatrix(mask,x)
           
           y = mask*x
           # Segmentation netowrk
           output_image = self.SegNet(y)
                  
           return output_image.unsqueeze(4)

    def expandSampleMatrix(self,mask,input_image):
            B, C, H, W = input_image.shape  
            mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            return mask   

class ADPSMask(nn.Module):
    def __init__(self,args):
           # %% initialization
           super(ADPSMask,self).__init__()
           print(args.device)
           self.device = args.device
           self.no_classes = 6
           args.mux_in = 51
           
           self.no_iter = args.bands
           self.mux_out = 1

           # create the layers as a module list
           self.SegNet = nn.ModuleList([(
                  ResnetGenerator(args.mux_in, self.no_classes, n_blocks=args.resnet_blocks)
                  ) for i in range(self.no_iter)])
           
           self.lstm_size = 256
           self.SampleNet = nn.ModuleList([nn.Sequential(
                 nn.Conv2d(6, 32, 3, padding=1),
                     nn.BatchNorm2d(32),
                     nn.ReLU(),
                     nn.Conv2d(32, 64, 3, padding=1),
                     nn.BatchNorm2d(64),
                     nn.ReLU(),
                     nn.Conv2d(64, 128, 3, padding=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(),
                     nn.Conv2d(128, self.lstm_size, 3, padding=1),
                     nn.BatchNorm2d(self.lstm_size),
                     nn.ReLU(),
                     nn.AdaptiveAvgPool2d(1),
                     nn.Flatten()
                 )for i in range(self.no_iter)])
           self.lstm = nn.LSTM(self.lstm_size,self.lstm_size,num_layers=1)
           
           self.final_fc = nn.ModuleList([
                 nn.Linear(self.lstm_size,args.mux_in)
                 for i in range(self.no_iter)])
        
           self.DPS = DPS(args)

        
    # %% forward call    
    def forward(self, x):
           # get the batch size
           batch_size, C, H, W = x.size()
           #initialize the DPS memory
           self.DPS.initialize_sample_memory(x)
           
           # create the initial hidden states for the lstm
           h_var = torch.zeros(1,batch_size,self.lstm_size).to(self.device)
           c_var = torch.zeros(1,batch_size,self.lstm_size).to(self.device)
           lstm_in = torch.zeros(1,batch_size,self.lstm_size).to(self.device)
           
           # initalize all outputs for s_hat
           output_image_all = torch.zeros(batch_size,self.no_classes,H,W,self.no_iter).to(self.device)
           
           # iterate over the lstm
           for i in range(self.no_iter):
                  #ADPS generate logits from the hidden state
                  logits = self.final_fc[i](h_var.reshape(batch_size,self.lstm_size))

                  # generate mask from the logits
                  mask = self.DPS(logits,1)

                  mask = self.expandSampleMatrix(mask,x)
                  y = mask*x
                  
                  # encode'                  
                  output_image = self.SegNet[i](y)
                  output_image_all[:,:,:,:,i] = output_image

                  # Generate LSTM Feature for next (iteration) logits
                  Bands_chosen = mask[:,:,0,0]
                  output_conv = self.SampleNet[i](output_image)
              #     input_lstm = torch.cat((output_conv,Bands_chosen),dim=1)
                  _,(h_var,c_var) = self.lstm(output_conv.unsqueeze(0),(h_var,c_var))
                  
           return output_image_all

    def expandSampleMatrix(self,mask,input_image):
            B, C, H, W = input_image.shape  
            mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            return mask   
    
class PGADPSMask(nn.Module):
    def __init__(self,args):
           # %% initialization
           super(PGADPSMask,self).__init__()
           print(args.device)
           self.device = args.device
           self.no_classes = 6
           args.mux_in = 51
           
           self.logit_parameter = nn.Parameter(torch.randn(args.mux_in)/4.)
           self.mux_out_first = int(args.bands*args.percentage_Ps/100)
           self.mux_out = int(args.bands*args.percentage_As/100)
           left_bands = args.bands-self.mux_out_first
           # "left_bands" represents the total number of bands to be selected by ADPS after the first selection by prior-aware DPS.
                  
           # Minimum samples to select in active sampling is 1.
           # If the self.mux_out is calculated as 0 ore lower than 1, set it to 1.
           # "more_iter" indicates whether the one last iteration is needed to collect all target samples with less samples to select than mux_out.
           # If more_iter is True, the last iteration will select all remaining samples which are less samples than mux_out.
           if self.mux_out == 0:
                  self.mux_out = 1
                  self.no_iter = left_bands+1
                  self.more_iter = False 
           elif self.mux_out == 1:
                  self.no_iter = int(left_bands/self.mux_out)+1
                  self.more_iter = False      
           elif left_bands%self.mux_out != 0: # Handling left bands 
                  self.no_iter = int(left_bands/self.mux_out)+2
                  self.more_iter = True
                  self.mux_out_last = left_bands - (self.no_iter-2)*self.mux_out
           else:
                  self.no_iter = int(left_bands/self.mux_out)+1
                  self.more_iter = False

           # create the layers as a module list
           self.SegNet = nn.ModuleList([(
                  ResnetGenerator(args.mux_in, self.no_classes, n_blocks=args.resnet_blocks)
                  ) for i in range(self.no_iter)])
           
           self.lstm_size = 256
           self.SampleNet = nn.ModuleList([nn.Sequential(
                     nn.Conv2d(6, 32, 3, padding=1),
                     nn.BatchNorm2d(32),
                     nn.ReLU(),
                     nn.Conv2d(32, 64, 3, padding=1),
                     nn.BatchNorm2d(64),
                     nn.ReLU(),
                     nn.Conv2d(64, 128, 3, padding=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(),
                     nn.Conv2d(128, self.lstm_size, 3, padding=1),
                     nn.BatchNorm2d(self.lstm_size),
                     nn.ReLU(),
                     nn.AdaptiveAvgPool2d(1),
                     nn.Flatten()
                 )for i in range(self.no_iter)])
           
           self.lstm = nn.LSTM(self.lstm_size,self.lstm_size,num_layers=1)
           self.final_fc = nn.ModuleList([
                 nn.Linear(self.lstm_size,args.mux_in)
                 for i in range(self.no_iter)])
        
           self.DPS = DPS(args)
        
    # %% forward call    
    def forward(self, x):
           # get the batch size
           batch_size, C, H, W = x.size()
           #initialize the DPS memory
           self.DPS.initialize_sample_memory(x)
           
           # create the initial hidden states for the lstm
           h_var = torch.zeros(1,batch_size,self.lstm_size).to(self.device)
           c_var = torch.zeros(1,batch_size,self.lstm_size).to(self.device)
           lstm_in = torch.zeros(1,batch_size,self.lstm_size).to(self.device)
           
           # initalize all outputs for s_hat
           output_image_all = torch.zeros(batch_size,self.no_classes,H,W,self.no_iter).to(self.device)
           
           # iterate over the lstm
           for i in range(self.no_iter):
                  # generate logits and corresponding mask
                  # first selection (i=0) by prior-aware DPS (deterministic sampling), then active GROUP sampling
                  if i == 0:
                     logits = self.logit_parameter.unsqueeze(0).repeat(batch_size,1)
                     mask = self.DPS(logits,self.mux_out_first)
                  else:
                     logits = self.final_fc[i](h_var.reshape(batch_size,self.lstm_size))
                     if self.more_iter == True:
                            if i == (self.no_iter-1):
                                   mask = self.DPS(logits,self.mux_out_last)
                            else:
                                   mask = self.DPS(logits,self.mux_out)
                     else:
                            mask = self.DPS(logits,self.mux_out)

                  mask = self.expandSampleMatrix(mask,x)
                  y = mask*x
                  
                  # encode'                  
                  output_image = self.SegNet[i](y)
                  output_image_all[:,:,:,:,i] = output_image

                  # Generate LSTM Feature for next (iteration) logits
                  Bands_chosen = mask[:,:,0,0]
                  output_conv = self.SampleNet[i](output_image)
              #     input_lstm = torch.cat((output_conv,Bands_chosen),dim=1)
                  _,(h_var,c_var) = self.lstm(output_conv.unsqueeze(0),(h_var,c_var))
                  
           return output_image_all

    def expandSampleMatrix(self,mask,input_image):
            B, C, H, W = input_image.shape  
            mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            return mask   