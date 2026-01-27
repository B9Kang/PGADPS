"""
===================================================================================
    Source Name   : MyModel.py
    Description   : Specification of the models used for MNIST
===================================================================================
"""

# %% import dependancies
import torch
import torch.nn as nn

import DPSLayerMemory

# %% network
class Network(nn.Module):
    def __init__(self,args):
           # %% initialization
           super(Network,self).__init__()
           self.device = args.device
                                          
           # make distinction between DPS and A-DPS
           self.sampling = args.sampling
           
           if self.sampling == 'DPS':
                  self.no_iter = 1
                  self.logit_parameter = nn.Parameter(torch.randn(784)/4.)
                  self.mux_out = args.no_pixels
           elif self.sampling == 'ADPS':
                  self.no_iter = args.no_pixels
                  self.mux_out = 1
           elif self.sampling == 'PGADPS':                  
                  self.logit_parameter = nn.Parameter(torch.randn(784)/4.)
                  self.mux_out_first = int(args.no_pixels*args.percentage_Ps/100)
                  self.mux_out = int(args.no_pixels*args.percentage_As/100)
                  left_pixels = args.no_pixels-self.mux_out_first
                  # "left pixels" represents the total number of pixels to be selected by ADPS after the first selection by prior-aware DPS.
                  
                  # Minimum samples to select in active sampling is 1.
                  # If the self.mux_out is calculated as 0 ore lower than 1, set it to 1.
                  # "more_iter" indicates whether the one last iteration is needed to collect all target samples with less samples to select than mux_out.
                  # If more_iter is True, the last iteration will select all remaining samples which are less samples than mux_out.
                  if self.mux_out == 0:
                        self.mux_out = 1
                        self.no_iter = left_pixels+1
                        self.more_iter = False 
                  elif self.mux_out == 1:
                     self.no_iter = int(left_pixels/self.mux_out)+1
                     self.more_iter = False      
                  elif left_pixels%self.mux_out != 0:
                     self.no_iter = int(left_pixels/self.mux_out)+2
                     self.more_iter = True
                     self.mux_out_last = left_pixels - (self.no_iter-2)*self.mux_out
                  else:
                     self.no_iter = int(left_pixels/self.mux_out)+1
                     self.more_iter = False
                  
           else:
                  raise Exception('invalid sampling strategy selected, choose DPS or ADPS or PGADPS')
                  
           # create the layers as a module list
           self.f1 = nn.ModuleList([
                  nn.Sequential(
                  nn.Linear(784,784),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(p=0.3),
                  
                  nn.Linear(784,256),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(p=0.3),
               
                  nn.Linear(256,128),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(p=0.3),
               
                  nn.Linear(128,128),
                  nn.LeakyReLU(0.2),
                  ) for i in range(self.no_iter)])
           
           self.f2 = nn.ModuleList([
                  nn.Sequential(
                  nn.Linear(128,10),
                  ) for i in range(self.no_iter)])
           
           if self.sampling == 'ADPS' or self.sampling =='PGADPS':
                  self.g = nn.ModuleList([
                         nn.Sequential(
                         nn.Linear(128,256),
                         nn.LeakyReLU(0.2),
                         nn.Dropout(p=0.3),
                         nn.Linear(256,784),
                         ) for i in range(self.no_iter)])
                  
                  self.lstm = nn.LSTM(128,128,1)
        
           args.mux_in = 28**2
           self.DPS = DPSLayerMemory.DPS(args)
        
    # %% forward call    
    def forward(self, x):
           # get the batch size
           batch_size = x.size(0)
           # convert the input image into a vector
           x_vector = x.reshape(batch_size,784)
           
           #initialize the DPS memory
           self.DPS.initialize_sample_memory()
           
           # create the initial hidden states for the lstm
           h_var = torch.zeros(1,batch_size,128).to(self.device)
           c_var = torch.zeros(1,batch_size,128).to(self.device)
           lstm_in = torch.zeros(1,batch_size,128).to(self.device)
           
           # initalize all outputs for s_hat
           s_hat_all = torch.zeros(batch_size,10,self.no_iter).to(self.device)
           
           # iterate over the lstm
           for i in range(self.no_iter):
                  # generate logits from the hidden state
                  if self.sampling == 'DPS':
                         #DPS
                         logits = self.logit_parameter.unsqueeze(0).repeat(batch_size,1)
                  elif self.sampling == 'ADPS':
                         #ADPS
                         _,(h_var,c_var) = self.lstm(lstm_in,(h_var,c_var))
                         logits = self.g[i](h_var.squeeze())
                  else:
                         if i == 0:
                                logits = self.logit_parameter.unsqueeze(0).repeat(batch_size,1)
                         else:
                                _,(h_var,c_var) = self.lstm(lstm_in,(h_var,c_var))
                                logits = self.g[i](h_var.squeeze())

                  if self.sampling == 'PGADPS':
                         # first selection (i=0) by prior-aware DPS (deterministic sampling), then active GROUP sampling
                         if i == 0:
                               mask = self.DPS(logits,self.mux_out_first)
                         else:
                            if self.more_iter == True:
                                   if i == (self.no_iter-1):
                                          mask = self.DPS(logits,self.mux_out_last)
                                   else:
                                          mask = self.DPS(logits,self.mux_out)
                            else:
                                   mask = self.DPS(logits,self.mux_out)
                  else:      
                     # sampling
                     if i == 0 :
                            mask = self.DPS(logits,self.mux_out)
                     else:
                            mask = self.DPS(logits,1)

                  y = mask*x_vector
                  
                  # encode
                  linear1_out = self.f1[i](y)
                  
                  # get the inputs ready for the lstm
                  lstm_in = linear1_out.unsqueeze(0)
                  
                  #last dense layer
                  s_hat = self.f2[i](linear1_out.squeeze())
                  
                  #save this result
                  s_hat_all[:,:,i] = s_hat
           
           return s_hat_all