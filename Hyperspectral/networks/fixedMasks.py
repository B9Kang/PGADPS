"""
===================================================================================
    Source Name   : fixedMasks.py
    Description   : Script used to create the various sampling masks that are not
                    changed once created.
===================================================================================
"""

import torch

def createMask(args):
   
    # initialize the sample matrix A
    A = torch.zeros(1,args.mux_in)
    
    # %% random_uniform
    if args.sampling == "Uniform":
        # select only the number of lines needed from the 51 lines
        indices = torch.linspace(0, args.mux_in - 1, steps=args.bands).round().long()
        
        # fill in those lines with ones 
        A[0, indices] = 1

        # %% GSS
    elif args.sampling == "GSS":
        if args.bands == 10:
            indices_to_sample = torch.tensor([13, 28, 31, 32, 33, 34, 36, 48, 49, 50])
        elif args.bands == 5 :
            indices_to_sample = torch.tensor([13, 27, 30, 34, 50])
        else:
            raise Exception('Cannot use GSS except for 5 bands lines because they are not available.')

        A[:,indices_to_sample] = 1
        
    elif args.sampling == 'SLR':
        if args.bands == 5:
            indices_to_sample = torch.tensor([49, 42, 41, 36, 50])
            
        else:
            raise Exception('Cannot use SLR except for 5 bands lines because they are not available.')
        
        A[:,indices_to_sample] = 1

    elif  args.sampling =='all':
        A = torch.ones(1, args.mux_in)
        
    
    # expand along H,W dimension
    A = A.reshape(1,args.mux_in,1,1).repeat(1,1,args.window_size,args.window_size).to(args.device)

    return A
