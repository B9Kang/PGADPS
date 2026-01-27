
"""
===================================================================================
    Source Name   : Fourier_Operations.py
    Description   : Some Fourier operations that map in and out of k-space
===================================================================================
"""

import torch

# %% forward Fourier
def Forward_Fourier(pixels):
    k_space = torch.fft.fft2(pixels)

    return k_space

# %% inverse Fourier
# def Inverse_Fourier(k_space):
#     k_space = torch.view_as_complex(k_space)
#     complex_pixels = torch.fft.ifft2(k_space)
#     envelop_pixels = torch.view_as_real(complex_pixels)

#     return envelop_pixels

def Inverse_Fourier(k_space):
    complex_pixels = torch.fft.ifft2(k_space)

    return complex_pixels.abs()

# %% full mapping with sampling mask
def Full_Map(pixels_in,mask):
  
    # go through fourier space
    # print(pixels_in.dtype)
    full_k_space = Forward_Fourier(pixels_in)
    # print(full_k_space.dtype)
    sampled_k_space = full_k_space*mask
    pixels_out = Inverse_Fourier(sampled_k_space) 
    # print(pixels_out.dtype)

    return pixels_out

# %% full mapping with sampling mask
def Full_Map_Pineda(pixels_in,mask):
    full_k_space = Forward_Fourier(pixels_in)
    sampled_k_space = full_k_space*mask
    pixels_out = torch.fft.ifft2(sampled_k_space)
    pixels_out = pixels_out.transpose(1,4).reshape(pixels_in.size(0),2,pixels_in.size(2),pixels_in.size(3))
    return pixels_out

