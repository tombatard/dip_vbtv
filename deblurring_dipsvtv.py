from __future__ import print_function

import os
import numpy as np
from models import *
from models.vbtv import*

import torch
import torch.optim

import kornia

from utils.denoising_utils import *
from utils.blur_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# Load clean image
fname = 'data/deblurring/kodim1_small.png' 
img_pil = crop_image(get_image(fname,-1)[0], d=32)
img_np = pil_to_np(img_pil)


# Generate degraded image (blur+noise) 
blur_type = 'gauss_blur'
noise_sigma = 2**.5
blurred = blur(img_np, blur_type)  # blur, and the line below adds noise
img_degraded_np = np.clip(blurred + np.random.normal(scale=noise_sigma/255., size=blurred.shape), 0, 1).astype(np.float32)
img_degraded_pil = np_to_pil(img_degraded_np)
img_degraded_pil.save(f'data/deblurring/degraded_kodim1_small.png')
img_degraded_torch = np_to_torch(img_degraded_np).type(dtype)

# Generta the degradation operator H
NOISE_SIGMA = 2**.5
BLUR_TYPE = 'gauss_blur'
GRAY_SCALE = False 
USE_FOURIER = False

ORIGINAL  = 'Clean'
CORRUPTED = 'Blurred'

data_dict = { ORIGINAL: Data(img_np), 
	                 CORRUPTED: Data(img_degraded_np, compare_PSNR(img_np, img_degraded_np, on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE)) }

H = get_h(data_dict[CORRUPTED].img.shape[0], BLUR_TYPE, USE_FOURIER, dtype)

degraded2 = np.zeros((img_degraded_np.shape))
degraded2[0,:,:]=(1./(0.6*sqrt(3)))*(np.sum(degraded,axis=0))
degraded2[1,:,:]=(1/np.sqrt(2))*(degraded[0,:,:]-degraded[1,:,:])
degraded2[2,:,:]=(1/np.sqrt(6))*(degraded[0,:,:]+degraded[1,:,:]-2*degraded[2,:,:])
img_degraded_torch2 = np_to_torch(degraded2).type(dtype)

# # Setup
INPUT = 'noise'
pad = 'reflection'
OPT_OVER='net'

reg_noise_std = 0.01 
LR = 0.001
Lambda = 0.0001
beta = 0.

OPTIMIZER = 'adam'
exp_weight = 0.99

num_iter = 22000
input_depth = 32

full_net = VectorBundleTotalVariationDenoising(input_depth, pad, beta, height=img_pil.size[1], width=img_pil.size[0],  upsample_mode='bilinear' ).type(dtype)

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in full_net.parameters()]); 

# Loss
mse = torch.nn.MSELoss().type(dtype)
mae = torch.nn.L1Loss().type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_im_avg = img_degraded_np

i = 0
def closure():
    
	global i, exp_weight, img_degraded_torch2, out_im_avg, net_input, H

	net_input = net_input_saved + (noise.normal_() * reg_noise_std)    

	net_output, vbtv = full_net(net_input)	

	# Conversion of the network output from rgb to opp
	Hnet_output=H(net_output)
	opp1=(1./(0.6*sqrt(3)))*torch.unsqueeze(torch.sum(Hnet_output,dim=1),dim=1)
	opp2= (1/sqrt(2))*torch.narrow(Hnet_output,1,0,1) - (1/sqrt(2))*torch.narrow(Hnet_output,1,1,1)
	opp3= (1/sqrt(6))*torch.narrow(Hnet_output,1,0,1) + (1/sqrt(6))*torch.narrow(Hnet_output,1,1,1) - (2/sqrt(6))*torch.narrow(Hnet_output,1,2,1)
	opp=torch.cat((opp1,opp2,opp3),dim=1)

	loss_dataterm = mse(opp,img_degraded_torch2) 
	loss_regularizer = mae(vbtv,torch.zeros(1,img_pil.size[1],img_pil.size[0]).type(dtype))

	total_loss = loss_dataterm + Lambda*loss_regularizer
	total_loss.backward(retain_graph=True)
        
	out_im_avg = out_im_avg * exp_weight + net_output.detach().cpu().numpy()[0] * (1 - exp_weight)

	i += 1

	return total_loss

p = get_params(OPT_OVER, full_net, net_input, input_depth)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_img_avg_pil = np_to_pil(out_im_avg)
out_img_avg_pil.save(f'data/restoration/deblurred_kodim1_small.png')