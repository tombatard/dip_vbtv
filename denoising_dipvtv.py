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
fname = 'data/denoising/kodim1.png' 
img_pil = crop_image(get_image(fname,-1)[0], d=32)
img_np = pil_to_np(img_pil)

# Add synthetic noise
sigma = 25
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma/255.)    
img_noisy_pil.save('data/denoising/noisy_kodim1.png')
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

# Setup
INPUT = 'noise'
pad = 'reflection'
OPT_OVER='net'

reg_noise_std = 1./30.
LR = 0.01
Lambda = 0.01

OPTIMIZER = 'adam' 
exp_weight = 0.99

num_iter = 4500
input_depth = 32

full_net = VectorialTotalVariation(input_depth, pad, height=img_pil.size[1], width=img_pil.size[0], upsample_mode='bilinear' ).type(dtype)

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in full_net.parameters()]); 

# Loss
mse = torch.nn.MSELoss().type(dtype)
mae = torch.nn.L1Loss().type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_img_avg = img_noisy_np

i = 0
def closure():
    
	global i, exp_weight, out_img_avg, net_input

	net_input = net_input_saved + (noise.normal_() * reg_noise_std)    

	net_output, vbtv = full_net(net_input)
 
	loss_regularizer = mae(vbtv,torch.zeros(1,img_pil.size[1],img_pil.size[0]).type(dtype))

	loss_dataterm = mse(net_output,img_noisy_torch)
 
	total_loss = loss_dataterm + Lambda*loss_regularizer
	total_loss.backward(retain_graph=True)
        
	out_img_avg = out_img_avg * exp_weight + net_output.detach().cpu().numpy()[0] * (1 - exp_weight)
	
	i += 1

	return total_loss

p = get_params(OPT_OVER, full_net, net_input, input_depth)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_img_avg_pil = np_to_pil(out_img_avg)
out_img_avg_pil.save(f'data/denoising/denoised_kodim1.png')