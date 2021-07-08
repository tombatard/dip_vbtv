import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init
from .common import *

class Skip2Net(nn.Module):
	def __init__(self, input_depth, pad,
                skip_n33d, 
                skip_n33u, 
                skip_n11, 
                num_scales,
                upsample_mode):
		super(Skip3Net,self).__init__()
		self.net1 = skip(input_depth, pad,
                skip_n33d, 
                skip_n33u, 
                skip_n11, 
                num_scales,
                upsample_mode)
		self.net2 = skip(input_depth, pad,
                skip_n33d, 
                skip_n33u, 
                skip_n11, 
                num_scales,
                upsample_mode)

	def forward(self, input1,input2):
		output1 = self.net1(input1)
		output2 = self.net2(input2)
		return output1,output2
