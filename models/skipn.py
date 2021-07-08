import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init
import kornia
from .common import *
from models import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

class Skip2Net(nn.Module):
	def __init__(self,input_depth, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(Skip2Net,self).__init__()
		self.net1 = skip(int(input_depth/2), n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
		self.net2 = skip(int(input_depth/2), n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
		self.input_depth=input_depth
	def forward(self, input):
		split_input = torch.split(input,int(self.input_depth/2), dim=1)
		input1 = split_input[0]
		input2 = split_input[1]
		output1 = self.net1(input1)
		output2 = self.net2(input2)
		output = torch.cat((output1,output2),dim=1)
		return output


class Skip4Net(nn.Module):
	def __init__(self,input_depth,  pad, width,height, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(Skip4Net,self).__init__()
		self.net1 = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.params2=nn.Parameter(2*torch.rand((3,width,height))-torch.ones((3,width,height)))
		self.params2.requires_grad = True
		self.params3=nn.Parameter(2*torch.rand((6,width,height))-torch.ones((6,width,height)))
		self.params3.requires_grad = True
		self.params4=nn.Parameter(2*torch.rand((18,width,height))-torch.ones((18,width,height)))
		self.params4.requires_grad = True

		self.input_depth=input_depth
		self.width=width,
		self.height=height,

	def forward(self, input):
		split_input = torch.split(input,[self.input_depth,3,6,18], dim=1)

		input1 = split_input[0]
		output1 = self.net1(input1)

		
		out1reshaped = torch.squeeze(output1)
		out1reshaped = torch.transpose(out1reshaped,0,2)
		out1reshaped = torch.reshape(out1reshaped, (self.width[0]*self.height[0],3))		

		differential = kornia.filters.SpatialGradient()(output1)

		differential_x = torch.narrow(differential,2,0,1)
		differential_x = torch.squeeze(differential_x)
		differential_x = torch.transpose(differential_x,0,2)
		differential_x = torch.reshape(differential_x, (self.width[0]*self.height[0],3))

		differential_y = torch.narrow(differential,2,1,1)
		differential_y = torch.squeeze(differential_y)	
		differential_y = torch.transpose(differential_y,0,2)
		differential_y = torch.reshape(differential_y, (self.width[0]*self.height[0],3))
		
			
		# Construction of the metric of the base manifold

		input2 = split_input[1] 
		output2 = input2 + self.params2
		out2 = torch.squeeze(output2)
		out2= torch.transpose(out2,0,2)
		out2 = torch.reshape(out2, (self.width[0]*self.height[0],3))		
		out2 = torch.cat( (torch.narrow(out2,1,0,2),torch.narrow(out2,1,1,2)),dim=1)

		# Construction of the vector bundle metric

		input3 = split_input[2]
		output3 = input3 + self.params3 
		out3 = torch.squeeze(output3)
		out3 = torch.transpose(out3,0,2)
		out3 = torch.reshape(out3, (self.width[0]*self.height[0],6))
		out3 = torch.cat ( (torch.narrow(out3,1,0,3),torch.narrow(out3,1,1,1),torch.narrow(out3,1,3,2),torch.narrow(out3,1,2,1),torch.narrow(out3,1,4,2)), dim=1)
			
		# Construction of the covariant derivative
		input4 = split_input[3]
		output4 = input4 + self.params4
		out4 = torch.squeeze(output4)
		out4 = torch.transpose(out4,0,2)
		out4 = torch.reshape(out4, (self.width[0]*self.height[0],18))

		norm_covariant_derivative = torch.zeros(1)

		superout = torch.cat((out1reshaped,out2,out3,out4,differential_x,differential_y),dim=1)
		for m in torch.unbind(superout, dim=0):
			out1reshaped = torch.narrow(m,0,0,3)
			out1reshaped = torch.t(out1reshaped)

			out2 = torch.narrow(m,0,3,4)
			out3 = torch.narrow(m,0,7,9)
			connection1form_x = torch.narrow(m,0,16,9)
			connection1form_y = torch.narrow(m,0,25,9)
			differential_x = torch.narrow(m,0,34,3)
			differential_x = torch.t(differential_x)
			differential_y = torch.narrow(m,0,37,3)
			differential_y = torch.t(differential_y)
			
			mat = torch.reshape(out2,(2,2))
			val,vec = torch.symeig(mat,eigenvectors=True)
			if torch.max(torch.abs(val))<0.01:
				metric_basemanifold = torch.eye(2)
			else:
				expval = torch.exp(val)
				diagexp = torch.diag(expval)
				metric_basemanifold = torch.matmul(vec,torch.matmul(diagexp,torch.t(vec)))
			#inv_metric_basemanifold = torch.inverse(metric_basemanifold)

			mat = torch.reshape(out3,(3,3))
			val,vec=torch.symeig(mat,eigenvectors=True)
			if torch.max(torch.abs(val))<0.01:
				metric_vectorbundle = torch.eye(3)
			else:	
				expval=torch.exp(val)
				diagexp=torch.diag(expval)
				metric_vectorbundle = torch.matmul(vec,torch.matmul(diagexp,torch.t(vec)))

			connection1form_x = torch.reshape(connection1form_x,(3,3))
			covariant_derivative_x = differential_x + torch.matmul(connection1form_x, out1reshaped)

			connection1form_y = torch.reshape(connection1form_y,(3,3))	
			covariant_derivative_y = differential_y + torch.matmul(connection1form_y, out1reshaped)

			tmp = torch.mul(metric_basemanifold[0,0],torch.matmul(covariant_derivative_x,torch.matmul(metric_vectorbundle,torch.t(covariant_derivative_x)))) \
                                                + torch.mul(torch.mul(2,metric_basemanifold[0,1]),torch.matmul(covariant_derivative_x,torch.matmul(metric_vectorbundle,torch.t(covariant_derivative_y)))) \
                                                + torch.mul(metric_basemanifold[1,1],torch.matmul(covariant_derivative_y,torch.matmul(metric_vectorbundle,torch.t(covariant_derivative_y))))
			tmp = torch.sqrt(tmp) 
			tmp=torch.unsqueeze(tmp,dim=0)
			

			norm_covariant_derivative = torch.cat((norm_covariant_derivative,tmp),dim=0)	

		norm_covariant_derivative = torch.narrow(norm_covariant_derivative,0,1,self.width[0]*self.height[0])	
		norm_covariant_derivative = torch.unsqueeze(norm_covariant_derivative,dim=0)
			                                                                           
		return output1, output2, output3, output4, norm_covariant_derivative