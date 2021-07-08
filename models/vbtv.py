import torch
import torch.nn as nn
from math import sqrt
import torch.nn.init
import kornia
from .common import *
from models import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

class VectorialTotalVariation(nn.Module):	

	def __init__(self,input_depth, pad, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorialTotalVariation,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth=input_depth

	def forward(self, input):

		output = self.net(input)

		differential = kornia.filters.SpatialGradient()(output)
		differential_squared = torch.mul(differential,differential)

		norm_squared = torch.sum(torch.sum(differential_squared,dim=1),dim=1)
		norm_squared_regularized = norm_squared+0.00001*torch.ones((1,self.height,self.width)).type(torch.cuda.FloatTensor)
		norm = torch.sqrt(norm_squared_regularized)
		return output, norm

class VectorBundleTotalVariationDenoising(nn.Module):
	def __init__(self,input_depth, pad, beta, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationDenoising,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta

	def forward(self, input):	

		output = self.net(input)

		luminance = (1/(sqrt(3)))*torch.unsqueeze(torch.sum(output,dim=1),dim=1) 
		differential_luminance = kornia.filters.SpatialGradient()(luminance)
		differential_luminance = differential_luminance/0.6

		chrominance1 = (1/sqrt(2))*torch.narrow(output,1,0,1) - (1/sqrt(2))*torch.narrow(output,1,1,1)
		chrominance2 = (1/sqrt(6))*torch.narrow(output,1,0,1) + (1/sqrt(6))*torch.narrow(output,1,1,1) - (2/sqrt(6))*torch.narrow(output,1,2,1)
		chrominance = torch.cat((chrominance1,chrominance2),dim=1)

		differential_chrominance = kornia.filters.SpatialGradient()(chrominance)
		
		covariant_derivative = torch.cat((differential_luminance,differential_chrominance),dim=1)
		
		covariant_derivative_x = torch.narrow(covariant_derivative,2,0,1)	
		covariant_derivative_x = torch.squeeze(covariant_derivative_x)
		covariant_derivative_y = torch.narrow(covariant_derivative,2,1,1)
		covariant_derivative_y = torch.squeeze(covariant_derivative_y)

		covariant_derivative_squared_x = torch.mul(covariant_derivative_x,covariant_derivative_x)	
		covariant_derivative_squared_y = torch.mul(covariant_derivative_y,covariant_derivative_y)		

		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*torch.sum(covariant_derivative_squared_x,dim=0)
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*torch.sum(covariant_derivative_squared_y,dim=0)
		g12 = self.beta*torch.sum(torch.mul(covariant_derivative_x,covariant_derivative_y),dim=0)
		detg = torch.mul(g11,g22) - torch.mul(g12,g12)

		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)

		norm_squared = torch.mul(invg11,torch.sum(covariant_derivative_squared_x,dim=0)) \
		                   +2*torch.mul(invg12, torch.sum(torch.mul(covariant_derivative_x,covariant_derivative_y),dim=0)) \
		                   +torch.mul(invg22,torch.sum(covariant_derivative_squared_y,dim=0))

		norm_squared_regularized = norm_squared+0.00001*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		norm = torch.sqrt(norm_squared_regularized)
		norm = torch.unsqueeze(norm,dim=0)

		return output, norm

class VectorBundleTotalVariationDeblurring(nn.Module):
	def __init__(self,input_depth, pad, omega11, omega23, beta, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.omega11 = omega11
		self.omega23 = omega23
		self.beta = beta

	def forward(self, input):	

		output = self.net(input)

		luminance = (1/sqrt(3))*torch.unsqueeze(torch.sum(output,dim=1),dim=1) 
		differential_luminance = kornia.filters.SpatialGradient()(luminance)
		luminance = torch.unsqueeze(luminance,dim=2)
		luminance = luminance.repeat(1,1,2,1,1)

		chrominance1 = (1/sqrt(2))*torch.narrow(output,1,0,1) - (1/sqrt(2))*torch.narrow(output,1,1,1)
		differential_chrominance1 = kornia.filters.SpatialGradient()(chrominance1)
		chrominance2 = (1/sqrt(6))*torch.narrow(output,1,0,1) + (1/sqrt(6))*torch.narrow(output,1,1,1) - (2/sqrt(6))*torch.narrow(output,1,2,1)
		differential_chrominance2 = kornia.filters.SpatialGradient()(chrominance2)
		differential_lumchrom = torch.cat((differential_luminance,differential_chrominance1,differential_chrominance2),dim=1)

		chrominance1 = torch.unsqueeze(chrominance1,dim=2)
		chrominance1 = chrominance1.repeat(1,1,2,1,1)
		chrominance2 = torch.unsqueeze(chrominance2,dim=2)
		chrominance2 = chrominance2.repeat(1,1,2,1,1)

		zero_order_term1 = torch.mul(self.omega11,luminance)
		zero_order_term2 = torch.mul(self.omega23,chrominance2)
		zero_order_term3 = - torch.mul(self.omega23,chrominance1)
		zero_order_term = torch.cat((zero_order_term1,zero_order_term2,zero_order_term3),dim=1)

		covariant_derivative = differential_lumchrom + zero_order_term 

		covariant_derivative_x = torch.narrow(covariant_derivative,2,0,1)	
		covariant_derivative_x = torch.squeeze(covariant_derivative_x)
		covariant_derivative_x_squared = torch.mul(covariant_derivative_x,covariant_derivative_x)
		norm_covariant_derivative_x_squared = torch.sum(covariant_derivative_x_squared,dim=0)

		covariant_derivative_y = torch.narrow(covariant_derivative,2,1,1)
		covariant_derivative_y = torch.squeeze(covariant_derivative_y)
		covariant_derivative_y_squared = torch.mul(covariant_derivative_y,covariant_derivative_y)
		norm_covariant_derivative_y_squared = torch.sum(covariant_derivative_y_squared,dim=0)

		covariant_derivative_x_covariant_derivative_y = torch.mul(covariant_derivative_x,covariant_derivative_y)	
		scal_product_covariant_derivative_x_covariant_derivative_y= torch.sum(covariant_derivative_x_covariant_derivative_y,dim=0)
		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*norm_covariant_derivative_x_squared
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*norm_covariant_derivative_y_squared 
		g12 = self.beta*scal_product_covariant_derivative_x_covariant_derivative_y #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]

		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)

		norm_squared = torch.mul(invg11,norm_covariant_derivative_x_squared) \
		               +2*torch.mul(invg12, scal_product_covariant_derivative_x_covariant_derivative_y) \
		               +torch.mul(invg22,norm_covariant_derivative_y_squared)

		norm_squared_regularized = norm_squared+0.00001*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		norm = torch.sqrt(norm_squared_regularized)
		norm = torch.unsqueeze(norm,dim=0)

		return output, norm