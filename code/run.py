import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self, config):    	
		numChannels = config['channels']
		useBias = config['bias']
		kernelSize = config['kernel_size']
		padding = config['padding']
		convLayers = config['conv_layers']
		outputFeatures = config['output_features']
		stride = config['stride']
		norm_layer = nn.BatchNorm3d
		super(Net, self).__init__()
		sequence = [nn.Conv3d(numChannels, outputFeatures, kernel_size=kernelSize, stride=stride, padding=padding),nn.MaxPool3d(2), nn.LeakyReLU(0.2, True)]
		'''
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, convLayers):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
			nn.Conv3d(outputFeatures * nf_mult_prev, outputFeatures * nf_mult, kernel_size=kernelSize, stride=stride, padding=padding, bias=useBias),
			norm_layer(outputFeatures * nf_mult),
			nn.LeakyReLU(0.2, True)]

		sequence += [nn.Conv3d(outputFeatures * nf_mult, 1, kernel_size=kernelSize, stride=stride, padding=padding)]
		'''  
		self.sequence = nn.Sequential(*sequence)

	def getSequence(self):
		return self.sequence


config = dict()
config['channels'] = 1
config['bias'] = False
config['kernel_size'] = 3
config['padding'] = 1
config['conv_layers'] = 1
config['output_features'] = 20
config['stride'] = 1
n = Net(config)
s = n.getSequence()
print(s)
t = torch.rand(1,1,20,20,20)
t = s(t)
print(t.shape)