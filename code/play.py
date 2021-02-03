from Discriminator import *

import torch
import torch.nn.functional as F



firstLayer = nn.Conv3d(1, 10, kernel_size=3, padding =1)
secondLayer = nn.Conv3d(10, 20, kernel_size=3, padding =1)
t = torch.rand(1,1,20,20,20)

class Net(nn.Module):
	def __init__(self, config):
		numChannels = config['channels']
		useBias = config['bias']
		kernelSize = config['kernel_size']
		padding = config['padding']
		convLayers = config['conv_layers']
		linearLayers = config['linear_layers']
		outputFeatures = config['output_features']
		stride = config['stride']
		numFeatures  = 100

		norm_layer = nn.BatchNorm3d    	
		
		super(Net, self).__init__()
		sequence = [nn.Conv3d(numChannels, outputFeatures, kernel_size=kernelSize, stride=stride, padding=padding),nn.MaxPool3d(2), nn.ReLU()]
		
		prevFeatures = outputFeatures

		for i in range(1, convLayers):
			outputFeatures*=2
			if i == 1:
				sequence+= [nn.Conv3d(prevFeatures, outputFeatures, kernel_size=kernelSize, stride=stride, padding=padding), nn.MaxPool3d(2), nn.ReLU()]
			else:
				sequence+= [nn.Conv3d(prevFeatures, outputFeatures, kernel_size=kernelSize, stride=stride, padding=padding), nn.ReLU()]

			prevFeatures = outputFeatures

		self.totalLinearFeatures = outputFeatures * 5 * 5 * 5
		self.sequence = nn.Sequential(*sequence)
		
		self.linearLayer = [nn.Linear(self.totalLinearFeatures, numFeatures), nn.ReLU()]
		prevFeatures = numFeatures
		for i in range(1, linearLayers):
			numFeatures*=.5
			numFeatures = int(numFeatures)
			self.linearLayer += [nn.Linear(prevFeatures, numFeatures), nn.ReLU()]
			prevFeatures = numFeatures

		self.linearLayer += [nn.Linear(prevFeatures, 2)]
		self.linearLayer = nn.Sequential(*self.linearLayer)
		print(self.sequence)
		print(self.linearLayer)

	def forward(self, x):
		x = self.sequence(x)
		print(x.shape)
		x = x.view(-1, self.totalLinearFeatures)
		x = self.linearLayer(x)
		return F.log_softmax(x, dim=1)

config = dict()
config['channels'] = 1
config['bias'] = False
config['kernel_size'] = 3
config['linear_layers'] = 3
config['padding'] = 1
config['conv_layers'] = 3
config['output_features'] = 20
config['stride'] = 1
n = Net(config)

t = torch.rand(1,1,20,20,20)
t = n.forward(t)
print(t.shape)