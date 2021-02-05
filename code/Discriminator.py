import torch
import torch.nn as nn
import torch.nn.functional as F



class ConfigNet(nn.Module):
    def __init__(self, config):
        """
        Build net with parameters from the config
        parameters. Most changed ones are conv layers,
        linear layers, output features
        """
        numChannels = config['channels']
        useBias = config['bias']
        kernelSize = config['kernel_size']
        padding = config['padding']
        convLayers = config['conv_layers']
        linearLayers = config['linear_layers']
        outputFeatures = config['output_features']
        stride = config['stride']
        numFeatures  = config['l1']

        norm_layer = nn.BatchNorm3d     
        
        super(ConfigNet, self).__init__()
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


    def forward(self, x):
        x = self.sequence(x)
        x = x.view(-1, self.totalLinearFeatures)
        x = self.linearLayer(x)
        return F.log_softmax(x, dim=1)
