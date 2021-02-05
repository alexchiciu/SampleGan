from Discriminator import ConfigNet
import random
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from functools import partial
import os
import ray
from ray import tune

import tempfile

logger = logging.getLogger(__name__)
ray.init(num_cpus=20, num_gpus=1)

path = "/home/alex/workspace/git/SampleGan/data/"



numepocs = 10

def get_train_validate_test_data(realImages, fakeImages):
	"""
	Builds train validate and test data by
	shuffling it and assigning it to 50%, 25%, 25% respectively
	"""
	random.shuffle(realImages)
	random.shuffle(fakeImages)

	trainData = []
	trainDataSize = len(realImages) //2
	addInfoToList(realImages[:trainDataSize], 0., trainData)
	addInfoToList(fakeImages[:trainDataSize], 1., trainData)

	validateData = []
	validateDataSize = len(realImages) //4
	validateDataSize +=trainDataSize
	addInfoToList(realImages[trainDataSize: validateDataSize], 0., validateData)
	addInfoToList(fakeImages[trainDataSize: validateDataSize], 1., validateData)

	testData = []
	addInfoToList(realImages[validateDataSize:], 0., testData)
	addInfoToList(fakeImages[validateDataSize:], 1., testData)

	trainData = getDataLoaderFromList(trainData)
	validateData = getDataLoaderFromList(validateData)
	testData = getDataLoaderFromList(testData)
	return trainData, validateData, testData

def addInfoToList(imageList, label, dataList):
	"""
	Add label to list
	"""
    for image in imageList:
        dataList.append((image,label))

def getDataLoaderFromList(dataList):
	"""
	Build dataloader from list
	"""
    return DataLoader(dataList, shuffle=True)


def get_optimizer(model, learning_rate, momentum):
	return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def run(config):
	"""
	Create a temporary file to save the best model to 
	and evaluate the test data on the best model saved from the epocs
	"""
	tf = tempfile.NamedTemporaryFile(delete = False)
	tempFileName = tf.name	
	print("Temporary file created: " + tempFileName)

	realImages = np.load(path + "originalSubsets.npy")
	fakeImages = np.load(path + "rescaledSubsets.npy")
	
	learning_rate = config['lr']
	momentum = config['momentum']
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model = ConfigNet(config).to(device)
	optimizer = get_optimizer(model,learning_rate,momentum)
	trainData, validateData, testData = get_train_validate_test_data(realImages, fakeImages)
	train(model, optimizer, trainData, validateData, device, tempFileName)
	model = torch.load(tempFileName)
	accuracy = getAccuracyFromModel(model, testData, device) * 100.
	print("Accuracy on testData " + str(accuracy))
	tune.report(test_accuracy=accuracy)
	os.remove(tempFileName)

def getAccuracyFromModel(model, dataSet, device):
    test_losses = []
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for img,label in dataSet:
            img, label = img.unsqueeze(0).float().to(device), label.long().to(device)
            output = model(img)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            if pred == label:
                correct+=1
        test_loss /= len(dataSet)
        test_losses.append(test_loss)
        model.train()
        return float(correct/len(dataSet))

def train(model, optimizer,trainData, validateData, device, tempFileName):


	prevAccuracy = None

	for epoch in range(1, numepocs+1):
		train_loss = 0
		for (img,target) in trainData:
			img, target = img.unsqueeze(0).float().to(device), target.long().to(device)
			optimizer.zero_grad()
			output = model(img)
			loss = F.nll_loss(output, target)
			loss.backward()
			train_loss+=loss.item()
			optimizer.step()
		validationAccuracy = getAccuracyFromModel(model, validateData,device) * 100.

		if prevAccuracy is None or validationAccuracy > prevAccuracy:
			prevAccuracy = validationAccuracy
			print("Accuracy increased, saving model to file: " + tempFileName)
			torch.save(model, tempFileName)




config = {
    "l1": tune.grid_search([100,200]),
    "lr": tune.grid_search([1e-4, 1e-2]), 
    "momentum": tune.grid_search([.5, 0.9]),
    "channels": 1,
    "kernel_size":3,
    "conv_layers": tune.grid_search([2,3]),
    "linear_layers":  tune.grid_search([2,3]),
    "output_features": tune.grid_search([10,20]),
    "stride": 1,
    "bias": False,
    "padding":1

}

def main():
	analysis = tune.run(
		partial(run),
		config=config,
		resources_per_trial={"cpu": 10, "gpu": .5})
	print("Best config: ", analysis.get_best_config(
		metric="test_accuracy", mode="max"))

main()
