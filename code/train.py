from Discriminator import Net
import random
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

path = "/home/alex/workspace/git/SampleGan/data/"
realImages = np.load(path + "originalSubsets.npy")
fakeImages = np.load(path + "rescaledSubsets.npy")

learning_rate = 0.001
momentum = 0.5
numepocs = 1

def get_train_validate_test_data():
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
    for image in imageList:
        dataList.append((image,label))

def getDataLoaderFromList(dataList):
    return DataLoader(dataList, shuffle=True)


def get_optimizer(model):
	return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def run():
	model = Net().to(device)
	optimizer = get_optimizer(model)
	trainData, validateData, testData = get_train_validate_test_data()
	train(model, optimizer, trainData, validateData)
	model = torch.load('bestModel.pkl')
	accuracy = getAccuracyFromModel(model, testData) * 100.
	print("Accuracy on testData " + str(accuracy))

def getAccuracyFromModel(model, dataSet):
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

def train(model, optimizer,trainData, validateData):
	epochList = []
	trainLossList = []
	trainAccuracyList = []
	validationAccuracyList = []

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
		validationAccuracy = getAccuracyFromModel(model, validateData) * 100.
		trainingAccuracy = getAccuracyFromModel(model, trainData) * 100.

		epochList.append(epoch)
		trainLossList.append(train_loss)
		trainAccuracyList.append(trainingAccuracy)
		validationAccuracyList.append(validationAccuracy)

		if prevAccuracy is None or validationAccuracy > prevAccuracy:
			prevAccuracy = validationAccuracy
			torch.save(model, 'bestModel.pkl')


if __name__ == "__main__":
	run()