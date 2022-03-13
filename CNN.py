import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, classification_report
from utils import *
import numpy as np
import math

# hyper parameters
batchSize = 100
lr = 0.01
epochs = 5
n_classes = 20

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(128000, 200)
        self.fc2 = nn.Linear(200, n_classes)

    def forward(self, x):  # (1, imageSize[0], imageSize[1])
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(modely, optimizer, criterion, trainLoader):
    modely.train()
    for imgBatch, labelBatch in trainLoader:
        optimizer.zero_grad()
        output = modely(imgBatch)
        loss = criterion(output, labelBatch)
        loss.backward()
        optimizer.step()


def validation(modely, criterion, valLoader):
    modely.eval()
    lossTotal, accuracy = 0, 0
    with torch.no_grad():
        for imgBatch, labelBatch in valLoader:
            outputBatch = modely(imgBatch)
            loss = criterion(outputBatch, labelBatch)
            lossTotal += loss.item()

    lossTotal /= len(valLoader)
    return lossTotal


def test(modely, imageSize):
    # return one hot encoded label
    def oneHot(label):
        vec = [0] * n_classes
        vec[label] = 1
        return np.array(vec)

    # return accuracy-k of pred given its real label
    def accuracyK(pred, label, k):
        topkInds = torch.topk(pred, k).indices
        if label in topkInds:
            return 1
        return 0

    # read data
    testData = readImages("test", imageSize, True)
    testLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)

    # initialize variables to count correct predictions, and total number of samples
    corrects1, corrects2, corrects3, samplesNum = 0, 0, 0, 0
    labels, preds = [sample[1] for sample in testData], []
    # for each batch, for each prediction sample, if it's correct add one to the counter, and save prediction
    modely.eval()
    with torch.no_grad():
        for imgBatch, labelBatch in testLoader:
            outputBatch = modely(imgBatch)
            for output, label in zip(outputBatch, labelBatch):
                corrects1 += accuracyK(output, label, 1)
                corrects2 += accuracyK(output, label, 2)
                corrects3 += accuracyK(output, label, 3)
                samplesNum += 1
                preds.append(torch.argmax(output))

    # print accuracies and confusion matrix
    print(f"test accuracies:   accuracy1={corrects1 / samplesNum}   accuracy2={corrects2 / samplesNum}   accuracy3={corrects3 / samplesNum}")
    print("\nconfusion matrix:\n", confusion_matrix(labels, preds))





    # doe roc curve
    # fpr, tpr, roc_auc = dict(), dict(), dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    #
    # plt.figure()
    # plt.title("CNN ROC curve")
    # plt.plot(fpr, tpr)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    # plt.savefig("CNN ROC curve")


# initiate, train, and return cnn
def makeModel(trainLoader, valLoader):

    # model
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # validation and train
    for epoch in range(epochs):
        loss = validation(model, criterion, valLoader)
        print(f"CNN:     epoch {epoch}    loss {loss}")
        if train != epochs - 1:
            train(model, optimizer, criterion, trainLoader)

    return model
