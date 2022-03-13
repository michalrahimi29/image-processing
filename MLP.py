import torch
from sklearn import metrics
from torch import nn
from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from autoencoder import encodeImages
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# hyper parameters
lr = 0.001
epochs = 5
batchSize = 200


class MLPClassifier(nn.Module):
    def __init__(self, latentDim):
        super(MLPClassifier, self).__init__()
        self.layer_1 = nn.Linear(latentDim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.softmax(self.layer_3(x))
        return x


def train(modely, optimizer, criterion, trainLoader):
    modely.train()
    for epoch in range(epochs):
        for xBatch, yBatch in trainLoader:
            optimizer.zero_grad()
            pred = modely(xBatch)
            loss = criterion(pred, yBatch.view_as(pred))
            loss.backward()
            optimizer.step()


def validation(modely, criterion, valLoader):
    # do predictions and save them. also save real labels
    modely.eval()
    loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for xBatch, yBatch in valLoader:
            output = modely(xBatch.view(1, -1))
            loss += criterion(output, yBatch.reshape(-1, 1)).item()
            curPreds = [torch.round(pred[0]) for pred in output]
            preds.extend(curPreds)
            labels.extend(yBatch)

    # take measurements
    predictions = preds
    accuracy = metrics.accuracy_score(labels, predictions)
    return loss / len(valLoader), accuracy


def test(testEncoded):
    pass


# train model and save it in file
def makeClassifier(trainEncodedLoader, valEncodedLoader):
    # prepare model
    modely = MLPClassifier(len(trainEncodedLoader[0][0]))
    optimizer = torch.optim.Adam(modely.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # run
    for epoch in range(epochs):
        # do validation first, to start seeing results before any training
        curLoss = validation(modely, criterion, valEncodedLoader)
        print(f"MLP classifier:   epoch {epoch}    loss {curLoss}")
        # do training unless it's the last epoch, because we are not going to see the result of last epoch training
        if epoch != epochs - 1:
            train(modely, optimizer, criterion, trainEncodedLoader)

    # save trained model
    return modely