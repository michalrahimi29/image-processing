import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *

# hyper parameters
batchSize = 100
epochs = 10
lr = 0.002


# encode from image size to latentDim
class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
        )

    def forward(self, x):
        return self.encode(x)


# decode from latentDim to image size
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(dims[-1], dims[-2]),
            nn.ReLU(),
            nn.Linear(dims[-2], dims[-3]),
            nn.ReLU(),
            nn.Linear(dims[-3], dims[-4]),
            nn.Sigmoid()  # to put all values back in range of (0,1). assuming that was their range at first
        )

    def forward(self, x):
        return self.decode(x)


# encode, decode, and return decoded output
class AutoEncoder(nn.Module):
    def __init__(self, inputSize, latentDim):
        dims = [inputSize, 1000, 200, latentDim]
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(dims)
        self.decoder = Decoder(dims)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded


# do train on given model
def train(modely, optimizer, criterion, trainLoader, imageSize):
    modely.train()
    for batchSamples in trainLoader:
        batchImgs = batchSamples[0]  # batchSamples[1] is ages
        batchImgs = batchImgs.reshape(-1, 1, imageSize[0] * imageSize[1])  # (batchSize, 1, inputSize)
        pred = modely(batchImgs)
        loss = criterion(pred, batchImgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# do dev on given model
def dev(modely, criterion, devLoader, imageSize):
    modely.eval()
    totalLoss = 0
    with torch.no_grad():
        for batchSamples in devLoader:
            batchImgs = batchSamples[0]  # batchSamples[1] is ages
            batchImgs = batchImgs.reshape(-1, 1, imageSize[0] * imageSize[1])  # (batchSize, 1, inputSize)
            pred = modely(batchImgs)
            loss = criterion(pred, batchImgs)
            totalLoss += loss.item()
    return loss


# initiate, train, and return autoencoder
def makeModel(imageSize, latentDim, trainLoader, valLoader):
    # init model
    model = AutoEncoder(imageSize[0] * imageSize[1], latentDim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # validation and and train
    for epoch in range(epochs):
        loss = dev(model, criterion, valLoader, imageSize)
        print(f"autoencoder:     epoch {epoch}   loss {loss}")
        if train != epochs - 1:
            train(model, optimizer, criterion, trainLoader, imageSize)

    return model


# encode given data using given model's encoder, to list of (encoded_img, age)
def encodeImages(autoencoder, imageSize, images):
    imagesLoader = DataLoader(images, batch_size=batchSize)
    encoded = []
    with torch.no_grad():
        for batchSamples in imagesLoader:
            batchImgs, batchAges = batchSamples[0], batchSamples[1].tolist()
            batchImgs = batchImgs.reshape(-1, 1, imageSize[0] * imageSize[1])                          # (batchSize, 1, inputSize)
            encodedImgs = autoencoder.encoder(batchImgs).tolist()
            encoded.extend([(encodedImgs[i][0], batchAges[i]) for i in range(len(batchImgs))])   # extend in list of (encoded_img, age)
    return encoded



