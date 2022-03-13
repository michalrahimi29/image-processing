import CNN
import autoencoder
import utils
import regression
import time
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

start_time = time.time()

# hyper parameters
imageSize = (160, 200)  # all images will be reduced to this
latentDim = 80
batchSize = 100

if __name__ == "__main__":
    # read data
    trainValData = utils.readImages("train", imageSize, True)
    trainData, valData = train_test_split(trainValData)
    trainLoader, valLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True), DataLoader(valData, batch_size=batchSize, shuffle=True)
    testData = utils.readImages("test", imageSize, True)
    testLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)

    # make autoencoder, use its encoder to encode the images, then do linear regression, mlp regressor, and mlp classifier
    autoencoderModel = autoencoder.makeModel(imageSize, latentDim, trainLoader, valLoader)
    trainEncoded, valEncoded, testEncoded = autoencoder.encodeImages(autoencoderModel, imageSize, trainData), autoencoder.encodeImages(autoencoderModel, imageSize,
                                                                                                                                       valData), autoencoder.encodeImages(
        autoencoderModel, imageSize, testData)
    trainEncodedLoader, valEncodedLoader = DataLoader(trainEncoded, batch_size=batchSize, shuffle=True), DataLoader(valEncoded, batch_size=batchSize, shuffle=True)
    regression.MLPRegression(trainEncoded, testEncoded)
    regression.linearRegression(trainEncoded, testEncoded)

    # make cnn model, use it to do classification
    cnnModel = CNN.makeModel(trainLoader, valLoader)
    torch.save(cnnModel, "cnnModel")
    cnnModel = torch.load("cnnModel")
    CNN.test(cnnModel, imageSize)

    # print the time it all took
    minutes = (time.time() - start_time) / 60
    print(f"\n--- {minutes:.1f} minutes ---")
