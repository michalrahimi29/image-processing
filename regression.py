from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from math import floor
from sklearn.metrics import accuracy_score


# plot scatter plot of original stability and score1 of machine output, with trend line
def plotScatter(realAges, preds, title):
    plt.figure()
    # scatter plot
    plt.scatter(realAges, preds)
    plt.xlabel("real ages")
    plt.ylabel("predicted ages")
    plt.title(title)
    # trend line
    z = np.polyfit(realAges, preds, 1)
    p = np.poly1d(z)
    plt.plot(realAges, p(realAges), "r--")
    # save
    plt.savefig(title)


# return class prediction according to given regressive score
def regToClass(regScore):
    return floor(regScore)

# return accuracy-k of pred given its real label
def accuracyK(pred, label, k):
    topkInds = print(sorted(range(len(pred)), key=lambda x: pred[x])[-k:])
    if label in topkInds:
        return 1
    return 0

def MLPRegression(trainDataEncoded, testDataEncoded):
    # extract data
    trainImgs, trainAges = [sample[0] for sample in trainDataEncoded], [sample[1] for sample in trainDataEncoded]
    testImgs, testAges = [sample[0] for sample in testDataEncoded], [sample[1] for sample in testDataEncoded]

    # make model, fit to train data
    model = MLPRegressor(random_state=7, max_iter=40000).fit(trainImgs, trainAges)

    # make prediction
    preds = model.predict(testImgs)

    # plot and print MSE
    plotScatter(testAges, preds, "MLPRegression")
    MSE = mean_squared_error(testAges, preds)
    print("MLP regressor MSE:    ", MSE)

    # make classification, to compare accuracy with the CNN
    classPreds = [regToClass(reg) for reg in preds]
    accuracy = accuracy_score(testAges, classPreds)
    print(f"MLP Regressor accuracy:  ", accuracy)


def linearRegression(trainDataEncoded, testDataEncoded):
    # extract data
    trainVecs, trainAges = torch.FloatTensor([sample[0] for sample in trainDataEncoded]), torch.FloatTensor([sample[1] for sample in trainDataEncoded])
    testVecs, testAges = torch.FloatTensor([sample[0] for sample in testDataEncoded]), torch.FloatTensor([sample[1] for sample in testDataEncoded])

    # model
    model = nn.Linear(len(trainDataEncoded[0][0]), 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

    # train
    epochs = 100
    for epoch in range(epochs):
        pred = model(trainVecs)
        loss = criterion(pred, trainAges.view(-1, 1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f" linear regression:   epoch {epoch}   loss {loss.item():.4f}")

    # make prediction
    preds = model(testVecs).detach().numpy()
    preds = np.squeeze(preds)
    # plot, and print MSE
    plotScatter(testAges, preds, "linearRegression")
    MSE = mean_squared_error(testAges, preds)
    print("linear regression MSE = ", MSE)




