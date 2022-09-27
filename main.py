import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlp

# Open the appropriate files and retrieve the data.
xTrain_file = 'train/X_train.txt'
yTrain_file = 'train/y_train.txt'


# Process the data by removing spaces and \n symbols
def load_x(xFile):
    dataList = pd.read_csv(xFile, delim_whitespace=True, header=None)
    return dataList


def load_y(yFile):
    data = np.loadtxt(yFile, encoding='bytes', delimiter=' ')
    dataList = np.reshape(data, (-1, 1))
    dataList = dataList.astype(int)
    return dataList


# Train, Test, and Valid sets of the x(input) and y(output) data.
# split into different data sets
xData = load_x(xTrain_file)
yData = load_y(yTrain_file)

# Data set #1
# Stratified non-random sets
xTest = xData[0:2000]
xTrain = xData[3000:5000]
xValid = xData[5000:]

yTest = yData[0:2000]
yTrain = yData[3000:5000]
yValid = yData[5000:]
'''
# Odds set
xTest = xTest[1::2]
xTrain = xTrain[1::2]
xValid = xValid[1::2]

yTest = yTest[1::2]
yTrain = yTrain[1::2]
yValid = yValid[1::2]

# Evens set
xTest = xTest[::2]
xTrain = xTrain[::2]
xValid = xValid[::2]

yTest = yTest[::2]
yTrain = yTrain[::2]
yValid = yValid[::2]

# Statistical model sets
# train 70% - valid 20% - test 10%
xTrain = xData[0:round(len(xData)*0.70)]
xValid = xData[round(len(xTrain)):round(len(xData)*0.20)]
xTest = xData[round(len(xValid)):round(len(xData)*0.10)]

yTrain = yData[0: round(len(yData)*0.70)]
yValid = yData[round(len(yTrain)):round(len(yData)*0.20)]
yTest = yData[round(len(yValid)):round(len(yData)*0.10)]
'''
# target set ---------
train_tgt = np.zeros((len(xTrain), 6))
for i in range(len(xTrain)):
    train_tgt[i, yTrain[i] - 1] = 1

# test set ------------
test_tgt = np.zeros((len(xTest), 6))
for i in range(len(xTest)):
    test_tgt[i, yTest[i] - 1] = 1

# valid set -----------
valid_tgt = np.zeros((len(xValid), 6))
for i in range(len(xValid)):
    valid_tgt[i, yValid[i] - 1] = 1


def trainData():
    # train and test neural networks with different number of hidden layers of neurons (i)
    for i in [1, 2, 5, 10, 20]:
        print("----- " + str(i))
        net = mlp.mlp(xTrain, train_tgt, i, outtype='softmax')
        net.earlystopping(xTrain, train_tgt, xValid, valid_tgt, eta=0.5)
        net.confmat(xTest, test_tgt)


def fitnessFunction(population):
    fitness = np.zeros((np.shape(population)[0], 1))
    # Enumerate to get the gene and it's position.
    fittest_genes = []
    for pos, item in enumerate(population):
        # Filter out genes that equal 1
        filtered_train = filterItems(xTrain, item)
        filtered_valid = filterItems(xValid, item)
        filtered_test = filterItems(xTest, item)
        m = mlp.mlp(filtered_train, train_tgt,
                    nhidden=3, outtype="softmax")
        m.earlystopping(filtered_train, train_tgt, filtered_valid, valid_tgt,
                                eta=0.25, niterations=200)
        # Minimize the features - Maximise accuracy
        accuracy = obtainAccuracy(filtered_test, test_tgt, m)
        # Sum of the chromosome- selected features, minus the shape of the chromosome - being the unselected
        returnValue = accuracy*(np.shape(item)[0]-sum(item))
        fitness[pos] = returnValue
    return fitness

def obtainAccuracy(test,tgt, mlp):
    a = mlp.confmat(test, tgt)
    return a

# Take a chromosome and find the genes that equal 1
# Iterate through the selected genes and find them in the training data
# Return the reduced features
def filterItems(xtrain, chromosome):
    xtrain = xtrain.to_numpy()
    filtered = []
    for y in range(len(xtrain)):
        item = []
        for pos, x in enumerate(chromosome):
            if x == 1:
                item.append(xtrain[y][pos])
        filtered.append(item)
    return filtered


trainData()
