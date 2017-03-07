#!/opt/anaconda3/bin/python
import scipy.io as sio
import numpy as np

def readMNIST(fname='./data/mnist.mat', asImage=True):
    data = sio.loadmat(fname)
    trainX = data['train_X']
    trainY = data['train_Y']
    trainY = trainY.reshape([-1])
    testX = data['test_X']
    testY = data['test_Y']
    testY = testY.reshape([-1])
    if asImage:
        trainX = trainX.reshape((-1, 28, 28))
        testX = testX.reshape((-1, 28, 28)) 
    trainX = trainX/np.max(trainX)
    testX = testX / np.max(testX)
    return trainX, trainY, testX, testY


    
def batchGenerator(X, Y, batchSize=4,):
    length = X.shape[0]
    if Y.shape[0] != length:
        raise ValueError('X and Y are not aligned')
    while True:
        index = np.random.randint(0, length, batchSize)
        yield X[index, :],Y[index]

