from KerasEarlyStopCallback import KerasEarlyStopCallback as kesc
import tensorflow as tf
import numpy as np
import pickle


inFile = '../fashionMNISTsubset.pkl'
with open(inFile, 'rb') as IN:
    ((xTrain, yTrain), (xTest, yTest)) = pickle.load(IN)

def testLoadPickle():
    assert np.shape(xTrain) == (500, 28, 28)
    assert np.shape(yTrain) == (500,)
    assert np.shape(xTest) == (250, 28, 28)
    assert np.shape(yTest) == (250,)

