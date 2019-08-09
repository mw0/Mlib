from utility import *
from time import sleep
import numpy as np
import pandas as pd

@timeUsage
def sleeper(seconds):
    sleep(seconds)


def testTimeUsage(capsys):
    seconds = 2.15
    sleeper(seconds)
    captured = capsys.readouterr()
    assert captured.out == "Δt:  2.15s.\n"


def testSplitDataFrameByClasses():

    randState = np.random.RandomState(26)
    vals = randState.randint(0, 100000, size = 100000)
    classes = ['a' if v < 100 else 'b' if v < 1000
               else 'c' if v < 10000 else 'd' for v in vals]
    df = pd.DataFrame({'values': vals, 'class': classes})
    
    # Using this randState, with seed 26, should get:

    # In [10]: df['class'].value_counts()
    # Out[10]:
    # d    90098
    # c     8908
    # b      908
    # a       86

    classColumn = 'class'
    testFrac = 0.30

    dfTrain, dfTest = splitDataFrameByClasses(df, classColumn,
                                              testFrac=testFrac,
                                              myRandomState=randState)

    expectedTrainValCts = pd.Series([63068, 6235, 635, 60],
                                    index = ['d', 'c', 'b', 'a'],
                                    name = 'class', dtype='int64')
    expectedTestValCts = pd.Series([27030, 2673, 273, 26],
                                   index = ['d', 'c', 'b', 'a'],
                                   name = 'class', dtype='int64')

    assert expectedTrainValCts.equals(dfTrain[classColumn].value_counts())
    assert expectedTestValCts.equals(dfTest[classColumn].value_counts())


def testSplitBalanceDataFrameByClasses():

    randState = np.random.RandomState(26)
    vals = randState.randint(0, 100000, size = 100000)
    classes = ['a' if v < 100 else 'b' if v < 1000
               else 'c' if v < 10000 else 'd' for v in vals]
    df = pd.DataFrame({'values': vals, 'class': classes})
    
    # Using this randState, with seed 26, should get:

    # In [10]: df['class'].value_counts()
    # Out[10]:
    # d    90098
    # c     8908
    # b      908
    # a       86

    classColumn = 'class'
    targetClassSize = 1000
    testFrac = 0.30

    dfTrain, dfTest = splitBalanceDataFrameByClasses(df, classColumn,
                                                     targetClassSize,
                                                     testFrac=0.33,
                                                     myRandomState=randState)

    expectedTrainValCts = pd.Series([1000, 1000, 1000, 1000],
                                    index = ['a', 'b', 'c', 'd'],
                                    name = 'class', dtype='int64')
    expectedTestValCts = pd.Series([29733, 2940, 300, 29],
                                   index = ['d', 'c', 'b', 'a'],
                                   name = 'class', dtype='int64')
    print(expectedTrainValCts)

    # Note: .value_counts() returns in descending count order. When the
    # counts are identical, as in the case for the training set (balanced
    # classes), the indices of the series are whatever the dict provides.
    # In order to make the comparison, apply .sort_index() to the result
    # when comparing to the hand-created expectedTrainValCts pd.Series.
    assert expectedTrainValCts.equals(dfTrain[classColumn].value_counts()
                                      .sort_index())
    assert expectedTestValCts.equals(dfTest[classColumn].value_counts())

