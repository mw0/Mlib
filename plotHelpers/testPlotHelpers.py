from plotHelpers import *
from time import sleep
import hashlib
import numpy as np
from numpy.random import RandomState
from collections import OrderedDict
from matplotlib import __version__ as mpVersion


@timeUsage
def sleeper(seconds):
    sleep(seconds)


def testPlotConfusionMatrix():

    print(f"mpVersion: {mpVersion}")

    confusionMatrix = np.array([[220, 12, 58, 3, 17],
                                [7, 330, 15, 22, 5],
                                [41, 3, 406, 8, 21],
                                [41, 72, 36, 308, 16],
                                [6, 11, 8, 19, 441]], dtype='int64')

    print(confusionMatrix)
    xlabels = ['a', 'b', 'c', 'd', 'e']
    ylabels = ['a', 'b', 'c', 'd', 'e']
    titleText = "Example 1"

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, saveAs='png')

    fileName = 'ConfusionMatrixCountsExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '7ea26d0b16cd0a72e161bb424fc641fb'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='recall', saveAs='png')

    fileName = 'ConfusionMatrixRecallExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '3c932c1d93ad7937ad2b749e25396096'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='precision', saveAs='png')

    fileName = 'ConfusionMatrixPrecisionExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = 'f16a33f98d58510df069c9016a20637c'

    assert expectedMD5 == actualMD5


def testDetailedHistogram():

    randState = RandomState(20)
    values = randState.randint(0, 101, size=1000)
    titleText = "Example 1"

    detailedHistogram(values, xlabel='values', ylabel='freqs',
                      titleText=titleText, saveAs='png')

    fileName = 'DetailedHistExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = 'aeae7309070e8a71a0b31397ef9b2639'

    assert expectedMD5 == actualMD5


def testPlotValueCounts():

    classes = ['Ugly']*14 + ['Loathsome']*15 + ['Weensy']*16 + \
        ['Stanky']*22 + ['Icky']*23 + ['Bad']*28 + ['Good']*31 + ['Smelly']*41
    values = ['14']*14 + ['15']*15 + ['16']*16 + ['22']*22 + ['23']*23 + \
        ['28']*28 + ['31']*31 + ['41']*41
    values = [int(v) for v in values]

    randState = RandomState(20)
    randState.shuffle(values)
    randState = RandomState(20)
    randState.shuffle(classes)

    df = pd.DataFrame({'value': values, 'class': classes})
    print(df.head(10))
    print(df['class'].value_counts())

    titleText = "Example 1"

    plotValueCounts(df, 'class', titleText=titleText, saveAs='png')

    fileName = 'classFrequenciesExample1.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = 'db82217006096d6b0a8f8ef4fc0aa2ca'

    assert expectedMD5 == actualMD5


def testTimeUsage(capsys):
    seconds = 2.15
    sleeper(seconds)
    captured = capsys.readouterr()
    assert captured.out == "Δt:  2.15s.\n"
