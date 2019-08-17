from plotHelpers import *
from time import sleep
import hashlib
import numpy as np
from numpy.random import RandomState
from collections import OrderedDict


@timeUsage
def sleeper(seconds):
    sleep(seconds)


def testPlotConfusionMatrix():

    confusionMatrix = np.array([[220, 12, 8],
                                [7, 330, 15],
                                [6, 11, 441]], dtype='int64')

    xlabels = ['a', 'b', 'c']
    ylabels = ['a', 'b', 'c']
    titleText = "It's a bunch of bunk/crapola."
    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, saveAs='png')

    fileName = 'ConfusionMatrixCountsItsABunchOfBunkcrapola.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = 'a53b4182d30b1c11fba08abce579b710'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='recall', saveAs='png')

    fileName = 'ConfusionMatrixRecallItsABunchOfBunkcrapola.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = 'f8e37ac0cf93f1d5a11eaabf68b13ef2'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='precision', saveAs='png')

    fileName = 'ConfusionMatrixPrecisionItsABunchOfBunkcrapola.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = 'a4be7e90255878e2c11705534f4668b8'

    assert expectedMD5 == actualMD5


def testDetailedHistogram():

    randState = RandomState(20)
    values = randState.randint(0, 101, size=1000)

    titleText = "It's a bunch of bunk/crapola."
    detailedHistogram(values, xlabel='values', ylabel='freqs',
                      titleText=titleText, saveAs='png')

    fileName = 'DetailedHistItsABunchOfBunkcrapola.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '008bef5fa5a64b01d739b1100829cbe5'

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

    titleText = "It's a bunch of bunk/crapola."
    plotValueCounts(df, 'class', titleText=titleText, saveAs='png')

    fileName = 'classFrequenciesItsABunchOfBunkcrapola.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '9057012c6ff3049cbde4bef8b700a0f4'

    assert expectedMD5 == actualMD5


def testTimeUsage(capsys):
    seconds = 2.15
    sleeper(seconds)
    captured = capsys.readouterr()
    assert captured.out == "Δt:  2.15s.\n"
