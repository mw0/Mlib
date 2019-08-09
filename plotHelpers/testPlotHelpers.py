from plotHelpers import *
from time import sleep
import hashlib

@timeUsage
def sleeper(seconds):
    sleep(seconds)


def testTimeUsage(capsys):
    seconds = 2.15
    sleeper(seconds)
    captured = capsys.readouterr()
    assert captured.out == "Δt:  2.15s.\n"

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

    expectedMD5 = '6c8b1444223e0c7e8c67a2025f78e1fa'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='recall', saveAs='png')

    fileName = 'ConfusionMatrixRecallItsABunchOfBunkcrapola.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '856ce3eb6edeb5b28b617e41e91ca16e'

    assert expectedMD5 == actualMD5

    plotConfusionMatrix(confusionMatrix, xlabels=xlabels, ylabels=ylabels,
                        titleText=titleText, type='precision', saveAs='png')

    fileName = 'ConfusionMatrixPrecisionItsABunchOfBunkcrapola.png'
    with open(fileName, 'rb') as veriFile:
        datums = veriFile.read()
        actualMD5 = hashlib.md5(datums).hexdigest()

    expectedMD5 = '982159b8c3422514c557f65ce3b0b273'

    assert expectedMD5 == actualMD5
