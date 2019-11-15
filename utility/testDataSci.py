from DataSci import *
from time import sleep
import numpy as np
import pandas as pd
import scipy.sparse as sp
from random import seed
from collections import OrderedDict
from os import system


def testSplitDataFrameByClasses():

    randState = np.random.RandomState(26)
    vals = randState.randint(0, 100000, size=100000)
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
                                    index=['d', 'c', 'b', 'a'],
                                    name='class', dtype='int64')
    expectedTestValCts = pd.Series([27030, 2673, 273, 26],
                                   index=['d', 'c', 'b', 'a'],
                                   name='class', dtype='int64')

    assert expectedTrainValCts.equals(dfTrain[classColumn].value_counts())
    assert expectedTestValCts.equals(dfTest[classColumn].value_counts())


def testSplitBalanceDataFrameByClasses():

    randState = np.random.RandomState(26)
    vals = randState.randint(0, 100000, size=100000)
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
                                    index=['a', 'b', 'c', 'd'],
                                    name='class', dtype='int64')
    expectedTestValCts = pd.Series([29733, 2940, 300, 29],
                                   index=['d', 'c', 'b', 'a'],
                                   name='class', dtype='int64')
    print(expectedTrainValCts)

    # Note: .value_counts() returns in descending count order. When the
    # counts are identical, as in the case for the training set (balanced
    # classes), the indices of the series are whatever the dict provides.
    # In order to make the comparison, apply .sort_index() to the result
    # when comparing to the hand-created expectedTrainValCts pd.Series.
    assert expectedTrainValCts.equals(dfTrain[classColumn].value_counts()
                                      .sort_index())
    assert expectedTestValCts.equals(dfTest[classColumn].value_counts())


def testMoveValidationSubsets():
    myClassDirs = ['a', 'b', 'c']
    myHeadDir = './moveDataTest'
    myTrainDir = 'train'
    myValidationDir = 'valid'
    myTestDir = 'test'
    myValidateFrac = 0.20
    myTestFrac = 0.20

    seed(32)
    head = Path(myHeadDir)
    print(head)
    if not head.is_dir():
        head.mkdir()

    train = head / myTrainDir
    print(train)
    if not train.is_dir():
        train.mkdir()

    for classDir in myClassDirs:
        classy = train / classDir
        print(classy)
        if not classy.is_dir():
            classy.mkdir()
        for i in range(100):
            f = classy / f"{i:02d}.jpg"
            f.write_text(f"{i:02d}")

    trainors, validators, testors = \
        moveValidationSubsets(myClassDirs,
                              headDir=myHeadDir,
                              trainDir=myTrainDir,
                              testDir=myTestDir,
                              validationDir=myValidationDir,
                              validateFrac=myValidateFrac,
                              testFrac=myTestFrac,
                              testOnly=False)
    print(trainors)
    print(validators)
    print(testors)
    expectedTrain = \
        OrderedDict({'a': ['96.jpg', '25.jpg', '70.jpg', '13.jpg', '81.jpg',
                           '45.jpg', '14.jpg', '50.jpg', '19.jpg', '89.jpg',
                           '90.jpg', '17.jpg', '76.jpg', '59.jpg', '79.jpg',
                           '80.jpg', '95.jpg', '22.jpg', '18.jpg', '23.jpg',
                           '55.jpg', '66.jpg', '63.jpg', '38.jpg', '78.jpg',
                           '09.jpg', '04.jpg', '62.jpg', '52.jpg', '36.jpg',
                           '49.jpg', '15.jpg', '85.jpg', '44.jpg', '46.jpg',
                           '11.jpg', '69.jpg', '93.jpg', '60.jpg', '39.jpg',
                           '47.jpg', '86.jpg', '97.jpg', '72.jpg', '82.jpg',
                           '48.jpg', '32.jpg', '41.jpg', '06.jpg', '12.jpg',
                           '98.jpg', '56.jpg', '40.jpg', '68.jpg'],
                     'b': ['31.jpg', '42.jpg', '08.jpg', '25.jpg', '33.jpg',
                           '45.jpg', '14.jpg', '50.jpg', '00.jpg', '64.jpg',
                           '89.jpg', '17.jpg', '76.jpg', '29.jpg', '61.jpg',
                           '34.jpg', '75.jpg', '77.jpg', '35.jpg', '74.jpg',
                           '27.jpg', '83.jpg', '80.jpg', '95.jpg', '23.jpg',
                           '55.jpg', '01.jpg', '66.jpg', '38.jpg', '67.jpg',
                           '04.jpg', '62.jpg', '52.jpg', '51.jpg', '07.jpg',
                           '94.jpg', '87.jpg', '11.jpg', '93.jpg', '88.jpg',
                           '21.jpg', '39.jpg', '47.jpg', '97.jpg', '72.jpg',
                           '91.jpg', '71.jpg', '82.jpg', '48.jpg', '41.jpg',
                           '06.jpg', '12.jpg', '10.jpg', '05.jpg', '03.jpg',
                           '43.jpg', '56.jpg', '30.jpg', '68.jpg'],
                     'c': ['31.jpg', '08.jpg', '96.jpg', '70.jpg', '33.jpg',
                           '13.jpg', '81.jpg', '00.jpg', '73.jpg', '19.jpg',
                           '02.jpg', '64.jpg', '89.jpg', '90.jpg', '76.jpg',
                           '61.jpg', '59.jpg', '84.jpg', '99.jpg', '75.jpg',
                           '77.jpg', '35.jpg', '27.jpg', '83.jpg', '80.jpg',
                           '22.jpg', '18.jpg', '55.jpg', '37.jpg', '92.jpg',
                           '01.jpg', '38.jpg', '67.jpg', '04.jpg', '62.jpg',
                           '24.jpg', '52.jpg', '51.jpg', '54.jpg', '07.jpg',
                           '87.jpg', '57.jpg', '15.jpg', '85.jpg', '11.jpg',
                           '69.jpg', '88.jpg', '39.jpg', '97.jpg', '26.jpg',
                           '53.jpg', '48.jpg', '32.jpg', '41.jpg', '98.jpg',
                           '05.jpg', '03.jpg', '16.jpg', '56.jpg', '40.jpg',
                           '30.jpg', '68.jpg']}
        )
    expectedValidate = \
        OrderedDict({'a': ['31.jpg', '33.jpg', '73.jpg', '02.jpg', '64.jpg',
                           '28.jpg', '61.jpg', '34.jpg', '84.jpg', '99.jpg',
                           '75.jpg', '77.jpg', '74.jpg', '83.jpg', '67.jpg',
                           '24.jpg', '20.jpg', '51.jpg', '94.jpg', '87.jpg',
                           '57.jpg', '21.jpg', '91.jpg', '71.jpg', '53.jpg',
                           '10.jpg', '03.jpg', '16.jpg'],
                     'b': ['70.jpg', '13.jpg', '81.jpg', '19.jpg', '28.jpg',
                           '84.jpg', '99.jpg', '79.jpg', '22.jpg', '18.jpg',
                           '63.jpg', '09.jpg', '24.jpg', '20.jpg', '57.jpg',
                           '15.jpg', '85.jpg', '60.jpg', '86.jpg', '32.jpg',
                           '65.jpg', '98.jpg', '16.jpg'],
                     'c': ['42.jpg', '14.jpg', '17.jpg', '28.jpg', '74.jpg',
                           '95.jpg', '23.jpg', '66.jpg', '63.jpg', '78.jpg',
                           '09.jpg', '20.jpg', '94.jpg', '49.jpg', '46.jpg',
                           '58.jpg', '60.jpg', '47.jpg', '86.jpg', '82.jpg',
                           '65.jpg', '06.jpg', '10.jpg', '43.jpg']}
        )

    expectedTest = \
        OrderedDict({'a': ['42.jpg', '08.jpg', '00.jpg', '29.jpg', '35.jpg',
                           '27.jpg', '37.jpg', '92.jpg', '01.jpg', '54.jpg',
                           '07.jpg', '88.jpg', '58.jpg', '26.jpg', '65.jpg',
                           '05.jpg', '43.jpg', '30.jpg'],
                     'b': ['96.jpg', '73.jpg', '02.jpg', '90.jpg', '59.jpg',
                           '37.jpg', '92.jpg', '78.jpg', '54.jpg', '36.jpg',
                           '49.jpg', '44.jpg', '46.jpg', '69.jpg', '58.jpg',
                           '26.jpg', '53.jpg', '40.jpg'],
                     'c': ['25.jpg', '45.jpg', '50.jpg', '29.jpg', '34.jpg',
                           '79.jpg', '36.jpg', '44.jpg', '93.jpg', '21.jpg',
                           '72.jpg', '91.jpg', '71.jpg', '12.jpg']}
        )

    system('rm -rf ' + myHeadDir)

    assert expectedTrain == trainors
    assert expectedValidate == validators
    assert expectedTest == testors


def testCreateGloVeDict():

    GloVeDir = '/home/wilber/work/GloVe'
    myGloVeDict = GloVeDict(GloVeDir, embeddingSz=50)

    expectedWomanVect = np.array([-1.8153e-01,  6.4827e-01, -5.8210e-01,
                                  -4.9451e-01,  1.5415e+00,  1.3450e+00,
                                  -4.3305e-01,  5.8059e-01,  3.5556e-01,
                                  -2.5184e-01,  2.0254e-01, -7.1643e-01,
                                  3.0610e-01,  5.6127e-01,  8.3928e-01,
                                  -3.8085e-01, -9.0875e-01,  4.3326e-01,
                                  -1.4436e-02,  2.3725e-01, -5.3799e-01,
                                  1.7773e+00, -6.6433e-02,  6.9795e-01,
                                  6.9291e-01, -2.6739e+00, -7.6805e-01,
                                  3.3929e-01,  1.9695e-01, -3.5245e-01,
                                  2.2920e+00, -2.7411e-01, -3.0169e-01,
                                  8.5286e-04,  1.6923e-01,  9.1433e-02,
                                  -2.3610e-02,  3.6236e-02,  3.4488e-01,
                                  -8.3947e-01, -2.5174e-01,  4.2123e-01,
                                  4.8616e-01,  2.2325e-02,  5.5760e-01,
                                  -8.5223e-01, -2.3073e-01, -1.3138e+00,
                                  4.8764e-01, -1.0467e-01], dtype='float32')

    assert np.array_equal(expectedWomanVect, myGloVeDict['woman'])

    myGloVeDict = GloVeDict(GloVeDir, embeddingSz=100)

    tmp = [0.30817,  0.30938,  0.52803, -0.92543, -0.73671,
           0.63475,  0.44197,  0.10262, -0.09142, -0.56607,
           -0.5327,  0.2013,  0.7704, -0.13983,  0.13727,
           1.1128,  0.89301, -0.17869, -0.0019722,  0.57289,
           0.59479,  0.50428, -0.28991, -1.3491,  0.42756,
           1.2748, -1.1613, -0.41084,  0.042804,  0.54866,
           0.18897,  0.3759,  0.58035,  0.66975,  0.81156,
           0.93864, -0.51005, -0.070079,  0.82819, -0.35346,
           0.21086, -0.24412, -0.16554, -0.78358, -0.48482,
           0.38968, -0.86356, -0.016391,  0.31984, -0.49246,
           -0.069363,  0.018869, -0.098286,  1.3126, -0.12116,
           -1.2399, -0.091429,  0.35294,  0.64645,  0.089642,
           0.70294,  1.1244,  0.38639,  0.52084,  0.98787,
           0.79952, -0.34625,  0.14095,  0.80167,  0.20987,
           -0.86007, -0.15308,  0.074523,  0.40816,  0.019208,
           0.51587, -0.34428, -0.24525, -0.77984,  0.27425,
           0.22418,  0.20164,  0.017431, -0.014697, -1.0235,
           -0.39695, -0.0056188,  0.30569,  0.31748,  0.021404,
           0.11837, -0.11319,  0.42456,  0.53405, -0.16717,
           -0.27185, -0.6255,  0.12883,  0.62529, -0.52086]
    expectedDogVect = np.array(tmp, dtype='float32')

    assert np.array_equal(expectedDogVect, myGloVeDict['dog'])

    myGloVeDict = GloVeDict(GloVeDir, embeddingSz=200)

    tmp = [3.4820e-01,  5.0612e-02,  4.6200e-01,  5.0187e-01,  1.0113e+00,
           2.7263e-01, -2.7686e-01, -3.7214e-01,  1.2201e-01, -4.5424e-01,
           -2.8875e-01,  2.3155e-01, -2.4348e-01,  4.5515e-01, -5.3815e-01,
           9.6373e-01,  1.4293e-01,  6.5552e-01, -5.6063e-01, -8.2300e-02,
           -2.8989e-01,  5.4887e-01, -6.0532e-01,  1.2149e-03,  8.7728e-01,
           -1.1679e-01,  6.1865e-02, -4.1655e-01,  2.1256e-01, -3.6832e-01,
           1.1676e-01,  8.7019e-01, -3.5418e-01, -6.3829e-01,  5.9640e-01,
           1.9156e-02,  2.0407e-01, -1.0453e+00,  1.9886e-01, -7.1201e-01,
           -2.4562e-01, -2.0316e-01, -6.7376e-01,  4.3171e-01,  1.8629e-01,
           1.0512e+00,  3.0116e-01, -4.5309e-02,  2.9385e-01,  2.0767e-02,
           -1.9536e-01, -5.9731e-01,  4.5437e-01,  1.2029e-01, -2.3031e-01,
           -3.8509e-01,  5.2826e-01, -1.2200e-01, -3.4997e-01, -3.6673e-02,
           -5.8293e-01, -7.0358e-02, -4.3221e-01, -5.1393e-01,  1.8583e-01,
           8.6244e-02,  1.2557e-01,  3.3545e-01, -4.2991e-01,  7.3322e-01,
           -4.7827e-01,  3.2281e-01, -4.4488e-01,  3.9233e-01,  1.3054e-01,
           -1.6376e-01,  1.2841e-01, -9.2026e-02, -3.2208e-01, -6.6586e-01,
           1.3258e-01,  1.4186e-01, -4.5162e-02,  2.1694e-01,  3.0942e-01,
           1.5047e-01, -5.4610e-01, -1.3289e-01,  5.3320e-01, -5.3975e-01,
           7.1144e-01, -5.4131e-02,  5.0530e-02,  7.4321e-01, -4.6833e-01,
           1.1119e-01,  7.1684e-01,  1.6587e-01,  2.7966e-01, -3.3668e-01,
           -3.2763e-01, -1.6562e-01,  5.8516e-02,  6.2019e-01, -2.4049e-01,
           -4.4339e-01,  1.7958e-01, -9.3943e-01, -2.9431e-01, -2.2581e-01,
           1.7766e-01, -7.6842e-01,  3.8649e-01, -3.7371e-01,  3.5115e-01,
           6.0244e-01, -6.9971e-01, -4.5115e-01,  2.3760e-01,  8.4735e-01,
           4.7702e-01,  6.6835e-02,  2.1270e-03,  1.3251e-01, -6.1201e-01,
           8.1657e-02,  4.9101e-01,  4.0888e-01, -2.1803e-02,  8.5172e-02,
           -3.9371e-01,  3.4054e-01,  5.2744e-01,  6.0344e-01, -3.1511e-01,
           -1.9968e-01, -2.4638e-01, -3.2337e-01,  5.5739e-02, -4.8721e-01,
           4.6538e-02,  1.7905e-01,  3.0219e-01, -3.1807e-01,  9.9946e-01,
           -5.3974e-02, -8.4791e-02, -4.3045e-01, -2.9209e-01,  2.2494e-01,
           -1.2301e+00,  4.2445e-01,  1.9698e-01,  2.1908e-01,  7.0144e-03,
           -3.2947e-01,  1.0969e+00,  4.5518e-03, -5.6118e-01, -3.7629e-01,
           -9.6898e-02,  1.7623e-01,  1.7840e-01,  1.4336e-01,  3.4675e-01,
           -6.4431e-01,  9.6800e-02, -2.1671e-01, -5.4083e-01,  2.6933e-01,
           3.4605e-01,  1.7484e-01,  4.4845e-01, -4.4811e-01,  1.2063e-01,
           -2.4531e-01, -1.3070e-01,  3.7554e-01,  4.7565e-01,  4.4303e-01,
           7.0039e-01, -5.9083e-01, -3.9620e-01,  4.4860e-01,  2.4397e-01,
           2.9852e-01,  1.7463e-01,  2.9816e-01,  1.7134e-01,  3.1868e-01,
           -8.1689e-01, -1.6444e-01,  8.3786e-01,  3.6278e-01, -2.1517e-01,
           -1.1054e+00,  7.2521e-02, -3.2189e-01,  4.4823e-01, -4.3356e-01]
    expectedHypothesisVect = np.array(tmp, dtype='float32')

    assert np.array_equal(expectedHypothesisVect, myGloVeDict['hypothesis'])


def testSparseUniq():

    a = np.array([[0, 5, 0, 0, 8, 0, 4, 0, 0, 8, 7, 0, 0, 0, 5],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [9, 2, 0, 2, 0, 0, 0, 0, 4, 0, 0, 9, 8, 2, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 8, 0, 0, 2, 2, 4, 0, 0, 0, 7, 8, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 7, 0, 9, 7, 0, 0, 0, 9, 12, 0, 0, 0, 0],
                  [9, 2, 0, 2, 0, 0, 0, 0, 4, 0, 0, 9, 8, 2, 2]])

    spM = sp.coo_matrix(a)
    origFormat = spM.getformat()
    print(f"spM's origFormat: {origFormat}")

    spU, inds = sparseUniq(spM)
    origFormat = spU.getformat()
    print(f"spU's origFormat: {origFormat}")

    spActual = spU.todense()
    spExpected = \
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 5, 0, 0, 8, 0, 4, 0, 0, 8, 7, 0, 0, 0, 5],
                  [0, 0, 7, 0, 9, 7, 0, 0, 0, 9, 12, 0, 0, 0, 0],
                  [0, 0, 0, 8, 0, 0, 2, 2, 4, 0, 0, 0, 7, 8, 0],
                  [9, 2, 0, 2, 0, 0, 0, 0, 4, 0, 0, 9, 8, 2, 2]])
    iExpected = np.array([1, 0, 7, 5, 2])

    assert np.array_equal(spActual, spExpected)
    assert np.array_equal(inds, iExpected)

    spU, inds = sparseUniq(spM, axis=1)

    spActual = spU.todense()
    spExpected = \
        np.array([[0, 0, 4, 0, 5, 0, 7, 0, 8, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 2, 0, 4, 2, 0, 0, 8, 0, 9],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [2, 8, 2, 4, 0, 0, 0, 7, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 7, 12, 0, 9, 0],
                  [0, 2, 0, 4, 2, 0, 0, 8, 0, 9]])
    iExpected = np.array([7, 3, 6, 8, 1, 2, 10, 12, 4, 0])

    assert np.array_equal(spActual, spExpected)
    assert np.array_equal(inds, iExpected)


@timeUsage
def sleeper(seconds):
    sleep(seconds)


def testTimeUsage(capsys):
    seconds = 2.15
    sleeper(seconds)
    captured = capsys.readouterr()
    assert captured.out == "Δt:  2.15s.\n"
