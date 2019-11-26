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
            f = classy / f"{classDir}-{i:02d}.jpg"
            f.write_text(f"{classDir}-{i:02d}")

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
        OrderedDict({'a': ['a-18.jpg', 'a-73.jpg', 'a-84.jpg', 'a-26.jpg',
                           'a-27.jpg', 'a-32.jpg', 'a-07.jpg', 'a-30.jpg',
                           'a-19.jpg', 'a-67.jpg', 'a-42.jpg', 'a-90.jpg',
                           'a-60.jpg', 'a-15.jpg', 'a-99.jpg', 'a-44.jpg',
                           'a-57.jpg', 'a-28.jpg', 'a-17.jpg', 'a-93.jpg',
                           'a-95.jpg', 'a-24.jpg', 'a-52.jpg', 'a-65.jpg',
                           'a-45.jpg', 'a-10.jpg', 'a-75.jpg', 'a-79.jpg',
                           'a-11.jpg', 'a-59.jpg', 'a-09.jpg', 'a-14.jpg',
                           'a-54.jpg', 'a-74.jpg', 'a-20.jpg', 'a-62.jpg',
                           'a-69.jpg', 'a-70.jpg', 'a-37.jpg', 'a-25.jpg',
                           'a-04.jpg', 'a-33.jpg', 'a-48.jpg', 'a-43.jpg',
                           'a-36.jpg', 'a-56.jpg', 'a-72.jpg', 'a-49.jpg',
                           'a-63.jpg', 'a-64.jpg', 'a-39.jpg', 'a-85.jpg',
                           'a-94.jpg', 'a-80.jpg', 'a-13.jpg', 'a-53.jpg',
                           'a-12.jpg', 'a-50.jpg', 'a-81.jpg', 'a-00.jpg'],
                     'b': ['b-55.jpg', 'b-92.jpg', 'b-47.jpg', 'b-65.jpg',
                           'b-91.jpg', 'b-33.jpg', 'b-98.jpg', 'b-76.jpg',
                           'b-57.jpg', 'b-43.jpg', 'b-70.jpg', 'b-05.jpg',
                           'b-73.jpg', 'b-97.jpg', 'b-06.jpg', 'b-01.jpg',
                           'b-40.jpg', 'b-30.jpg', 'b-25.jpg', 'b-08.jpg',
                           'b-54.jpg', 'b-38.jpg', 'b-94.jpg', 'b-88.jpg',
                           'b-86.jpg', 'b-31.jpg', 'b-66.jpg', 'b-16.jpg',
                           'b-11.jpg', 'b-13.jpg', 'b-74.jpg', 'b-67.jpg',
                           'b-77.jpg', 'b-95.jpg', 'b-04.jpg', 'b-36.jpg',
                           'b-10.jpg', 'b-85.jpg', 'b-78.jpg', 'b-03.jpg',
                           'b-64.jpg', 'b-37.jpg', 'b-60.jpg', 'b-72.jpg',
                           'b-02.jpg', 'b-20.jpg', 'b-58.jpg', 'b-19.jpg',
                           'b-35.jpg', 'b-69.jpg', 'b-56.jpg', 'b-22.jpg',
                           'b-18.jpg', 'b-68.jpg', 'b-39.jpg', 'b-59.jpg',
                           'b-17.jpg', 'b-93.jpg', 'b-32.jpg', 'b-90.jpg'],
                     'c': ['c-08.jpg', 'c-92.jpg', 'c-07.jpg', 'c-55.jpg',
                           'c-26.jpg', 'c-33.jpg', 'c-95.jpg', 'c-37.jpg',
                           'c-10.jpg', 'c-69.jpg', 'c-16.jpg', 'c-41.jpg',
                           'c-31.jpg', 'c-24.jpg', 'c-56.jpg', 'c-59.jpg',
                           'c-57.jpg', 'c-18.jpg', 'c-49.jpg', 'c-70.jpg',
                           'c-01.jpg', 'c-43.jpg', 'c-84.jpg', 'c-42.jpg',
                           'c-96.jpg', 'c-51.jpg', 'c-94.jpg', 'c-15.jpg',
                           'c-06.jpg', 'c-45.jpg', 'c-68.jpg', 'c-00.jpg',
                           'c-98.jpg', 'c-11.jpg', 'c-38.jpg', 'c-28.jpg',
                           'c-91.jpg', 'c-90.jpg', 'c-71.jpg', 'c-83.jpg',
                           'c-81.jpg', 'c-22.jpg', 'c-35.jpg', 'c-53.jpg',
                           'c-75.jpg', 'c-46.jpg', 'c-50.jpg', 'c-72.jpg',
                           'c-30.jpg', 'c-86.jpg', 'c-99.jpg', 'c-65.jpg',
                           'c-76.jpg', 'c-32.jpg', 'c-34.jpg', 'c-64.jpg',
                           'c-61.jpg', 'c-09.jpg', 'c-12.jpg', 'c-27.jpg']
        })

    expectedValidate = \
        OrderedDict({'a': ['a-71.jpg', 'a-98.jpg', 'a-82.jpg', 'a-38.jpg',
                           'a-58.jpg', 'a-51.jpg', 'a-68.jpg', 'a-46.jpg',
                           'a-87.jpg', 'a-92.jpg', 'a-78.jpg', 'a-21.jpg',
                           'a-29.jpg', 'a-34.jpg', 'a-97.jpg', 'a-40.jpg',
                           'a-88.jpg', 'a-01.jpg', 'a-23.jpg', 'a-02.jpg'],
                     'b': ['b-62.jpg', 'b-96.jpg', 'b-07.jpg', 'b-53.jpg',
                           'b-42.jpg', 'b-23.jpg', 'b-50.jpg', 'b-12.jpg',
                           'b-79.jpg', 'b-52.jpg', 'b-00.jpg', 'b-15.jpg',
                           'b-84.jpg', 'b-89.jpg', 'b-87.jpg', 'b-46.jpg',
                           'b-71.jpg', 'b-21.jpg', 'b-75.jpg', 'b-29.jpg'],
                     'c': ['c-14.jpg', 'c-66.jpg', 'c-40.jpg', 'c-25.jpg',
                           'c-13.jpg', 'c-19.jpg', 'c-54.jpg', 'c-47.jpg',
                           'c-79.jpg', 'c-20.jpg', 'c-88.jpg', 'c-63.jpg',
                           'c-82.jpg', 'c-36.jpg', 'c-03.jpg', 'c-44.jpg',
                           'c-93.jpg', 'c-89.jpg', 'c-73.jpg', 'c-52.jpg']
        })

    expectedTest = \
        OrderedDict({'a': ['a-31.jpg', 'a-22.jpg', 'a-76.jpg', 'a-35.jpg',
                           'a-05.jpg', 'a-41.jpg', 'a-66.jpg', 'a-55.jpg',
                           'a-86.jpg', 'a-16.jpg', 'a-91.jpg', 'a-47.jpg',
                           'a-03.jpg', 'a-89.jpg', 'a-83.jpg', 'a-61.jpg',
                           'a-77.jpg', 'a-06.jpg', 'a-96.jpg', 'a-08.jpg'],
                     'b': ['b-49.jpg', 'b-82.jpg', 'b-41.jpg', 'b-44.jpg',
                           'b-34.jpg', 'b-83.jpg', 'b-26.jpg', 'b-99.jpg',
                           'b-80.jpg', 'b-45.jpg', 'b-28.jpg', 'b-61.jpg',
                           'b-27.jpg', 'b-48.jpg', 'b-81.jpg', 'b-63.jpg',
                           'b-14.jpg', 'b-51.jpg', 'b-09.jpg', 'b-24.jpg'],
                     'c': ['c-67.jpg', 'c-48.jpg', 'c-21.jpg', 'c-78.jpg',
                           'c-87.jpg', 'c-23.jpg', 'c-74.jpg', 'c-60.jpg',
                           'c-39.jpg', 'c-02.jpg', 'c-58.jpg', 'c-97.jpg',
                           'c-17.jpg', 'c-29.jpg', 'c-85.jpg', 'c-62.jpg',
                           'c-05.jpg', 'c-80.jpg', 'c-04.jpg', 'c-77.jpg']
        })

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
