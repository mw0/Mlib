from dynamiclist import dynamiclist
import numpy as np


def testDynamicList():

    l = dynamiclist()
    l[20][1] = 10
    l[21][1] = 20
    print(l)
    expected = [[],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [[], 10],
                [[], 20]
               ]

    assert np.array_equal(l, expected)

    myList = dynamiclist()
    myList[2][1][1] = 13
    myList[2][2][3] = 24
    myList[4][2][2] = 14
    print(myList)
    expected = [[],
                [],
                [[],
                 [[], 13],
                 [[], [], [], 24]
                ],
                [],
                [[],
                 [],
                 [[], [], 14]]
               ]
    print(np.shape(myList))
    assert np.shape(myList) == np.shape(expected)
    assert np.array_equal(myList, expected)
