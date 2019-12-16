import timeit
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from pathlib import Path
from random import random, shuffle
from collections import OrderedDict
from functools import wraps


# Decorator for timing functions.
def timeUsage(func):
    """
    INPUT:
        func	obj, function you want timed

    This is a decorator that prints out the duration of execution for a
    function. Formats differ for cases < 1 min, < 1 hour and >= 1 hour.
    """

    @wraps(func)
    def _timeUsage(*args, **kwds):

        t0 = timeit.default_timer()

        retval = func(*args, **kwds)

        t1 = timeit.default_timer()
        Δt = t1 - t0
        if Δt > 86400.0:
            print(f"Δt: {Δt//86400}d, {int((Δt % 86400)//3600)}h, "
                  f"{int((Δt % 3600)//60)}m, {Δt % 60.0:4.1f}s.")
        elif Δt > 3600.0:
            print(f"Δt: {int(Δt//3600)}h, {int((Δt % 3600)//60)}m, "
                  f"{Δt % 60.0:4.1f}s.")
        elif Δt > 60.0:
            print(f"Δt: {int(Δt//60)}m, {Δt % 60.0:4.1f}s.")
        else:
            print(f"Δt: {Δt % 60.0:5.2f}s.")
        return retval

    return _timeUsage


@timeUsage
def moveValidationSubsets(classDirs, headDir='./data', trainDir='train',
                          validationDir='validation', testDir=None,
                          validateFrac=0.20, testFrac=None, fileSuffix='jpg',
                          testOnly=False):
    """
    INPUT:
        classDirs	list(type=str), list of sub-directories, one for each
                        class type, into which train and validation data of the
                        particular class will be put.
        headDir		str, path to directory containing all training and
                        test data, default: './data'
        trainDir	str, default: 'train'
        testDir		str, default: None
        validationDir	str, default: 'validation'
        validateFrac	float, fraction of files to be moved to validation
                        subdirectories, default: 0.20
        testFrac	float, fraction of files to be moved to validation
                        subdirectories, default: None. If not None, testDir
                        must also be set to a str != None.
        fileSuffix	str, indicating type of image file, default: 'jpg'
        testOnly	bool, set True if only want to generate names of files
                        that would be moved into each validation sub-directory.
                        No files will actually be moved. Default: False

    RETURNS:
	trainors	list(type=str) list of list of files remaining in
                        training sub-directories.
	validators	list(type=str) list of list of files remaining in
                        validation sub-directories.

    This routine assumes that all train data have been put into sub-directories
    indicated by classDirs, under {headDir}/{trainDir}.
    Randomly selects a proportion validationFrac of files from each
    {trainDir}/{classDir} and puts them into the corresponding
    {headDir}/{validationDir}/{classDir}. If testFrac is not None, will put
    testFrac of files from each {trainDir}/{classDir} into the corresponding
    {headDir}/{testDir}/{classDir}.
    """

    if (testFrac is not None) and (testFrac > 0.0) and testDir is None:
        raise ValueError(f"You have testFrac: {testFrac}.\nWhen not None, "
                         "you must specify a non-None testDir value.")

    head = Path(headDir)
    train = head / trainDir
    validate = head / validationDir
    if not validate.is_dir():
        validate.mkdir()
    if testFrac is not None:
        testing = head / testDir
        if not testing.is_dir():
            testing.mkdir()

    trainors = OrderedDict()
    validators = OrderedDict()
    if testFrac is not None:
        testors = OrderedDict()

    # If validationDir / classDir not empty, put the files back into
    # trainDir / classDir, so that the random selections & moves are done from
    # scratch.

    mvCts = {}
    for classDir in classDirs:
        trainSubdir = train / classDir
        validateSubdir = validate / classDir
        filePaths = list(validateSubdir.glob('*.' + fileSuffix))
        totCt = len(filePaths)
        if totCt > 0:
            print(f"Moving {totCt} files from {validateSubdir}"
                  f" to {trainSubdir}.")
            for path in filePaths:
                file = path.name
                if mvCts.get(classDir, None) is None:
                    mvCts[classDir] = 1
                else:
                    mvCts[classDir] += 1
                destPath = trainSubdir / file
                path.rename(destPath)

    print(mvCts.keys())
    for className in mvCts.keys():
        trainSubdir = train / classDir
        validateSubdir = validate / classDir
        print(f"Moved {mvCts[className]} files from {validateSubdir} to "
              f"{trainSubdir}.")

    if testFrac is not None:
        mvCts = {}
        for classDir in classDirs:
            trainSubdir = train / classDir
            testSubdir = testing / classDir
            filePaths = list(testSubdir.glob('*.' + fileSuffix))
            totCt = len(filePaths)
            if totCt > 0:
                print(f"Moving {totCt} files from {testSubdir}"
                      f" to {trainSubdir}.")
                for path in filePaths:
                    # print(f"path: {path}")
                    file = path.name
                    if mvCts.get(classDir, None) is None:
                        mvCts[classDir] = 1
                    else:
                        mvCts[classDir] += 1
                    destPath = trainSubdir / file
                    path.rename(destPath)

        print(mvCts.keys())
        for className in mvCts.keys():
            trainSubdir = train / classDir
            testSubdir = testing / classDir
            print(f"Moved {mvCts[className]} files from {testSubdir} to "
                  f"{trainSubdir}.")

    for classDir in classDirs:
        trainSubdir = train / classDir
        totCt = len(list(trainSubdir.glob('*.' + fileSuffix)))
        if totCt <= 0:
            raise Exception(f"No files in {trainSubdir}; perhaps you haven't"
                  " yet moved your files there?")

        validationSubdir = validate / classDir
        if not validationSubdir.is_dir():
            print(f"Path {validationSubdir} does not exist -- creating ...")
            validationSubdir.mkdir()
            if not validationSubdir.is_dir():
                raise Exception(f"Unable to create {validationSubdir}.")
        validationSubdirCt = len(list(validationSubdir.glob('*.' + fileSuffix)))
        if validationSubdirCt > 0:
            raise Exception(f"There are already {validationSubdirCt} files in "
                        f"{validationSubdir}. You should figure out why before"
                        " attempting to re-run this.")

        if testFrac is not None:
            testSubdir = testing / classDir
            if not testSubdir.is_dir():
                print(f"Path {testSubdir} does not exist -- creating ...")
                testSubdir.mkdir()
                if not testSubdir.is_dir():
                    raise Exception(f"Unable to create {testSubdir}.")
            testSubdirCt = len(list(testSubdir.glob('*.' + fileSuffix)))
            if testSubdirCt > 0:
                print(list(testSubdir.glob('*.' + fileSuffix)))
                raise Exception(f"There are already {testSubdirCt} files in "
                                f"{testSubdir}. You should figure out why"
                                " before attempting to re-run this.")

        trainPaths = trainSubdir.glob('*.' + fileSuffix)
        trainList = list(trainPaths)
        trainCt = len(trainList)
        shuffle(trainList)

        validateLimit = round(trainCt*validateFrac)
        validateList = trainList[:validateLimit]
        print(f"validateList = trainList[:{validateLimit}]")
        print("validateList:\n", sorted([Path(t).name for t in validateList]))

        for filePath in validateList:
            path = Path(filePath)
            file = path.name
            if path.stat().st_size == 0:
                path.remove_p()
            else:
                if validators.get(classDir, None) is None:
                    validators[classDir] = [file]
                else:
                    validators[classDir].append(file)
                if testOnly is False:
                    destPath = validationSubdir / file
                    path.rename(destPath)

        if testFrac is None:
            trainLimit = round(trainCt*validateFrac)
            newTrainList = trainList[trainLimit:]
            print(f"newTrainList = trainList[{trainLimit}:]")
            print("newTrainList:\n",
                  sorted([Path(t).name for t in newTrainList]))
        else:
            testLimit = round(trainCt*(validateFrac + testFrac))
            testList = trainList[validateLimit: testLimit]
            print(f"testList = trainList[{validateLimit}: {testLimit}]")
            print("testList:\n", sorted([Path(t).name for t in testList]))
            newTrainList = trainList[testLimit:]
            print(f"newTrainList = trainList[{testLimit}:]")
            print("newTrainList:\n",
                  sorted([Path(t).name for t in newTrainList]))

            for filePath in testList:
                path = Path(filePath)
                file = path.name
                if path.stat().st_size == 0:
                    path.remove_p()
                else:
                    if testors.get(classDir, None) is None:
                        testors[classDir] = [file]
                    else:
                        testors[classDir].append(file)
                    if testOnly is False:
                        destPath = testSubdir / file
                        path.rename(destPath)

        for filePath in newTrainList:
            path = Path(filePath)
            file = path.name
            if path.stat().st_size == 0:
                path.remove_p()
            else:
                if trainors.get(classDir, None) is None:
                    trainors[classDir] = [file]
                else:
                    trainors[classDir].append(file)

    if testFrac is not None:
        return trainors, validators, testors
    else:
        return trainors, validators


@timeUsage
def splitDataFrameByClasses(df, classColumn, testFrac=0.33, volubility=1,
                            randomizeResult=True, myRandomState=None):
    """
    Conducts train/test splits of df separately for each class, and then
    concatenates them together.

    Returns dfTrain, dfTest.
    """

    if volubility > 0:
        print(f"df.shape: {df.shape}")

    labels = list(set(df[classColumn]))
    if volubility > 1:
        print(f"labels: {labels}")

    # If not passed, create a RandomState object to feed into each
    # randomizing call:
    if myRandomState is None:
        myRandomState = np.random.RandomState(21)
    elif isinstance(myRandomState, int):
        myRandomState = np.random.RandomState(myRandomState)

    for i, label in enumerate(labels):
        dfLabel = df[df[classColumn] == label]
        if volubility > 1:
            print(f"dfLabel.shape: {dfLabel.shape}")

        dfLabelTrain, dfLabelTest = \
            train_test_split(dfLabel, test_size=testFrac,
                             random_state=myRandomState)
        if volubility > 1:
            print(f"dfLabelTr.shape: {dfLabelTr.shape}"
                  f"\tdfLabelTe.shape: {dfLabelTe.shape}")

        if i == 0:
            dfTest = dfLabelTest
            dfTrain = dfLabelTrain
        else:
            dfTest = pd.concat([dfTest, dfLabelTest])
            dfTrain = pd.concat([dfTrain, dfLabelTrain])

    if volubility > 0:
        print(f"dfTrain.shape: {dfTrain.shape}\tdfTest.shape: {dfTest.shape}")

    if randomizeResult:
        dfTrain = dfTrain.sample(frac=1, random_state=myRandomState)\
                         .reset_index(drop=True)
        dfTest = dfTest.sample(frac=1, random_state=myRandomState)\
                       .reset_index(drop=True)
    return dfTrain, dfTest


@timeUsage
def splitBalanceDataFrameByClasses(df, classColumn, targetClassSize,
                                   testFrac=0.33, randomizeResult=True,
                                   myRandomState=None, volubility=1):
    """
    Conducts train/test splits of df separately for each class, then balances
    the classes of the training splits, and concatentates together. Balancing
    is done by sampling with replacement when the training set for a class
    is < targetClassSize, and sampling without replacement when it
    is > targetClassSize.

    Returns dfTr, dfTe, where the testFrac ratio corresponds to the splits
    prior to balancing dfTr classes.
    """

    if volubility > 0:
        print(f"df.shape: {df.shape}")

    labels = list(set(df[classColumn]))
    if volubility > 1:
        print(f"labels: {labels}")

    # If not passed, create a RandomState object to feed into each randomizing
    # call:
    if myRandomState is None:
        myRandomState = np.random.RandomState(21)
    elif isinstance(myRandomState, int):
        myRandomState = np.random.RandomState(myRandomState)

    for i, label in enumerate(labels):
        dfLabel = df[df[classColumn] == label]
        if volubility > 1:
            print(f"dfLabel.shape: {dfLabel.shape}")

        dfLabelTr, dfLabelTe = \
            train_test_split(dfLabel, test_size=testFrac,
                             random_state=myRandomState)
        if volubility > 1:
            print(f"dfLabelTr.shape: {dfLabelTr.shape}\tdfLabelTe.shape: "
                  f"{dfLabelTe.shape}")

        ct = dfLabelTe.shape[0]
        if i == 0:
            dfTest = dfLabelTe
            if ct < targetClassSize:
                dfTrain = dfLabelTr.sample(n=targetClassSize, replace=True,
                                           random_state=myRandomState)
            elif ct > targetClassSize:
                dfTrain = dfLabelTr.sample(n=targetClassSize, replace=False,
                                           random_state=myRandomState)
            else:
                dfTrain = dfLabelTr
        else:
            dfTest = pd.concat([dfTest, dfLabelTe])
            if ct < targetClassSize:
                dfTrain = \
                    pd.concat([dfTrain,
                               dfLabelTr.sample(n=targetClassSize,
                                                replace=True,
                                                random_state=myRandomState)])
            elif ct > targetClassSize:
                dfTrain = \
                    pd.concat([dfTrain,
                               dfLabelTr.sample(n=targetClassSize,
                                                replace=False,
                                                random_state=myRandomState)])
            else:
                dfTrain = pd.concat([dfTrain, dfLabelTr])

    if volubility > 1:
        print(f"type(dfTrain): {type(dfTrain)}"
              f"\ttype(dfTest): {type(dfTest)}")
    if volubility > 0:
        print(f"dfTrain.shape: {dfTrain.shape}"
              f"\tdfTest.shape: {dfTest.shape}")

    if randomizeResult:
        dfTrain = dfTrain.sample(frac=1, random_state=myRandomState)\
                         .reset_index(drop=True)
        dfTest = dfTest.sample(frac=1, random_state=myRandomState)\
                       .reset_index(drop=True)
    return dfTrain, dfTest


@timeUsage
def GloVeDict(GloVeDir, embeddingSz=200):
    """
    INPUT:
        GloVeDir	str, path to GloVe embedding data
        embeddingSz	int, one of [50, 100, 200, 300], the dimension of
                        the embedding vectors you want returned.

    Returns a dict containing GloVe word embedding vectors of specified
    dimensions, indexed by un-cased words, as computed by Stanford from the
    Wikipedia 2014 data set. If a corresponding pickle file is found, it will
    simply load and return the dict. Otherwise, it will extract the embeddings
    from the source file, create the dict and save the pickled dict, before
    returning the dict.
    """

    if embeddingSz not in [50, 100, 200, 300]:
        raise ValueError(f"You supplied embeddingSz: {embeddingSz}, but it"
                         " must be one of [50, 100, 200, 300].")

    GloVeFile = Path(GloVeDir) / f"GloVe.6B.{embeddingSz:03d}.pkl"
    if GloVeFile.exists():
        print(f"Loading GloVe vectors from {GloVeFile} ...")
        with open(GloVeFile, 'rb') as pickledGloVe:
            return pickle.load(pickledGloVe)
    else:
        RawGloVeFile = Path(GloVeDir) / f"glove.6B.{embeddingSz:03d}d.txt"
        print(f"{GloVeFile} does not (yet) exist.")
        print(RawGloVeFile)
        if RawGloVeFile.exists():
            print("Reading GloVe vectors from raw text file, "
                  f"{RawGloVeFile} ...")
            GloVeDict = {}
            GloVeVects = RawGloVeFile.open(encoding='utf-8')
            for l in GloVeVects:
                parts = l.split()
                word = parts[0]
                coeffs = np.asarray(parts[1:], dtype='float32')
                GloVeDict[word] = coeffs
            GloVeVects.close()

            print(f"vocab size: {len(GloVeDict.keys())}.")

            print(f"Saving embeddings dict to {GloVeFile}.")
            pFile = open(GloVeFile, 'wb')
            pickle.dump(GloVeDict, pFile, -1)
            pFile.close()

            return GloVeDict
        else:
            raise FileNotFoundError("Can't extract GloVe vectors from non-"
                                    f"existent raw text file, {RawGloVeFile}.")
            return 1


def sparseUniq(spMatrix, axis=0):
    '''
    INPUT:
        spMatrix	spMatrix, a sparse matrix
        axis		int (0 or 1), indicating the dimension to be extracted,
                        with 0 for returning rows (and their indices), and 1
                        for returning columns, default: 0

    RETURNS:
        spUniq		sparse matrix, containing only unique rows or columns
        inds		list of corresponding indices

    Refer to https://stackoverflow.com/questions/46126840/.
    ./get-unique-rows-from-a-scipy-sparse-matrix#52891452

    Extracts sparse matrix containing only unique rows (axis=0) or columns
    (axis=1) of spMatrix.
    Also, returns corresponding indices of the unique rows or columns.

    The returned indices can be helpful for slicing paired arrays, when
    you want to extract the corresponding rows/columns.
    '''

    if axis == 1:
        spMatrix = spMatrix.T

    origFormat = spMatrix.getformat()
    # print(f"origFormat: {origFormat}")
    dt = np.dtype(spMatrix)
    ncols = spMatrix.shape[1]

    if origFormat != 'lil':
        spMatrix = spMatrix.tolil()

    _, inds = np.unique(spMatrix.data + spMatrix.rows, return_index=True)
    rows = spMatrix.rows[inds]
    data = spMatrix.data[inds]
    nrows_uniq = data.shape[0]

    # spMatrix.resize(nrows_uniq, ncols)
    spMatrix = sp.lil_matrix((nrows_uniq, ncols), dtype=dt)
    spMatrix.data = data
    spMatrix.rows = rows

    spUniq = spMatrix.asformat(origFormat)
    if axis == 1:
        spUniq = spUniq.T

    return spUniq, inds
