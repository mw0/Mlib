import timeit
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

def timeUsage(func):
    """
    INPUT:
        func	obj, function you want timed

    This is a decorator that prints out the duration of execution for a
    function. Formats differ for cases < 1 min, < 1 hour and >= 1 hour.
    """

    def wrapper(*args, **kwargs):
        t0 = timeit.default_timer()

        retval= func(*args, **kwargs)

        t1 = timeit.default_timer()
        Δt = t1 - t0
        if Δt > 86400.0:
            print(f"Δt: {Δt//86400}d, {int((Δt % 86400)//3600)}h, {int((Δt % 3600)//60)}m, {Δt % 60.0:4.1f}s.")
        elif Δt > 3600.0:
            print(f"Δt: {int(Δt//3600)}h, {int((Δt % 3600)//60)}m, {Δt % 60.0:4.1f}s.")
        elif Δt > 60.0:
            print(f"Δt: {int(Δt//60)}m, {Δt % 60.0:4.1f}s.")
        else:
            print(f"Δt: {Δt % 60.0:5.2f}s.")
        return retval

    return wrapper


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
        dfLabel = df[df[classColumn]==label]
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
        dfLabel = df[df[classColumn]==label]
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
                dfTrain = pd.concat([dfTrain,
                                     dfLabelTr.sample(n=targetClassSize,
                                                  replace=True,
                                                  random_state=myRandomState)])
            elif ct > targetClassSize:
                dfTrain = pd.concat([dfTrain,
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
