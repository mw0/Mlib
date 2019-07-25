#!/bin/python3

# from sklearn.metrics import confusion_matrix
import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

### decorator for timing functions.
def timeUsage(func):

    def wrapper(*args, **kwargs):
        t0 = timeit.default_timer()
        retval= func(*args, **kwargs)
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

    return wrapper 


@timeUsage
def plotConfusionMatrix(confusionMat, xlabels=None, ylabels=None,
                        type='counts', titleText=None, ax=None,
                        saveAs=None):

    if xlabels is None:
        xlabels = list(range(len(confusionMat) + 1))
    if ylabels is None:
        ylabels = list(range(len(confusionMat) + 1))

    if type == 'counts':
        name = 'Counts'
        fmtType = 'd'
        confusionMatNorm = confusionMat.copy()
    elif type == 'recall':
        name = 'Recall'
        fmtType = '0.2f'
        confusionMatNorm = (confusionMat.astype('float')/
                            confusionMat.sum(axis=1)[:, np.newaxis])
    elif type == 'precision':
        name = 'Precision'
        fmtType = '0.2f'
        confusionMatNorm = (confusionMat.astype('float')/
                            confusionMat.sum(axis=0)[np.newaxis, :])

    if titleText is not None:
        fileNameAugmentString = "".join([w.lstrip('(').rstrip(')')
                                         .capitalize() for w in \
                                         titleText.split(" ")])
    else:
        fileNameAugmentString = ""

    if ax is None:
        fig, ax = plt.subplots(figsize=(30,25))
    sns.heatmap(confusionMatNorm, annot=True, fmt=fmtType, cmap=cc.cm.rainbow,
                xticklabels=xlabels, yticklabels=ylabels)
    plt.ylabel('Actual', fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    plt.title(", ".join(['Confusion matrix', name, titleText]), fontsize=18)
    if saveAs == 'pdf':
        plt.savefig("".join(['ConfusionMatrix', name,
                             fileNameAugmentString, '.pdf']))
    elif saveAs == 'png':
        plt.savefig("".join(['ConfusionMatrix', name,
                             fileNameAugmentString, '.png']))
    elif saveAs == 'svg':
        plt.savefig("".join(['ConfusionMatrix', name,
                             fileNameAugmentString, '.svg']))

    return


@timeUsage
def splitBalancedDataFrameClasses(df, classColumn, targetClassSize,
                                  testFrac=0.33, randomizeResult=True,
                                  randomState=None, volubility=1):
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

    # Create a RandomState object to feed into each randomizing call:
    myRandomState = np.random.RandomState(randomState)

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
            dfTe = dfLabelTe
            if ct < targetClassSize:
                dfTr = dfLabelTr.sample(n=targetClassSize, replace=True,
                                        random_state=myRandomState)
            elif ct > targetClassSize:
                dfTr = dfLabelTr.sample(n=targetClassSize, replace=False,
                                        random_state=myRandomState)
            else:
                dfTr = dfLabelTr
        else:
            dfTe = pd.concat([dfTe, dfLabelTe])
            if ct < targetClassSize:
                dfTr = pd.concat([dfTr,
                                  dfLabelTr.sample(n=targetClassSize,
                                                   replace=True,
                                                   random_state=myRandomState)])
            elif ct > targetClassSize:
                dfTr = pd.concat([dfTr,
                                  dfLabelTr.sample(n=targetClassSize,
                                                   replace=False,
                                                   random_state=myRandomState)])
            else:
                dfTr = pd.concat([dfTr, dfLabelTr])

    if volubility > 1:
        print(f"type(dfTr): {type(dfTr)}\ttype(dfTe): {type(dfTe)}")
    if volubility > 0:
        print(f"dfTr.shape: {dfTr.shape}\tdfTe.shape: {dfTe.shape}")

    if randomizeResult:
        dfTr = dfTr.sample(frac=1, random_state=myRandomState).reset_index(drop=True)
        dfTe = dfTe.sample(frac=1, random_state=myRandomState).reset_index(drop=True)
    return dfTr, dfTe
