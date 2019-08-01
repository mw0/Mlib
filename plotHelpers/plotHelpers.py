#!/bin/python3

# from sklearn.metrics import confusion_matrix
import timeit
import numpy as np
import pandas as pd
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
    """
    INPUTS:
        confustionMat	np.array (square, floats), containing 
        xlabels		list (type=str), containing labels for predict
        ylabels		list (type=str), containing labels for actual
        type		str, in ['counts', 'recall', 'precision'], indicating
                        whether to plot raw counts, normalized along predicted,
                        normalized along actual
        titleText	str, title for plot
        ax		optional matplotlib.axis object, default: None
        saveAs	str, in ['pdf', 'png', 'svg']

    Creates heatmap representing confusion matrix passed via confusionMat. When
    type == 'recall', normalization across predicted values ensures that
    diagonal elements represent recall for each class, and 'precision'
    normalizes across actual values so that diagonal elements represent class
    precisions. (For recall and precision, max values are 1.0.)
    """

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
def plotValueCounts(df, colName, barWidth=0.9, figSz=(16.0, 10.0),
                    xrot=65.0, titleText=None, ax=None, saveAs=None):
    """
    INPUTS:
        df	        Pandas DataFrame
        colName	        str, column whose item counts are to be histogrammed.
        barWidth        float, fractional width of histogram bars (1.0 for no
                        gaps), default: 0.90
        figSz	        tuple (type=float), size of figure in inches, default:
                        (16.0, 10.0)
        xrot	        float, angle by which column values are rotated
                        (x-axis), default: 65.0
        titleText	str, title for plot
        ax		optional matplotlib.axis object, default: None
        saveAs	str, in ['pdf', 'png', 'svg']
    """

    classCts = pd.DataFrame(df[colName].value_counts())

    if ax is None:
        ax = classCts.plot(kind='bar', width=barWidth, figsize=figSz, rot=xrot)
    else:
        classCts.plot(kind='bar', width=barWidth, figsize=figSz,
                      rot=xrot, ax=ax)

    rects = ax.patches

    # Number of points between bar and label. Change to your liking.
    spac=e = 5

    # Vertical alignment for positive values
    va = 'bottom'

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = "top"
        else:
            va = "bottom"

        # Use Y value as label and format number with one decimal place
        label = "{:d}".format(y_value)

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            rotation=90.0,
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha="center",                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

    ax.set_ylim([0.0, 187500.0])
    if titleText is not None:
        ax.set_title(titleText)
        fileNameAugmentString = "".join([w.lstrip('(').rstrip(')')
                                         .capitalize() for w in \
                                         titleText.split(" ")])
    else:
        fileNameAugmentString = ""

    if saveAs == 'pdf':
        plt.savefig("".join([colName + ' frequencies', name,
                             fileNameAugmentString, '.pdf']))
    elif saveAs == 'png':
        plt.savefig("".join([colName + ' frequencies', name,
                             fileNameAugmentString, '.png']))
    elif saveAs == 'svg':
        plt.savefig("".join([colName + ' frequencies', name,
                             fileNameAugmentString, '.svg']))

    return
