import numpy as np
import matplotlib.pyplot as plt


def plotLearningCurves(history, figsize=(12, 6)):
    """
    INPUT:
        history		dict, containing metric values at each epoch, obtained
                        as history instance variable from keras model.fit().
        figsize		tuple, size of figure (two panels plotted
                        side-by-side), default: (12, 6)

    Plots learning curves for loss and accuracy in side-by-side panels. If fit
    was invoked with `validation_data`, will overplot as open circles the test
    results.

    Minimum values (loss) or maximum values (accuracy) are highlighted; and
    if the minimum loss does not occur at the same epoch as the maximum
    accuracy, the corresponding epoch's accuracy is also highlighted.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    trainLoss = history['loss']
    if 'val_loss' in history.keys():
        valLoss = history['val_loss']
        lossMin = np.argmin(valLoss)
    trainAcc = history['acc']
    if 'val_acc' in history.keys():
        valAcc = history['val_acc']
        accMax = np.argmax(valAcc)
    epochs = range(1, len(history['loss'])+1)

    Δy = -0.0075
    axes[0].plot(epochs, trainLoss,'bo', mfc='none',label='Training Loss')
    if 'val_loss' in history.keys():
        axes[0].plot(epochs, valLoss, 'bo',
                     label='Validation Loss')
        axes[0].plot(epochs[lossMin], valLoss[lossMin], 'bo', mfc='#7070d0')
        axes[0].annotate(f"({epochs[lossMin]:d}, {valLoss[lossMin]:5.3f})",
                         (epochs[lossMin], valLoss[lossMin]), va='top',
                         ha='center', xytext=(epochs[lossMin],
                         valLoss[lossMin] + Δy))
        axes[0].plot(epochs[lossMin], trainLoss[lossMin], 'bo', mfc='#c7c7ff')
        axes[0].annotate(f"({epochs[lossMin]:d}, {trainLoss[lossMin]:5.3f})",
                         (epochs[lossMin], trainLoss[lossMin]), va='top',
                         ha='center', xytext=(epochs[lossMin],
                         trainLoss[lossMin] + Δy))
        axes[0].set_title('Training and Validation Loss')
    else:
        axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    Δy = 0.0025
    axes[1].plot(epochs, trainAcc,'bo', mfc='none',label='Training Accuracy')
    if 'val_acc' in history.keys():
        axes[1].plot(epochs, valAcc, 'bo',
                     label='Validation Accuracy')
        axes[1].plot(epochs[accMax], valAcc[accMax], 'bo', mfc='#7070d0')
        axes[1].annotate(f"({epochs[accMax]:d}, {valAcc[accMax]:5.3f})",
                         (epochs[accMax], valAcc[accMax]), va='bottom',
                         ha='center',
                         xytext=(epochs[accMax], valAcc[accMax] + Δy))
        axes[1].plot(epochs[accMax], trainAcc[accMax], 'bo', mfc='#c7c7ff')
        axes[1].annotate(f"({epochs[accMax]:d}, {trainAcc[accMax]:5.3f})",
                         (epochs[accMax], trainAcc[accMax]), va='bottom',
                         ha='center',
                         xytext=(epochs[accMax], trainAcc[accMax] + Δy))
        if lossMin != accMax:
            axes[1].plot(epochs[lossMin], valAcc[lossMin], 'bo', mfc='#7070d0')
            axes[1].annotate(f"({epochs[lossMin]:d}, {valAcc[lossMin]:5.3f})",
                             (epochs[lossMin], valAcc[lossMin]), va='top',
                             ha='center', xytext=(epochs[lossMin],
                             valAcc[lossMin] - 2*Δy))
            axes[1].plot(epochs[lossMin], trainAcc[lossMin], 'bo',
                         mfc='#c7c7ff')
            axes[1].annotate(f"({epochs[lossMin]:d}, {trainAcc[lossMin]:5.3f})",
                             (epochs[lossMin], trainAcc[lossMin]), va='top',
                             ha='center', xytext=(epochs[lossMin],
                             trainAcc[lossMin] - 2*Δy))

        axes[1].set_title('Training and Validation Accuracy')
    else:
        axes[0].set_title('Training Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
