from tensorflow.keras.callbacks import Callback

class KerasEarlyStopCallback(Callback):
    """
    INPUT:
        threshold	float, threshold to attain for early stopping, as a
                        fraction on (0, 1), default: 0.95
        metric		str, one of std metrics to use for stopping,
                        default: 'acc'.

    User initializes with the stopping test metric and a threshold. Once
    attained, fit will stop early.

    If 'loss' in metric, will test for metric <= threshold; if 'acc' in metric,
    will test for metric >= threshold.
    """

    def __init__(self, threshold=0.95, metric='acc'):
        super(Callback, self).__init__()

        self.threshold = threshold
        self.metric = metric

    def on_epoch_end(self, epoch, logs={}):
        if 'loss' in self.metric:
            if(logs.get(self.metric) <= self.threshold):
                print(f"\nReached {self.threshold} {self.metric}, so "
                      "cancelling training!")
                self.model.stop_training = True
        elif 'acc' in self.metric:
            if(logs.get(self.metric) >= self.threshold):
                print(f"\nReached {self.threshold} {self.metric}, so "
                      "cancelling training!")
                self.model.stop_training = True
