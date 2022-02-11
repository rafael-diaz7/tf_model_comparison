from tensorflow.keras.callbacks import Callback, EarlyStopping
import time

class SaveModelWeightsCallback(Callback):
    "Saves the Model after each iteration of training"
    def __init__(self, classifier, weight_filename):
        Callback.__init__(self)
        self._classifier = classifier
        self._weight_filename = weight_filename
    
    def on_epoch_end(self, epoch, logs=None):
        self._classifier.save_weights(self._weight_filename)
        

class WriteMetrics(Callback):
    '''
    Writes metrics at end of every epoch... 
    given metrics is:
        num_neg, macro_precision, macro_recall, macro_F1, micro_precision, micro_recall, micro_F1
    writes:
        epoch, time for epoch, num_neg, macro_precision, macro_recall, macro_F1, micro_precision, micro_recall, micro_F1
    '''
    def __init__(self, metric_filename):
        self.mf = metric_filename
        self.start = 0

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("At start; log keys: ".format(keys))

    def on_epoch_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start of epoch {}; log keys: {}".format(epoch+1, keys))
        start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End of epoch {}; log keys;: {}".format(epoch+1, keys))
        print(list(logs.values()))
        vals = list(logs.values())
        end = time.time()
        time_taken = time.time() - start
        with open(mf, 'a') as file:
            file.write("{},{}s,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                epoch+1, time_taken, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]))