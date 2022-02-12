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
        loss, num_neg, macro_precision, macro_recall, macro_F1, micro_precision, micro_recall, micro_F1
    writes:
        epoch, time, loss, num_neg, macro_precision, macro_recall, macro_F1, micro_precision, micro_recall, micro_F1
    '''
    def __init__(self, metric_filename):
        self.mf = metric_filename
        self.start = 0

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("At start; log keys: ".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()
        keys = list(logs.keys())
        print("Start of epoch {}; log keys: {}".format(epoch+1, keys))

    def on_epoch_end(self, epoch, logs=None):
        time_taken = int(time.time() - self.start)
        keys = list(logs.keys())
        val_ind = keys.index('val_loss')
        print("End of epoch {}; log keys;: {}".format(epoch+1, keys))
        print(list(logs.values()))
        vals = list(logs.values())
        print("via logs: {}; via index: {}".format(logs['val_loss'], vals[val_ind])
       	with open(self.mf, 'a') as file:
		file.write("{},{}s,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, time_taken, vals[val_ind], vals[val_ind+1], vals[val_ind+2], vals[val_ind+3], vals[val_ind+4], vals[val_ind+5], vals[val_ind+6], vals[val_ind+7]))
