from Classifier import *
from Dataset import *
import sys

# Simple example to test multilabel text classification datasets
def run_i2b2_dataset():
    # hyperparameter search
    # need to get system arguments [file, learning rate, drop out]
    cmdargs = sys.argv
    lr = float(cmdargs[1]) # learning rate
    do = float(cmdargs[2]) # drop out
    bpS = cmdargs[3] # back propogation
    bp = True if bpS == "True" else False
    arch = cmdargs[4] # architecture
   
    # training parameters
    max_epoch = 1000
    batch_size = 20 if bp else 200
    print(batch_size)
    early_stopping_patience = 5
    early_stopping_monitor = 'loss'

    # model hyperparameters
    #learning_rate = 0.01
    #dropout_rate = 0.8
    learning_rate = lr
    dropout_rate = do
    language_model_trainable = bp
    
    # parameters to load and save a model
    #model_in_file_name = "my_models/model_out_trainable_true" # to load a model, need to uncomment some code below
    #model_out_file_name = "my_models/model_out_trainable_true_then_finetune" # to save a model, need to uncomment some code below
    
    #set up the language model
    language_model_name = Classifier.BIODISCHARGE_SUMMARY_BERT
    max_length = 512
    
    #load the dataset
    data_filepath = '../data/train_split_2.tsv'
    num_classes = 8
    data = i2b2Dataset(data_filepath, validation_set_size=0.2)
    #data = i2b2Dataset(data_filepath)
    #exit()

    # creating the file for writing metrics to
    metric_file = "grid_search/{}/{}/{}_{}.csv".format(arch, ("BP" if bp else "noBP"), learning_rate, dropout_rate)
    with open(metric_file, 'w') as file:
        file.write("epoch,time,loss,num_neg,macro_precision,macro_recall,macro_F1,micro_precision,micro_recall,micro_F1\n")
    
    #create classifier and load data for a multiclass text classifier
    if arch == "CLS_1L":
        classifier = CLS_1L(language_model_name, num_classes, metric_file, max_length=max_length, learning_rate=learning_rate, language_model_trainable=language_model_trainable, dropout_rate=dropout_rate)
    elif arch == "CLS_3L": 
        classifier = CLS_3L(language_model_name, num_classes, metric_file, max_length=max_length, learning_rate=learning_rate, language_model_trainable=language_model_trainable, dropout_rate=dropout_rate)
    elif arch == "biLSTM_1L":
        classifier = biLSTM_1L(language_model_name, num_classes, metric_file, max_length=max_length, learning_rate=learning_rate, language_model_trainable=language_model_trainable, dropout_rate=dropout_rate)
    elif arch == "biLSTM_3L":
        classifier = biLSTM_3L(language_model_name, num_classes, metric_file, max_length=max_length, learning_rate=learning_rate, language_model_trainable=language_model_trainable, dropout_rate=dropout_rate)
    elif arch == "BERT_SIG":
        classifier = BERT_SIG(language_model_name, num_classes, metric_file, max_length=max_length, learning_rate=learning_rate, language_model_trainable=language_model_trainable, dropout_rate=dropout_rate)       

    print("USING learning_rate: {} dropout: {}, model: {}, back_prop: {}".format(learning_rate, dropout_rate, language_model_name, bp))
    #load a model's weights from file, use this code
    #classifier.load_weights(model_in_file_name)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    ###### BONUS STUFF ########
    #summarize the model in text
    classifier.model.summary()
    #plot the model (an image)
    tf.keras.utils.plot_model(
        classifier.model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     epochs=max_epoch,
                     batch_size=batch_size,
                     #model_out_file_name=model_out_file_name,
                     early_stopping_patience=5, early_stopping_monitor=early_stopping_monitor,
                     class_weights=data.get_train_class_weights()
    )



#This is the main running method for the script
if __name__ == '__main__':
    run_i2b2_dataset()
