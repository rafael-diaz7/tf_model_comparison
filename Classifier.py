from transformers import AutoTokenizer, TFAutoModel, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow_addons as tfa
import os


from DataGenerator import *
from CustomCallbacks import *
from Metrics import *
from abc import ABC, abstractmethod

class Classifier(ABC):
    '''
    Classifier class, which holds a language model and a classifier
    This class can be modified to create whatever architecture you want,
    however it requres the following instance variables:
    self.language_mode_name - this is a passed in variable and it specifies
       which HuggingFace language model to use
    self.tokenizer - this is created, but is just an instance of the tokenizer
       corresponding to the HuggingFace language model you use
    self.model - this is the Keras/Tensor flow classification model. This is
       what you can modify the architecture of, but it must be set

    Upon instantiation, the model is constructed. It should then be trained, and
    the model will then be saved and can be used for prediction.

    Training uses a DataGenerator object, which inherits from a sequence object
    The DataGenerator ensures that data is correctly divided into batches. 
    This could be done manually, but the DataGenerator ensures it is done 
    correctly, and also allows for variable length batches, which massively
    increases the speed of training.
    '''
    # only using BIODISCHARGE SUMMARY BERT
    # not typically tf model, has been converted
    BIODISCHARGE_SUMMARY_BERT = '../models/biobert_pretrain_output_disch_100000'
    
    #some default parameter values
    EPOCHS = 1000
    BATCH_SIZE = 20
    MAX_LENGTH = 512
    #Note: MAX_LENGTH varies depending on the model. For Roberta, max_length = 768.
    #      For BERT its 512
    LEARNING_RATE = 0.01
    DROPOUT_RATE = 0.8
    MODEL_OUT_FILE_NAME = ''
    
    @abstractmethod
    def __init__(self, language_model_name, metric_file, language_model_trainable=False, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE):
        '''
        Initializer for a language model. This class should be extended, and
        the model should be built in the constructor. This constructor does
        nothing, since it is an abstract class. In the constructor however
        you must define:
        self.tokenizer 
        self.model
        '''
        self.tokenizer = None
        self.model = None
        self.metric_file = metric_file
        self._language_model_name = language_model_name
        self._language_model_trainable = language_model_trainable
        self._max_length = max_length
        self._learning_rate=learning_rate
        self._dropout_rate=dropout_rate
        self.tokenizer = AutoTokenizer.from_pretrained(self._language_model_name)

    def load_language_model(self):
        # either load the language model locally or grab it from huggingface
        if os.path.isdir(self._language_model_name):
             language_model = TFBertModel.from_pretrained(self._language_model_name, from_pt=True)
        # else the language model can be grabbed directly from huggingface
        else:
            language_model = TFAutoModel.from_pretrained(self._language_model_name)

        # set properties
        language_model.trainable = self._language_model_trainable
        language_model.output_hidden_states = False

        #return the loaded model
        return language_model

        
    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS, model_out_file_name=MODEL_OUT_FILE_NAME, early_stopping_monitor='loss', early_stopping_patience=0, class_weights=None):
        '''
        Trains the classifier
        :param x: the training data
        :param y: the training labels

        :param batch_size: the batch size
        :param: validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
                If validation data is passed in, then all metrics (including loss) will 
                report performance on the validation data
        :param: epochs: the number of epochs to train for
        '''
        
        #create a DataGenerator from the training data
        training_data = DataGenerator(x, y, batch_size, self)
        
        # generate the validation data (if it exists)
        if validation_data is not None:
            validation_data = DataGenerator(validation_data[0], validation_data[1], batch_size, self)        
        
        # set up callbacks
        callbacks = []
        if not model_out_file_name == '':
            callbacks.append(SaveModelWeightsCallback(self, model_out_file_name))
        if early_stopping_patience > 0:
            callbacks.append(EarlyStopping(monitor=early_stopping_monitor, min_delta=0, patience=early_stopping_patience,
             restore_best_weights=True))
        callbacks.append(WriteMetrics(self.metric_file))
            
        # fit the model to the training data
        self.model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            class_weight=class_weights,
            verbose=2,
            callbacks=callbacks
        )


    #function to predict using the NN
    def predict(self, x, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(x, padding=True, truncation=True, max_length=self._max_length, return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])

        return self.model.predict(x, batch_size=batch_size)


    #function to save the model weights
    def save_weights(self, filepath):
        """
        Saves the model weights
        :return: None
        """
        self.model.save_weights(filepath)

    #function to load the model weights
    def load_weights(self, filepath):
        """
        Loads weights for the model
        :param filepath: the filepath (without extension) of the model #TODO - is that the file_path?
        :return:
        """
        self.model.load_weights(filepath)


class MultiLabel_Text_Classifier(Classifier):

    def __init__(self, language_model_name, num_classes, metric_file, language_model_trainable=False, max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        
        Classifier.__init__(self, language_model_name, metric_file, language_model_trainable=language_model_trainable, max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # set instance attributes
        self._num_classes = num_classes
        
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")
 
        # create the language model
        language_model = self.load_language_model()
        
        #create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]
        
        #We can create a sentence embedding using the one directly from BERT, or using a biLSTM
        # OR, we can return the sequence from BERT (just don't slice) or the BiLSTM (use retrun_sequences=True)
        #create the sentence embedding layer - using the BERT sentence representation (cls token)    
        #embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:,0,:]
        sentence_representation_language_model = embeddings[:,0,:]
        #Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        lstm_size=256
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)
        
        # now, create three layer, sigmoid
        # dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_biLSTM))
    
        #dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(self._dropout_rate)
        output2 = dropout2(dense2(output1))

        #dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(self._dropout_rate)
        output3 = dropout3(dense3(output2))

        #softmax
        sigmoid_layer = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')
        final_output = sigmoid_layer(output3)
        
        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        
        #create the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

        # create the merics
        #from Metrics import MyMetrics
        my_metrics = MyTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )