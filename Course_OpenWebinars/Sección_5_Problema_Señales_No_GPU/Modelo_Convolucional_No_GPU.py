"""
Author: @gabvaztor

Style: "Google Python Style Guide" 
https://google.github.io/styleguide/pyguide.html

"""
"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

'''LOCAL IMPORTS'''
from Course_OpenWebinars.UsefulTools.TensorFlowUtils import *

''' TensorFlow: https://www.tensorflow.org/
To upgrade TensorFlow to last version:
*CPU: pip3 install --upgrade tensorflow
*GPU: pip3 install --upgrade tensorflow-gpu
'''
import tensorflow as tf
# noinspection PyUnresolvedReferences
print("TensorFlow: " + tf.__version__)

''' Numpy is an extension to the Python programming language, adding support for large,
multi-dimensional arrays and matrices, along with a large library of high-level
mathematical functions to operate on these arrays.
It is mandatory to install 'Numpy+MKL' before scipy.
Install 'Numpy+MKL' from here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
http://www.numpy.org/
https://en.wikipedia.org/wiki/NumPy '''
import numpy as np

''' Matlab URL: http://matplotlib.org/users/installing.html'''
import matplotlib.pyplot as plt

'''"Best image library"
pip install opencv-python'''
import cv2

"""Python libraries"""
""" Random to shuffle lists """
import random
""" Time """
import time

class Modelo():
    def __init__(self,input, test, input_labels, test_labels, number_of_classes, option_problem=None):
        self.input = input
        self.test = test
        self.input_labels = input_labels
        self.test_labels = test_labels
        self.number_of_classes = number_of_classes
        self.input_batch = None
        self.label_batch = None
        # CONFIGURATION VARIABLES
        self.show_advanced_info = False  # Labels and logits info.
        self.show_images = False  # If True show images when show_info is True
        self.save_model_configuration = True  # If True, then all attributes will be saved in a settings_object path.
        self.shuffle_data = True  # If True, then the train and validation data will be shuffled separately.
        self.save_graphs_images = False #  If True, then save graphs images from statistical values. NOTE that this will
        # decrease the performance during training. Although this is true or false, for each time an epoch has finished,
        # the framework will save a graph
        # TRAIN MODEL VARIABLES
        self.input_rows_numbers = 60
        self.input_columns_numbers = 60
        self.kernel_size = [7, 7]  # Kernel patch size
        self.epoch_numbers = 100 # Epochs number
        self.batch_size = 16  # Batch size
        self.input_size = len(input)  # Change if necessary
        self.test_size = len(test)  # Change if necessary
        self.train_dropout = 0.5  # Keep probably to dropout to avoid overfitting
        self.first_label_neurons = 16
        self.second_label_neurons = 32
        self.third_label_neurons = 64
        self.learning_rate = 1e-3  # Learning rate
        self.number_epoch_to_change_learning_rate = 2
        self.trains = int(self.input_size / self.batch_size) + 1 # Total number of trains for epoch
        # INFORMATION VARIABLES
        self.index_buffer_data = 0  # The index for mini_batches during training
        self.num_trains_count = 1
        self.num_epochs_count = 1
        self.train_accuracy = None
        self.validation_accuracy = None
        self.test_accuracy = None
        # OPTIONS
        # Options represent a list with this structure:
        #               - First position: "string_option" --> unique string to represent problem in question
        #               - Others positions: all variables you need to process each input and label elements
        # noinspection PyUnresolvedReferences
        self.options = [option_problem, cv2.IMREAD_GRAYSCALE,
                   self.input_rows_numbers, self.input_columns_numbers]

    def convolucion_imagenes(self):
        """
        Generic convolutional model
        """
        # Print actual configuration
        self.print_actual_configuration()
        # Placeholders
        x_input, y_labels, keep_probably = self.placeholders(args=None, kwargs=None)
        # Reshape x placeholder into a specific tensor
        x_reshape = tf.reshape(x_input, [-1, self.input_rows_numbers, self.input_columns_numbers, 1])
        # Network structure
        y_prediction = self.network_structure(x_reshape, args=None, keep_probably=keep_probably)
        cross_entropy, train_step, correct_prediction, accuracy = self.model_evaluation(y_labels=y_labels,
                                                                                        y_prediction=y_prediction)
        # Session
        sess = initialize_session()
        self.train_model(args=None, kwargs=locals())

    def update_inputs_and_labels_shuffling(self, inputs, inputs_labels):
        """
        Update inputs_processed and labels_processed variables with an inputs and inputs_labels shuffled
        :param inputs: Represent input data
        :param inputs_labels:  Represent labels data
        """
        c = list(zip(inputs, inputs_labels))
        random.shuffle(c)
        self.inputs_processed, self.labels_processed = zip(*c)

    def data_buffer_generic_class(self, inputs, inputs_labels, shuffle_data=False, batch_size=None, is_test=False,
                                  options=None):
        """
        Create a data buffer having necessaries class attributes (inputs,labels,...)
        :param inputs: Inputs
        :param inputs_labels: Inputs labels
        :param shuffle_data: If it is necessary shuffle data.
        :param batch_size: The batch size.
        :param is_test: if the inputs are the test set.
        :param options: options       
        :return: Two numpy arrays (x_batch and y_batch) with input data and input labels data batch_size like shape.
        """
        x_batch = []
        y_batch = []
        if is_test:
            x_batch, y_batch = process_test_set(inputs,inputs_labels,options)
        else:
            if shuffle_data and self.index_buffer_data == 0:
                self.input, self.input_labels = get_inputs_and_labels_shuffled(self.input,self.input_labels)
            else:
                self.input, self.input_labels = self.input, self.input_labels  # To modify if is out class
            batch_size, out_range = self.get_out_range_and_batch()  # out_range will be True if
            # next batch is out of range
            for _ in range(batch_size):
                x, y = process_input_unity_generic(self.input[self.index_buffer_data],
                                                   self.input_labels[self.index_buffer_data],
                                                   options)
                x_batch.append(x)
                y_batch.append(y)
                self.index_buffer_data += 1
            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)
            if out_range:  # Reset index_buffer_data
                self.index_buffer_data = 0
        return x_batch, y_batch

    def get_out_range_and_batch(self):
        """
        Return out_range flag and new batch_size if necessary. It is necessary when batch is bigger than input rest of
        self.index_buffer_data
        :return: out_range (True or False), batch_size (int)
        """
        out_range = False
        batch_size = self.batch_size
        if self.input_size - self.index_buffer_data == 0:  # When is all inputs
            out_range = True
        elif self.input_size - self.index_buffer_data < self.batch_size:
            batch_size = self.input_size - self.index_buffer_data
            out_range = True
        return batch_size, out_range

    def placeholders(self, *args, **kwargs):
        """
        This method will contains all TensorFlow code about placeholders (variables which will be modified during 
        process)
        :return: Inputs, labels and others placeholders
        """
        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, self.input_rows_numbers * self.input_columns_numbers])  # All images will be 60x60
        y_ = tf.placeholder(tf.float32, shape=[None, self.number_of_classes])  # Number of labels
        keep_probably = tf.placeholder(tf.float32)  # Value of dropout. With this you can set a value for each data set
        return x, y_, keep_probably

    def network_structure(self, input, *args, **kwargs):
        """
        This method will contains all TensorFlow code about your network structure. 
        :param input: inputs 
        :return: The prediction (network output)
        """
        keep_dropout = kwargs['keep_probably']
        # First Convolutional Layer
        convolution_1 = tf.layers.conv2d(
            inputs=input,
            filters=self.first_label_neurons,
            kernel_size=self.kernel_size,
            padding="same")
        # Pool Layer 1 and reshape images by 2
        pool1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)
        convolution_2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.second_label_neurons,
            kernel_size=[4,4],
            padding="same")
        # Pool Layer 2 nd reshape images by 2
        pool2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)
        dropout3 = tf.nn.dropout(pool2, keep_dropout)
        # Dense Layer
        pool2_flat = tf.reshape(dropout3, [-1, int(self.input_rows_numbers / 4) * int(self.input_columns_numbers / 4)* self.second_label_neurons])
        dense = tf.layers.dense(inputs=pool2_flat, units=self.third_label_neurons)
        dropout4 = tf.nn.dropout(dense, keep_dropout)
        # Readout Layer
        w_fc2 = weight_variable([self.third_label_neurons, self.number_of_classes])
        b_fc2 = bias_variable([self.number_of_classes])
        y_convolution = (tf.matmul(dropout4, w_fc2) + b_fc2)
        return y_convolution

    def model_evaluation(self, y_labels, y_prediction, *args, **kwargs):
        """
        This methods will contains all TensorFlow about model evaluation. 
        :param y_labels: Labels
        :param y_prediction: The prediction
        :return: The output must contains all necessaries variables that it used during training
        """
        # Evaluate model
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_labels,
                                                    logits=y_prediction))  # Cross entropy between y_ and y_conv

        #train_step = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cross_entropy)  # Adadelta Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)  # Adam Optimizer
        #train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)  # Adam Optimizer

        # Sure is axis = 1
        correct_prediction = tf.equal(tf.argmax(y_prediction, axis=1),tf.argmax(y_labels, axis=1))  # Get Number of right values in tensor
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Get accuracy in float

        return cross_entropy, train_step, correct_prediction, accuracy

    def show_advanced_information(self, y_labels, y_prediction, feed_dict):
        y__ = y_labels.eval(feed_dict)
        argmax_labels_y_ = [np.argmax(m) for m in y__]
        pt('y__shape', y__.shape)
        pt('argmax_labels_y__', argmax_labels_y_)
        pt('y__[-1]', y__[-1])
        y__conv = y_prediction.eval(feed_dict)
        argmax_labels_y_convolutional = [np.argmax(m) for m in y__conv]
        pt('argmax_y_conv', argmax_labels_y_convolutional)
        pt('y_conv_shape', y__conv.shape)
        pt('index_buffer_data', self.index_buffer_data)

    def show_save_statistics(self, accuracies_train, accuracies_validation=None, accuracies_test=None,
                        loss_train=None, loss_validation=None, loss_test=None, show_graphs=True):
        """
        Show all necessary visual and text information.
        """
        accuracy_plot = plt.figure(0)
        plt.title(str(self.options[0]))
        plt.xlabel("ITERATIONS | Batch Size=" + str(self.batch_size) + " | Trains for epoch: " + str(self.trains))
        plt.ylabel("ACCURACY (BLUE = Train | RED = Validation | GREEN = Test)")
        plt.plot(accuracies_train, 'b')
        if accuracies_validation:
            plt.plot(accuracies_validation, 'r')
        if accuracies_test:
            plt.plot(accuracies_test, 'g')
        if (accuracies_train or accuracies_validation or accuracies_test) and show_graphs:
            accuracy_plot.show()
            plt.show()
        loss_plot = plt.figure(1)
        plt.title("LOSS")
        plt.xlabel("ITERATIONS | Batch Size=" + str(self.batch_size) + " | Trains for epoch: " + str(self.trains))
        plt.ylabel("LOSS (BLUE = Train | RED = Validation | GREEN = Test)")
        plt.plot(loss_train, 'b')
        if loss_validation:
            plt.plot(loss_validation, 'r')
        if loss_test:
            plt.plot(loss_test, 'g')
        if (loss_train or loss_validation or loss_test) and show_graphs:
            loss_plot.show()
            plt.show()
    def print_actual_configuration(self):
        """
        Print all attributes to console
        """
        pt('first_label_neurons', self.first_label_neurons)
        pt('second_label_neurons', self.second_label_neurons)
        pt('third_label_neurons', self.third_label_neurons)
        pt('input_size', self.input_size)
        pt('batch_size', self.batch_size)

    def update_batch(self, is_test=False):
        if not is_test:
            self.input_batch, self.label_batch = self.data_buffer_generic_class(inputs=self.input,
                                                                                inputs_labels=self.input_labels,
                                                                                shuffle_data=self.shuffle_data,
                                                                                batch_size=self.batch_size,
                                                                                is_test=False,
                                                                                options=self.options)
        elif is_test:
            x_test_feed, y_test_feed = self.data_buffer_generic_class(inputs=self.test,
                                                                  inputs_labels=self.test_labels,
                                                                  shuffle_data=self.shuffle_data,
                                                                  batch_size=None,
                                                                  is_test=True,
                                                                  options=self.options)
            return x_test_feed, y_test_feed


    def train_model(self, *args, **kwargs):

        x = kwargs['kwargs']['x_input']
        y_labels = kwargs['kwargs']['y_labels']
        keep_probably = kwargs['kwargs']['keep_probably']
        accuracy = kwargs['kwargs']['accuracy']
        train_step = kwargs['kwargs']['train_step']
        cross_entropy = kwargs['kwargs']['cross_entropy']
        y_prediction = kwargs['kwargs']['y_prediction']

        # Batching values and labels from input and labels (with batch size)
        self.update_batch()
        x_test_feed, y_test_feed = self.update_batch(is_test=True)

        # TRAIN VARIABLES
        start_time = time.time()  # Start time

        # TO STATISTICS
        # To load accuracies and losses
        accuracies_train, accuracies_test, loss_train, loss_test = [],[],[],[]

        # Update test feeds ( will be not modified during training)
        feed_dict_test_100 = {x: x_test_feed, y_labels: y_test_feed, keep_probably: 1}
        # Update real num_train:
        num_train_start = int(self.num_trains_count % self.trains)
        if num_train_start == self.trains:
            num_train_start = 0
        # START  TRAINING
        parar_entrenamiento = False

        for epoch in range(self.num_epochs_count, self.epoch_numbers):  # Start with load value or 0
            if parar_entrenamiento:
                break
            for num_train in range(num_train_start, self.trains):  # Start with load value or 0
                # Update feeds
                feed_dict_train_100 = {x: self.input_batch, y_labels: self.label_batch, keep_probably: 1}
                feed_dict_train_dropout = {x: self.input_batch, y_labels: self.label_batch,
                                      keep_probably: self.train_dropout}
                # Setting values
                self.train_accuracy = accuracy.eval(feed_dict_train_100) * 100
                train_step.run(feed_dict_train_dropout)
                self.test_accuracy = accuracy.eval(feed_dict_test_100) * 100
                cross_entropy_train = cross_entropy.eval(feed_dict_train_100)
                cross_entropy_test = cross_entropy.eval(feed_dict_test_100)

                # To generate statistics
                accuracies_train.append(self.train_accuracy)
                accuracies_test.append(self.test_accuracy)
                loss_train.append(cross_entropy_train)
                loss_test.append(cross_entropy_test)
                if num_train % 10 == 0:
                    percent_advance = str(num_train * 100 / self.trains)
                    pt('Time', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                    pt('TRAIN NUMBER: ' + str(self.num_trains_count) + ' | Percent Epoch ' +
                       str(epoch) + ": " + percent_advance + '%')
                    pt('train_accuracy', self.train_accuracy)
                    pt('cross_entropy_train', cross_entropy_train)
                    pt('test_accuracy', self.test_accuracy)
                    pt('index_buffer_data', self.index_buffer_data)
                # To decrement learning rate during training
                if epoch % self.number_epoch_to_change_learning_rate == 0 and num_train == 9 and epoch != 0:
                    self.learning_rate = float(self.learning_rate / 1.5)
                # Update indexes
                # Update num_epochs_counts
                if num_train +1 == self.trains:  # +1 because start in 0
                    self.num_epochs_count += 1

                if self.show_advanced_info:
                    self.show_advanced_information(y_labels=y_labels, y_prediction=y_prediction,
                                                   feed_dict=feed_dict_train_100)
                # Update num_trains_count and num_epoch_count
                self.num_trains_count += 1
                # Para parar entrenamiento
                if epoch % 200 == 0 and num_train == 9 and epoch != 0:
                    respuesta = str(input("¿Paramos entrenamiento?: Pulsa 'S' para sí y 'N' para no." + " ")).upper()
                    if respuesta == "S": # Paramos entrenamiento
                        parar_entrenamiento = True
                        break
                    elif respuesta != "N":
                        pass
                # Update batches values
                self.update_batch()
        pt('FIN DEL ENTRENAMIENTO')
        self.show_save_statistics(accuracies_train=accuracies_train, accuracies_test=accuracies_test,
                             loss_train=loss_train, loss_test=loss_test)
        self.make_predictions()

    def make_predictions(self):
        pass

"""
STATIC METHODS: Not need "self" :argument
"""


def get_inputs_and_labels_shuffled(inputs, inputs_labels):
    """
    Get inputs_processed and labels_processed variables with an inputs and inputs_labels shuffled
    :param inputs: Represent input data
    :param inputs_labels:  Represent labels data
    :returns inputs_processed, labels_processed
    """
    c = list(zip(inputs, inputs_labels))
    random.shuffle(c)
    inputs_processed, labels_processed = zip(*c)
    return inputs_processed, labels_processed


def process_input_unity_generic(x_input, y_label, options=None, is_test=False):
    """
    Generic method that process input and label across a if else statement witch contains a string that represent
    the option (option = how process data)
    :param x_input: A single input
    :param y_label: A single input label 
    :param options: All attributes to process data. First position must to be the option.
    :param is_test: Sometimes you don't want to do some operation to test set.
    :return: x_input and y_label processed
    """
    x_input = process_image_signals_problem(x_input, options[1], options[2],
                                                    options[3], is_test=is_test)
    return x_input, y_label

# noinspection PyUnresolvedReferences
def process_image_signals_problem(image, image_type, height, width, is_test=False):
    """
    Process signal image
    :param image: The image to change
    :param image_type: Gray Scale, RGB, HSV
    :param height: image height
    :param width: image width
    :param is_test: flag with True if image is in test set
    :return: 
    """
    # 1- Get image in GrayScale
    # 2- Modify intensity and contrast
    # 3- Transform to gray scale
    # 4- Return image
    image = cv2.imread(image, 0)
    image = cv2.resize(image, (height, width))
    image = cv2.equalizeHist(image)
    image = cv2.equalizeHist(image)
    image = cv2.equalizeHist(image)
    if not is_test:

        random_percentage = random.randint(3, 20)
        to_crop_height = int((random_percentage * height) / 100)
        to_crop_width = int((random_percentage * width) / 100)
        image = image[to_crop_height:height - to_crop_height, to_crop_width:width - to_crop_width]
        image = cv2.copyMakeBorder(image, top=to_crop_height,
                                   bottom=to_crop_height,
                                   left=to_crop_width,
                                   right=to_crop_width,
                                   borderType=cv2.BORDER_CONSTANT)
    image = image.reshape(-1)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)  # Wait until press key to destroy image
    return image


def process_test_set(test, test_labels, options):
    """
    Process test set and return it
    :param test: Test set
    :param test_labels: Test labels set
    :param options: All attributes to process data. First position must to be the option.
    :return: x_test and y_test
    """
    x_test = []
    y_test = []
    for i in range(len(test)):
        x, y = process_input_unity_generic(test[i],test_labels[i],options,is_test=True)
        x_test.append(x)
        y_test.append(y)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_test, y_test



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    # initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def initialize_session():
    """
    Initialize interactive session and all local and global variables
    :return: Session
    """
    sess = tf.InteractiveSession()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    return sess
