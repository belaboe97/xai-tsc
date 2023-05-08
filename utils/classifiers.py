
import numpy as np
import pandas as pd
import sklearn 
import os 
import tensorflow as tf

SEED =0

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    print("set global determinism")



def create_classifier(classifier_name, input_shape, nb_classes, output_directory, epochs, batch_size, verbose=True):
    set_global_determinism(0)
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, epochs, batch_size,  verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, epochs, batch_size,  verbose)


def create_classifier_mt(classifier_name, 
                         input_shape, nb_classes, 
                         output_directory, gamma, 
                         epochs, batch_size,
                         verbose=False):
    set_global_determinism(0)
    if classifier_name == 'fcn_mt_ae': 
        from classifiers_mtl.fcn import fcn_mt_ae
        return fcn_mt_ae.Classifier_FCN_MT_AE(output_directory, input_shape, nb_classes, 
                                              gamma, epochs, batch_size, verbose)
    if classifier_name == 'fcn_mt_dense': 
        from classifiers_mtl.fcn import fcn_mt_dense
        return fcn_mt_dense.Classifier_FCN_MT_DENSE(output_directory, input_shape, nb_classes, 
                                                    gamma, epochs, batch_size, verbose)
    if classifier_name == 'fcn_mt_sigmoid': 
        from classifiers_mtl.fcn import fcn_mt_sigmoid
        return fcn_mt_sigmoid.Classifier_FCN_MT_SIGMOID(output_directory, input_shape, 
                                                        nb_classes, gamma, epochs, batch_size, verbose)
    if classifier_name == 'resnet_mt_dense': 
        from classifiers_mtl.resnet import resnet_mt_dense
        return resnet_mt_dense.Classifier_RESNET_MT_DENSE(output_directory, input_shape, 
                                                          nb_classes, gamma, epochs, batch_size, verbose)
    
def fit_classifier(classifier_name, mode, datasets_dict, datasets_dict_2, 
                   output_directory, gamma, epochs, batch_size):
    set_global_determinism(0)
    x_train, y_train, x_test, y_test = datasets_dict

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    if mode == 'singletask':

        classifier = create_classifier(classifier_name, input_shape, nb_classes, 
                                       output_directory,  epochs, batch_size)
        classifier.fit(x_train, y_train, x_test, y_test, y_true)

    if mode == 'multitask': 

        y_len = len(x_train[0])
        
        _ , y_train_2, _ , y_test_2 = datasets_dict_2

        classifier = create_classifier_mt(classifier_name, input_shape, nb_classes, 
                                          output_directory, gamma, epochs, batch_size)
        classifier.fit(x_train, y_train, y_train_2, x_test, y_test , y_test_2, y_true ,y_true_2 = None)