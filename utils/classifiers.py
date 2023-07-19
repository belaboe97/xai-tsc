
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
    #set_global_determinism(0)
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, epochs, batch_size,  verbose)
    if classifier_name == 'fcn_test':
        from classifiers import fcn_test
        return fcn_test.Classifier_FCN_TEST(output_directory, input_shape, nb_classes, epochs, batch_size,  verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder 
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, epochs, batch_size,  verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, epochs, batch_size,  verbose)


def create_classifier_mt(classifier_name, 
                         input_shape, nb_classes, lossf, 
                         output_directory, gamma, 
                         epochs, batch_size,
                         verbose=False):
    #set_global_determinism(0)

    ### EXPERIMENT 1 AND 2: 
    if classifier_name == 'fcn_mt_ae': 
        from classifiers_mtl.fcn import fcn_mt_ae
        return fcn_mt_ae.Classifier_FCN_MT_AE(output_directory, input_shape, nb_classes, lossf,
                                              gamma, epochs, batch_size, verbose)
    

    if classifier_name == 'resnet_mt_ae': 
        from classifiers_mtl.resnet import resnet_mt_ae
        return resnet_mt_ae.Classifier_RESNET_MT_AE(output_directory, input_shape, nb_classes, lossf, 
                                                          gamma, epochs, batch_size, verbose)

    if classifier_name == 'fcn_mt_conv': 
        from classifiers_mtl.fcn import fcn_mt_conv
        return fcn_mt_conv.Classifier_FCN_MT_CONV(output_directory, input_shape, nb_classes, lossf,
                                                        gamma, epochs, batch_size, verbose)
    
    if classifier_name == 'resnet_mt_conv': 
        from classifiers_mtl.resnet import resnet_mt_conv
        return resnet_mt_conv.Classifier_RESNET_MT_CONV(output_directory, input_shape, nb_classes, lossf, 
                                                            gamma, epochs, batch_size, verbose)
    

    if classifier_name == 'fcn_mt_linear': 
        from legacy_code import fcn_mt_linear
        return fcn_mt_linear.Classifier_FCN_MT_Linear(output_directory, input_shape, nb_classes, lossf,
                                                    gamma, epochs, batch_size, verbose)


    ### EXPERIMENT 3: 
    if classifier_name == 'fcn_mt_conv_cas': 
        from classifiers_mtl_cascade.fcn import fcn_mt_conv_cas
        return fcn_mt_conv_cas.Classifier_FCN_MT_CONV_CAS(output_directory, input_shape, nb_classes, lossf,
                                                        gamma, epochs, batch_size, verbose)
    
    if classifier_name == 'fcn_mt_ae_cas': 
        from classifiers_mtl_cascade.fcn import fcn_mt_ae_cas
        return fcn_mt_ae_cas.Classifier_FCN_MT_AE_CAS(output_directory, input_shape, nb_classes, lossf,
                                                        gamma, epochs, batch_size, verbose)
    
    if classifier_name == 'resnet_mt_conv_cas': 
        from classifiers_mtl_cascade.resnet import resnet_mt_conv_cas
        return resnet_mt_conv_cas.Classifier_RESNET_MT_CONV_CAS(output_directory, input_shape, nb_classes, lossf, 
                                                            gamma, epochs, batch_size, verbose)
    
    if classifier_name == 'resnet_mt_ae_cas': 
        from classifiers_mtl_cascade.resnet import resnet_mt_ae_cas
        return resnet_mt_ae_cas.Classifier_RESNET_MT_AE_CAS(output_directory, input_shape, nb_classes, lossf, 
                                                            gamma, epochs, batch_size, verbose)

    ### EXPERIMENT 4: 
    if classifier_name == 'fcn_mt_ae_iter': 
        from classifiers_mtl_iterative.fcn import fcn_mt_ae_iter
        return fcn_mt_ae_iter.Classifier_FCN_MT_AE_ITER(output_directory, input_shape, nb_classes, lossf, 
                                                            gamma, epochs, batch_size, verbose)
    
    if classifier_name == 'fcn_mt_ae_iter_freeze': 
        from classifiers_mtl_iterative.fcn import fcn_mt_ae_iter_freeze
        return fcn_mt_ae_iter_freeze.Classifier_FCN_MT_AE_ITER_FREEZE(output_directory, input_shape, nb_classes, lossf, 
                                                            gamma, epochs, batch_size, verbose)

    if classifier_name == 'fcn_mt_conv_iter': 
        from classifiers_mtl_iterative.fcn import fcn_mt_conv_iter
        return fcn_mt_conv_iter.Classifier_FCN_MT_CONV_ITER(output_directory, input_shape, nb_classes, lossf,
                                                        gamma, epochs, batch_size, verbose)
    
def fit_classifier(classifier_name, mode, datasets_dict, datasets_dict_2, 
                   output_directory, lossf, gamma, epochs, batch_size):
    
    #set_global_determinism(0)
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
        acc = classifier.fit(x_train, y_train, x_test, y_test, y_true)
        return acc

    if mode == 'multitask': 

        y_len = len(x_train[0])
        _ , y_train_2, _ , y_test_2 = datasets_dict_2

        classifier = create_classifier_mt(classifier_name, input_shape, nb_classes, lossf, 
                                          output_directory, gamma, epochs, batch_size)
        classifier.fit(x_train, y_train, y_train_2, x_test, y_test , y_test_2, y_true ,y_true_2 = None)