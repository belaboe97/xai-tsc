from utils.utils import generate_results_csv, save_attributions
from utils.utils import create_directory
from utils.utils import read_dataset
import os
import numpy as np
import sys
import sklearn
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import calculate_attributions, calculate_pointwise_attributions
from utils.explanations import create_pointwise_explanations
from utils.explanations import save_explanations
import tensorflow as tf


def fit_classifier():
    x_train, y_train, x_test, y_test = datasets_dict[dataset_name]

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
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def fit_classifier_mt():

    """ 
    For Task 1: Classification
    """

    x_train_1, y_train_1, x_test_1, y_test_1 = datasets_dict_1[dataset_name]
    nb_classes_1 = len(np.unique(np.concatenate((y_train_1, y_test_1), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train_1, y_test_1), axis=0).reshape(-1, 1))
    y_train_1 = enc.transform(y_train_1.reshape(-1, 1)).toarray()
    y_test_1 = enc.transform(y_test_1.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true_1 = np.argmax(y_test_1, axis=1)

    """
        if len(x_train_1.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train_1 = x_train_1.reshape((x_train_1.shape[0], x_train_1.shape[1], 1))
        x_test_1 = x_test_1.reshape((x_test_1.shape[0], x_test_1.shape[1], 1))

    """
    if len(x_train_1.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train_1 = x_train_1.reshape((x_train_1.shape[0], x_train_1.shape[1], 1))
        x_test_1 = x_test_1.reshape((x_test_1.shape[0], x_test_1.shape[1], 1))

    input_shape = x_train_1.shape[1:]

    """
    For Task 2: Explanation 
    Extract labels: 
    """    
    _ , y_train_2, _ , y_test_2 = datasets_dict_2[dataset_name] 

    # save orignal y because later we will use binary
    # y_true = np.argmax(y_test, axis=1)

    """
    Instatiate Classifier
    - create_classifier
    - fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true)
    """ 


    classifier = create_classifier_mt(classifier_name, input_shape, nb_classes_1, None, output_directory, gamma)

    classifier.fit(x_train_1, y_train_1,y_train_2, x_test_1, y_test_1,y_test_2, y_true_1,y_true_2 = 'dummy')


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, EPOCHS, BATCH_SIZE,  verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, EPOCHS, BATCH_SIZE,  verbose)


def create_classifier_mt(classifier_name, input_shape, nb_classes_1, nb_classes_2, output_directory, gamma, verbose=False):
    if classifier_name == 'fcn_mt': 
        from classifiers_mtl import fcn_mt
        return fcn_mt.Classifier_FCN_MT(output_directory, input_shape, nb_classes_1, nb_classes_2,gamma,EPOCHS, BATCH_SIZE, verbose)
    if classifier_name == 'fcn_mt_ae': 
        from classifiers_mtl import fcn_mt_ae
        return fcn_mt_ae.Classifier_FCN_MT_AE(output_directory, input_shape, nb_classes_1, gamma,EPOCHS, BATCH_SIZE, verbose)
    if classifier_name == 'fcn_mt_dense': 
        from classifiers_mtl import fcn_mt_dense
        return fcn_mt_dense.Classifier_FCN_MT_DENSE(output_directory, input_shape, nb_classes_1, gamma,EPOCHS, BATCH_SIZE, verbose)
    if classifier_name == 'resnet_mt_dense': 
        from classifiers_mtl import resnet_mt_dense
        return resnet_mt_dense.Classifier_RESNET_MT_DENSE(output_directory, input_shape, nb_classes_1, gamma,EPOCHS, BATCH_SIZE, verbose)


############################################### main

# change this directory for your machine

import os

if os.getenv("COLAB_RELEASE_TAG"):
    print("Google Colab Environment detected")
    root_dir =  "/content/drive/My Drive/master thesis/code/xai-tsc"
    EPOCHS = 1000
    BATCH_SIZE = 16
    print('Epochs',EPOCHS, 'Batch size', BATCH_SIZE)
else: 
    print("Local Environment detected")
    root_dir = "G:/Meine Ablage/master thesis/code/xai-tsc"
    EPOCHS = 1
    BATCH_SIZE = 16
    print('Epochs',EPOCHS, 'Batch size', BATCH_SIZE)

#Set random seed 
#https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
SEED = 0
SLICES = 5
DATASET_NAMES = ['GunPoint','Coffee'] # #'wafer'

print(f'In fixed SEED mode: {SEED}')
print(f'Epochs for each classifier is set to {EPOCHS} and Batchsize set to {BATCH_SIZE}')

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    #random.seed(seed)
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


set_global_determinism(SEED)

# this is the code used to launch an experiment on a dataset
mode = sys.argv[1]

if mode == 'sequential':
    gamma = 0.5
    archive_name = 'ucr'
    dataset_name = ''
    itr = '_itr_0' 

else: 
    archive_name = sys.argv[2]
    dataset_name = sys.argv[3]
    classifier_name = sys.argv[4]
    itr = sys.argv[5] if sys.argv[5] != '_itr_0' else ''
    data_source = sys.argv[6] 
    data_dest = sys.argv[7]
    #print(sys.argv[7])
    gamma = 0.5 if sys.argv[8] == None else sys.argv[8]
    gamma = np.float64(gamma)
    #classifier = classifier_name + itr 

    classifier = classifier_name + '_' + str(gamma) #+ mode
def output_path():
    #classifier = classifier_name + str(gamma) #+ mode

    output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/{classifier_name.split("_")[0]}/{classifier}/{data_source}/' 
    print(output_directory)
    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)
    return output_directory, test_dir_df_metrics


if mode == 'sequential': 
    
    for dataset in DATASET_NAMES:

        data_source = 'original'
        data_dest = 'pointwise'
        classifier_name = 'fcn'
        classifier = classifier_name + itr
        dataset_name = dataset

        output_directory, test_dir_df_metrics = output_path()

        if os.path.exists(test_dir_df_metrics):
            print('Already done')
        else:
            create_directory(output_directory)

        """
        Building Block 1: Fit Single Task Classifier and Build Explanations
        """

        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original')
        fit_classifier()
        att = calculate_pointwise_attributions(root_dir, archive_name, classifier, dataset_name, data_source, 'stl', task=1)
        exp = create_pointwise_explanations(att)
        save_explanations(exp, root_dir, archive_name, data_dest, dataset_name)


        """
        Building Block 2: Fit Multi Task Classifier
        """
        classifier_name = 'fcn_mt_ae'
        classifier = classifier_name + itr
        data_source = 'pointwise'

        output_directory, test_dir_df_metrics = output_path()

        if os.path.exists(test_dir_df_metrics):
            print('Already done')
        else:
            create_directory(output_directory)


        datasets_dict_1 = read_dataset(root_dir, archive_name, dataset_name,  'original')

        def readucr(filename):
            data = np.loadtxt(filename, delimiter=',')
            Y = data[:, 150:]
            X = data[:, :150]
            return X, Y

    
        def read_dataset(root_dir, archive_name, dataset_name, data_source):
            datasets_dict = {}
            cur_root_dir = root_dir.replace('-temp', '')
            if data_source == 'original': 
                file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name 
            else: 
                file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name  + '/' + data_source + '/'

            x_train, y_train = readucr(file_name + '/' + dataset_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '/' + dataset_name + '_TEST')
            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),y_test.copy())
            return datasets_dict




        datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name,   data_source)
        print("SHAPE DICT", datasets_dict_2['GunPoint'][1].shape)
        fit_classifier_mt()
        
        """
        # For task 1
        att = calculate_pointwise_attributions(root_dir, archive_name, classifier, dataset, data_source, 'mtl', task=1)
        save_attributions(output_directory, att, task = 1)

        # For task 2
        att = calculate_pointwise_attributions(root_dir, archive_name, classifier, dataset, data_source, 'mtl', task=2)
        save_attributions(output_directory, att, task = 2)
        
        """

        # the creation of this directory means
        create_directory(output_directory + '/DONE')

else: 

    output_directory, test_dir_df_metrics = output_path()

    """
    if os.path.exists(test_dir_df_metrics):
        print('Already done')"""
    if True:
        create_directory(output_directory)
        print(mode)
        if mode == 'mtl': 
            datasets_dict_1 = read_dataset(root_dir, archive_name, dataset_name,  'original')

            def readucr(filename):
                data = np.loadtxt(filename, delimiter=',')
                Y = data[:, :150]
                X = data[:, 150:]

                print("LENGTHS",len(X),len(Y))
                
                return X, Y
    
            def read_dataset(root_dir, archive_name, dataset_name, data_source):
                datasets_dict = {}
                cur_root_dir = root_dir.replace('-temp', '')
                if data_source == 'original': 
                    file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name 
                else: 
                    file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name  + '/' + data_source + '/'

                x_train, y_train = readucr(file_name + '/' + dataset_name + '_TRAIN')
                x_test, y_test = readucr(file_name + '/' + dataset_name + '_TEST')
                datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),y_test.copy())
                return datasets_dict



            datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name,   data_source)
            fit_classifier_mt()

        elif mode == 'stl': 
            datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original')
            fit_classifier()
            att = calculate_pointwise_attributions(root_dir, archive_name, classifier, dataset_name, data_source, mode, task=1)
            exp = create_pointwise_explanations(att)
            save_explanations(exp, root_dir, archive_name, data_dest, dataset_name)

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
