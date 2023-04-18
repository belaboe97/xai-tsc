from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
import os
import numpy as np
import sys
import sklearn
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
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


    nb_classes_2 = len(np.unique(np.concatenate((y_train_2, y_test_2), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train_2, y_test_2), axis=0).reshape(-1, 1))
    y_train_2 = enc.transform(y_train_2.reshape(-1, 1)).toarray()
    y_test_2 = enc.transform(y_test_2.reshape(-1, 1)).toarray()

    y_true_2 =  np.argmax(y_test_2, axis=1)

    # save orignal y because later we will use binary
    # y_true = np.argmax(y_test, axis=1)

    """
    Instatiate Classifier
    - create_classifier
    - fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true)
    """ 

    classifier = create_classifier_mt(classifier_name, input_shape, nb_classes_1, nb_classes_2, output_directory, gamma)

    classifier.fit(x_train_1, y_train_1,y_train_2, x_test_1, y_test_1,y_test_2, y_true_1,y_true_2)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, EPOCHS, BATCH_SIZE,  verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)


def create_classifier_mt(classifier_name, input_shape, nb_classes_1, nb_classes_2, output_directory, gamma, verbose=False):
    if classifier_name == 'fcn_mt': 
        from classifiers_mtl import fcn_mt
        return fcn_mt.Classifier_FCN_MT(output_directory, input_shape, nb_classes_1, nb_classes_2,gamma,EPOCHS, BATCH_SIZE, verbose)



############################################### main

# change this directory for your machine

import os

if os.getenv("COLAB_RELEASE_TAG"):
    print("Google Colab Environment detected")
    root_dir =  "/content/drive/My Drive/master thesis/code/dl-4-tsc-mtl"
else: 
    print("Local Environment detected")
    root_dir = "G:/My Drive/master thesis/code/dl-4-tsc-mtl"


#Set random seed 
#https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
SEED = 0
EPOCHS = 100
BATCH_SIZE = 16

print(f'In fixed SEED mode: {SEED}')
print(f'Epochs for each classifier is set to {EPOCHS} and Batchsize set to {BATCH_SIZE}')

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    #random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

"""
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
"""

# this is the code used to launch an experiment on a dataset
archive_name = sys.argv[1]
dataset_name = sys.argv[2]
classifier_name = sys.argv[3]
itr = sys.argv[4] if sys.argv[4] != '_itr_0' else ''
mtl = sys.argv[5]
appendix = sys.argv[6] 
gamma = 0.5 if sys.argv[7] == None else sys.argv[7]
gamma = np.float64(gamma)
#output_directory = root_dir + '/results/'  + archive_name + '/' +  dataset_name + classifier_name.split('_')[0] + '/' + classifier_name + '/' 

output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/{classifier_name.split("_")[0]}/{classifier_name + itr}_{mtl}/{appendix}/' 
print(output_directory)
test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', archive_name, dataset_name, classifier_name, itr)

if os.path.exists(test_dir_df_metrics):
    print('Already done')
else:
    create_directory(output_directory)

    """
    Read datasets for classification and 
    """

    if mtl == 'mtl': 
        datasets_dict_1 = read_dataset(root_dir, archive_name, dataset_name, 'original')
        datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name,  appendix )
        fit_classifier_mt()
    else: 
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original')
        fit_classifier()

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
