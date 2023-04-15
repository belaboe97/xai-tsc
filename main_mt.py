from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets


def fit_classifier():


    """ 
    For Task 1: Classification
    """


    x_train_1 = datasets_dict_1[dataset_name][0]
    y_train_1 = datasets_dict_1[dataset_name][1]
    x_test_1 = datasets_dict_1[dataset_name][2]
    y_test_1 = datasets_dict_1[dataset_name][3]

    nb_classes_1 = len(np.unique(np.concatenate((y_train_1, y_test_1), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train_1, y_test_1), axis=0).reshape(-1, 1))
    y_train_1 = enc.transform(y_train_1.reshape(-1, 1)).toarray()
    y_test_1 = enc.transform(y_test_1.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test_1, axis=1)

    if len(x_train_1.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train_1 = x_train_1.reshape((x_train_1.shape[0], x_train_1.shape[1], 1))
        x_test_1 = x_test_1.reshape((x_test_1.shape[0], x_test_1.shape[1], 1))

    input_shape = x_train_1.shape[1:]


    """
    For Task 2: Explanation 
    
    Extract labels: 
    """


    x_train_2 = datasets_dict_2[dataset_name + "_Exp"][0]
    y_train_2 = datasets_dict_2[dataset_name + "_Exp"][1]
    x_test_2 = datasets_dict_2[dataset_name  + "_Exp"][2]
    y_test_2 = datasets_dict_2[dataset_name  + "_Exp"][3]

    nb_classes_2 = len(np.unique(np.concatenate((y_train_2, y_test_2), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train_2, y_test_2), axis=0).reshape(-1, 1))
    y_train_2 = enc.transform(y_train_2.reshape(-1, 1)).toarray()
    y_test_2 = enc.transform(y_test_2.reshape(-1, 1)).toarray()


    # save orignal y because later we will use binary
    # y_true = np.argmax(y_test, axis=1)

    """
    Instatiate Classifier

    - create_classifier
    - fit(self, x_train, y_train_1,y_train_2, x_val, y_val_1, y_val_2, y_true)
    """ 

    #print(y_train_1.shape, y_train_2.shape)

    classifier = create_classifier(classifier_name, input_shape, nb_classes_1, nb_classes_2, output_directory)

    classifier.fit(x_train_1, y_train_1,y_train_2, x_test_1, y_test_1,y_test_2, y_true)


def create_classifier(classifier_name, input_shape, nb_classes_1, nb_classes_2, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'fcn_mt': 
        from classifiers import fcn_mt
        return fcn_mt.Classifier_FCN_MT(output_directory, input_shape, nb_classes_1, nb_classes_2, verbose)
        
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
root_dir = "/content/drive/My Drive/master thesis/code/dl-4-tsc"
#"G:/My Drive/master thesis/code/dl-4-tsc"

# "/content/drive/My Drive/master thesis/code/dl-4-tsc"

#"G:/My Drive/master thesis/code/dl-4-tsc"


#"Google Colab:" /content/drive/My Drive/master thesis/code/dl-4-tsc"

#"/content/drive/My Drive/master thesis/code/dl-4-tsc"

#'/b/home/uha/hfawaz-datas/dl-tsc-temp/'

if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(ITERATIONS):
                print('\t\titer', iter)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)

                tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

                for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = tmp_output_directory + dataset_name + '/'

                    create_directory(output_directory)

                    fit_classifier()

                    print('\t\t\t\tDONE')

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1] == 'viz_cam':
    viz_cam(root_dir)
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + \
                       dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)

        """
        Read datasets for classification and 
        """

        datasets_dict_1 = read_dataset(root_dir, archive_name, dataset_name)
        datasets_dict_2 = read_dataset(root_dir, archive_name, dataset_name +"_Exp")

        fit_classifier()

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')
