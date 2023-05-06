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
from utils.utils import calculate_pointwise_attributions
from utils.explanations import create_pointwise_explanations
from utils.explanations import save_explanations
import tensorflow as tf
from utils.classifiers import fit_classifier

###### SETTINGS


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


#ARGS  
mode = sys.argv[1]
archive_name = sys.argv[2]
dataset_name = sys.argv[3]
classifier_name = sys.argv[4]
itr = sys.argv[5] if sys.argv[5] != '_itr_0' else ''
data_source = sys.argv[6] 
data_dest = sys.argv[7]
gamma = 0.5 if sys.argv[8] == None else sys.argv[8]
gamma = np.float64(gamma)
classifier = classifier_name + '_' + str(gamma) 



def output_path():
    output_directory = f'{root_dir}/results/{archive_name}/{dataset_name}/{classifier_name.split("_")[0]}/{classifier}/{data_source}/' 
    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)
    return output_directory, test_dir_df_metrics


output_directory, test_dir_df_metrics = output_path()

"""
if os.path.exists(test_dir_df_metrics):
    print('Already done')
"""
if mode == 'singletask':
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)[dataset_name]
    fit_classifier(classifier_name, mode, datasets_dict, None, 
                   output_directory, gamma, EPOCHS, BATCH_SIZE)
    att = calculate_pointwise_attributions(root_dir, archive_name, classifier, 
                                           dataset_name, data_source, mode, task=1)
    exp = create_pointwise_explanations(att, minmax_norm=True)
    save_explanations(exp, root_dir, archive_name, data_dest, dataset_name)

print('DONE')

# the creation of this directory means
create_directory(output_directory + '/DONE')
