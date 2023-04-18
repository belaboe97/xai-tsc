from builtins import print
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

#matplotlib.rcParams['font.family'] = 'sans-serif'
#matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import ARCHIVE_NAMES  as ARCHIVE_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import ITERATIONS
from utils.constants import MTS_DATASET_NAMES

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

from scipy.interpolate import interp1d
from scipy.io import loadmat
import tensorflow as tf
#import tensorflow_addons as tfa


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/'  + archive_name + '/' + classifier_name.split('_')[0] + '/' + classifier_name + '/' 
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def read_dataset(root_dir, archive_name, dataset_name, appendix):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + appendix
    x_train, y_train = readucr(file_name + '/' + dataset_name + '_TRAIN')
    x_test, y_test = readucr(file_name + '/' + dataset_name + '_TEST')
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),y_test.copy())
    return datasets_dict


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def generate_results_csv(output_file_name, root_dir):
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir, archive_name)
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                for dataset_name in datasets_dict.keys():
                    output_dir = root_dir + '/results/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv(root_dir + output_file_name, index=False)
    # aggreagte the accuracy for iterations on same dataset
    res = pd.DataFrame({
        'accuracy': res.groupby(
            ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    }).reset_index()

    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()



def save_logs_stl(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics



def save_logs_mtl(output_directory, hist, y_pred_1, y_pred_2, y_true_1, y_true_2, duration, lr=True, y_true_val=None, y_pred_val=None):
    
    

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)


    """
    Task 1:
    """

    df_metrics_1 = calculate_metrics(y_true_1, y_pred_1, duration, y_true_val, y_pred_val)
    df_metrics_1.to_csv(output_directory + 'task1_df_metrics.csv', index=False)

    """
    Task 2: 
    """

    df_metrics_2 = calculate_metrics(y_true_2, y_pred_2, duration, y_true_val, y_pred_val)
    df_metrics_2.to_csv(output_directory + 'task2_df_metrics.csv', index=False)


    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]


    df_best_model = pd.DataFrame(data=np.zeros((1, 8), dtype=np.float64), index=[0],
                                columns=['best_model_train_loss', 'best_model_val_loss', 
                                'best_model_train_acc_1','best_model_train_acc_2',
                                'best_model_val_acc_1', 'best_model_val_acc_2',
                                'best_model_learning_rate', 'best_model_nb_epoch'])
    
    #val_task_1_output_accuracy,val_task_2_output_accuracy
    
    # Loss
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    
    # Accuracy
    df_best_model['best_model_train_acc_1'] = row_best_model['task_1_output_accuracy']
    df_best_model['best_model_train_acc_2'] = row_best_model['task_2_output_accuracy']

    df_best_model['best_model_val_acc_1'] = row_best_model['val_task_1_output_accuracy']
    df_best_model['best_model_val_acc_2'] = row_best_model['val_task_2_output_accuracy']

    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']

    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code
    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics_1, df_metrics_2



def calculate_attributions(root_dir, archive_name, classifier,  dataset_name, appendix, mode, itr, task=1):
    
    import tensorflow as tf
    import tensorflow_addons as tfa
    import tensorflow.keras as keras
    import sklearn


    if os.getenv("COLAB_RELEASE_TAG"):
        from tf.keras.utils import CustomObjectScope
    else: 
        from keras.utils.generic_utils import CustomObjectScope

    max_length = 2000
    

    datasets_dict = read_dataset(root_dir, archive_name, dataset_name, appendix)
    x_train, y_train, x_test, y_test = datasets_dict[dataset_name]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()
    
    orgx_train = x_train
    orgx_test = x_test
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


    with CustomObjectScope({'InstanceNormalization':tfa.layers.InstanceNormalization()}):
        model = keras.models.load_model( f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                                         + f'{classifier.split("_")[0]}/{classifier + itr}_{mode}/{appendix}/' \
                                         + f'/best_model.hdf5', compile=False )
    


    # filters
    # Output is both the original as well as the before last layer
    # layers[-3] : KerasTensor(type_spec=TensorSpec(shape=(None, 150, 128), dtype=tf.float32, name=None), name='activation_8/Relu:0', description="created by layer 'activation_8'") 
    # layers[-1] : KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name=None), name='dense/Softmax:0', description="created by layer 'dense'")
    if mode == 'stl': 
        relu, softm = (-3,-1)
    if mode == 'mtl': 
        if task == 1: 
            relu, softm = (-4,-2)
        elif task == 2:
            relu, softm = (-4,-1)
        
    w_k_c = model.layers[softm].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs

    new_output_layer = [model.layers[relu].output, model.layers[softm].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)
    
    print(x_train.shape,x_test.shape)
    
    output = []

    for orgx_vals,x_vals,y_vals in [[orgx_train,x_train,y_train],[orgx_test,x_test,y_test]]:
        attr = []
        classes = np.unique(y_vals)
        for c in classes:
            ts_class = np.where(y_vals == c)
            #get time series data 
            c_x_train = x_vals[ts_class]
            for idx,ts in enumerate(c_x_train):
                ts_nr=ts_class[0][idx]
                ts = ts.reshape(1, -1, 1)
                [conv_out, predicted] = new_feed_forward([ts])
                pred_label = np.argmax(predicted)
                orig_label = np.argmax(enc.transform([[c]]))
                if True: 
                    cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))
                    for k, w in enumerate(w_k_c[:, orig_label]):
                        cas += w * conv_out[0, :, k] 
                    minimum = np.min(cas)
                    cas = cas - minimum
                    cas = cas / max(cas)
                    cas = cas * 100
                    x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                    # linear interpolation to smooth
                    f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                    y = f(x)
                    f = interp1d(range(ts.shape[1]), cas)
                    cas = f(x).astype(int)
                    attr.append([ts_nr,x,y,cas,pred_label,orig_label,orgx_vals[ts_nr]])
        output.append(sorted(attr, key=lambda x: x[0]))
    return output