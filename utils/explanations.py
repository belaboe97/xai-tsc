import numpy as np 
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.constants import CAM_LAYERS
import tensorflow.keras as keras
import sklearn
import os

def get_layer_index(model, layer_name):
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(layer_name)
    return layer_idx

def calculate_cam_attributions(root_dir, archive_name, classifier, dataset_name, data_source):

    #load original data 
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)
    x_train, y_train, x_test, y_test = datasets_dict[dataset_name]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()
    
    orgx_train = x_train
    orgx_test = x_test
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    #load model 
    model_path = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                                        + f'{classifier.split("_")[0]}/{classifier}/{data_source}/' \
                                        + f'best_model.hdf5'
    
    model = keras.models.load_model(model_path ,compile=False)
    
    #get gap and output layer
    gap = CAM_LAYERS[classifier.split("_")[0]]["gap_layer"]
    gap = get_layer_index(model, gap)
    out = CAM_LAYERS[classifier.split("_")[0]]["task_1"]
    out = get_layer_index(model, out)

            
    w_k_c = model.layers[out].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    new_output_layer = [model.layers[gap].output, model.layers[out].output]
    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)
    output = []

    # Calculate classwise attribution gap to output
    for orgx_vals,x_vals,y_vals in [[orgx_train,x_train,y_train],[orgx_test,x_test,y_test]]:
        attr = list()
        for idx,ts in enumerate(x_vals):
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])
            cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))

            for k, w in enumerate(w_k_c[:,int(y_vals[idx]-1)]):
                cas += w * conv_out[0, :, k] 
            attr.append([y_vals[idx],orgx_vals[idx],cas])
        output.append(attr)
    return output


def create_cam_explanations(attributions, minmax_norm = False):
    output = []
    for split in attributions:
        explanations = []
        for ts in split: 
            x_values = ts[1]
            if minmax_norm:
                attributions = (ts[2] - np.min(ts[2])) / (np.max(ts[2]) - np.min(ts[2]))
            else: 
                attributions = ts[2]
            explanations.append(np.concatenate((attributions,np.array([x_values])), axis=None))    
        output.append(np.array(explanations))
    return output


def save_explanations(data, root_dir, archive_name, classifier_att_type, dataset_name):
    train_explanation,test_explanation = data
    print(train_explanation.shape, test_explanation.shape)
    dir_path = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + classifier_att_type + '/' 
    create_directory(dir_path)
    np.savetxt(dir_path + dataset_name + "_TRAIN", train_explanation, delimiter=',')
    np.savetxt(dir_path + dataset_name + "_TEST", test_explanation, delimiter=',')
    print("Successfully created explanation done.")