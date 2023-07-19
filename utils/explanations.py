import numpy as np 
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.constants import CAM_LAYERS
import tensorflow.keras as keras
from keras.utils import CustomObjectScope
import tensorflow_addons as tfa
import sklearn
from sklearn.preprocessing import normalize

import os

def minmax_norm(ts): 
    return (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

def norm(values): 
    if not type(values) == np.ndarray:
        return normalize(values.numpy().reshape(1,-1))[0]
    else: 
        return normalize(values.reshape(1,-1))[0]

def get_layer_index(model, layer_name):
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(layer_name)
    return layer_idx

def calculate_cam_attributions(root_dir, archive_name, classifier, dataset_name, data_source, experiment=1):

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
    
    model_path = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                    f'/experiment_{experiment}/{classifier.split("_")[:-1][0]}/'\
                    f'{classifier}/{data_source}/best_model.hdf5'  


    #load model 
    #model_path = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
    #                                    + f'{classifier.split("_")[0]}/{classifier}/{data_source}/' \
    #                                    + f'best_model.hdf5'
    
    model = keras.models.load_model(model_path ,compile=False)

    for layer in model.layers: 
        print(layer.name)
    
    #get gap and output layer
    #print(CAM_LAYERS[classifier.split("_")[0]])

    gap = CAM_LAYERS[classifier.split("_")[0]]["last_conv_layer"]
    gap = get_layer_index(model, gap)
    out = CAM_LAYERS[classifier.split("_")[0]]["task_1"]
    out = get_layer_index(model, out)

    w_k_c = model.layers[out].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    new_output_layer = [model.layers[gap].output, model.layers[out].output]
    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)
    output = []
    
    y_pos = list(np.unique(y_train))
    # Calculate classwise attribution gap to output
    for orgx_vals,x_vals,y_vals in [[orgx_train,x_train,y_train],[orgx_test,x_test,y_test]]:
        attr = list()
        for idx,ts in enumerate(x_vals):
            ts = ts.reshape(1, -1, 1)
            cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))
            for k, w in enumerate(w_k_c[:,np.argmax(predicted)]): #y_pos.index(y_vals[idx])]): #np.argmax(predicted)
                cas += w * conv_out[0, :, k] 
            attr.append([y_vals[idx],orgx_vals[idx],cas])
        output.append(attr)
    return output

def calculate_gradcam_attributions(root_dir, archive_name, classifier, dataset_name, data_source, experiment=1, scale='None'):

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
    
    model_path = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                    f'/experiment_{experiment}/{classifier.split("_")[:-1][0]}/'\
                    f'{classifier}/{data_source}/best_model.hdf5'  
    #load model 
    #model_path = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
    #                                    + f'{classifier.split("_")[0]}/{classifier}/{data_source}/' \
    #                                    + f'best_model.hdf5'
    
    model = keras.models.load_model(model_path ,compile=False)

    for layer in model.layers: 
        print(layer.name)
    
    #get gap and output layer
    #print(CAM_LAYERS[classifier.split("_")[0]])

    gap = CAM_LAYERS[classifier.split("_")[0]]["last_conv_layer"]
    gap = get_layer_index(model, gap)
    out = CAM_LAYERS[classifier.split("_")[0]]["task_1"]
    out = get_layer_index(model, out)

    w_k_c = model.layers[out].get_weights()[0]  # weights for each filter k for each class c

    grad_model = tf.keras.models.Model([model.inputs], [model.layers[gap].output,  model.layers[out].output])
    output = []
    # Calculate classwise attribution gap to output
    for orgx_vals,x_vals,y_vals in [[orgx_train,x_train,y_train],[orgx_test,x_test,y_test]]:
        attr = list()
        
        y_pos = list(np.unique(y_train))

        #conv_out, predicted = new_feed_forward([ts])
        with tf.GradientTape() as tape:
            [conv_out, predicted] = grad_model(x_vals) 
            pred_index = tf.math.argmax(predicted[0])
            tf.print(pred_index)
            class_channel = predicted[:, pred_index]
    
        grads = tape.gradient(tf.convert_to_tensor(class_channel), tf.convert_to_tensor(conv_out))
        #tape.gradient(class_channel, conv_out)
        #tape.gradient(tf.convert_to_tensor(class_channel), tf.convert_to_tensor(conv_out))
        pooled_grads = tf.reduce_mean(grads, axis=(0))

        tf.print(pooled_grads)

        for idx,ts in enumerate(x_vals):
            ts = ts.reshape(1, -1, 1)
            cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))
            [conv_out, predicted] = grad_model([ts]) 
            for k, w in enumerate(w_k_c[:,np.argmax(predicted)]): #y_pos.index(y_vals[idx])]): #np.argmax(predicted)
                cas += w * conv_out[0, :, k] * pooled_grads[:,k]
            #print(cas, np.mean(cas))
            cas = cas / np.mean(cas)
            if scale == 'minmax': 
                cas = cas - np.min(cas) / (np.max(cas) - np.min(cas))
            elif scale == 'normalized':
                cas = norm(cas)
            attr.append([y_vals[idx],orgx_vals[idx],cas])
        output.append(attr)
    return output



import tensorflow as tf 
def interpolate_series(baseline,
                       series,
                       alphas):
  alphas_x = alphas[:,tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(series, axis=0)
  delta = tf.expand_dims((series - baseline),axis=0)
  series = baseline_x +  alphas_x * delta
  return series

def compute_gradients(series,model,target_class_idx,task=0):
  #tf.print("SERIES",series.shape)
  with tf.GradientTape() as tape:
    tape.watch(series)
    logits = model(series)
    #check for singletask
    #tf.print(logits.shape)
    if task == 0: 
        logits = logits[:,target_class_idx]
        #tf.print(len(logits))
    elif task == 1: 
        logits = logits[0]
        #print(target_class_idx)
        logits = logits[:,target_class_idx]
    elif task== 2: 
        logits = logits[1]
  return tape.gradient(logits, series)


def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

@tf.function
def integrated_gradients(model,
                         baseline,
                         series,
                         target_class,
                         m_steps=64,
                         batch_size=4,
                         task = 0):
  # 1. Generate alphas.
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
  # Initialize TensorArray outside loop to collect gradients.    
  gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    # 2. Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_series(baseline=baseline,
                                                       series=series,
                                                       alphas=alpha_batch)
    
    # 3. Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(series=interpolated_path_input_batch,
                                       model=model,
                                       target_class_idx=target_class,
                                       task=task)

    # Write batch indices and gradients to extend TensorArray.
    gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

  # Stack path gradients together row-wise into single tensor.
  total_gradients = gradient_batches.stack()

  # 4. Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # 5. Scale integrated gradients with respect to input.
  #tf.print(avg_gradients)
  integrated_gradients = (series - baseline) * avg_gradients

  return integrated_gradients


def calculate_ig_attributions(root_dir, archive_name, classifier, dataset_name, 
                              data_source, datasets_dict = None, task=0, experiment=1, scale='None'):
     

    model_path = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
                    f'/experiment_{experiment}/{classifier.split("_")[:-1][0]}/'\
                    f'{classifier}/{data_source}/last_model.hdf5'  
    print(model_path)   
    #model_path = f'{root_dir}/results/{archive_name}/{dataset_name}/' \
    #                                + f'{classifier.split("_")[0]}/{classifier}/{data_source}/' \
    #                                + f'last_model.hdf5'
    model = keras.models.load_model(model_path ,compile=False)

    if datasets_dict == None: 
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name, 'original', 1)
        x_train, y_train, x_test, y_test = datasets_dict[dataset_name]
    
    else: 
        x_train, y_train, x_test, y_test = datasets_dict

    output = list()
    baseline = tf.zeros(len(x_train[0]))
    
    #tf.random.uniform((1,x_train.shape[1]),minval=-1,maxval=1) # tf.zeros(len(x_train[0]))
    y_pos = list(np.unique(y_train))
    for x_vals,y_vals in [[x_train,y_train],[x_test,y_test]]:
        pred = model.predict(x_vals) if task == 0 else model.predict(x_vals)[0]
        attr = list()
        for idx,ts in enumerate(x_vals):
            series = ts
            ig_att = integrated_gradients(model,baseline,series.astype('float32'),
                                        np.argmax(pred[idx]),
                                        task=task)
                                        #optimize for true values
                                        #y_pos.index(y_vals[idx]),
            if scale == 'minmax': 
                ig_att = minmax_norm(ig_att)
            if scale == 'normalized': 
                ig_att = norm(ig_att)
            attr.append([y_vals[idx],x_vals[idx],ig_att])
        output.append(attr)
    return output

    
def create_explanations(attributions, scaling='None'):
    output = []
    for split in attributions:
        explanations = []
        for ts in split: 
            x_values = ts[1]
            if scaling == 'minmax':
                attributions = (ts[2] - np.min(ts[2])) / (np.max(ts[2]) - np.min(ts[2]))
            elif scaling == 'normalized': 
                pass
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