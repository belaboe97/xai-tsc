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