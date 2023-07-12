from tensorflow import keras

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # Load new data for each epoch
        x_train, y_train = load_data()
        
        # Update the training data and labels in the model's parameters
        self.params['data'] = x_train
        self.params['labels'] = y_train


    def on_epoch_end(self, data, logs=None):
        import numpy as np
        import sklearn
        from ...utils.constants import CAM_LAYERS
        from ...utils.explanations import get_layer_index

        # Make predictions using model.predict

        # transform to binary labels
        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(np.concatenate((y_train), axis=0).reshape(-1, 1))
        y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()
        
        x_train = self.params['data']
        y_train = self.params['labels']
        
        predictions = self.model.predict(x_train)

        #get gap and output layer
        gap = CAM_LAYERS["fcn"]["gap_layer"]
        gap = get_layer_index(self.model, gap)
        out = CAM_LAYERS["fcn"]["task_1"]
        out = get_layer_index(self.model, out)

            
        w_k_c = self.model.layers[out].get_weights()[0]  # weights for each filter k for each class c

        # the same input
        new_input_layer = self.model.inputs
        new_output_layer = [self.model.layers[gap].output, self.model.layers[out].output]
        new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)
        output = []

        classes = np.unique(y_train)
        
                # Calculate classwise attribution gap to output
        for idx,x_vals,y_vals in enumerate(x_train,y_train):
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])
            cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))
            pred_label = np.argmax(predicted[0])
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                for k, w in enumerate(w_k_c[:,int(y_vals[idx]-1)]):
                    cas += w * conv_out[0, :, k] 
                y_train

            y_train[idx] = cas

        return output


        for c in classes:
            print(np.where(y_train == c)[0])
            c_x_train = x_train[np.where(y_train == c)[0]]
            for ts in c_x_train:
                ts = ts.reshape(1, -1, 1)
                #print(ts.shape)
                [conv_out, predicted] = new_feed_forward([ts])
                pred_label = np.argmax(predicted[0])
                orig_label = np.argmax(enc.transform([[c]]))
                if pred_label == orig_label:
                    cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))
                    for k, w in enumerate(w_k_c[:, orig_label]):
                        cas += w * conv_out[0, :, k]

        # Custom logic using the predictions
        custom_function(predictions)