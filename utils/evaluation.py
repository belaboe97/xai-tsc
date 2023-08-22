import numpy as np

def calculate_accuaracy_change(model,xvals,yvals,attributions): 
    
    # predict
    prediction =  model.predict(xvals)
    # get originally predicted values // we test for mtl learning // each dataset has more than one ts in STL
    original_prediction = prediction if len(prediction) < 2 else prediction[0] # prediciton 0 --> accuracy # prediction 1 --> feature att estimates 
    # mean scores
    mean_change_zero_imputation_lerf = 0
    mean_change_zero_imputation_morf = 0
    #mean_change_randn_imputation_morf = 0
    #mean_change_randn_imputation_lerf = 0

    #sore transformed timeseries
    lerf_tss = []
    morf_tss = []
    # loop over timeseries to replace higest and lowest percentile
    for ts in range(len(attributions)): 
        # sort values 
        sorted_vals = np.argsort(attributions[ts])
        #remove lowest 10th percentile
        lerf_sorted = sorted_vals[:int(sorted_vals*0.1)]
        morf_sorted = sorted_vals[:int(sorted_vals*0.9)]
        # replace and create new explanations 
        lerf_ts = sorted_vals.copy()[lerf_sorted] = 0
        morf_ts = sorted_vals.copy()[morf_sorted] = 0
        lerf_tss.append(lerf_ts)
        morf_tss.append(morf_ts)
        # predict 

    for pred, ls, ms in  zip(original_prediction,lerf_tss,morf_tss)
        ypred_label = np.argmax(pred)
        mean_change_zero_imputation_lerf += np.abs(pred[ypred_label] - ls[ypred_label]) 
        mean_change_zero_imputation_morf += np.abs(pred[ypred_label] - ms[ypred_label])
    

    mean_change_zero_imputation_lerf /= len(lerf_tss)
    mean_change_zero_imputation_morf /= len(morf_tss)

    #sup_x[sorted_vals[:idx]] = np.nan 
    # Indices of non-NaN values
    #not_nan_indices = np.arange(len(sup_x))[~np.isnan(sup_x)]
    # Linearly interpolate NaN values
    #interpolated_array = np.interp(np.arange(len(sup_x)), not_nan_indices, sup_x[not_nan_indices])
    #sup_array.append(interpolated_array)

def visualize_predictions_flipped(model,xvals,yvals,attributions): 

    for ts in range(len(attributions)): 

        sorted_vals = np.argsort(attributions[ts])
        xvals = xvals
        ytrue  = yvals[ts]

        pred_label = np.argmax(model.predict(ig_data_stl[])[0][ts])
        #print(xvals)
        sup_array = []
        if pred_label == 0:
            for idx in range(0, len(sorted_vals), 1):
                sup_x  = xvals.copy()
                #print(sorted_vals[:idx])
                #print(sorted_vals[:idx])
                sup_x[sorted_vals[:idx]] = 0
                sup_array.append(sup_x)
                #sup_x[sorted_vals[:idx]] = np.nan 
                # Indices of non-NaN values
                #not_nan_indices = np.arange(len(sup_x))[~np.isnan(sup_x)]
                # Linearly interpolate NaN values
                #interpolated_array = np.interp(np.arange(len(sup_x)), not_nan_indices, sup_x[not_nan_indices])
                #sup_array.append(interpolated_array)


            sup_array = np.array(sup_array)
            pred = ig_model.predict(sup_array)
            plt.plot(pred[0][:,pred_label])
    plt.show()
plt.draw()