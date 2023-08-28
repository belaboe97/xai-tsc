import numpy as np


def calculate_accuaracy_change(model,xvals,yvals,attributions, mtl=False): 
    
    # predict
    prediction =  model.predict(xvals)
    # get originally predicted values // we test for mtl learning // each dataset has more than one ts in STL
    if mtl: 
        original_prediction = prediction[0] 
    else:     
        original_prediction = prediction 
    #@todo
    #prediction if len(prediction) < 2 else prediction[0] # prediciton 0 --> accuracy # prediction 1 --> feature att estimates 
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
        lerf_sorted = sorted_vals[:int(len(sorted_vals)*0.1)]
        morf_sorted = sorted_vals[::-1][:int(len(sorted_vals)*0.1)]
        #print(lerf_sorted == morf_sorted)
        #print(lerf_sorted, morf_sorted)
        # replace and create new explanations 
        lerf_ts = xvals[ts].copy()
        morf_ts = xvals[ts].copy()
        lerf_ts[lerf_sorted] = 0
        morf_ts[morf_sorted] = 0
        lerf_tss.append(lerf_ts)
        morf_tss.append(morf_ts)
        #print(morf_tss)
        # predict 

    if mtl: 
        lerf_preds = model.predict(np.array(lerf_tss))[0]
        morf_preds = model.predict(np.array(morf_tss))[0]
    else: 
        lerf_preds = model.predict(np.array(lerf_tss))
        morf_preds = model.predict(np.array(morf_tss))
    #print(lerf_preds, morf_preds)

    for pred, ls, ms in  zip(original_prediction,lerf_preds,morf_preds):

        ypred_label = np.argmax(pred)

        #print(ypred_label)

        #print(pred[ypred_label],ls[ypred_label])

        #print(ypred_label,pred[ypred_label] - ls[ypred_label],pred[ypred_label],ls[ypred_label], ms[ypred_label])
        #print(pred[ypred_label],ms[ypred_label], ls[ypred_label], pred[ypred_label] - ls[ypred_label])
        mean_change_zero_imputation_lerf += (ls[ypred_label] - pred[ypred_label] ) / pred[ypred_label] #np.abs
        mean_change_zero_imputation_morf += (ms[ypred_label] - pred[ypred_label] ) / pred[ypred_label] #np.abs
    #print(mean_change_zero_imputation_lerf)
    mean_change_zero_imputation_lerf /= len(lerf_tss)
    mean_change_zero_imputation_morf /= len(morf_tss)

    return mean_change_zero_imputation_lerf, mean_change_zero_imputation_morf




"""

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

"""