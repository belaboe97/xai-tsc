import numpy as np
import sklearn 


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




def run_flipped_pred_test(model,xvals,yvals,attributions, mtl=False):
    
    lerf_flipped_preds = []
    morf_flipped_preds = []

    results_lerf_acc = []
    results_morf_acc = []

    #small workaround to get desired form for comparison
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(yvals.reshape(-1,1))
    transf = enc.transform(yvals.reshape(-1, 1)).toarray()
    yvals = np.array([np.argmax(tff) for tff in transf]) 
    




    baseline_pred = np.argmax(model.predict(np.expand_dims(np.zeros(len(xvals[0])),0)))
    

    # Check each iteration
    for delete_rate in np.arange(0,1.05,0.05): 
        #Split data by train and test
        attr =  attributions
        sup_array_lerf = []
        sup_array_morf = []
        #delete percentile for each timeseries 
        for ts in range(len(attr)):
            #print(new_data[ts],baseline_pred, new_data[ts] == baseline_pred)
            if yvals[ts] == baseline_pred:continue

            #Lerf
            sorted_vals = np.argsort(attr[ts])
            #print(np.sort(attr[ts][2]))
            sorted_vals = sorted_vals[:int(len(attr[ts])*delete_rate)]
            sup_x  = xvals[ts].copy()
            sup_x[sorted_vals] = 0 
            sup_array_lerf.append(sup_x)
            #Morf 
            sorted_vals = np.argsort(attr[ts])[::-1]
            #print(np.sort(attr[ts][2])[::-1])
            #print(sorted_vals)
            
            sorted_vals = sorted_vals[:int(len(attr[ts])*delete_rate)]
            sup_x  =  xvals[ts].copy()
            sup_x[sorted_vals] = 0 
            sup_array_morf.append(sup_x)

            # Indices of non-NaN values
            #not_nan_indices = np.arange(len(sup_x))[~np.isnan(sup_x)]
            # Linearly interpolate NaN values
            #interpolated_array = np.interp(np.arange(len(sup_x)), not_nan_indices, sup_x[not_nan_indices])
        #Get all transformed timeseries data for cycle // 1 indicates test data
        vals_lerf = sup_array_lerf
        vals_morf = sup_array_morf

        
        #lerf_acc = np.array([pr[np.argmax(opr)] for pr,opr in zip(loaded_model.predict(np.array(vals_lerf)),original_pred)]).flatten()
        #morf_acc = np.array([pr[np.argmax(opr)] for pr,opr in zip(loaded_model.predict(np.array(vals_morf)),original_pred)]).flatten()

        
        #results_lerf_acc.append(lerf_acc)
        #results_morf_acc.append(morf_acc)
        print(np.array(vals_lerf).shape)
        if mtl: 
            results_lerf = np.array([np.argmax(pr) for pr in model.predict(np.array(vals_lerf))[0]]).flatten()
            results_morf = np.array([np.argmax(pr) for pr in model.predict(np.array(vals_morf))[0]]).flatten()
        else: 
            results_lerf = np.array([np.argmax(pr) for pr in model.predict(np.array(vals_lerf))]).flatten()
            results_morf = np.array([np.argmax(pr) for pr in model.predict(np.array(vals_morf))]).flatten()
        #print(ogd.flatten()[np.where(ogd.flatten()!=baseline_pred)],results_lerf)

        print(len(results_lerf),len(yvals.flatten()[np.where(yvals.flatten()!=baseline_pred)]))

        lerf_flipped_preds.append(list(yvals.flatten()[np.where(yvals.flatten()!=baseline_pred)]==results_lerf).count(False))
        morf_flipped_preds.append(list(yvals.flatten()[np.where(yvals.flatten()!=baseline_pred)]==results_morf).count(False))

    return lerf_flipped_preds, morf_flipped_preds
