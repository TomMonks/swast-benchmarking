'''
Classes and functions to assist with model selection and evaluation

'''
import numpy as np
import pandas as pd

from amb_forecast.baseline import boot_prediction_intervals

def _drop_na(data):
    if isinstance(data, pd.Series):
        return data.dropna().to_numpy()
    else:
        return data[~np.isnan(data)]


def time_series_cv(model, error_func, train, val, horizons, step=1):
    '''
    Time series cross validation across multiple horizons for a single model.

    Incrementally adds additional training data to the model and tests
    across a provided list of forecast horizons. Note that function tests a
    model only against complete validation sets.  E.g. if horizon = 15 and 
    len(val) = 12 then no testing is done.  In the case of multiple horizons
    e.g. [7, 14, 28] then the function will use the maximum forecast horizon
    to calculate the number of iterations i.e if len(val) = 365 and step = 1
    then no. iterations = len(val) - max(horizon) = 365 - 28 = 337.
    
    Parameters:
    --------
    model - forecasting model

    error_func - function to measure forecast error

    train - np.array - vector of training data

    val - np.array - vector of validation data

    horizon - list of ints, forecast horizon e.g. [7, 14, 28] days

    step -- step taken in cross validation 
            e.g. 1 in next cross validation training data includes next point 
            from the validation set.
            e.g. 7 in the next cross validation training data includes next 7 points
            (default=1)
            
    Returns:
    -------
    np.array - vector of forecast errors from the CVs.
    '''
    cvs = []

    #change here: max(horizons) + 1
    for i in range(0, len(val) - max(horizons) + 1, step):
        
        train_cv = np.concatenate([train, val[:i]], axis=0)
        model.fit(train_cv)
        
        #predict the maximum horizon 
        preds = model.predict(horizon=len(val[i:i+max(horizons)]))


        horizon_errors = []
        for h in horizons:
            #would be useful to return multiple prediction errors in one go.
            pred_error = error_func(preds[:h], val[i:i+h])
            horizon_errors.append(pred_error)
        
        cvs.append(horizon_errors)
    
    return np.array(cvs)

def time_series_cv2(model, error_func, train, val, horizons, levels=None, step=1):
    '''
    Time series cross validation across multiple horizons for a single model.

    Incrementally adds additional training data to the model and tests
    across a provided list of forecast horizons. Note that function tests a
    model only against complete validation sets.  E.g. if horizon = 15 and 
    len(val) = 12 then no testing is done.  In the case of multiple horizons
    e.g. [7, 14, 28] then the function will use the maximum forecast horizon
    to calculate the number of iterations i.e if len(val) = 365 and step = 1
    then no. iterations = len(val) - max(horizon) = 365 - 28 = 337.
    
    Parameters:
    --------
    model - forecasting model

    error_func - function to measure forecast error

    train - np.array - vector of training data

    val - np.array - vector of validation data

    horizon - list of ints, forecast horizon e.g. [7, 14, 28] days

    step -- step taken in cross validation 
            e.g. 1 in next cross validation training data includes next point 
            from the validation set.
            e.g. 7 in the next cross validation training data includes next 7 points
            (default=1)
            
    Returns:
    -------
    np.array - vector of forecast errors from the CVs.
    '''
    cvs = []
    cvs_coverage = []

    #change here: max(horizons) + 1
    for i in range(0, len(val) - max(horizons) + 1, step):
        
        train_cv = np.concatenate([train, val[:i]], axis=0)
        model.fit(train_cv)
        
        #predict the maximum horizon 
        preds = model.predict(horizon=len(val[i:i+max(horizons)]))

        if levels != None:
            #construct prediction intervals
            pis = boot_prediction_intervals(preds, _drop_na(model.resid), 
                                            max(horizons), levels=levels)
        
        
        horizon_errors = []
        horizon_coverage = []
        for h in horizons:
            #would be useful to return multiple prediction errors in one go.
            pred_error = error_func(preds[:h], val[i:i+h])
            horizon_errors.append(pred_error)

            if levels != None:
                
                coverage = prediction_interval_coverage(_drop_na(val[i:i+h]), 
                                                        pis[0][0][:h],
                                                        pis[0][1][:h])
                
               
                horizon_coverage.append(coverage)
                            
        cvs.append(horizon_errors)
        cvs_coverage.append(horizon_coverage)
    
    return np.array(cvs), np.array(cvs_coverage)

def batch_ts_cv2(train, val, error_func, horizons, estimators, levels=None, step=1):
    '''
    Time series cross validation batched across a set of estimators and 
    forecast horizons
        
    Parameters:
    --------
    train - array like, (1d numpy vector or pandas series) of training data

    val - array like, (1d numpy vector or pandas series) of validation data

    error_func - func, error function with signature (actual, predications)

    h - int, forecast horizon

    estimators - dict, forecasting objects
    
    step -- step taken in cross validation 
    e.g. 1 in next cross validation training data includes next point 
    from the validation set.
    e.g. 7 in the next cross validation training data includes next 7 points
    (default=1)
            
    Returns:
    --------
    pandas.DataFrame, forecast error for each estimator in each split and forecast
    horizon
    '''
    results = []
    results_coverage = []
    
    for key, estimator in estimators.items():
        results_cv, results_cv_coverage = time_series_cv2(estimator, error_func, 
                                                          train, val, 
                                                          horizons=horizons, 
                                                          levels=levels,
                                                          step=step)
        results.append(results_cv)
        results_coverage.append(results_cv_coverage)
    
    results = np.concatenate(results, axis=1)
    headers = [key + '_h' + str(h) for key in list(estimators.keys()) for h in horizons]
    df_error = pd.DataFrame(results, columns=headers)

    results_coverage = np.concatenate(results_coverage, axis=1)
    headers = [key + '_cov_h' + str(h) for key in list(estimators.keys()) for h in horizons]
    df_coverage = pd.DataFrame(results_coverage, columns=headers)

    return df_error, df_coverage


def batch_ts_cv(train, val, error_func, horizons, estimators, step=1):
    '''
    Time series cross validation batched across a set of estimators and 
    forecast horizons
        
    Parameters:
    --------
    train - array like, (1d numpy vector or pandas series) of training data

    val - array like, (1d numpy vector or pandas series) of validation data

    error_func - func, error function with signature (actual, predications)

    h - int, forecast horizon

    estimators - dict, forecasting objects
    
    step -- step taken in cross validation 
    e.g. 1 in next cross validation training data includes next point 
    from the validation set.
    e.g. 7 in the next cross validation training data includes next 7 points
    (default=1)
            
    Returns:
    --------
    pandas.DataFrame, forecast error for each estimator in each split and forecast
    horizon

    Examples:
    -------
    from amb_forecast.baseline import Naive1, SNaive
    from statsmodels.tools.eval_measures import rmse

    #assume train_data and val_data contain daily level data.

    estimators = {}
    estimators['NF1] = Naive1()
    estimators['SNaive] = SNaive(seasonal_period=7)
    
    horizons = [7, 14, 28, 56, 84, 180, 365]

    results_rmse = batch_ts_cv(train=train_data, 
                               val=val_data, 
                               estimators=estimators, 
                               error_func=rmse,
                               horizons=horizons,
                               step=7)
    '''
    results = []
    
    for key, estimator in estimators.items():
        results_cv = time_series_cv(estimator, error_func, 
                                    train, val, horizons=horizons, step=step)
        results.append(results_cv)
    
    results = np.concatenate(results, axis=1)
    headers = [key + '_h' + str(h) for key in list(estimators.keys()) for h in horizons]
    df = pd.DataFrame(results, columns=headers)
    
    return df



def prediction_interval_coverage(actuals, lower, upper):
    '''
    What percentage of actual data used in CV are covered
    by the prediction intervals.
    '''   
    if isinstance(actuals, pd.Series):
        actuals = actuals.to_numpy()
    
    result = (actuals >= lower) * (actuals <= upper)

    return result.sum() / len(actuals)





