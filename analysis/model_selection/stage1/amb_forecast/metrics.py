'''
Metrics to measure forecast error 
These are measures currently not found in sklearn or statsmodels
'''
import numpy as np
from sklearn.metrics import mean_absolute_error

from amb_forecast.baseline import Naive1, SNaive


def mean_absolute_percentage_error(y_true, y_pred): 
    '''
    MAPE

    Parameters:
    --------
    y_true -- np.array actual observations from time series
    y_pred -- the predictions to evaluate

    Returns:
    -------
    float, scalar value representing the MAPE (0-100)
    '''
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_scaled_error(y_true, y_pred, y_train, period=7):
    '''
    Mean absolute scaled error (MASE)

    MASE = MAE / MAE_{insample, snaive}
    '''
    if period is None:
        in_sample = Naive1()
        in_sample.fit(y_train)
    else:
        in_sample = SNaive(period=period)
        in_sample.fit(y_train)
        y_train = y_train.copy()[period:]

    mae_insample = mean_absolute_error(y_train, in_sample.fittedvalues.dropna())
    
    return mean_absolute_error(y_true, y_pred) / mae_insample
