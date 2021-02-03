from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from datetime import timedelta

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#from amb_forecast.baseline import boot_prediction_intervals

def array_to_ts(arr, idx):
    '''
    Parameters:
    --------
    arr, - np.ndarray. vector of time series data
    
    idx - datetimeindex
    '''
    return pd.Series(arr, index=idx)

def quick_ts_plot(time_series, series_name=None, ylabel=None, min_x=0, color='black', ax=None):
    plt.style.use('ggplot')
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,9))
    
    if series_name == None: series_name = ''
    ax.plot(time_series[-min_x:], label=series_name, color=color)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')

    return ax





def plot_prediction(model, horizon, ax, model_name=None,
                    intervals=None):
    '''
    Parameters:
    train - pd.Series or np.array.  Contains training time series

    model - forecast model.  Must implemented fittedvalue, resid (properties)

    validation - pd.Series or np.array.  Validation/holdout/test set beyond end of train.
    If set to None it is not included in the plot.  (default=None)

    train_min_x - int, range 0 to len(train). The minimum value x can take.  E.g.
    if 12 then only the last 12 periods of train will be included in the plot.
    (default=0 i.e. the full time series)

    ylabel - str, set the y axis label (default=None)

    bootstrap_intervals - list of floats, if set bootstrap prediction intervals are included.

    '''
    if model_name == None: model_name = 'prediction'
    
    preds = model.predict(horizon)
    
    if isinstance(preds, np.ndarray):
        t = ax.lines[0].get_xdata().max()
        index = pd.date_range(t + np.timedelta64(1,'D'), 
                              periods=horizon, freq='D')
    
        preds = array_to_ts(preds, index)
    else:
        index = preds.index

    ax.plot(preds, linewidth=2, label=model_name)


    if intervals != None:

        if isinstance(model.resid, np.ndarray):
            resid = pd.DataFrame(model.resid).dropna()
        else:
            resid = model.resid.dropna()

        pis = boot_prediction_intervals(preds.to_numpy(), 
                                        resid,
                                        horizon,
                                        levels=intervals
                                        )
        
        pis = model.prediction_interval(horizon, levels=intervals)

        for pi, interval in zip(pis, intervals):
            limits = pd.DataFrame(pi, index=index, columns=['lower', 'upper'])
            ax.fill_between(index, limits['lower'], limits['upper'], 
                            alpha=.1, 
                            label=str(model_name)+' ' + str(interval*100)+'% PI')
    

    ax.legend(loc='upper left')
    return ax
    
def plot_predictions_grid(estimators, ylabel, horizon, train, val=None, 
                          min_x=10, figsize=None):
    '''
    Plots a grid of predictions

    Parameters:
    ---------
    estimators - dict of forecast objects

    ylabel - str, label for y axis

    horizon - int, forecast horizon

    train - array-like, training data for estimators to produce forecasts

    val - array-like, validation data (default=None)

    min_x - int, users may not wish to plot all training data. This limits the plot 
    to the last min_x data points. (default=10)

    figsize - tuple, size of figure (default = (20,8))
    '''
    if figsize is None:
        figsize = (20,8)
        
    fig, axes = plt.subplots(int(len(estimators)/2)+len(estimators)%2, 2, 
                             figsize=figsize)
    axes = axes.flatten() 
    ax_index = 0
    
    for key, model in estimators.items():
             
        model.fit(train)
        quick_ts_plot(train, series_name='Train', ylabel=ylabel, min_x=min_x, 
                      ax=axes[ax_index])
        plot_prediction(model=model, model_name=key, horizon=horizon, 
                        ax=axes[ax_index])
        
        if isinstance(val, (np.ndarray, pd.Series)):
            quick_ts_plot(val[:horizon], series_name='Validation', 
                          ylabel=ylabel, color='green', ax=axes[ax_index])
        
        ax_index += 1

    return fig, axes  