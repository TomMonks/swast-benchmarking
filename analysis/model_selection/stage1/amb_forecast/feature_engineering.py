'''
Classes and functions to support feature engineering for time series.
'''

import numpy as np
import pandas as pd
from itertools import combinations

def integer_month_or_day_to_string(i):
    if len(str(i)) == 1:
        return '0' + str(i)
    else:
        return str(i)


def regular_busy_calender_days(data, quantile=0.98):
    '''
    Returns an array of dates representing the top quantile
    days in a time series

    Parameters:
    --------
    data - pandas.Series, the univariate time series

    quantile - float, the percentile threshold e.g. 0.99 returns the top 0.01 of time periods
    default=(0.98)

    Returns:
    -------
    numpy.ndarray
    '''
    data.rename('actual', inplace=True)
    upper = data.quantile(quantile)
    exceptional = pd.DataFrame(data.loc[data > upper])

    exceptional['month'] = exceptional.index.month
    exceptional['day'] = exceptional.index.day
    
    count_of_days = exceptional.groupby(by=['month', 'day']).count()
    top = count_of_days.loc[count_of_days['actual'] > 1]
    top = top.index.to_numpy()
    
    years = np.arange(data.index.min().year, data.index.max().year+1)
    
    days = []
    for year in years:
        for row in top:
            days.append(integer_month_or_day_to_string(year)+'-'+ 
                        integer_month_or_day_to_string(row[0])+'-'+
                        integer_month_or_day_to_string(row[1]))
            
    return np.array(days, dtype='datetime64[ns]')  

def add_lags(data, to_add):
    '''
    Creates additional feastures that are 
    sequential lags of the data series passed in.
    
    Parameters:
    --------
    data - pandas.DataFrame, time series.  Can be one or more time series.
    index must be a datetimeindex.
    
    Returns:
    ------
    pandas.DataFrame
    '''
    lag_list = []
    index = 0
    for lag in to_add:
        lag_list.append(data.shift(lag))
        lag_list[index].columns = [col + '_lag' + str(lag) for col in data.columns]
        index += 1
    
    return pd.concat(lag_list, axis=1)


def featurize_time_series(y, X=None,
                          max_lags=7, 
                          include_interactions=True, 
                          exceptional_days_method='auto', 
                          exceptional_percentile=0.99,
                          exceptional_days=None):
    '''
    Creates Features of time series for dynamic regression.
    1. lags of the y variable
    2. week day dummy variables (6 variables)
    3. month dummy variables (11 variables)
    4. Exceptional days - days above exceptional_percentile (default=0.99)
    if include_interactions = True:
        4. Weekday and season interactions
        
    Parameters:
    ----------
    y - pandas.DataFrame, time series y variable. 
    X - pandas.DataFrame, the x values of the time series (default=None)
    
    max_y_lags - int, maximum lags of the y variable to include (default=7)
    
    include_interactions - bool, include interation between weekday and month (default=True)
    
    exceptional_days_method - str, how are exceptional days calculated?
    'auto' = include exceptional days above a given exceptional_percentile threshold 
    'manual' = you have supplied the exceptional_days arraylike parameter containing a list of datetimes
    'none' = no exceptional days are supplied.
    
    Returns:
    ------
    tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
    0. lagged y and x variables
    1. calendar / seasonal dummies and interactions
    2. exceptional day dummies
    
    '''
    
    allowed = ['auto', 'manual', 'none']
    
    if exceptional_days_method not in allowed:
        raise ValueError("exceptional_days_method, must be auto; manual; none")
    
    lags_to_create = [i for i in range(1, max_lags+1)]
    
    #create lags
    lags = add_lags(pd.DataFrame(pd.concat([y, X], axis=1)), lags_to_create)

    #if needed create exceptional days
    if exceptional_days_method != 'none':
        if exceptional_days_method == 'auto':
            exceptional = regular_busy_calender_days(y, quantile=exceptional_percentile)
        else:
            exceptional = exceptional_days
        exceptional = y.index.isin(exceptional).astype(int)
        exceptional = pd.DataFrame(exceptional)
        exceptional.columns = ['top']
        exceptional.index = y.index
    else:
        exceptional = None

    months = pd.get_dummies(y.index.month,  prefix='m', drop_first=True)

    days = pd.get_dummies(y.index.weekday, prefix='dow', drop_first=True)

    quarters = pd.get_dummies(y.index.quarter, prefix='q', drop_first=True)

    seasons = pd.concat([months, days], axis=1)

    df = seasons.copy()
    
    if include_interactions:
        for pair in combinations(df.columns, 2):
            new_col = '*'.join(pair)
            df[new_col] = df[pair[0]] * df[pair[1]]

        for col in df.columns:
            if len(df[col].unique()) == 1:
                df.drop(col,inplace=True,axis=1)
        seasonal_dummy_titles = list(df.columns)    

    seasonal_dummies = df
    seasonal_dummies = pd.concat([seasonal_dummies, quarters], axis=1)
   
    seasonal_dummies['t'] = seasons.index + 1
    seasonal_dummies.index = y.index
    
    return (lags, seasonal_dummies, exceptional)
