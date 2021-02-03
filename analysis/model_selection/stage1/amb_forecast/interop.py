from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

from fbprophet import Prophet
import pandas as pd
import numpy as np

class SimpleExpSmoothingWrapper(object):
    def __init__(self):
        pass

    def fit(self, train):
        return SimpleExpSmoothing(train).fit()
        

class HoltsWinters(object):
    '''
    A wrapper class for statsmodels.tsa.holtwinters.ExponentialSmoothing
    The facade allows the object to work with the cross validation framework.
    '''
    def __init__(self, trend=None, damped=False, seasonal_periods=None, seasonal=None):
        self._seasonal = seasonal
        self._seasonal_periods = seasonal_periods
        self._damped = damped
        self._trend = trend

    def fit(self, train):
        return ExponentialSmoothing(train,
                                    seasonal=self._seasonal,
                                    seasonal_periods=self._seasonal_periods,
                                    trend=self._trend,
                                    damped=self._damped).fit()



class ARIMAWrapper(object):
    def __init__(self, order, seasonal_order, exog=None):
        self._order = order
        self._seasonal_order = seasonal_order
        self._exog = exog
    
    def fit(self, train):
        model =  SARIMAX(train,
                       order=self._order,
                       seasonal_order=self._seasonal_order,
                       exog = self._exog,
                       initialization='approximate_diffuse')
        
        fitted = model.fit()

        return fitted

        

        


class StatsModelsForecastObject(object):
    '''
    Facade for statsmodels.tsa.holtswinters.ExponentialSmoothing
    Double Exponential Smoothing

    Useful because ES interface is different and allows
    for time series cross validation to be called in same 
    way as other baseline forecasts.
    '''
    def __init__(self, model):
        self._model = model

    def _get_resids(self):
        return self._fitted.resid

    def _get_preds(self):
        return self._fitted.fittedvalues

    def fit(self, train):
        self._fitted = self._model.fit(train)
        self._t = len(train)
    
    def predict(self, horizon):
        return self._fitted.predict(self._t, self._t+horizon-1)
    
    fittedvalues = property(_get_preds)
    resid = property(_get_resids)


class StateSpaceExponentialSmoothing(object):
    '''
    Facade for statsmodels.statespace
    '''
    def __init__(self, trend=False, damped_trend=False, seasonal=None):
        self._trend = trend
        self._seasonal= seasonal
        self._damped_trend = damped_trend

    def _get_resids(self):
        return self._fitted.resid

    def _get_preds(self):
        return self._fitted.fittedvalues

    def fit(self, train):
        self._model = sm.tsa.statespace.ExponentialSmoothing(endog=train,
                                                             trend=self._trend, 
                                                             damped_trend=self._damped_trend,
                                                             seasonal=self._seasonal)
        self._fitted = self._model.fit()
        self._t = len(train)
    
    def predict(self, horizon, return_conf_int=False, alpha=0.2):
        
        forecast = self._fitted.get_forecast(horizon)
        
        mean_forecast = forecast.summary_frame()['mean'].to_numpy()
        
        if return_conf_int:
            df = forecast.summary_frame(alpha=alpha)
            pi = df[['mean_ci_lower', 'mean_ci_upper']].to_numpy()
            return mean_forecast, pi
            
        #if return_conf_int:
        #    pi = []
        #    for interval in intervals:
                
        #        df = self._fitted.summary_frame(alpha=interval)
        #        pi.append(df[['mean_ci_lower', 'mean_ci_upper']].to_numpy())
                
        #    return mean_forecast, pi
        
        else:
            return mean_forecast

    fittedvalues = property(_get_preds)
    resid = property(_get_resids)

    
class StateSpaceARIMA(object):
    '''
    Facade for statsmodels.statespace
    '''
    def __init__(self, order, seasonal_order):
        self._order = order
        self._seasonal_order = seasonal_order

    def _get_resids(self):
        return self._fitted.resid

    def _get_preds(self):
        return self._fitted.fittedvalues

    def fit(self, train, exog=None):
        self._model = ARIMA(endog=train,
                            exog=exog,
                            order=self._order, 
                            seasonal_order=self._seasonal_order,
                            enforce_stationarity=False)
        self._fitted = self._model.fit()
        self._t = len(train)
    
    def predict(self, horizon, exog=None, return_conf_int=False, alpha=0.2):
        
        forecast = self._fitted.get_forecast(horizon, exog=exog)
        mean_forecast = forecast.summary_frame()['mean'].to_numpy()
        
        if return_conf_int:
            df = forecast.summary_frame(alpha=alpha)
            pi = df[['mean_ci_lower', 'mean_ci_upper']].to_numpy()
            return mean_forecast, pi
            
        #if return_conf_int:
        #    pi = []
        #    for interval in intervals:
                
        #        df = self._fitted.summary_frame(alpha=interval)
        #        pi.append(df[['mean_ci_lower', 'mean_ci_upper']].to_numpy())
                
        #    return mean_forecast, pi
        
        else:
            return mean_forecast

    fittedvalues = property(_get_preds)
    resid = property(_get_resids)    
    

    
class Regression(object):
    '''
    Facade for statsmodels.tsa.holtswinters.ExponentialSmoothing
    Dgouble Exponential Smoothing

    Useful because ES interface is different and allows
    for time series cross validation to be called in same 
    way as other baseline forecasts.
    '''
    def __init__(self):
        pass

    def _get_resids(self):
        return self._fitted.resid

    def _get_preds(self):
        return self._fitted.fittedvalues

    def fit(self, train):

        #dropped iloc - tjhis will need to be handled.
        self._y_train = train[:,0]
        train_exog = sm.add_constant(train[:,1:], prepend=False)
        self._model = sm.OLS(self._y_train, train_exog)
        self._fitted = self._model.fit()
        self._t = len(self._y_train)
    
    def predict(self, exog, return_pred_int=False, alpha=0.05):
        pred_exog = sm.add_constant(exog, prepend=False)
        return self._fitted.predict(pred_exog)

    fittedvalues = property(_get_preds)
    resid = property(_get_resids)


class FbProphetWrapper(object):
    '''
    Facade for FBProphet object - so that it can be
    used within Ensemble with methods from other packages

    '''
    def __init__(self, training_index, holidays=None, interval_width=0.8,
                 mcmc_samples=0, changepoint_prior_scale=0.05,
                 daily_season=False):
        self._training_index = training_index
        self._holidays = holidays
        self._interval_width = interval_width
        self._mcmc_samples = mcmc_samples
        self._cp_prior_scale = changepoint_prior_scale
        self._daily_season = daily_season

    def _get_resids(self):
        return self._train - self._forecast['yhat'][:-self._h]

    def _get_preds(self):
        return self._forecast['yhat'][:-self._h].to_numpy()

    def fit(self, train):
        
        self._model = Prophet(holidays=self._holidays, 
                              interval_width=self._interval_width,
                              mcmc_samples=self._mcmc_samples,
                              changepoint_prior_scale=self._cp_prior_scale,
                              daily_seasonality=self._daily_season)
        
        self._model.fit(self._pre_process_training(train))
        self._t = len(train)
        self._train = train
        self.predict(len(train))


    def _pre_process_training(self, train):

        if len(train.shape) > 1:
            y_train = train[:, 0]
        else:
            y_train = train

        y_train = np.asarray(y_train)
            
        #hack!!
        if len(y_train) > len(self._training_index):
            self._training_index = pd.date_range(start=self._training_index[0], 
                                                 periods=len(y_train),
                                                 freq=self._training_index.freq)
        
        
        prophet_train = pd.DataFrame(self._training_index)
        prophet_train['y'] = y_train
        prophet_train.columns = ['ds', 'y']
        
        return prophet_train

    def predict(self, h, return_conf_int=False, alpha=0.2):

        if isinstance(h, (np.ndarray, pd.DataFrame)):
            h = len(h)
        
        self._h = h
        future = self._model.make_future_dataframe(periods=h)
        self._forecast = self._model.predict(future)

        if return_conf_int:
            return (self._forecast['yhat'][-h:].to_numpy(), 
                    self._forecast[['yhat_lower', 'yhat_upper']][-h:].to_numpy())
        else:
            return self._forecast['yhat'][-h:].to_numpy()
            

    fittedvalues = property(_get_preds)
    resid = property(_get_resids)