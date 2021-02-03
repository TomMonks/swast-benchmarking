import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor


class AbstractEnsembleVote(ABC):

    @abstractmethod
    def fit(self, train):
        pass

    @abstractmethod
    def predict(self, predictions):
        pass

class UnweightedVote(AbstractEnsembleVote):
    def __init__(self):
        pass

    def fit(self, train, y=None):
        pass

    def predict(self, predictions):
        '''
        Parameters:
        ------
        predictions - numpy.array. matrix of predictions 
        from ensemble of models
        '''

        return np.nan_to_num(predictions).mean(axis=0)


class WeightedVote(AbstractEnsembleVote):
    def __init__(self, weights):
        self._weights = weights

    def fit(self, train, y=None):
        pass

    def predict(self, predictions):
        '''
        Parameters:
        ------
        predictions - numpy.array. matrix of predictions 
        from ensemble of models
        '''

        return np.average(np.nan_to_num(predictions), 
                          weights=self._weights)
        

class RandomForestMetaLearner(AbstractEnsembleVote):
    '''
    Meta Learner for Random Forests
    '''
    def __init__(self):
        pass

    def fit(self, train, y=None):
        self._rgr = RandomForestRegressor()
        print('RF trains using shape ', train.T.shape)
        self._rgr.fit(X=train.T, y=y)

    def predict(self, exog):
        '''
        Parameters:
        ------
        exog - numpy.array. matrix of predictions 
               from ensemble of models
        '''
        print('RF predicts using shape exog.T ', exog.T.shape )
        return self._rgr.predict(exog.T)


class Ensemble(object):
    '''
    BANANA!
    '''
    def __init__(self, estimators, meta_learner, include_exog=False, 
                 rescale_preds=False):
        '''
        Constructor 

        Parameters:
        -------
        estimators - dict of forecast estimators 
        vote - AbstractEnsembleVote object e.g. UnweightedVote

        '''
        self._estimators = estimators
        self._vote = meta_learner
        self._y_train = None
        self._exog = include_exog
        self._rescale_preds = rescale_preds
       

    def _get_fitted(self):
        #broken
        return self._fitted['pred']

    def _get_resid(self):
        #broken
        return self._fitted['resid']
       
    def fit(self, train):

        preds = []

        for key, estimator in self._estimators.items():
            estimator.fit(train)
            preds.append(estimator.fittedvalues)
                    
    
        if len(train.shape) > 1:
            y_train = train[:, 0]
            x_train = train[:, 1:]
        else:
            y_train = None
            x_train = None

        if isinstance(x_train, (np.ndarray)) and self._exog:
            x_train = np.concatenate([preds, x_train.T]).T
        else:
            x_train = np.array(preds)


        #print('fit shape ', x_train.shape)

        self._vote.fit(x_train, y=y_train)
        
        #This needs sorting...
        #self._fitted = pd.DataFrame(y_train)
        #self._fitted.columns=['actual']
        #self._fitted['pred'] = self._vote.combine(np.array(preds))
        #self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']
        
    def predict(self, horizon, return_conf_int=False, alpha=0.2):
        self._preds = []
        self._lower_pi = []
        self._upper_pi = []
        
        for key, estimator in self._estimators.items():
            results = estimator.predict(horizon,    
                                        return_conf_int=return_conf_int,
                                        alpha=alpha)
            if return_conf_int:
                preds, pis = results
                self._preds.append(preds)
                self._lower_pi.append(pis.T[0])
                self._upper_pi.append(pis.T[1])
                
            else:
                self._preds.append(results)

        #print(type(horizon))
        if isinstance(horizon, (np.ndarray, pd.DataFrame)) and self._exog:
            if isinstance(horizon, (pd.DataFrame)):
                horizon = horizon.to_numpy()
            x_train = np.concatenate([self._preds, horizon.T]).T
        else:
            x_train = np.array(self._preds)
            x_train_lower = np.array(self._lower_pi)
            x_train_upper = np.array(self._upper_pi)

        #print('predict shape ', x_train.shape)

        ensemble_preds = self._vote.predict(x_train)
        if return_conf_int:
            ensemble_lower = self._vote.predict(x_train_lower)
            ensemble_upper = self._vote.predict(x_train_upper)

            ensemble_pis = np.array([ensemble_lower, ensemble_upper])
            return ensemble_preds, ensemble_pis.T
        else:
            return ensemble_preds

        #return self._vote.predict(x_train), []
        
    
    fittedvalues = property(_get_fitted)
    resid = property(_get_resid)