import logging
from abc import ABC, abstractmethod

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor


class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        pass



class RandomForestModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(X_train, y_train)
        return reg
    
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int('n_estimator', 1, 200)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_sample_split = trial.suggest_int('min_sample_split',2,20)
        reg = self.train(X_train, y_train, n_estimators= n_estimators, max_depth=max_depth, min_samples_split=min_sample_split)
        return reg.score(X_test, y_test)

class LightGBMModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        reg = LightGBMModel(**kwargs)
        reg.fit(X_train, y_train)
        return reg
    
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimator = trial.suggest_int("n_estimator", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_int('learning_rate', 0.01, 0.99)
        reg = self.train(X_train, y_train, n_estimator=n_estimator, max_depth=max_depth, learning_rate=learning_rate)
        return reg.score(X_test, y_test)

# TODO: Create a XGBoost model like 
# TODO: Create a Linear RgressionModel
# TODO: Create a HyperParameterTuner  



