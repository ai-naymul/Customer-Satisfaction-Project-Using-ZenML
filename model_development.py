import logging
from abc import ABC, abstractmethod

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor

# TODO Create the base of the Model to use it making models using it in the Models(Linear,XGBoost etc.)
class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        pass


# TODO Create a RandomForest Model 
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

# TODO Create a LightGBMMOdel like the before one
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

class XGBoost(Model):
    def train(self, X_train, y_train, **kwargs):
        reg = XGBoost(**kwargs)
        reg.fit(X_train, y_train)
        return reg
    
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimator = trial.suggest_int('n_estimator', 1, 200)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        learning_rate = trial.suggest_int('learning_rate', 0.01, 0.99)
        reg = self.train(X_train, y_train, n_estimator=n_estimator, max_depth=max_depth, learning_rate=learning_rate)
        return reg.score(X_test, y_test)


# TODO: Create a Linear RgressionModel

class LinearRegression(Model):
    def train(self, X_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(X_train, y_train)
        return reg
    
    def optimize(self, X_train, y_train, X_test, y_test):
        reg = self.train(X_train, y_train)
        return reg.score(X_test, y_test)


# TODO: Create a HyperParameterTuner  

class HyperParameterTuner:
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train, self.X_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params
