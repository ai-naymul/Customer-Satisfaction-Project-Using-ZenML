# import logging
import logging

# import mlflow
import mlflow

#import pandas
import pandas as pd
# import all the models from the model development part
from model_development import (LightGBMModel, LinearRegression, XGBoost, RandomForestRegressor, HyperParameterTuner)

# import regressor mix from sklearn base
from sklearn.base import RegressorMixin

# import zenml steps
from zenml import steps
# import client from zenml
from zenml.client import Client
# import the modelname config from the config  
from .config import ModelNameConfig


# define  variable name experiment tracker using the cleint active dtakc
experiment_tracker = Client().active_stack.experiment_tracker
# define step with experiment trakcer with name of it own 
@steps(experiment_tracker=experiment_tracker)
# define the train function with the input of trains and test and the config to the MDOelnameConfig
def train_model(X_train, y_train, X_test, y_test, config=ModelNameConfig) -> RegressorMixin:
    model = None
    tune = None

    if config.model_name == 'lightgbm':
        mlflow.lightgbm.autolog()
        model = LightGBMModel()
    elif config.model_name == 'xgboost':
        mlflow.xgboost.autolog()
        model = XGBoost()
    elif config.model_name == 'randomforest':
        mlflow.sklearn.autolog()
        model = RandomForestRegressor()
    elif config.model_name == 'linearregression':
        mlflow.sklearn.autolog()
        model = LinearRegression()
    
    else:
        raise ValueError("Model doesn't supported")

    tuner = HyperParameterTuner(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    if config.fine_tuning:
        best_params = tuner.optimize()
        trained_model = model.train(X_train, y_train, **best_params)
    else:
        trained_model = model.train(X_train, y_train)
    
    return trained_model

