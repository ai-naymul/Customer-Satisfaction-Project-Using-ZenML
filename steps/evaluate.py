# import logging, mlflow, numpy, pandas, anoonateted, tuple
import logging
import mlflow
import numpy as np
import pandas as pd
from typing_extensions import Annotated
from typing_extensions import Tuple

# steps, client
from zenml import step
from zenml.client import Client
# import all mse, r2score, rmse from the model evaluation 
from evaluation import (MSE, RMSE, R2Score)

# regression mixin
from sklearn.base import RegressorMixin

# variable experiment-tracker to the client active stack tracker

experiment_tracker = Client().active_stack.experiment_tracker

# use step wiith the experiment tracker name

@step(experiment_tracker=experiment_tracker.name)
#define function evaluation
# params: model: regressormixin, x-test,y-test 
# -> tuple[anonnotated[float, r_2_score], annotated[float, rmse]]
def evaluation(model: RegressorMixin, X_test, y_test) -> Tuple[Annotated[float, 'r2_score'], Annotated[float, 'rmse']]:
    # create prediction variable to predict model using the x_test
    prediction = model.predict(X_test)
    # create the mse object and call the calculate_score method using the y_test and the prediction
    # log it into the mlflow
    
    # calculate the mean squared error
    mse_class = MSE()
    mse = mse_class.claculate_score(y_test, prediction)
    mlflow.log_metric('mse', mse)

    # calculate the r2score 
    r2_score_class = R2Score()
    r2_score = r2_score_class.claculate_score(y_test, prediction)
    mlflow.log_metric('r2_score', r2_score)

    # calculate the root mean squared error
    rmse_class = RMSE()
    rmse = rmse_class.claculate_score(y_test, prediction)
    mlflow.log_metric('rmse', rmse)

    return r2_score, rmse 