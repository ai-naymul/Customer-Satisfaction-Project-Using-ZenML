# import logging
import logging
# import abc abstractmethod from the abc
from abc import ABC, abstractmethod
# import numpy
import numpy as np
# import r_score , mse from sklearn
from sklearn.metrics import r2_score, mean_squared_error
# define the evaluation clss with abc

class Evaluation(ABC):
    def claculate_score(self, y_true, y_pred):
        pass
# a function name calculate_score params: y_true,y-pred passed

# define class mse with the evaluation calcscore with

class MSE(Evaluation):
    def claculate_score(self, y_true, y_pred):
        logging.info("Claculating the Mean Squared Error")
        mse = mean_squared_error(y_true, y_pred) 
        logging.info(f'The MSE is : {str(mse)}')
        return mse

#y-true, y-pred call mse with the y-true and y_pred 

# SAME AS FOR R2 SCORE CLASS 


class R2Score(Evaluation):
    def claculate_score(self, y_true, y_pred):
        logging.info("Calculating the r2 score")
        r2score = r2_score(y_true, y_pred)
        logging.info(f'R2 score is: {str(r2score)}')
        return r2score





# same as r2 call sqrt within the mse



class RMSE(Evaluation):
    def claculate_score(self, y_true, y_pred):
        logging.info('Calculating the RMSE')
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        logging.info(f"The RMSE is: {str(rmse)}")
        return rmse