# import base parameter from zenml steps
from zenml.steps import BaseParameters
# define class name modelnameconfig with the base parameter
class ModelNameConfig(BaseParameters):

# define tege model name lightgbm and the fine tune to false 
    model_name: str = 'lightgbm'
    fine_tuning: bool = False
