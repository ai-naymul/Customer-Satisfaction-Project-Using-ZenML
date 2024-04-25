# Import pandas
import pandas as pd
# import tuple from typing
from typing import Tuple
# import all those class from the data cleaning file
from data_cleaning import (DataCleaningStrategy, DivideData, DataCleaning)


# import annotated from typing_extension
from typing_extensions import Annotated
# import steps from Zenml
from zenml import steps

# Using the step create a function named clean-data that return the annnotated x_train y and so on, parameter would a data use the methods from the data cleaning then return all those X_train stuff.


@steps
def clean_data(data) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'y_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.DataFrame, 'y_test']
]:
    preprocess_strategy = DataCleaningStrategy()
    data_cleaning1 = DataCleaning(data, preprocess_strategy)
    preprocessed_data = data_cleaning1.handle_data()

    divide_strategy = DivideData()
    data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
    X_train, X_test, y_train, y_test = data_cleaning.handle_data()
    return X_train, X_test, y_train, y_test