import logging
import pandas as pd
 
from zenml import step


class DataIngest:
    def __init__(self) -> None:
        pass

    def get_data(self):
        df = pd.read_csv('data/olist_customers_dataset.csv')
        return df
    

@step
def ingest_data():
    ingest_data = DataIngest()
    df = ingest_data.get_data()
    return df


    