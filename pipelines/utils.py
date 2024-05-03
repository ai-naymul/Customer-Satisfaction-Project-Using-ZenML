# import logging
import logging
# import pandas
import pandas as pd 
# import datacleaning and the datacleaning strat
from data_cleaning import DataCleaning, DataCleaningStrategy
# method get_datafortest
def get_data_for_test():
    # df to read csv from the data
    df = pd.read_csv('.\data\olist_customers_dataset.csv')
    # df sample method n ==100
    df = df.sample(n=100)
    # define preprocessing strat
    preprocessing_strategy = DataCleaningStrategy()
    # define data clen with the df and preprocessing strat
    data_cleaning = DataCleaning(df, preprocessing_strategy)
    # call handle data data cleaning define it df again
    df = data_cleaning.handle_data()
    # drop review_score 
    df.drop(['review_score'], axis=1, inplace=True)
    # result= to json orient to split
    result = df.to_json(orient='split')
    # return result
    return result