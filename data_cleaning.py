from abc import ABC, abstractclassmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataWorks(ABC):
    '''Abstract class defining the strategy for handeling the data'''
    
    @abstractclassmethod
    def handle_data(self, data: pd.DataFrame):
        pass


class DataCleaning(DataWorks):
    def handle_data(self, data: pd.DataFrame):
        
        data.drop([
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date',
            'order_purchase_timestamp'
        ], axis=1)
        
        data['price'].fillna(data['price'].mean(), inplace=True)
        data['product_weight_g'].fillna(data['product_weight_g'].median, inplace=True)
        data['product_length_cm'].fillna(data['product_length_cm'].median, inplace=True)
        data['product_height_cm'].fillna(data['product_height_cm'].median, inplace=True)
        data['product_width_cm'].fillna(data['product_width_cm'].median, inplace=True)
        data['review_score'].fillna(data['review_score'].mean(), inplace=True)
        data['review_comment_message'].fillna("No review", inplace=True)

        data = data.select_dtypes(include=[np.number])
        columns_to_drop = ['customer_zip_code_prefix', 'order_item_id']
        data.drop(columns_to_drop, axis=1)
        
        return data

class DivideData(DataWorks):
    def handle_data(self, data: pd.DataFrame):
        X = data.drop("review_score", axis=1)
        y = data['review_score']

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    

