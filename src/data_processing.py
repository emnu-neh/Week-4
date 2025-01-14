import pandas as pd
import numpy as np

class DataProcessor:
    @staticmethod
    def load_data(train_path, store_path, test_path=None, nrows=None):
        types = {'StateHoliday': np.dtype(str)}
        train = pd.read_csv(train_path, parse_dates=[2], dtype=types, nrows=nrows)
        store = pd.read_csv(store_path)
        
        if test_path:
            test = pd.read_csv(test_path, parse_dates=[3], dtype=types)
            return train, store, test
        return train, store

    @staticmethod
    def remove_no_sales(train):
        return train.loc[train['Sales'] > 0]

    @staticmethod
    def fill_missing_values(df):
        return df.fillna(0)

    @staticmethod
    def encode_categorical(store, train):
        store['StoreType'] = store['StoreType'].astype('category').cat.codes
        store['Assortment'] = store['Assortment'].astype('category').cat.codes
        train["StateHoliday"] = train["StateHoliday"].astype('category').cat.codes
        return store, train
