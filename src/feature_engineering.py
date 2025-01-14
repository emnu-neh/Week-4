import pandas as pd
import numpy as np

class FeatureEngineer:
    def create_time_features(self, df):
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Day'] = df['Date'].dt.day
        return df

    def create_store_metrics(self, train, store):
        metrics = train.groupby('Store').agg({
            'Sales': ['mean', 'median', 'std'],
            'Customers': ['mean', 'median', 'std']
        })
        metrics.columns = ['AvgSales', 'MedSales', 'StdSales', 
                         'AvgCustomers', 'MedCustomers', 'StdCustomers']
        metrics.reset_index(inplace=True)
        return pd.merge(store, metrics, on='Store', how='left')

    def create_competition_features(self, df):
        df = df.copy()
        df['MonthsCompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                                     (df['Month'] - df['CompetitionOpenSinceMonth'])
        df['MonthsCompetitionOpen'] = df['MonthsCompetitionOpen'].fillna(0).clip(lower=0)
        return df

    def create_promo_features(self, df):
        df = df.copy()
        df['IsPromo'] = df['Promo'].astype(int)
        df['IsPromo2'] = df['Promo2'].astype(int)
        return df
