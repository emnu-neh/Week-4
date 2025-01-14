import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from src.feature_engineering import FeatureEngineer
from src.data_visualization import DataVisualizer

def setup_directories():
    os.makedirs('output/visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def load_data():
    print("Loading data...")
    train = pd.read_csv('data/master/train.csv', 
                       parse_dates=['Date'], 
                       low_memory=False,
                       dtype={
                           'StateHoliday': str,
                           'SchoolHoliday': int,
                           'StoreType': str,
                           'Assortment': str
                       })
    store = pd.read_csv('data/master/store.csv')
    test = pd.read_csv('data/master/test.csv', parse_dates=['Date'])
    return train, store, test

def process_data(train, store, test):
    fe = FeatureEngineer()
    
    # Create store metrics
    store = fe.create_store_metrics(train, store)
    
    # Create time features
    train = fe.create_time_features(train)
    test = fe.create_time_features(test)
    
    # Encode categoricals
    store['StoreType'] = store['StoreType'].astype('category').cat.codes
    store['Assortment'] = store['Assortment'].astype('category').cat.codes
    train['StateHoliday'] = train['StateHoliday'].astype('category').cat.codes
    test['StateHoliday'] = test['StateHoliday'].astype('category').cat.codes
    
    # Merge data
    merged_train = pd.merge(train, store, on='Store', how='left')
    merged_test = pd.merge(test, store, on='Store', how='left')
    
    # Create competition features
    merged_train = fe.create_competition_features(merged_train)
    merged_test = fe.create_competition_features(merged_test)
    
    # Fill missing values
    merged_train.fillna(0, inplace=True)
    merged_test.fillna(0, inplace=True)
    
    return merged_train, merged_test

def get_feature_columns():
    return [
        'Store', 'CompetitionDistance', 'Promo', 'Promo2', 
        'StateHoliday', 'StoreType', 'Assortment',
        'AvgSales', 'AvgCustomers', 'MedSales', 'MedCustomers',
        'DayOfWeek', 'Week', 'Day', 'Month', 'Year',
        'MonthsCompetitionOpen'
    ]

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        eta=0.02,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=42
    )
    
    model.fit(X_train[get_feature_columns()], np.log1p(y_train))
    return model

def predict_sales(model, test_data):
    features = get_feature_columns()
    y_hat = np.expm1(model.predict(test_data[features]))
    submission = pd.DataFrame({
        "Id": test_data.Id,
        'Sales': y_hat
    })
    submission.loc[test_data['Open'] == 0, 'Sales'] = 0
    submission.to_csv('output/submission.csv', index=False)
    return submission

def main():
    setup_directories()
    
    train, store, test = load_data()
    
    print("Creating visualizations...")
    viz = DataVisualizer()
    viz.plot_sales_trend(train)
    viz.plot_store_types(train, store)
    viz.plot_holidays(train)
    viz.plot_promotions(train)
    
    print("Processing data...")
    merged_train, merged_test = process_data(train, store, test)
    
    print("Training model...")
    X_train, X_val, y_train, y_val = train_test_split(
        merged_train, merged_train['Sales'], 
        test_size=0.2, random_state=42
    )
    
    model = train_model(X_train, y_train)
    
    print("Generating predictions...")
    predictions = predict_sales(model, merged_test)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
