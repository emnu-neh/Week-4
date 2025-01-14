# utilities.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def configure_plotting():
    """
    Set common plotting configurations to ensure consistent visualization style.
    """
    plt.rcParams['figure.figsize'] = (12.0, 10.0)
    plt.style.use('seaborn-darkgrid')


def calculate_rmspe(y_true, y_pred):
    """
    Calculate Root Mean Square Percentage Error (RMSPE).
    
    Parameters:
    - y_true: array-like of true values
    - y_pred: array-like of predicted values
    
    Returns:
    - rmspe: RMSPE value
    """
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def plot_importance(feature_names, feature_importances, output_file='feature_importance.png'):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    - feature_names: list of feature names
    - feature_importances: array-like of feature importances
    - output_file: file path to save the plot
    """
    indices = np.argsort(feature_importances)
    plt.barh(range(len(indices)), feature_importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def log_missing_values(df, name='DataFrame'):
    """
    Log missing values for each column in the DataFrame.
    
    Parameters:
    - df: DataFrame to check for missing values
    - name: Name of the DataFrame for logging purposes
    """
    missing = df.isnull().sum()
    logging.info(f'Missing values in {name}:')
    for col, val in missing.items():
        logging.info(f'{col}: {val} missing values')


def fill_na(df, column_name, fill_value):
    """
    Fill NaN values in a specified column of a DataFrame.
    
    Parameters:
    - df: DataFrame containing the column
    - column_name: name of the column to fill NaN values
    - fill_value: value to replace NaN with
    
    Returns:
    - df: DataFrame with NaN values filled in the specified column
    """
    df[column_name].fillna(fill_value, inplace=True)
    logging.info(f'Filled NaN values in {column_name} with {fill_value}')
    return df


def preprocess_data(df):
    """
    Preprocess the data to handle missing values and encode categorical variables.
    
    Parameters:
    - df: DataFrame containing the Rossmann sales data
    
    Returns:
    - df: Preprocessed DataFrame
    """
    # Fill missing values for 'CompetitionDistance'
    df = fill_na(df, 'CompetitionDistance', df['CompetitionDistance'].max())
    
    # Fill missing values for 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear'
    df['CompetitionOpenSinceMonth'].fillna(1, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(df['CompetitionOpenSinceYear'].min(), inplace=True)
    
    # Fill missing values for 'Promo2SinceWeek' and 'Promo2SinceYear'
    df['Promo2SinceWeek'].fillna(1, inplace=True)
    df['Promo2SinceYear'].fillna(df['Promo2SinceYear'].min(), inplace=True)
    
    # Encode 'StateHoliday' as numeric
    df['StateHoliday'] = df['StateHoliday'].map({'0': 0, 'a': 1, 'b': 2, 'c': 3})
    
    return df


def add_date_features(df):
    """
    Add additional date-based features to the DataFrame.
    
    Parameters:
    - df: DataFrame containing the Rossmann sales data
    
    Returns:
    - df: DataFrame with additional date features
    """
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    return df


def engineer_features(df):
    """
    Engineer features for the Rossmann sales prediction model.
    
    Parameters:
    - df: DataFrame containing the Rossmann sales data
    
    Returns:
    - df: DataFrame with engineered features
    """
    df = preprocess_data(df)
    df = add_date_features(df)
    
  
    
    return df
