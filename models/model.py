import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class SalesModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=7,
            eta=0.02,
            subsample=0.7,
            colsample_bytree=0.6,
            random_state=42
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(
            X_train, 
            np.log1p(y_train),
            eval_set=[(X_val, np.log1p(y_val))] if X_val is not None else None,
            early_stopping_rounds=50,
            verbose=100
        )
        
    def predict(self, X):
        return np.expm1(self.model.predict(X))
        
    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {
            'RMSE': rmse,
            'MAE': mae
        }
