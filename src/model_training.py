import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def rmspe(y, y_hat):
    return np.sqrt(np.mean(((y - y_hat) / y) ** 2))

class ModelTrainer:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=2,
            tree_method='exact',
            reg_alpha=0.05,
            silent=0,
            random_state=1023
        )

    def train(self, X, y, test_size=0.1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=10)
        
        self.model.fit(
            X_train, np.log1p(y_train),
            eval_set=[(X_train, np.log1p(y_train)), (X_test, np.log1p(y_test))],
            early_stopping_rounds=300
        )
        
        return self.model

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
