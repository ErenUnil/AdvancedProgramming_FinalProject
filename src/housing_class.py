import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from src.evaluation import calculate_metrics

class HousingModel:
    def __init__(self, model_type="RF"):
        self.model_type = model_type
        # Mapping for professional file naming in results folder
        self.full_names = {"OLS": "Linear_Regression", "RF": "Random_Forest", "XGB": "XGBoost"}
        self.model = None
        self.feature_names = None
        self.metrics = None

    def train(self, X_train, y_train):
        self.feature_names = X_train.columns.tolist()
        if self.model_type == "OLS":
            X_sm = sm.add_constant(X_train, has_constant='add')
            self.model = sm.OLS(y_train, X_sm).fit()
        elif self.model_type == "RF":
            from src.models import train_random_forest
            self.model = train_random_forest(X_train, y_train)
        elif self.model_type == "XGB":
            from src.models import train_xgboost
            self.model = train_xgboost(X_train, y_train)
        print(f"Model {self.model_type} trained successfully.")

    def evaluate(self, X_test, y_test):
        if self.model_type == "OLS":
            X_sm = sm.add_constant(X_test, has_constant='add')
            y_pred = self.model.predict(X_sm)
        else:
            y_pred = self.model.predict(X_test)
            
        # Passing the full name for clearer diagnostic plot filenames
        self.metrics = calculate_metrics(y_test, y_pred, self.full_names[self.model_type])
        return self.metrics

    def save(self, path):
        joblib.dump(self, path)