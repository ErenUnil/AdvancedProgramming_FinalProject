import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from src.evaluation import calculate_metrics

class HousingModel:
    def __init__(self, model_type="RF"):
        """
        Initializes the housing model instance.
        model_type: "OLS", "RF" (Random Forest), or "XGB" (XGBoost)
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.metrics = None

    def train(self, X_train, y_train):
        """
        Encapsulates the training logic based on the model type.
        Demonstrates Polymorphism by using a unified interface for different algorithms.
        """
        self.feature_names = X_train.columns.tolist()
        
        if self.model_type == "OLS":
            # OLS requires an explicit intercept constant (Lecture 6a)
            X_sm = sm.add_constant(X_train, has_constant='add')
            self.model = sm.OLS(y_train, X_sm).fit()
        
        elif self.model_type == "RF":
            # Encapsulating Scikit-Learn Random Forest logic (Lecture 8/9)
            from src.models import train_random_forest
            self.model = train_random_forest(X_train, y_train)
            
        elif self.model_type == "XGB":
            # Encapsulating Gradient Boosting logic (Lecture 10)
            from src.models import train_xgboost
            self.model = train_xgboost(X_train, y_train)
        
        print(f"Model {self.model_type} trained successfully.")

    def evaluate(self, X_test, y_test):
        """
        Encapsulates the prediction and evaluation process.
        Automatically handles the structural differences between Statsmodels and SKLearn.
        """
        if self.model_type == "OLS":
            # Re-adding constant for prediction to match training dimensions
            X_sm = sm.add_constant(X_test, has_constant='add')
            y_pred = self.model.predict(X_sm)
        else:
            y_pred = self.model.predict(X_test)
            
        # calculate_metrics now handles printing and saving diagnostic plots
        self.metrics = calculate_metrics(y_test, y_pred, self.model_type)
        return self.metrics

    def save(self, path):
        """Saves the encapsulated model object instance."""
        joblib.dump(self, path)
        print(f"Full class instance saved to {path}")