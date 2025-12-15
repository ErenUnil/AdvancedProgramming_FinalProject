import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import statsmodels.api as sm

def calculate_metrics(y_true, y_pred, model_name):
    """
    Calculates and prints the primary regression metrics (RMSE, MAE, R^2).
    
    Args:
        y_true (np.array): True target values (from y_test).
        y_pred (np.array): Predicted target values from the model.
        model_name (str): The name of the model being evaluated.
    
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R-squared (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    # Print results neatly
    print(f"\n--- Results for {model_name} ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    
    return metrics

def evaluate_model(model, X_test, y_test, model_name):
    """
    Generically handles prediction and metric calculation for scikit-learn/XGBoost models.
    """
    # 1. Predict on the test set
    y_pred = model.predict(X_test)
    
    # 2. Calculate and return metrics
    return calculate_metrics(y_test, y_pred, model_name)

def evaluate_ols_model(ols_model, X_test, y_test):
    """
    Handles prediction and metric calculation for the statsmodels OLS model.
    """
    # 1. statsmodels requires the constant column for prediction
    X_test_sm = sm.add_constant(X_test)
    
    # 2. Predict on the test set
    y_pred = ols_model.predict(X_test_sm)
    
    # 3. Calculate and return metrics
    return calculate_metrics(y_test, y_pred, "OLS Linear Regression (Baseline)")