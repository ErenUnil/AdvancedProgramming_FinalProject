import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def calculate_metrics(y_true, y_pred, model_name):
    """
    Calculates and prints the primary regression metrics (RMSE, MAE, R^2).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- Results for {model_name} ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    
    # Run diagnostic plots for every model evaluated
    run_diagnostic_plots(y_true, y_pred, model_name)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def evaluate_model(model, X_test, y_test, model_name):
    """
    Handles prediction and metric calculation for scikit-learn/XGBoost models.
    """
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred, model_name)

def evaluate_ols_model(ols_model, X_test, y_test):
    """
    Handles prediction and metric calculation for the statsmodels OLS model.
    """
    X_test_sm = sm.add_constant(X_test, has_constant='add')
    y_pred = ols_model.predict(X_test_sm)
    return calculate_metrics(y_test, y_pred, "OLS Linear Regression (Baseline)")

def run_diagnostic_plots(y_true, y_pred, model_name):
    """
    Generates residual diagnostic plots to check for Normality and Homoskedasticity.
    """
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram: Checking for Normal Distribution of errors
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title(f'Residual Distribution - {model_name}')
    
    # Scatter: Checking for constant variance (Homoskedasticity)
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title(f'Residuals vs. Predictions - {model_name}')
    
    plt.tight_layout()
    plt.savefig(f'results/diagnostics_{model_name.replace(" ", "_")}.png')
    plt.close() # Closes figure to save memory

def calculate_vif(X):
    """
    Calculates Variance Inflation Factor (VIF) to identify multicollinearity.
    """
    if 'const' not in X.columns:
        X = sm.add_constant(X)
        
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("\n--- Variance Inflation Factor (VIF) Analysis ---")
    print(vif_data[vif_data['feature'] != 'const']) 
    return vif_data