import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def calculate_metrics(y_true, y_pred, model_name):
    """
    Calculates primary regression metrics and triggers diagnostic plots.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- Results for {model_name} ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    
    # Save diagnostic plots to the results folder
    run_diagnostic_plots(y_true, y_pred, model_name)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def run_diagnostic_plots(y_true, y_pred, model_name):
    """
    Generates residual diagnostic plots to check for Normality and Homoskedasticity.
    Essential for validating OLS and ML assumptions.
    """
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram: Checking for Normal Distribution of errors
    sns.histplot(residuals, kde=True, ax=ax1, color='teal')
    ax1.set_title(f'Residual Distribution\n({model_name})')
    
    # Scatter: Checking for Homoskedasticity (Constant Variance)
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title(f'Residuals vs. Predictions\n({model_name})')
    
    plt.tight_layout()
    # Save with a clean filename in the results directory
    plt.savefig(f'results/diagnostics_{model_name.replace(" ", "_")}.png')
    plt.close()

def calculate_vif(X):
    """
    Calculates VIF to identify multicollinearity issues.
    High VIF (>5 or 10) suggests redundant locational features.
    """
    # Create a copy to avoid modifying the original dataframe
    X_vif = X.copy()
    if 'const' not in X_vif.columns:
        X_vif = sm.add_constant(X_vif)
        
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    
    print("\n--- Variance Inflation Factor (VIF) Analysis ---")
    # Hide the constant to focus on the features
    print(vif_data[vif_data['feature'] != 'const']) 
    return vif_data