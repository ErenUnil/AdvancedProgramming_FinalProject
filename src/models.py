import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import xgboost as xgb
import statsmodels.api as sm

# Use the same random state as the data loader for full reproducibility
RANDOM_STATE = 42

# --- 1. Baseline Model: OLS Linear Regression ---

def train_ols_model(X_train, y_train):
    """
    Trains an OLS model using statsmodels for full econometric reporting.
    (Note: statsmodels requires adding a constant for the intercept)
    """
    print("\n--- Training OLS Model with statsmodels ---")

    # Add a constant (intercept) term required by statsmodels
    X_train_sm = sm.add_constant(X_train)

    # Instantiate and fit the OLS model
    ols_model = sm.OLS(y_train, X_train_sm).fit()

    # Print the summary immediately for the terminal output
    print(ols_model.summary()) # THIS PROVIDES THE ECONOMETRIC TABLE

    return ols_model

# --- 2. ML Model 1: Random Forest Regression (with basic tuning) ---

def train_random_forest(X_train, y_train):
    """Trains a Random Forest Regressor with basic hyperparameter tuning."""
    print("\n--- 4. Training Random Forest Regressor ---")
    
    # Define the hyperparameters to search over (basic grid search)
    param_grid = {
        'n_estimators': [100, 200],  # Number of trees
        'max_depth': [10, 20],        # Max depth of each tree
        'random_state': [RANDOM_STATE]
    }
    
    # Initialize the base model
    base_model = RandomForestRegressor(random_state=RANDOM_STATE)
    
    # Use GridSearchCV for tuning (finds best combination of parameters)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,  # Use 3-fold cross-validation
        verbose=1,
        n_jobs=-1 # Use all available processors
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best RF Parameters found: {grid_search.best_params_}")
    
    # Return the best trained model
    return grid_search.best_estimator_

# --- 3. ML Model 2: XGBoost Regression (with basic tuning) ---

def train_xgboost(X_train, y_train):
    """Trains an XGBoost Regressor with basic hyperparameter tuning."""
    print("\n--- 5. Training XGBoost Regressor ---")
    
    # Define the hyperparameters to search over
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.01],
        'random_state': [RANDOM_STATE]
    }
    
    # Initialize the base model
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)
    
    # Use GridSearchCV for tuning
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best XGBoost Parameters found: {grid_search.best_params_}")
    
    # Return the best trained model
    return grid_search.best_estimator_
