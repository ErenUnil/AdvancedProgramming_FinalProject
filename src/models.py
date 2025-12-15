import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import xgboost as xgb

# Use the same random state as the data loader for full reproducibility
RANDOM_STATE = 42

# --- 1. Baseline Model: OLS Linear Regression ---

def train_ols_model(X_train, y_train):
    """Trains the Ordinary Least Squares (OLS) Linear Regression baseline model."""
    print("\n--- 3. Training OLS Linear Regression (Baseline) ---")
    
    # 1. Add a constant term for the OLS model
    # statsmodels requires an explicit constant for the intercept
    X_train_sm = sm.add_constant(X_train)
    
    # 2. Fit the OLS model
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    
    # For OLS, we return the fitted object which contains summary statistics
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

    def train_svr_model(X_train, y_train):
    """
    Trains a Support Vector Regressor model with a basic grid search.
    """
    print("\n--- Training Support Vector Regressor (SVR) ---")
    
    # SVR requires features to be scaled, which is best handled within the pipeline,
    # but for simplicity, we will rely on the model handling minor differences here.
    base_model = SVR(kernel='rbf') # Radial Basis Function kernel for non-linearity
    
    # Define a basic hyperparameter grid for tuning (keep it small to prevent timeouts)
    param_grid = {
        'C': [1, 10],  # Regularization parameter
        'gamma': ['scale', 0.1], # Kernel coefficient
    }
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2,  # Reduced cross-validation for speed
        verbose=0, # Reduced verbosity
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best SVR Parameters found: {grid_search.best_params_}")
    
    return grid_search.best_estimator_