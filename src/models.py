import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Use the same random state as the data loader for full reproducibility
RANDOM_STATE = 42

# --- 1. ML Model: Random Forest Regression ---

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Regressor using Grid Search Cross-Validation.
    Bagging helps reduce variance in locational price predictions.
    """
    print("\n--- Training Random Forest (Grid Search) ---")
    
    param_grid = {
        'n_estimators': [100, 200],  # Number of trees
        'max_depth': [10, 20, None], # Depth control
        'random_state': [RANDOM_STATE]
    }
    
    base_model = RandomForestRegressor(random_state=RANDOM_STATE)
    
    # 3-fold CV ensures the model generalizes well to new house data
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3, 
        verbose=1,
        n_jobs=-1 
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best RF Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

# --- 2. ML Model: XGBoost Regression ---

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost Regressor. 
    Boosting focuses on minimizing residuals from previous trees.
    """
    print("\n--- Training XGBoost (Grid Search) ---")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.05],
        'random_state': [RANDOM_STATE]
    }
    
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_
