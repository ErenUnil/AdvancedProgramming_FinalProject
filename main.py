import pandas as pd
import numpy as np
import os
import time
from src.data_loader import load_and_preprocess_data
from src.models import train_ols_model, train_random_forest, train_xgboost
from src.evaluation import evaluate_model, evaluate_ols_model

# Set a consistent random state for full reproducibility
RANDOM_STATE = 42

def main():
    """
    Runs the complete housing price prediction analysis pipeline:
    1. Loads and splits data.
    2. Trains three models (OLS, RF, XGBoost).
    3. Evaluates models and compares performance.
    """
    start_time = time.time()
    print("=========================================================")
    print("|| STARTING HOUSING PRICE PREDICTION PROJECT PIPELINE ||")
    print("=========================================================")
    
    # 1. DATA LOADING AND PREPROCESSING
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 2. MODEL TRAINING
    
    # A. Train OLS Baseline Model
    ols_model = train_ols_model(X_train, y_train)
    
    # B. Train Random Forest Model (with tuning)
    rf_model = train_random_forest(X_train, y_train)
    
    # C. Train XGBoost Model (with tuning)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 3. MODEL EVALUATION
    print("\n\n=========================================================")
    print("|| MODEL EVALUATION AND COMPARISON ||")
    print("=========================================================")
    
    # Evaluate OLS Model
    ols_metrics = evaluate_ols_model(ols_model, X_test, y_test)
    
    # Evaluate Random Forest Model
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest Regressor")
    
    # Evaluate XGBoost Model
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost Regressor")
    
    
    # 4. FEATURE IMPORTANCE ANALYSIS (for ML Models)
    print("\n\n=========================================================")
    print("|| FEATURE IMPORTANCE ANALYSIS (RF vs. XGBoost) ||")
    print("=========================================================")
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # --- Random Forest Importance ---
    rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    print("\nRandom Forest Top 5 Feature Importances:")
    print(rf_importances.nlargest(5))
    
    # --- XGBoost Importance ---
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
    print("\nXGBoost Top 5 Feature Importances:")
    print(xgb_importances.nlargest(5))
    
    
    # 5. FINAL SUMMARY
    print("\n\n=========================================================")
    print("|| FINAL METRICS SUMMARY ||")
    print("=========================================================")
    
    summary_df = pd.DataFrame({
        'OLS Baseline': ols_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }).T
    
    print(summary_df)
    
    end_time = time.time()
    print(f"\nTotal Pipeline Execution Time: {(end_time - start_time):.2f} seconds.")
    print("=========================================================")


if __name__ == '__main__':
    main()