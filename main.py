import pandas as pd
import os
import time
import joblib
from src.data_loader import load_and_preprocess_data
from src.evaluation import calculate_vif
from src.housing_class import HousingModel

def main():
    start_time = time.time()
    print("=========================================================")
    print("|| STARTING ENCAPSULATED HOUSING PREDICTION PIPELINE   ||")
    print("=========================================================")
    
    # 1. DATA LOADING AND PREPROCESSING
    # Modular data handling as per software engineering standards (Lecture 13)
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # --- 2. ACADEMIC DIAGNOSTICS ---
    # Detecting Multicollinearity using Variance Inflation Factor (Lecture 6c)
    calculate_vif(X_train)
    
    # --- 3. MODELING VIA ENCAPSULATION (OOP Approach) ---
    # A. OLS Baseline (Econometric Approach)
    ols_project = HousingModel(model_type="OLS")
    ols_project.train(X_train, y_train)
    ols_metrics = ols_project.evaluate(X_test, y_test)
    
    # B. Random Forest (Ensemble Learning - Bagging)
    rf_project = HousingModel(model_type="RF")
    rf_project.train(X_train, y_train)
    rf_metrics = rf_project.evaluate(X_test, y_test)
    
    # C. XGBoost (Ensemble Learning - Boosting)
    xgb_project = HousingModel(model_type="XGB")
    xgb_project.train(X_train, y_train)
    xgb_metrics = xgb_project.evaluate(X_test, y_test)
    
    # 4. FEATURE IMPORTANCE ANALYSIS
    print("\n\n=========================================================")
    print("|| FEATURE IMPORTANCE ANALYSIS (RF vs XGB) ||")
    print("=========================================================")
    
    # Extracting importance from Random Forest (Lecture 8/9)
    rf_importances = pd.Series(rf_project.model.feature_importances_, index=X_train.columns)
    print("\nRandom Forest Top 5 Features:")
    print(rf_importances.nlargest(5))
    
    # Extracting importance from XGBoost (Lecture 10/11)
    # This allows comparing how Bagging vs Boosting weight locational factors
    xgb_importances = pd.Series(xgb_project.model.feature_importances_, index=X_train.columns)
    print("\nXGBoost Top 5 Features:")
    print(xgb_importances.nlargest(5))
    
    # 5. FINAL SUMMARY TABLE
    print("\n\n=========================================================")
    print("|| FINAL METRICS SUMMARY ||")
    print("=========================================================")
    summary_df = pd.DataFrame({
        'OLS Baseline': ols_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }).T
    print(summary_df)

    # 6. SAVING ASSETS FOR DASHBOARD
    os.makedirs('results', exist_ok=True)
    joblib.dump(rf_project.model, 'results/best_random_forest_model.joblib')
    joblib.dump(X_train.columns.tolist(), 'results/feature_names.joblib')
    
    print(f"\nPipeline Execution Complete in {(time.time() - start_time):.2f}s.")
    print("=========================================================")

if __name__ == '__main__':
    main()