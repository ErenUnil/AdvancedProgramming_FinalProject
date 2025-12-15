import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm # Used to fetch the Wooldridge dataset

# Set a consistent random state for reproducibility across all steps
RANDOM_STATE = 42

def load_and_preprocess_data():
    """
    Loads the Wooldridge hprice3 dataset, performs basic preprocessing,
    and splits the data into training and testing sets (80/20).
    
    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable (price).
        y_test (pd.Series): Testing target variable (price).
    """
    print("\n--- 1. Loading Data ---")
    
    # Use statsmodels to load the built-in Wooldridge hprice3 dataset
    data = sm.datasets.get_rdataset("hprice3", "wooldridge").data
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    
    # --- 2. Feature Selection and Cleanup ---
    
    # The proposal focuses on environmental factors and location.
    # Dropping 'assess', 'bdrms', and 'baths' to simplify the model comparison
    # and focus on the primary features of interest (price, dist, nearinc, etc.)
    data = data.drop(columns=['assess', 'bdrms', 'baths', 'lprice', 'land', 'lland', 'area', 'larea', 'rooms'], errors='ignore')

    # Define Target (y) and Features (X)
    # The target variable (y) is the house price.
    y = data['price']
    
    # The features (X) are all remaining columns excluding the target
    X = data.drop(columns=['price']) 
    
    # --- 3. Splitting Data ---
    print("--- 2. Preprocessing and Splitting Data ---")
    
    # Split the data into 80% training and 20% testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Train set size: {X_train.shape[0]} samples.")
    print(f"Test set size: {X_test.shape[0]} samples.")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # This block allows for direct testing of the function
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("\nData loading and split successful.")
    print(f"Feature set example: {X_train.head(1).columns.tolist()}")