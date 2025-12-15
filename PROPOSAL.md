# Project Proposal: Predicting Housing Prices from Location & Environmental Factors

## 1. Project Goal and Research Question

This project investigates how location-based and environmental factors influence housing prices, and evaluates whether advanced machine learning models can improve price prediction relative to a classical linear regression baseline.

**Research Question:** How do location-based and environmental factors influence housing prices, and can machine learning models (Random Forest, Gradient Boosting) significantly improve price prediction relative to a classical Linear Regression baseline?

## 2. Data and Methodology

### 2.1. Data Acquisition and Preprocessing
The project will use the **Wooldridge `hprice3` housing dataset**. This dataset is robust and contains key features for our analysis, including geographic distance to amenities, proximity to a waste facility, and year indicators.

The `src/data_loader.py` module will be responsible for:
* **Loading:** Importing the data using the `pandas` library.
* **Cleaning:** Initial checks for missing values or extreme outliers. Given the nature of the dataset, minimal cleaning is anticipated.
* **Feature Engineering:** Creating necessary dummy variables for categorical features (if any exist) and ensuring all features are numerical for model input.
* **Splitting:** Separating the data into features ($\mathbf{X}$) and the target variable (price, $y$).

### 2.2. Model Comparison and Implementation
The project will compare three distinct models to evaluate performance across varying levels of complexity:

1.  **Baseline Model (Linearity Check):** Multiple Linear Regression (OLS) from the `statsmodels` library. This model serves as a benchmark for interpretability and linear relationship testing.
2.  **Machine Learning Model 1 (Ensemble/Bagging):** **Random Forest Regression** (`scikit-learn`). This model handles non-linear relationships and interactions well, often reducing overfitting compared to single decision trees.
3.  **Machine Learning Model 2 (Ensemble/Boosting):** **Gradient Boosting Regression** (specifically, XGBoost). This model is known for high predictive accuracy by iteratively correcting the errors of preceding models.

### 2.3. Training and Hyperparameter Tuning
Models will be trained using a standard **80% training / 20% testing split** for evaluation. To ensure reproducibility and fair comparison, a fixed `random_state=42` will be used for all splits and model initializations.

A basic **Grid Search** or **Random Search** cross-validation approach will be utilized on the training set to identify optimal hyperparameters for the Random Forest and XGBoost models. The baseline OLS model requires no tuning.

## 3. Evaluation

Model performance will be rigorously compared using standard regression metrics, implemented within the `src/evaluation.py` module:

* **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors, with larger errors weighted more heavily.
* **Mean Absolute Error (MAE):** Measures the average magnitude of the errors without considering their direction.
* **Coefficient of Determination ($R^2$ score):** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

Beyond predictive accuracy, a crucial part of the evaluation will involve analyzing the **Feature Importance** output from the Random Forest and XGBoost models to determine which environmental factors are the most significant drivers of house prices.