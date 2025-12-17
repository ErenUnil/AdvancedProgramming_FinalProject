# Housing Price Prediction: Locational & Environmental Analysis

This project investigates the impact of geographic and environmental factors on housing prices using the **Wooldridge `hprice3` dataset**. It demonstrates a Data Science workflow, combining econometric with machine learning techniques.

---

## 1. Project Goal & Research Question
**Research Question:** How do locational factors (distance to the city center, proximity to services) impact house prices, and can non-linear models (Random Forest, XGBoost) outperform a classical linear baseline?

The project focuses on:
* **Object-Oriented Programming (OOP):** Modular and reusable code architecture.
* **Econometric Rigor:** VIF analysis for multicollinearity and residual diagnostics.
* **Machine Learning:** Hyperparameter tuning via Cross-Validation (`GridSearchCV`).

---

## 2. Project Architecture
The repository follows industry-standard modularity to ensure code maintainability:

* **`src/`**: Core logic and backend modules.
    * `data_loader.py`: Automated data fetching and 80/20 splitting.
    * `housing_class.py`: Encapsulates all models into a single OOP interface.
    * `models.py`: Handles ML logic and Hyperparameter tuning.
    * `evaluation.py`: Computes RMSE, MAE, R² and generates diagnostic plots.
* **`notebooks/`**: Contains `data_exploration.ipynb` for initial data visualization and feature analysis.
* **`results/`**: Stores saved models (`.joblib`), feature lists, and diagnostic charts.
* **`data/raw/`**: Contains documentation regarding dynamic data acquisition.
* **`main.py`**: The central pipeline that orchestrates training and evaluation.
* **`dashboard.py`**: A Streamlit web application for real-time price simulation.

---

## 3. Installation & Reproducibility
This project uses **Conda** to ensure that all library versions are consistent across different machines.

1. **Create the environment:** `conda env create -f environment.yml`
2. **Activate the environment:** `conda activate housing-project`

 A fixed `RANDOM_STATE = 42` is implemented across all scripts to guarantee reproducible results.

---

## 4. Execution Workflow

### Step 1: Exploratory Data Analysis (EDA)
Before running the models, you can explore the data distribution and correlations in the notebook: `notebooks/data_exploration.ipynb`.

### Step 2: Running the Pipeline
Execute the main script to fetch data, train models, and generate all necessary assets:
**Command:** `python main.py`

* **VIF Analysis:** Printed in the console to detect redundant features.
* **Metrics:** RMSE, MAE, and R² are displayed for OLS, Random Forest, and XGBoost.
* **Assets:** Trained models and plots are automatically saved to the `results/` folder.

### Step 3: Interactive Dashboard
To use the trained model for real-time price simulations based on locational factors:
**Command:** `streamlit run dashboard.py`

---

## 5. Methodology & Evaluation
The pipeline generates diagnostic plots in the `results/` folder for each model to verify:

1. **Normality:** Checking if prediction errors are normally distributed using residual histograms.
2. **Homoskedasticity:** Checking if the error variance is constant using scatter plots.