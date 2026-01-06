import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Housing Price Predictor", layout="wide")

st.title("🏠 Housing Price Prediction: Locational Effects (XGBoost)")
st.markdown("Move the sliders to see how each factor changes the predicted house price using the best-performing XGBoost model.")

# 1. Load the saved XGBoost model and features
@st.cache_resource
def load_assets():
    # Updated path to point to the XGBoost model
    model = joblib.load('results/best_xgboost_model.joblib')
    feature_names = joblib.load('results/feature_names.joblib')
    return model, feature_names

try:
    xgb_model, feature_names = load_assets()

    # 2. Create Input Sliders for Features
    st.sidebar.header("Adjust House Characteristics")
    input_data = {}
    
    # Direct use of feature_names to ensure matching consistency
    for col in feature_names:
        if 'dist' in col or 'cbd' in col or 'inst' in col:
            # Range adapted for geographical data (0 to 50k feet)
            input_data[col] = st.sidebar.slider(f"{col}", 0.0, 50000.0, 10000.0)
        elif 'nbh' in col:
            input_data[col] = st.sidebar.slider(f"{col}", 1, 10, 5)
        else:
            input_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

    # 3. Make Prediction with FIXED COLUMN ORDER
    # Force column order using [feature_names] to match XGBoost requirements
    input_df = pd.DataFrame([input_data])[feature_names] 
    prediction = xgb_model.predict(input_df)[0]

    # 4. Display Result
    st.metric(label="Predicted House Price (XGBoost)", value=f"${prediction:,.2f}")

    # 5. Visualizing "The Effect"
    st.subheader("Factor Sensitivity Analysis")
    st.write("This shows how the price changes if you vary one factor while keeping others constant.")
    
    selected_factor = st.selectbox("Select a factor to analyze:", feature_names)
    
    # Generate data for the 'Effect' plot
    plot_x = np.linspace(0, 50000, 100)
    plot_preds = []
    
    for val in plot_x:
        temp_input = input_data.copy()
        temp_input[selected_factor] = val
        # Always force the column order here as well
        temp_df = pd.DataFrame([temp_input])[feature_names]
        plot_preds.append(xgb_model.predict(temp_df)[0])
    
    chart_data = pd.DataFrame({selected_factor: plot_x, 'Predicted Price': plot_preds})
    st.line_chart(chart_data.set_index(selected_factor))

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.write("Ensure 'main.py' was executed to generate 'results/best_xgboost_model.joblib'.")