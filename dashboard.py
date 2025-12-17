import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Housing Price Predictor", layout="wide")

st.title("🏠 Housing Price Prediction: Locational Effects")
st.markdown("Move the sliders to see how each factor changes the predicted house price.")

# 1. Load the saved model and features
@st.cache_resource
def load_assets():
    model = joblib.load('results/best_random_forest_model.joblib')
    feature_names = joblib.load('results/feature_names.joblib')
    return model, feature_names

try:
    rf_model, feature_names = load_assets()

    # 2. Create Input Sliders for Features
    st.sidebar.header("Adjust House Characteristics")
    input_data = {}
    
    # We will create sliders based on the feature names
    for col in feature_names:
        # Provide sensible ranges based on common housing datasets
        if 'dist' in col or 'cbd' in col or 'inst' in col:
            input_data[col] = st.sidebar.slider(f"{col}", 0.0, 50000.0, 10000.0)
        elif 'nbh' in col:
            input_data[col] = st.sidebar.slider(f"{col}", 1, 10, 5)
        else:
            input_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

    # 3. Make Prediction
    input_df = pd.DataFrame([input_data])
    prediction = rf_model.predict(input_df)[0]

    # 4. Display Result
    st.metric(label="Predicted House Price", value=f"${prediction:,.2f}")

    # 5. Visualizing "The Effect"
    st.subheader("Factor Sensitivity Analysis")
    st.write("This shows how the price changes if you vary one factor while keeping others constant.")
    
    selected_factor = st.selectbox("Select a factor to analyze:", feature_names)
    
    # Generate data for the 'Effect' plot
    plot_x = np.linspace(0, 50000, 100)
    plot_preds = []
    
    temp_df = input_df.copy()
    for val in plot_x:
        temp_df[selected_factor] = val
        plot_preds.append(rf_model.predict(temp_df)[0])
    
    chart_data = pd.DataFrame({selected_factor: plot_x, 'Predicted Price': plot_preds})
    st.line_chart(chart_data.set_index(selected_factor))

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.write("Make sure you ran main.py recently to generate the .joblib files.")