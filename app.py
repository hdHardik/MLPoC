import streamlit as st
import pickle
import os
import pandas as pd
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

# Set the directory where models are stored
MODEL_DIR = "./pkl_model"

def load_model(product_id='Overall Forecast...'):
    """Load the appropriate model based on `product_id`."""
    if product_id != 'Overall Forecast':
        model_filename = f"product_{product_id}_model.pkl"
    else:
        model_filename = "overall_model.pkl"

    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        return None  # Model file not found

    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

# ---- STREAMLIT LAYOUT ----
st.set_page_config(page_title="Demand Forecast Prediction App", layout="wide")

# Sidebar for product selection
with st.sidebar:
    st.title("⚙️ Settings")
    product_ids = ["Overall Forecast...", "4A297", "4A306", "4A299", "4A183", "4A887"]
    product_id = st.selectbox("Select Product ID:", product_ids)

# Main content layout (Left: Sidebar, Right: Full-width Display)
col1, col2 = st.columns([1, 4])  # Allocating 20% width to sidebar, 80% to main content

with col2:  # Full-width area for charts and tables
    st.title("Demand Forecast Prediction App")

    # Load the model
    model_data = load_model(product_id if product_id else None)

    if model_data is None:
        st.error(f"Model for product_id={product_id} not found!")
    else:
        st.success(f"Loaded model: {'Overall Model' if not product_id else f'Product {product_id} Model'}")

        # Extract model and parameters
        loaded_model = model_data["best_model"]
        actual_df = model_data["actual_df"]
        periods = model_data["periods"]
        freq = model_data["freq"]

        # User input for forecasting periods
        user_periods = st.sidebar.slider("Select Forecast Periods", min_value=1, max_value=periods, value=periods)

        # Generate future dataframe
        future = loaded_model.make_future_dataframe(periods=user_periods, freq=freq)

        # Make predictions
        forecast = loaded_model.predict(future)

        # Select the last 'user_periods' records and rename 'yhat' to 'y'
        predicted_df = forecast[["ds", "yhat"]].tail(user_periods).rename(columns={"yhat": "y"}).astype({"ds": str, "y": int})

        # Plot the forecast similar to Facebook Prophet's built-in visualization
        st.subheader("Demand Forecast Plot")
        fig1 = plot_plotly(loaded_model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        # Show trend, seasonality, and other components from Prophet
        st.subheader("Demand Forecast Components (Trend, Seasonality)")
        fig2 = plot_components_plotly(loaded_model, forecast)
        st.plotly_chart(fig2, use_container_width=True)

        # Display forecast table
        st.subheader("Demand    Forecasted Predictions")
        st.dataframe(predicted_df, height=400)  # Better UI for large tables
