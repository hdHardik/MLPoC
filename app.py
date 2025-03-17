import streamlit as st
import pickle
import os
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# Set the directory where models are stored
MODEL_DIR = "model_files"

def load_model(product_id='Overall Forecast'):
    """Load the appropriate model based on `product_id`."""
    model_filename = f"product_{product_id}_model.pkl"
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
    product_ids = ["4A297", "4A306", "4A299", "4A183", "4A887"]
    product_id = st.selectbox("Select Product ID:", product_ids)

# Main content layout (Left: Sidebar, Right: Full-width Display)
col1, _ = st.columns([1, 1])  # Allocating 20% width to sidebar, 80% to main content

st.write("")  # Forces a new line

col2 = st.container()

with col2:  # Full-width area for charts and tables
    st.title("Demand Forecast Prediction App")

    # Load the model and parameters
    model_data = load_model(product_id)
    if model_data is None:
        st.error(f"Model for product_id={product_id} not found!")
    else:
        msg  = f'Sales prediction for product: {product_id}'
        st.success(msg)

        # Extract model, parameters, and actual data
        loaded_model = model_data["best_model"]
        params = model_data["best_params"]
        actual_df = model_data["actual_df"]
        periods = model_data["periods"]
        freq = model_data["freq"]

        # User input for forecasting periods (for the original model forecast)
        user_periods = st.sidebar.slider("Select Forecast Periods", min_value=1, max_value=periods, value=periods)


        # Generate future dataframe for the original loaded model forecast
        future = loaded_model.make_future_dataframe(periods=user_periods, freq=freq)
        forecast = loaded_model.predict(future)

        # Plot the forecast similar to Facebook Prophet's built-in visualization
        st.subheader("Demand Forecast - Visualization")

        # Add explanation text for the chart
        st.markdown("""
               - **─  Prediction Line**: This is the model's predicted demand over time.
               - **🔵 Actual Data Points**: These markers represent the real demand values from the test dataset.
               - **Prediction Confidence Interval (Shaded Area)**: 
                   - The shaded region shows the **95% confidence interval**.
                   - The model is **95% confident** that the true demand will fall within this range.
               """)

        fig1 = plot_plotly(loaded_model, forecast)
        st.plotly_chart(fig1, use_container_width=True)


        # Show trend, seasonality, and other components from Prophet
        st.subheader("Demand Forecast Components (Trend, Seasonality)")
        # Add explanation text for components chart
        st.markdown("""
              - **📈 Trend**: Shows the overall direction of demand over time.
              - **📆 Weekly & Yearly Seasonality**:
                  - Peaks and dips in demand **based on weekly & yearly patterns**.
                  - Useful for identifying high-sales and low-sales periods.
              """)

        fig2 = plot_components_plotly(loaded_model, forecast)
        st.plotly_chart(fig2, use_container_width=True)





        # Display forecast table
        st.subheader("Demand Forecasted Predictions - Data Table")
        # Select the last 'user_periods' records and rename columns
        predicted_df = (forecast[["ds", "yhat"]].tail(user_periods).rename(columns={"ds": "Date", "yhat": "Predicted"}).astype({"Date": str, "Predicted": int}))

        # Display the table
        st.dataframe(predicted_df, height=400, width=400)

        st.subheader("Actual vs. Predicted Demand - Visualization")
        train_size = int(len(actual_df) * 0.95)
        train_df = actual_df.iloc[:train_size][["ds", "y"]]
        test_df = actual_df.iloc[train_size:][["ds", "y"]]

        # Initialize and train a new Prophet model on the training set with stored parameters
        model_95_5 = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            seasonality_mode=params['seasonality_mode']
        )
        model_95_5.fit(train_df)

        # Create a future dataframe covering the test period
        future_95_5 = model_95_5.make_future_dataframe(periods=len(test_df), freq=freq)
        forecast_95_5 = model_95_5.predict(future_95_5)

        # Plot the forecast using Prophet's Plotly function
        fig_95_5 = plot_plotly(model_95_5, forecast_95_5)

        # Overlay the entire actual data as markers
        fig_95_5.add_trace(go.Scatter(
            x=actual_df['ds'],
            y=actual_df['y'],
            mode='markers',
            name='Actual Data',
            marker=dict(color='black', size=4)
        ))

        st.plotly_chart(fig_95_5, use_container_width=True)
        # ---------------------------------------------------------------------

        # Display forecast table
        st.subheader("Actual vs. Predicted Demand - Data Table")
        # Select the last 'user_periods' records and rename 'yhat' to 'y'
        predicted_95_5_df = forecast_95_5[["ds", "yhat"]].tail(len(test_df))

        # Merge both DataFrames on 'ds'
        combined_df = pd.merge(predicted_95_5_df, test_df, on="ds", how="inner")

        # Rename columns
        combined_df = combined_df.rename(columns={"ds": "Date", "yhat": "Predicted", "y": "Actual"}).astype({"Date": str, "Predicted": int})

        # Display the DataFrame
        st.dataframe(combined_df, height=400, width=400)



        # # Display forecast table
        # st.subheader("Actual vs. Predicted Demand - Data Table")
        #
        # # Select the last 'user_periods' records and include 'yhat_lower' and 'yhat_upper'
        # predicted_95_5_df = forecast_95_5[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(len(test_df))
        #
        # # Merge both DataFrames on 'ds'
        # combined_df = pd.merge(predicted_95_5_df, test_df, on="ds", how="inner")
        #
        # # Rename columns
        # combined_df = combined_df.rename(columns={
        #     "ds": "Date",
        #     "yhat": "Predicted",
        #     "y": "Actual",
        #     "yhat_lower": "Lower Trend",
        #     "yhat_upper": "Upper Trend"
        # })
        #
        # # Rearrange columns to show "Predicted Lower Bound" and "Predicted Upper Bound" at the end
        # combined_df = combined_df[["Date", "Predicted", "Actual", "Lower Trend", "Upper Trend"]]
        #
        # # Convert data types
        # combined_df = combined_df.astype(
        #     {"Date": str, "Predicted": int, "Actual": int, "Lower Trend": int, "Upper Trend": int})
        #
        # # Display the DataFrame
        # st.dataframe(combined_df, height=400, width=700)