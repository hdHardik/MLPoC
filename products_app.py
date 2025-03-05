import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

# Product Model Mapping
MODEL_PATHS = {
    "4A183": "product_4A183_model.pkl",
    "4A297": "product_4A297_model.pkl",
    "4A299": "product_4A299_model.pkl",
    "4A306": "product_4A306_model.pkl",
    "4A887": "product_4A887_model.pkl"
}

st.title("📊 Sales Forecasting App")
st.write(
    "Welcome to the Sales Forecasting App! This app uses pre-trained Prophet models to predict future sales for selected products.")


# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model_dict = pickle.load(file)
        model = model_dict["best_model"]
    return model


# Sidebar for User Input
st.sidebar.header("User Input")
product = st.sidebar.selectbox("Select Product", list(MODEL_PATHS.keys()))
future_weeks = st.sidebar.number_input("Enter number of future weeks to predict", min_value=1, max_value=52, value=12)

if st.sidebar.button("Predict Sales"):
    with st.spinner("Predicting..."):
        model = load_model(MODEL_PATHS[product])

        # Create future dataframe
        future = model.make_future_dataframe(periods=future_weeks, freq='W-MON')
        forecast = model.predict(future)
        st.success("Prediction Completed ✅")

        # Display Forecast Data
        st.subheader(f"Forecasted Sales Data for {product}")
        st.write(forecast[['ds', 'yhat']].tail(future_weeks))
        # st.write(pd.concat([forecast['ds'].tail(future_weeks),forecast['yhat'].tail(future_weeks)], axis=1))

        # Plot Forecast
        # st.subheader(f"Forecast Visualization for {product}")
        # plt.figure(figsize=(10, 5))
        # plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        # plt.xlabel("Date")
        # plt.ylabel("Sales")
        # plt.legend()
        # st.pyplot(plt)

        # Plot Forecast with Plotly
        st.subheader(f"Forecast Visualization for {product}")
        fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast for {product}", markers=True,  labels={'ds': 'Date', 'yhat': 'Sales'})
        fig.update_traces(marker=dict(size=8), hoverinfo='x+y')
        st.plotly_chart(fig)

st.write("Note: The model predicts future sales based on historical data trends.")
