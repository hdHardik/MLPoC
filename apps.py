import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the Prophet model locally
MODEL_PATH = "overall_model.pkl"

st.title("📊 Sales Forecasting App")
st.write("Welcome to the Sales Forecasting App! This app uses a pre-trained Prophet model to predict future sales.")

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model_dict = pickle.load(file)
        model = model_dict["best_model"]
    return model

model = load_model()

# Sidebar for User Input
st.sidebar.header("User Input")
future_weeks = st.sidebar.number_input("Enter number of future weeks to predict", min_value=1, max_value=52, value=12)

if st.sidebar.button("Predict Sales"):
    with st.spinner("Predicting..."):
        # Create future dataframe
        future = model.make_future_dataframe(periods=future_weeks, freq='W')
        forecast = model.predict(future)
        st.success("Prediction Completed ✅")


        # Display Forecast Data
        st.subheader("Forecasted Sales Data")
        st.write(pd.concat([forecast['ds'].tail(future_weeks),  forecast['yhat'].apply(lambda x: "{:.6e}".format(x)).tail(future_weeks)], axis=1))

        # Plot Forecast
        st.subheader("Forecast Visualization")
        plt.figure(figsize=(10, 5))
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        st.pyplot(plt)

st.write("Note: The model predicts future sales based on historical data trends.")
