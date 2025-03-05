from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pyngrok import ngrok

app = Flask(__name__)

# Set Pandas to show scientific notation
pd.set_option("display.float_format", "{:.6e}".format)

# Load model
model_dict = joblib.load("overall_model.pkl")
model = model_dict["best_model"]
print("Model Loaded Successfully")

@app.route("/")
def home():
    return "Prophet Model API is Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        periods = data.get("periods", 12)
        freq = data.get("freq", "W")  # Weekly forecast by default

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Convert 'yhat' to scientific notation
        forecast["yhat"] = forecast["yhat"].apply(lambda x: "{:.6e}".format(x))


        result = forecast[["ds", "yhat"]].tail(periods).to_dict(orient="records")
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = 5000
    # ngrok_tunnel = ngrok.connect(addr='5000', proto="http")
    # print(f"Public URL: {ngrok_tunnel.public_url}")
    # app.run(host='0.0.0.0', port=5000)
    app.run(debug=True, port=port)