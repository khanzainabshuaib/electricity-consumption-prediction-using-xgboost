from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "electricity_xgboost_model.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from HTML form
        global_reactive_power = float(request.form["Global_reactive_power"])
        voltage = float(request.form["Voltage"])
        global_intensity = float(request.form["Global_intensity"])
        sub_metering_1 = float(request.form["Sub_metering_1"])
        sub_metering_2 = float(request.form["Sub_metering_2"])
        sub_metering_3 = float(request.form["Sub_metering_3"])
        year = int(request.form["Year"])
        month = int(request.form["Month"])
        day = int(request.form["Day"])
        hour = int(request.form["Hour"])

        # Arrange data exactly as used during training
        features = np.array([[
            global_reactive_power,
            voltage,
            global_intensity,
            sub_metering_1,
            sub_metering_2,
            sub_metering_3,
            year,
            month,
            day,
            hour
        ]])

        # Prediction
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Electricity Consumption: {prediction:.3f} kW"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)