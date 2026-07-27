from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Loan Prediction.csv", encoding="latin1")

# Get unique values for dropdowns
professions = sorted(df["Profession"].unique())
cities = sorted(df["CITY"].unique())
states = sorted(df["STATE"].unique())


# Load trained model
model = load_model("loan_risk_ann.keras")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load label encoders
label_encoders = joblib.load("label_encoders.pkl")

@app.route("/")
def home():
    return render_template(
        "index.html",
        professions=professions,
        cities=cities,
        states=states
    )

@app.route("/predict", methods=["POST"])
def predict():

    print(request.form)

    # Get form data
    income = int(request.form["Income"])
    age = int(request.form["Age"])
    experience = int(request.form["Experience"])
    married = request.form["Married/Single"]
    house = request.form["House_Ownership"]
    car = request.form["Car_Ownership"]
    profession = request.form["Profession"]
    city = request.form["CITY"]
    state = request.form["STATE"]
    current_job = int(request.form["CURRENT_JOB_YRS"])
    current_house = int(request.form["CURRENT_HOUSE_YRS"])

    # Create DataFrame
    data = pd.DataFrame([{
        "Income": income,
        "Age": age,
        "Experience": experience,
        "Married/Single": married,
        "House_Ownership": house,
        "Car_Ownership": car,
        "Profession": profession,
        "CITY": city,
        "STATE": state,
        "CURRENT_JOB_YRS": current_job,
        "CURRENT_HOUSE_YRS": current_house
    }])

    # Encode
    categorical_cols = [
        "Married/Single",
        "House_Ownership",
        "Car_Ownership",
        "Profession",
        "CITY",
        "STATE"
    ]

    for col in categorical_cols:
        data[col] = label_encoders[col].transform(data[col])

    # Scale
    data_scaled = scaler.transform(data)

    # Predict
    probability = model.predict(data_scaled)[0][0]

    if probability >= 0.5:
        prediction = "High Risk"
    else:
        prediction = "Low Risk"

    confidence = round(probability * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        professions=professions,
        cities=cities,
        states=states
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)